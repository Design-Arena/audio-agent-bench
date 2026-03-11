import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from benchmarks.appointment_bench.system import system_instruction
from benchmarks.appointment_bench.turns import get_relevant_dimensions, turns as expected_turns
from src.audio_arena.judging.llm_judge import (
    build_judge_system_prompt,
    build_judge_user_prompt,
    format_turns_for_judge,
    load_transcript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated OpenAI judge evaluations at different reasoning efforts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--turns", type=str, default="9,23")
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--efforts", type=str, default="none,low")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def extract_kb_text() -> str:
    kb_marker = "### **KNOWLEDGE BASE**"
    tools_marker = "### **AVAILABLE TOOLS**"
    start = system_instruction.index(kb_marker) + len(kb_marker)
    end = system_instruction.index(tools_marker)
    return system_instruction[start:end].strip()


def build_prompt(run_dir: Path, target_turns: list[int]) -> tuple[str, str]:
    records = load_transcript(run_dir)
    selected_records = [record for record in records if record["turn"] in set(target_turns)]
    formatted_turns = format_turns_for_judge(
        selected_records,
        expected_turns,
        only_turns=set(target_turns),
        turn_taking_data=None,
        get_relevant_dimensions_fn=get_relevant_dimensions,
        kb_text=extract_kb_text(),
    )
    user_prompt = build_judge_user_prompt(
        formatted_turns=formatted_turns,
        turn_numbers=target_turns,
        cross_turn_realignment=False,
    )
    system_prompt = build_judge_system_prompt(cross_turn_realignment=False)
    return system_prompt, user_prompt


async def judge_once(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str,
    attempt_index: int,
) -> dict[str, Any]:
    max_attempts = 6
    for retry_index in range(max_attempts):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=reasoning_effort,
            )
            response_text = response.choices[0].message.content or ""
            parsed = json.loads(response_text)
            return {
                "attempt_index": attempt_index,
                "reasoning_effort": reasoning_effort,
                "response_text": response_text,
                "usage": response.usage.model_dump() if response.usage else {},
                "parsed": parsed,
            }
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "rate limit" not in message.lower() and "429" not in message:
                raise
            if retry_index == max_attempts - 1:
                raise
            time.sleep(2 ** retry_index)
    raise RuntimeError("Unreachable retry loop")


def judgment_rows(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        final_judgments = result["parsed"].get("final_judgments", [])
        for judgment in final_judgments:
            rows.append(
                {
                    "attempt_index": result["attempt_index"],
                    "reasoning_effort": result["reasoning_effort"],
                    "turn": judgment["turn"],
                    "instruction_following": judgment.get("instruction_following"),
                    "kb_grounding": judgment.get("kb_grounding"),
                    "tool_use_correct": judgment.get("tool_use_correct"),
                    "ambiguity_handling": judgment.get("ambiguity_handling"),
                    "state_tracking": judgment.get("state_tracking"),
                    "reasoning": judgment.get("reasoning", ""),
                    "prompt_tokens": result["usage"].get("prompt_tokens"),
                    "completion_tokens": result["usage"].get("completion_tokens"),
                    "total_tokens": result["usage"].get("total_tokens"),
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {key: str(value) for key, value in df.dtypes.items()},
        "missingness": {key: int(value) for key, value in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "per_effort": {},
    }

    for effort, effort_df in df.groupby("reasoning_effort"):
        per_turn: dict[int, Any] = {}
        for turn, turn_df in effort_df.groupby("turn"):
            fail_count = int((~turn_df["kb_grounding"].fillna(False)).sum())
            pass_count = int(turn_df["kb_grounding"].fillna(False).sum())
            reason_counter = Counter(turn_df["reasoning"].tolist())
            state_true_count = int(turn_df["state_tracking"].eq(True).sum())
            per_turn[int(turn)] = {
                "kb_grounding_passes": pass_count,
                "kb_grounding_fails": fail_count,
                "instruction_following_passes": int(turn_df["instruction_following"].fillna(False).sum()),
                "state_tracking_passes": state_true_count if "state_tracking" in turn_df else None,
                "top_reasonings": reason_counter.most_common(3),
                "avg_total_tokens": float(turn_df["total_tokens"].dropna().mean()),
            }
        summary["per_effort"][effort] = per_turn
    return summary


def write_markdown(
    output_path: Path,
    run_dir: Path,
    model: str,
    repeats: int,
    target_turns: list[int],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# OpenAI Judge Effort Experiment",
        "",
        f"- Run: `{run_dir}`",
        f"- Model: `{model}`",
        f"- Repeats per effort: `{repeats}`",
        f"- Turns: `{', '.join(str(turn) for turn in target_turns)}`",
        "",
        "## Data Sanity",
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Duplicate rows: `{summary['duplicate_rows']}`",
        f"- Missingness: `{json.dumps(summary['missingness'], sort_keys=True)}`",
        "",
        "## Findings",
        "",
    ]

    for effort, per_turn in summary["per_effort"].items():
        lines.append(f"### `{effort}`")
        lines.append("")
        for turn, turn_summary in per_turn.items():
            lines.append(
                f"- Turn `{turn}`: kb pass `{turn_summary['kb_grounding_passes']}/{repeats}`, "
                f"kb fail `{turn_summary['kb_grounding_fails']}/{repeats}`, "
                f"instruction pass `{turn_summary['instruction_following_passes']}/{repeats}`, "
                f"avg total tokens `{turn_summary['avg_total_tokens']:.1f}`"
            )
            for reasoning_text, count in turn_summary["top_reasonings"]:
                compact = " ".join(str(reasoning_text).split())
                if len(compact) > 240:
                    compact = compact[:237] + "..."
                lines.append(f"  Reason x{count}: {compact}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    args = parse_args()
    target_turns = [int(part) for part in args.turns.split(",") if part.strip()]
    efforts = [part.strip() for part in args.efforts.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, user_prompt = build_prompt(args.run_dir, target_turns)
    client = AsyncOpenAI()

    raw_results: list[dict[str, Any]] = []
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    for effort in efforts:
        for attempt_index in range(args.repeats):
            result = await judge_once(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                reasoning_effort=effort,
                attempt_index=attempt_index,
            )
            raw_results.append(result)
            raw_path = raw_dir / f"{effort}_attempt_{attempt_index:02d}.json"
            raw_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    df = pd.DataFrame(judgment_rows(raw_results)).sort_values(
        ["reasoning_effort", "attempt_index", "turn"]
    ).reset_index(drop=True)
    summary = summarize(df)

    df.to_csv(args.output_dir / "judge_effort_results.csv", index=False)
    (args.output_dir / "judge_effort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        output_path=args.output_dir / "judge_effort_summary.md",
        run_dir=args.run_dir,
        model=args.model,
        repeats=args.repeats,
        target_turns=target_turns,
        summary=summary,
    )


if __name__ == "__main__":
    asyncio.run(main())
