import argparse
import base64
import html
import copy
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORE_COLUMNS = [
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
    "ambiguity_handling",
    "state_tracking",
]

DISPLAY_DIMENSIONS = {
    "tool_use_correct": "Tool Use",
    "instruction_following": "Instruction",
    "kb_grounding": "KB Grounding",
    "ambiguity_handling": "Ambiguity",
    "state_tracking": "State",
}

REQUIRED_BATCH_FILES = [
    "all_runs.csv",
    "overall_summary.csv",
    "latency_summary.csv",
    "grader_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained HTML review page for a conversation-bench "
            "audio repeat batch."
        )
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help=(
            "Batch results directory. Defaults to the latest "
            "results/conversation_bench_audio_repeats_* directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path. Defaults to <batch-dir>/review.html.",
    )
    return parser.parse_args()


def find_default_batch_dir() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "results").glob("conversation_bench_audio_repeats_*")
    )
    if not candidates:
        raise FileNotFoundError(
            "No results/conversation_bench_audio_repeats_* directories were found."
        )
    return candidates[-1].resolve()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_benchmark_turns(benchmark_name: str) -> list[dict]:
    turns_path = PROJECT_ROOT / "benchmarks" / benchmark_name / "turns.py"
    if not turns_path.exists():
        raise FileNotFoundError(f"Benchmark turns file not found: {turns_path}")

    spec = importlib.util.spec_from_file_location(
        f"{benchmark_name}_turns_module",
        turns_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load benchmark turns from {turns_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.turns


def normalize_tool_entries(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def make_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def format_golden_context_block(turn_number: int, benchmark_turn: dict) -> str:
    lines = [
        f"Turn {turn_number}",
        f"User: {benchmark_turn.get('input', '')}",
    ]

    tool_calls = normalize_tool_entries(benchmark_turn.get("required_function_call"))
    tool_results = normalize_tool_entries(benchmark_turn.get("function_call_response"))

    for index, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name", "unknown_tool")
        tool_args = json.dumps(tool_call.get("args", {}), ensure_ascii=False)
        lines.append(f"  [Tool call: {tool_name}({tool_args})]")
        if index < len(tool_results):
            lines.append(
                f"  [Tool result: {json.dumps(tool_results[index], ensure_ascii=False)}]"
            )

    lines.append(f"Assistant: {benchmark_turn.get('golden_text', '')}")
    return "\n".join(lines)


def path_to_file_url(path: Path) -> str:
    return path.resolve().as_uri()


def turn_audio_path(benchmark_name: str, turn: int) -> Path:
    return PROJECT_ROOT / "benchmarks" / benchmark_name / "audio" / f"turn_{turn:03d}.wav"


def coerce_percent_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    converted = dataframe.copy()
    for column in converted.columns:
        if column == "Audio" or column == "Failed Turns By Repeat":
            continue
        try:
            converted[column] = pd.to_numeric(converted[column])
        except (TypeError, ValueError):
            continue
    return converted


def build_record(raw_record: dict, run_info: dict, benchmark_turn: dict, benchmark_name: str) -> dict:
    scores = raw_record.get("scores", {})
    normalized_scores = {column: scores.get(column) for column in SCORE_COLUMNS}
    failed_dimensions = [
        column for column, value in normalized_scores.items() if value is False
    ]
    completed_dimensions = [
        column for column, value in normalized_scores.items() if value is not None
    ]
    assistant_text = raw_record.get("assistant_text", "")
    response_status = (
        "empty_response"
        if assistant_text.startswith("[EMPTY_RESPONSE")
        else "normal"
    )
    if not completed_dimensions:
        overall_status = "incomplete"
    elif failed_dimensions:
        overall_status = "fail"
    else:
        overall_status = "pass"

    turn_number = int(raw_record.get("turn"))
    audio_path = turn_audio_path(benchmark_name, turn_number)

    tool_calls = raw_record.get("tool_calls", [])
    tool_results = raw_record.get("tool_results", [])
    return {
        "row_id": f"{run_info['audio_label']}|r{run_info['repeat']}|t{turn_number}",
        "audio_label": run_info["audio_label"],
        "repeat": int(run_info["repeat"]),
        "run_id": run_info["run_id"],
        "run_dir": run_info["run_dir"],
        "run_dir_url": path_to_file_url(Path(run_info["run_dir"])),
        "runtime_json_url": path_to_file_url(Path(run_info["run_dir"]) / "runtime.json"),
        "summary_json_url": path_to_file_url(
            Path(run_info["run_dir"]) / "openai_summary.json"
        ),
        "judged_jsonl_url": path_to_file_url(
            Path(run_info["run_dir"]) / "openai_judged.jsonl"
        ),
        "turn": turn_number,
        "timestamp": raw_record.get("ts"),
        "model_name": raw_record.get("model_name"),
        "user_text": raw_record.get("user_text", ""),
        "assistant_text": assistant_text,
        "latency_ms": raw_record.get("latency_ms"),
        "ttfb_ms": raw_record.get("ttfb_ms"),
        "reconnection_count": raw_record.get("reconnection_count"),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "tool_calls_text": json.dumps(tool_calls, ensure_ascii=False, indent=2),
        "tool_results_text": json.dumps(tool_results, ensure_ascii=False, indent=2),
        "judge_reasoning": raw_record.get("judge_reasoning", ""),
        "scores": normalized_scores,
        "failed_dimensions": failed_dimensions,
        "failed_dimensions_text": ", ".join(failed_dimensions),
        "completed_dimensions": completed_dimensions,
        "response_status": response_status,
        "overall_status": overall_status,
        "failure_count": len(failed_dimensions),
        "failed_turns_for_run": run_info["failed_turns"],
        "golden_user_text": benchmark_turn.get("input", ""),
        "golden_assistant_text": benchmark_turn.get("golden_text", ""),
        "golden_tool_calls": normalize_tool_entries(
            benchmark_turn.get("required_function_call")
        ),
        "golden_tool_results": normalize_tool_entries(
            benchmark_turn.get("function_call_response")
        ),
        "golden_tool_calls_text": json.dumps(
            normalize_tool_entries(benchmark_turn.get("required_function_call")),
            ensure_ascii=False,
            indent=2,
        ),
        "golden_tool_results_text": json.dumps(
            normalize_tool_entries(benchmark_turn.get("function_call_response")),
            ensure_ascii=False,
            indent=2,
        ),
        "golden_context_block": format_golden_context_block(turn_number, benchmark_turn),
        "user_audio_path": str(audio_path),
        "user_audio_url": path_to_file_url(audio_path),
        "user_audio_exists": audio_path.exists(),
    }


def build_sanity_report(flattened_rows: pd.DataFrame) -> dict:
    duplicate_keys = int(
        flattened_rows[["audio_label", "repeat", "turn"]].duplicated().sum()
    )
    missingness = {}
    for column in [
        "audio_label",
        "repeat",
        "turn",
        "timestamp",
        "user_text",
        "assistant_text",
        "latency_ms",
        "judge_reasoning",
    ]:
        missingness[column] = int(flattened_rows[column].isna().sum())

    dtypes = {
        key: str(value)
        for key, value in flattened_rows.dtypes.astype(str).to_dict().items()
    }

    return {
        "shape": [int(flattened_rows.shape[0]), int(flattened_rows.shape[1])],
        "duplicate_primary_keys": duplicate_keys,
        "missingness": missingness,
        "dtypes": dtypes,
    }


def encode_image_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def badge_html(status: str, label: str | None = None) -> str:
    text = html.escape(label or status)
    class_name = {
        "pass": "badge badge-pass",
        "fail": "badge badge-fail",
    }.get(status, "badge badge-incomplete")
    return f'<span class="{class_name}">{text}</span>'


def score_status(value: object) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "incomplete"


def sort_rows(rows: list[dict]) -> list[dict]:
    status_order = {"fail": 0, "incomplete": 1, "pass": 2}
    return sorted(
        rows,
        key=lambda row: (
            status_order[row["overall_status"]],
            -row["failure_count"],
            row["audio_label"],
            row["repeat"],
            row["turn"],
        ),
    )


def render_hero_metadata(payload: dict) -> str:
    entries = [
        ("Batch", payload["metadata"]["batch_name"]),
        ("Model", payload["metadata"]["model_name"]),
        ("Judge", payload["metadata"]["judge_model"]),
        ("Judge Version", payload["metadata"]["judge_version"]),
        ("Generated", payload["metadata"]["generated_at"]),
        ("Rows", str(payload["metadata"]["row_count"])),
        ("Runs", str(payload["metadata"]["run_count"])),
        ("Batch Dir", payload["metadata"]["batch_dir"]),
    ]
    cards = []
    for label, value in entries:
        cards.append(
            f"""
        <div class="hero-card">
          <div class="label">{html.escape(label)}</div>
          <div class="value mono">{html.escape(value)}</div>
        </div>
"""
        )
    return "".join(cards)


def render_metrics_cards(payload: dict) -> str:
    cards = []
    for card in payload["summary_cards"]:
        cards.append(
            f"""
        <div class="metric-card">
          <div class="label">{html.escape(card["label"])}</div>
          <div class="metric-number">{html.escape(card["value"])}</div>
          <div class="metric-note subtle">{html.escape(card["note"])}</div>
        </div>
"""
        )
    return "".join(cards)


def render_overall_summary_rows(payload: dict) -> str:
    rows = []
    for row in payload["overall_summary"]:
        rows.append(
            f"""
        <tr>
          <td>{html.escape(row["Audio"])}</td>
          <td>{float(row["Pass Rows Avg"]):.2f}</td>
          <td>{float(row["Fail Rows Avg"]):.2f}</td>
          <td>{float(row["Pass Rate Avg"]):.2f}%</td>
          <td>{float(row["Empty Resp Avg"]):.2f}</td>
          <td>{float(row["End Session Avg"]):.2f}</td>
          <td class="mono">{html.escape(row["Failed Turns By Repeat"])}</td>
        </tr>
"""
        )
    return "".join(rows)


def render_latency_summary_rows(payload: dict) -> str:
    rows = []
    for row in payload["latency_summary"]:
        rows.append(
            f"""
        <tr>
          <td>{html.escape(row["Audio"])}</td>
          <td>{float(row["Latency Mean"]):.2f} ms</td>
          <td>{float(row["Latency p50"]):.2f} ms</td>
          <td>{float(row["Latency p90"]):.2f} ms</td>
          <td>{float(row["Latency p95"]):.2f} ms</td>
          <td>{float(row["Latency p99"]):.2f} ms</td>
        </tr>
"""
        )
    return "".join(rows)


def render_grader_summary_rows(payload: dict) -> str:
    rows = []
    for row in payload["grader_summary"]:
        rows.append(
            f"""
        <tr>
          <td>{html.escape(row["Audio"])}</td>
          <td>{float(row["tool_use_correct"]):.2f}%</td>
          <td>{float(row["instruction_following"]):.2f}%</td>
          <td>{float(row["kb_grounding"]):.2f}%</td>
          <td>{float(row["ambiguity_handling"]):.2f}%</td>
          <td>{float(row["state_tracking"]):.2f}%</td>
        </tr>
"""
        )
    return "".join(rows)


def render_run_summary_rows(payload: dict, share_safe: bool) -> str:
    runs = sorted(
        payload["runs"],
        key=lambda row: (
            row["pass_row_rate"],
            row["audio_label"],
            row["repeat"],
        ),
    )
    rendered_rows = []
    for row in runs:
        artifacts_html = (
            '<span class="subtle">local-only artifacts omitted in share-safe export</span>'
            if share_safe
            else (
                f'<a href="{html.escape(row["run_dir_url"])}" target="_blank" rel="noreferrer">run</a> · '
                f'<a href="{html.escape(row["runtime_json_url"])}" target="_blank" rel="noreferrer">runtime</a> · '
                f'<a href="{html.escape(row["summary_json_url"])}" target="_blank" rel="noreferrer">summary</a> · '
                f'<a href="{html.escape(row["judged_jsonl_url"])}" target="_blank" rel="noreferrer">judged</a>'
            )
        )
        rendered_rows.append(
            f"""
        <tr>
          <td>{html.escape(row["audio_label"])}</td>
          <td>{row["repeat"]}</td>
          <td class="mono">{html.escape(row["run_id"])}</td>
          <td>{row["pass_row_rate"]:.2f}%</td>
          <td>{row["pass_rows"]}</td>
          <td>{row["fail_rows"]}</td>
          <td>{row["latency_ms_p95"]:.2f} ms</td>
          <td class="mono">{html.escape(row["failed_turns_text"])}</td>
          <td>{artifacts_html}</td>
        </tr>
"""
        )
    return "".join(rendered_rows)


def render_table_row(row: dict, is_selected: bool) -> str:
    selected_class = " selected" if is_selected else ""
    source = f"{row['audio_label']} · r{row['repeat']}"
    response_badge = badge_html(
        "fail" if row["response_status"] == "empty_response" else "pass",
        row["response_status"].replace("_", " "),
    )
    latency = "n/a" if row["latency_ms"] is None else f"{row['latency_ms']} ms"
    return f"""
        <tr data-row-id="{html.escape(row["row_id"])}" class="{selected_class.strip()}" onclick="if (window.__reviewSelectRow) window.__reviewSelectRow(this.getAttribute('data-row-id'))">
          <td class="mono column-source">{html.escape(source)}</td>
          <td class="mono column-turn">{row["turn"]}</td>
          <td class="column-status">{badge_html(row["overall_status"])}</td>
          <td class="column-response">{response_badge}</td>
          <td class="mono column-latency">{html.escape(latency)}</td>
          <td class="column-failed">{html.escape(row["failed_dimensions_text"] or "none")}</td>
          <td class="column-user-prompt">{html.escape(row["user_text"])}</td>
          <td class="column-assistant-output">{html.escape(row["assistant_text"])}</td>
        </tr>
"""


def render_curated_rows(rows: list[dict], selected_row_id: str | None) -> tuple[str, str]:
    failed_rows = [row for row in rows if row["overall_status"] == "fail"]
    curated_rows = failed_rows[:30]
    rendered = []
    for row in curated_rows:
        is_selected = row["row_id"] == selected_row_id
        selected_class = "selected" if is_selected else ""
        source = f"{row['audio_label']} · r{row['repeat']}"
        rendered.append(
            f"""
        <tr data-row-id="{html.escape(row["row_id"])}" class="{selected_class}" onclick="if (window.__reviewSelectRow) window.__reviewSelectRow(this.getAttribute('data-row-id'))">
          <td class="mono column-source">{html.escape(source)}</td>
          <td class="mono column-turn">{row["turn"]}</td>
          <td class="column-status">{badge_html(row["overall_status"])}</td>
          <td class="column-failed">{html.escape(row["failed_dimensions_text"] or "none")}</td>
          <td class="column-user-prompt">{html.escape(row["user_text"])}</td>
          <td class="column-assistant-output">{html.escape(row["assistant_text"])}</td>
        </tr>
"""
        )
    count_text = f"Showing {len(curated_rows)} of {len(failed_rows)} failing rows after filters."
    return "".join(rendered), count_text


def render_full_rows(rows: list[dict], selected_row_id: str | None, all_count: int) -> tuple[str, str]:
    rendered = [render_table_row(row, row["row_id"] == selected_row_id) for row in rows]
    count_text = f"Showing {len(rows)} of {all_count} rows. Failures are sorted first."
    return "".join(rendered), count_text


def render_detail_panel(
    rows: list[dict],
    selected_row_id: str | None,
    *,
    share_safe: bool,
) -> str:
    if not rows:
        return '<p class="subtle">No rows match the current filters.</p>'

    row = next((item for item in rows if item["row_id"] == selected_row_id), rows[0])
    previous_context = "\n\n".join(
        item["golden_context_block"]
        for item in sorted(
            (
                candidate
                for candidate in rows
                if candidate["audio_label"] == row["audio_label"]
                and candidate["repeat"] == row["repeat"]
                and candidate["turn"] < row["turn"]
            ),
            key=lambda item: item["turn"],
        )
    )
    grader_rows = []
    for dimension, value in row["scores"].items():
        status = score_status(value)
        reason = "Not applicable on this turn."
        if value is False:
            reason = "Failed. See the turn-level judge reasoning for decisive evidence."
        elif value is True:
            reason = "Passed."
        grader_rows.append(
            f"""
          <tr>
            <td class="mono" style="width: 220px;">{html.escape(dimension)}</td>
            <td style="width: 140px;">{badge_html(status, "n/a" if value is None else status)}</td>
            <td>{html.escape(reason)}</td>
          </tr>
"""
        )

    if share_safe:
        audio_html = (
            '<div class="subtle" style="margin-top: 8px;">Audio is omitted in the share-safe export.</div>'
        )
        artifacts_html = (
            '<div class="subtle">Local artifact links are omitted in the share-safe export.</div>'
        )
    else:
        audio_html = (
            f'<audio controls preload="none" src="{html.escape(row["user_audio_url"])}"></audio>'
            if row["user_audio_exists"]
            else '<div class="subtle" style="margin-top: 8px;">No per-turn input audio file found.</div>'
        )
        artifacts_html = f"""
                  <a href="{html.escape(row["run_dir_url"])}" target="_blank" rel="noreferrer">run directory</a><br>
                  <a href="{html.escape(row["runtime_json_url"])}" target="_blank" rel="noreferrer">runtime.json</a><br>
                  <a href="{html.escape(row["summary_json_url"])}" target="_blank" rel="noreferrer">openai_summary.json</a><br>
                  <a href="{html.escape(row["judged_jsonl_url"])}" target="_blank" rel="noreferrer">openai_judged.jsonl</a>
"""
    response_badges = (
        f"{badge_html(row['overall_status'])} "
        f"{badge_html('fail' if row['response_status'] == 'empty_response' else 'pass', row['response_status'].replace('_', ' '))}"
    )
    failed_turns_text = ", ".join(str(value) for value in row["failed_turns_for_run"]) or "none"
    compact_latency = "n/a" if row["latency_ms"] is None else f"{row['latency_ms']}ms"
    compact_ttfb = "n/a" if row["ttfb_ms"] is None else f"{row['ttfb_ms']}ms"
    return f"""
        <div class="detail-panel">
          <div class="detail-meta">
            <div class="hero-card">
              <div class="label">Source</div>
              <div class="value mono">{html.escape(f"{row['audio_label']} · repeat {row['repeat']}")}</div>
            </div>
            <div class="hero-card">
              <div class="label">Identifiers</div>
              <div class="value mono">turn={row["turn"]} · ts={html.escape(str(row["timestamp"]))} · run={html.escape(row["run_id"])}</div>
            </div>
            <div class="hero-card">
              <div class="label">Response Status</div>
              <div class="value">{response_badges}</div>
            </div>
          </div>
          <div class="detail-layout">
            <div class="stack">
              <div class="text-card">
                <div class="label">Current Turn Audio</div>
                <div class="subtle">Benchmark input audio for the selected user turn.</div>
                {audio_html}
              </div>
              <div class="text-card">
                <div class="label">Gold Previous Turns Context</div>
                <div class="scroll-box context-box">{html.escape(previous_context or "No previous turns.")}</div>
              </div>
              <div class="text-card">
                <div class="label">User Query</div>
                <div class="scroll-box short-box">{html.escape(row["user_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Model Output</div>
                <div class="scroll-box tall-box">{html.escape(row["assistant_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Judge Reasoning</div>
                <div class="scroll-box reason-box">{html.escape(row["judge_reasoning"] or "No reasoning provided.")}</div>
              </div>
            </div>
            <div class="stack">
              <div class="text-card">
                <div class="label">Compact Metadata</div>
                <div class="value mono">
                  latency={html.escape(compact_latency)}<br>
                  ttfb={html.escape(compact_ttfb)}<br>
                  reconnections={html.escape(str(row["reconnection_count"]))}<br>
                  failed={html.escape(row["failed_dimensions_text"] or "none")}<br>
                  run_failed_turns={html.escape(failed_turns_text)}
                </div>
              </div>
              <div class="text-card">
                <div class="label">Per-Dimension Verdicts</div>
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th style="width: 220px;">Dimension</th>
                        <th style="width: 140px;">Verdict</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>{''.join(grader_rows)}</tbody>
                  </table>
                </div>
              </div>
              <div class="text-card">
                <div class="label">Expected Gold Response</div>
                <div class="scroll-box short-box">{html.escape(row["golden_assistant_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Tool Calls</div>
                <div class="scroll-box short-box mono">{html.escape(row["tool_calls_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Tool Results</div>
                <div class="scroll-box short-box mono">{html.escape(row["tool_results_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Expected Tool Calls / Results</div>
                <div class="scroll-box short-box mono">{html.escape(row["golden_tool_calls_text"] + "\n\n" + row["golden_tool_results_text"])}</div>
              </div>
              <div class="text-card">
                <div class="label">Local Artifacts</div>
                <div class="value">
                  {artifacts_html}
                </div>
              </div>
            </div>
          </div>
        </div>
"""


def build_batch_payload(batch_dir: Path) -> tuple[dict, pd.DataFrame]:
    batch_dir = batch_dir.resolve()
    missing = [
        str(batch_dir / name)
        for name in REQUIRED_BATCH_FILES
        if not (batch_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required batch artifacts:\n" + "\n".join(missing)
        )

    all_runs_df = pd.read_csv(batch_dir / "all_runs.csv")
    overall_summary_df = coerce_percent_columns(pd.read_csv(batch_dir / "overall_summary.csv"))
    latency_summary_df = coerce_percent_columns(pd.read_csv(batch_dir / "latency_summary.csv"))
    grader_summary_df = coerce_percent_columns(pd.read_csv(batch_dir / "grader_summary.csv"))

    benchmark_names = {
        Path(run_dir).resolve().parent.name for run_dir in all_runs_df["run_dir"].tolist()
    }
    if len(benchmark_names) != 1:
        raise ValueError(f"Expected one benchmark in batch, found: {sorted(benchmark_names)}")
    benchmark_name = next(iter(benchmark_names))
    benchmark_turns = load_benchmark_turns(benchmark_name)

    all_rows: list[dict] = []
    run_payload_rows: list[dict] = []
    judge_versions: set[str] = set()
    judge_models: set[str] = set()
    model_names: set[str] = set()
    total_failed_api_turns = 0

    for run_info in all_runs_df.sort_values(["audio_label", "repeat"]).to_dict("records"):
        run_dir = Path(run_info["run_dir"]).resolve()
        judged_path = run_dir / "openai_judged.jsonl"
        summary_path = run_dir / "openai_summary.json"
        runtime_path = run_dir / "runtime.json"
        if not judged_path.exists() or not summary_path.exists() or not runtime_path.exists():
            raise FileNotFoundError(
                f"Missing run artifacts for {run_info['run_id']} in {run_dir}"
            )

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        judged_rows = load_jsonl(judged_path)
        if not judged_rows:
            raise ValueError(f"No judged rows found in {judged_path}")

        judge_versions.add(summary_payload.get("judge_version", "unknown"))
        judge_models.add(summary_payload.get("judge_model", "unknown"))
        model_names.add(summary_payload.get("model_name", "unknown"))

        failed_turns = json.loads(run_info.get("failed_turns", "[]"))
        total_failed_api_turns += len(failed_turns)

        run_payload_rows.append(
            {
                "audio_label": run_info["audio_label"],
                "repeat": int(run_info["repeat"]),
                "run_id": run_info["run_id"],
                "run_dir": str(run_dir),
                "run_dir_url": path_to_file_url(run_dir),
                "runtime_json_url": path_to_file_url(runtime_path),
                "summary_json_url": path_to_file_url(summary_path),
                "judged_jsonl_url": path_to_file_url(judged_path),
                "rows": int(len(judged_rows)),
                "failed_turns": failed_turns,
                "failed_turns_text": ", ".join(str(value) for value in failed_turns) or "none",
                "pass_rows": int(run_info["pass_rows"]),
                "fail_rows": int(run_info["fail_rows"]),
                "incomplete_rows": int(run_info["incomplete_rows"]),
                "pass_row_rate": round(float(run_info["pass_row_rate"]) * 100.0, 2),
                "empty_response_count": int(run_info["empty_response_count"]),
                "model_ended_session_count": int(run_info["model_ended_session_count"]),
                "latency_ms_mean": round(float(run_info["latency_ms_mean"]), 2),
                "latency_ms_p50": round(float(run_info["latency_ms_p50"]), 2),
                "latency_ms_p90": round(float(run_info["latency_ms_p90"]), 2),
                "latency_ms_p95": round(float(run_info["latency_ms_p95"]), 2),
                "latency_ms_p99": round(float(run_info["latency_ms_p99"]), 2),
                "tool_use_correct_pass_rate": round(
                    float(run_info["tool_use_correct_pass_rate"]) * 100.0, 2
                ),
                "instruction_following_pass_rate": round(
                    float(run_info["instruction_following_pass_rate"]) * 100.0, 2
                ),
                "kb_grounding_pass_rate": round(
                    float(run_info["kb_grounding_pass_rate"]) * 100.0, 2
                ),
                "ambiguity_handling_pass_rate": round(
                    float(run_info["ambiguity_handling_pass_rate"]) * 100.0, 2
                ),
                "state_tracking_pass_rate": round(
                    float(run_info["state_tracking_pass_rate"]) * 100.0, 2
                ),
                "judge_version": summary_payload.get("judge_version"),
                "judge_model": summary_payload.get("judge_model"),
                "mode": runtime_payload.get("mode"),
                "parallel": runtime_payload.get("parallel"),
                "disable_vad": runtime_payload.get("disable_vad"),
                "audio_source": runtime_payload.get("audio_source"),
            }
        )

        enriched_run_info = {
            "audio_label": run_info["audio_label"],
            "repeat": int(run_info["repeat"]),
            "run_id": run_info["run_id"],
            "run_dir": str(run_dir),
            "failed_turns": failed_turns,
        }
        for raw_record in judged_rows:
            turn_number = int(raw_record["turn"])
            benchmark_turn = benchmark_turns[turn_number]
            all_rows.append(
                build_record(raw_record, enriched_run_info, benchmark_turn, benchmark_name)
            )

    flattened_rows = pd.DataFrame(all_rows).sort_values(
        ["audio_label", "repeat", "turn"]
    ).reset_index(drop=True)
    sanity_report = build_sanity_report(flattened_rows)
    if sanity_report["duplicate_primary_keys"]:
        raise ValueError(
            "Duplicate (audio_label, repeat, turn) keys found: "
            f"{sanity_report['duplicate_primary_keys']}"
        )

    overview_records = overall_summary_df.to_dict(orient="records")
    latency_records = latency_summary_df.to_dict(orient="records")
    grader_records: list[dict] = []
    for row in grader_summary_df.to_dict(orient="records"):
        grader_records.append(
            {
                "Audio": row["Audio"],
                "tool_use_correct": row["tool_use_correct"],
                "instruction_following": row["instruction_following"],
                "kb_grounding": row["kb_grounding"],
                "ambiguity_handling": row["ambiguity_handling"],
                "state_tracking": row["state_tracking"],
            }
        )

    payload = {
        "metadata": {
            "batch_name": batch_dir.name,
            "batch_dir": str(batch_dir),
            "benchmark_name": benchmark_name,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            ),
            "model_name": ", ".join(sorted(model_names)),
            "judge_model": ", ".join(sorted(judge_models)),
            "judge_version": ", ".join(sorted(judge_versions)),
            "audio_conditions": int(all_runs_df["audio_label"].nunique()),
            "repeats": int(all_runs_df["repeat"].nunique()),
            "run_count": int(len(all_runs_df)),
            "row_count": int(len(flattened_rows)),
            "total_failed_api_turns": int(total_failed_api_turns),
            "source_artifacts": [str(batch_dir / name) for name in REQUIRED_BATCH_FILES],
        },
        "chart_data_url": encode_image_data_url(batch_dir / "metrics_avg_barplot.png"),
        "summary_cards": [
            {
                "label": "Audio Conditions",
                "value": str(int(all_runs_df["audio_label"].nunique())),
                "note": "Synthetic plus two real-audio conditions.",
            },
            {
                "label": "Repeats",
                "value": str(int(all_runs_df["repeat"].nunique())),
                "note": "Three repeats per condition.",
            },
            {
                "label": "Turn Rows",
                "value": str(int(len(flattened_rows))),
                "note": "All judged turn rows across the nine runs.",
            },
            {
                "label": "API Runtime Failures",
                "value": str(int(total_failed_api_turns)),
                "note": "Turns finalized as explicit failed or empty-response runtime rows.",
            },
        ],
        "overall_summary": overview_records,
        "latency_summary": latency_records,
        "grader_summary": grader_records,
        "runs": run_payload_rows,
        "sanity_checks": sanity_report,
        "rows": flattened_rows.to_dict(orient="records"),
    }
    return make_json_safe(payload), flattened_rows


def build_share_safe_payload(payload: dict) -> dict:
    shared = copy.deepcopy(payload)
    shared["metadata"]["batch_dir"] = "omitted in share-safe export"
    shared["metadata"]["source_artifacts"] = ["embedded in HTML"]

    for run in shared["runs"]:
        run["run_dir"] = ""
        run["run_dir_url"] = ""
        run["runtime_json_url"] = ""
        run["summary_json_url"] = ""
        run["judged_jsonl_url"] = ""

    for row in shared["rows"]:
        row["run_dir"] = ""
        row["run_dir_url"] = ""
        row["runtime_json_url"] = ""
        row["summary_json_url"] = ""
        row["judged_jsonl_url"] = ""
        row["user_audio_path"] = ""
        row["user_audio_url"] = ""
        row["user_audio_exists"] = False

    return shared


def build_client_script() -> str:
    return """
    (function () {
    function setBootError(message) {
      var el = document.getElementById("js-status");
      if (!el) {
        return;
      }
      el.style.display = "block";
      el.className = "status-banner status-banner-error";
      el.textContent = "Interactive mode failed to initialize: " + message;
    }

    try {
    var payloadElement = document.getElementById("payload");
    if (!payloadElement) {
      throw new Error("Missing payload script element.");
    }

    var payload = JSON.parse(payloadElement.textContent);
    var allRows = payload.rows.slice();
    var audioFilter = document.getElementById("audio-filter");
    var repeatFilter = document.getElementById("repeat-filter");
    var dimensionFilter = document.getElementById("dimension-filter");
    var statusFilter = document.getElementById("status-filter");
    var searchFilter = document.getElementById("search-filter");
    var curatedBody = document.getElementById("curated-body");
    var fullBody = document.getElementById("full-body");
    var curatedCount = document.getElementById("curated-count");
    var fullCount = document.getElementById("full-count");
    var detailRoot = document.getElementById("detail-root");
    var jsStatus = document.getElementById("js-status");

    var selectedRowId = allRows.length ? allRows[0].row_id : null;

    function escapeHtml(text) {
      var value = text == null ? "" : String(text);
      return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function badgeClass(status) {
      if (status === "pass") {
        return "badge badge-pass";
      }
      if (status === "fail") {
        return "badge badge-fail";
      }
      return "badge badge-incomplete";
    }

    function badge(status, label) {
      var text = label || status;
      return '<span class="' + badgeClass(status) + '">' + escapeHtml(text) + "</span>";
    }

    function scoreStatus(value) {
      if (value === true) {
        return "pass";
      }
      if (value === false) {
        return "fail";
      }
      return "incomplete";
    }

    function compareRows(left, right) {
      var statusOrder = { fail: 0, incomplete: 1, pass: 2 };
      if (left.overall_status !== right.overall_status) {
        return statusOrder[left.overall_status] - statusOrder[right.overall_status];
      }
      if (left.failure_count !== right.failure_count) {
        return right.failure_count - left.failure_count;
      }
      if (left.audio_label !== right.audio_label) {
        return left.audio_label < right.audio_label ? -1 : 1;
      }
      if (left.repeat !== right.repeat) {
        return left.repeat - right.repeat;
      }
      return left.turn - right.turn;
    }

    function rowMatches(row) {
      var audio = audioFilter.value;
      var repeat = repeatFilter.value;
      var dimension = dimensionFilter.value;
      var status = statusFilter.value;
      var needle = searchFilter.value ? searchFilter.value.toLowerCase() : "";
      var haystack = "";
      var dimensionStatus = "incomplete";

      if (audio !== "all" && row.audio_label !== audio) {
        return false;
      }

      if (repeat !== "all" && String(row.repeat) !== repeat) {
        return false;
      }

      if (dimension === "all") {
        if (status !== "all" && row.overall_status !== status) {
          return false;
        }
      } else {
        dimensionStatus = scoreStatus(row.scores[dimension]);
        if (status !== "all" && dimensionStatus !== status) {
          return false;
        }
      }

      if (!needle) {
        return true;
      }

      haystack = [
        row.audio_label,
        row.repeat,
        row.turn,
        row.user_text,
        row.assistant_text,
        row.judge_reasoning,
        row.failed_dimensions_text,
        row.tool_calls_text,
        row.tool_results_text,
        row.golden_assistant_text
      ].join(" ").toLowerCase();
      return haystack.indexOf(needle) !== -1;
    }

    function filteredRows() {
      var rows = [];
      var index;
      for (index = 0; index < allRows.length; index += 1) {
        if (rowMatches(allRows[index])) {
          rows.push(allRows[index]);
        }
      }
      rows.sort(compareRows);
      return rows;
    }

    function setSelectedRowId(rows) {
      var index;
      if (!rows.length) {
        selectedRowId = null;
        return;
      }
      for (index = 0; index < rows.length; index += 1) {
        if (rows[index].row_id === selectedRowId) {
          return;
        }
      }
      selectedRowId = rows[0].row_id;
    }

    function renderTableRow(row, isSelected) {
      var selectedClass = isSelected ? "selected" : "";
      var source = row.audio_label + " · r" + row.repeat;
      var responseBadge = badge(
        row.response_status === "empty_response" ? "fail" : "pass",
        row.response_status.replace("_", " ")
      );
      var latency = row.latency_ms == null ? "n/a" : String(row.latency_ms) + " ms";
      return ""
        + '<tr data-row-id="' + escapeHtml(row.row_id) + '" class="' + selectedClass + '" onclick="if (window.__reviewSelectRow) window.__reviewSelectRow(this.getAttribute(\\'data-row-id\\'))">'
        + '<td class="mono column-source">' + escapeHtml(source) + "</td>"
        + '<td class="mono column-turn">' + escapeHtml(String(row.turn)) + "</td>"
        + '<td class="column-status">' + badge(row.overall_status) + "</td>"
        + '<td class="column-response">' + responseBadge + "</td>"
        + '<td class="mono column-latency">' + escapeHtml(latency) + "</td>"
        + '<td class="column-failed">' + escapeHtml(row.failed_dimensions_text || "none") + "</td>"
        + '<td class="column-user-prompt">' + escapeHtml(row.user_text) + "</td>"
        + '<td class="column-assistant-output">' + escapeHtml(row.assistant_text) + "</td>"
        + "</tr>";
    }

    function renderCuratedTableRow(row, isSelected) {
      var selectedClass = isSelected ? "selected" : "";
      var source = row.audio_label + " · r" + row.repeat;
      return ""
        + '<tr data-row-id="' + escapeHtml(row.row_id) + '" class="' + selectedClass + '" onclick="if (window.__reviewSelectRow) window.__reviewSelectRow(this.getAttribute(\\'data-row-id\\'))">'
        + '<td class="mono column-source">' + escapeHtml(source) + "</td>"
        + '<td class="mono column-turn">' + escapeHtml(String(row.turn)) + "</td>"
        + '<td class="column-status">' + badge(row.overall_status) + "</td>"
        + '<td class="column-failed">' + escapeHtml(row.failed_dimensions_text || "none") + "</td>"
        + '<td class="column-user-prompt">' + escapeHtml(row.user_text) + "</td>"
        + '<td class="column-assistant-output">' + escapeHtml(row.assistant_text) + "</td>"
        + "</tr>";
    }

    function renderCurated(rows) {
      var failures = [];
      var htmlParts = [];
      var limit;
      var index;
      for (index = 0; index < rows.length; index += 1) {
        if (rows[index].overall_status === "fail") {
          failures.push(rows[index]);
        }
      }

      limit = failures.length < 30 ? failures.length : 30;
      for (index = 0; index < limit; index += 1) {
        htmlParts.push(renderCuratedTableRow(failures[index], failures[index].row_id === selectedRowId));
      }

      curatedCount.textContent = "Showing " + limit + " of " + failures.length + " failing rows after filters.";
      curatedBody.innerHTML = htmlParts.join("");
    }

    function renderFull(rows) {
      var htmlParts = [];
      var index;
      for (index = 0; index < rows.length; index += 1) {
        htmlParts.push(renderTableRow(rows[index], rows[index].row_id === selectedRowId));
      }
      fullCount.textContent = "Showing " + rows.length + " of " + allRows.length + " rows. Failures are sorted first.";
      fullBody.innerHTML = htmlParts.join("");
    }

    function renderDetail(rows) {
      var row = null;
      var index;
      var previousParts = [];
      var graderRows = [];
      var keys;
      var dimension;
      var value;
      var status;
      var reason;
      var audioHtml;
      var responseBadges;
      var failedTurnsText;
      var compactLatency;
      var compactTtfb;

      if (!rows.length || selectedRowId == null) {
        detailRoot.innerHTML = '<p class="subtle">No rows match the current filters.</p>';
        return;
      }

      for (index = 0; index < rows.length; index += 1) {
        if (rows[index].row_id === selectedRowId) {
          row = rows[index];
          break;
        }
      }
      if (!row) {
        row = rows[0];
      }

      for (index = 0; index < allRows.length; index += 1) {
        if (
          allRows[index].audio_label === row.audio_label &&
          allRows[index].repeat === row.repeat &&
          allRows[index].turn < row.turn
        ) {
          previousParts.push(allRows[index].golden_context_block);
        }
      }
      previousParts.sort();

      keys = [];
      for (dimension in row.scores) {
        if (Object.prototype.hasOwnProperty.call(row.scores, dimension)) {
          keys.push(dimension);
        }
      }
      keys.sort();

      for (index = 0; index < keys.length; index += 1) {
        dimension = keys[index];
        value = row.scores[dimension];
        status = scoreStatus(value);
        reason = "Not applicable on this turn.";
        if (value === false) {
          reason = "Failed. See the turn-level judge reasoning for decisive evidence.";
        } else if (value === true) {
          reason = "Passed.";
        }
        graderRows.push(
          "<tr>"
          + '<td class="mono" style="width: 220px;">' + escapeHtml(dimension) + "</td>"
          + '<td style="width: 140px;">' + badge(status, value == null ? "n/a" : status) + "</td>"
          + "<td>" + escapeHtml(reason) + "</td>"
          + "</tr>"
        );
      }

      if (row.user_audio_exists) {
        audioHtml = '<audio controls preload="none" src="' + escapeHtml(row.user_audio_url) + '"></audio>';
      } else {
        audioHtml = '<div class="subtle" style="margin-top: 8px;">No per-turn input audio file found.</div>';
      }

      responseBadges = badge(row.overall_status) + " " + badge(
        row.response_status === "empty_response" ? "fail" : "pass",
        row.response_status.replace("_", " ")
      );
      failedTurnsText = row.failed_turns_for_run && row.failed_turns_for_run.length
        ? row.failed_turns_for_run.join(", ")
        : "none";
      compactLatency = row.latency_ms == null ? "n/a" : String(row.latency_ms) + "ms";
      compactTtfb = row.ttfb_ms == null ? "n/a" : String(row.ttfb_ms) + "ms";

      detailRoot.innerHTML = ""
        + '<div class="detail-panel">'
        + '<div class="detail-meta">'
        + '<div class="hero-card"><div class="label">Source</div><div class="value mono">' + escapeHtml(row.audio_label + " · repeat " + row.repeat) + "</div></div>"
        + '<div class="hero-card"><div class="label">Identifiers</div><div class="value mono">turn=' + escapeHtml(String(row.turn)) + " · ts=" + escapeHtml(row.timestamp) + " · run=" + escapeHtml(row.run_id) + "</div></div>"
        + '<div class="hero-card"><div class="label">Response Status</div><div class="value">' + responseBadges + "</div></div>"
        + "</div>"
        + '<div class="detail-layout">'
        + '<div class="stack">'
        + '<div class="text-card"><div class="label">Current Turn Audio</div><div class="subtle">Benchmark input audio for the selected user turn.</div>' + audioHtml + "</div>"
        + '<div class="text-card"><div class="label">Gold Previous Turns Context</div><div class="scroll-box context-box">' + escapeHtml(previousParts.length ? previousParts.join("\\n\\n") : "No previous turns.") + "</div></div>"
        + '<div class="text-card"><div class="label">User Query</div><div class="scroll-box short-box">' + escapeHtml(row.user_text) + "</div></div>"
        + '<div class="text-card"><div class="label">Model Output</div><div class="scroll-box tall-box">' + escapeHtml(row.assistant_text) + "</div></div>"
        + '<div class="text-card"><div class="label">Judge Reasoning</div><div class="scroll-box reason-box">' + escapeHtml(row.judge_reasoning || "No reasoning provided.") + "</div></div>"
        + "</div>"
        + '<div class="stack">'
        + '<div class="text-card"><div class="label">Compact Metadata</div><div class="value mono">'
        + "latency=" + escapeHtml(compactLatency) + "<br>"
        + "ttfb=" + escapeHtml(compactTtfb) + "<br>"
        + "reconnections=" + escapeHtml(row.reconnection_count == null ? "n/a" : String(row.reconnection_count)) + "<br>"
        + "failed=" + escapeHtml(row.failed_dimensions_text || "none") + "<br>"
        + "run_failed_turns=" + escapeHtml(failedTurnsText)
        + "</div></div>"
        + '<div class="text-card"><div class="label">Per-Dimension Verdicts</div><div class="table-wrap"><table><thead><tr><th style="width: 220px;">Dimension</th><th style="width: 140px;">Verdict</th><th>Reason</th></tr></thead><tbody>'
        + graderRows.join("")
        + "</tbody></table></div></div>"
        + '<div class="text-card"><div class="label">Expected Gold Response</div><div class="scroll-box short-box">' + escapeHtml(row.golden_assistant_text) + "</div></div>"
        + '<div class="text-card"><div class="label">Tool Calls</div><div class="scroll-box short-box mono">' + escapeHtml(row.tool_calls_text) + "</div></div>"
        + '<div class="text-card"><div class="label">Tool Results</div><div class="scroll-box short-box mono">' + escapeHtml(row.tool_results_text) + "</div></div>"
        + '<div class="text-card"><div class="label">Expected Tool Calls / Results</div><div class="scroll-box short-box mono">' + escapeHtml(row.golden_tool_calls_text + "\\n\\n" + row.golden_tool_results_text) + "</div></div>"
        + "</div>"
        + "</div>"
        + "</div>";
    }

    function render(rows) {
      setSelectedRowId(rows);
      renderCurated(rows);
      renderFull(rows);
      renderDetail(rows);
    }

    function findRowElement(node, root) {
      var current = node;
      while (current && current !== root) {
        if (current.tagName && current.tagName.toLowerCase() === "tr" && current.getAttribute("data-row-id")) {
          return current;
        }
        current = current.parentNode;
      }
      return null;
    }

    function selectRow(rowId) {
      selectedRowId = rowId;
      render(filteredRows());
      if (detailRoot && detailRoot.scrollIntoView) {
        detailRoot.scrollIntoView(true);
      }
    }

    function initialize() {
      window.__reviewSelectRow = function (rowId) {
        selectRow(rowId);
      };

      window.__reviewApplyFilters = function () {
        render(filteredRows());
      };

      function handleTableClick(event) {
        var rowElement = findRowElement(event.target || event.srcElement, this);
        if (!rowElement) {
          return;
        }
        selectRow(rowElement.getAttribute("data-row-id"));
      }

      if (jsStatus) {
        jsStatus.style.display = "none";
      }
      audioFilter.onchange = window.__reviewApplyFilters;
      repeatFilter.onchange = window.__reviewApplyFilters;
      dimensionFilter.onchange = window.__reviewApplyFilters;
      statusFilter.onchange = window.__reviewApplyFilters;
      searchFilter.oninput = window.__reviewApplyFilters;
      curatedBody.onclick = handleTableClick;
      fullBody.onclick = handleTableClick;

      render(filteredRows());
    }

    initialize();
    } catch (error) {
      setBootError(error && error.message ? error.message : String(error));
      if (window.console && window.console.error) {
        window.console.error(error);
      }
    }
    }());
"""


def build_html(payload: dict, *, share_safe: bool) -> str:
    payload_json = json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("</", "<\\/")
    generated_at = payload["metadata"]["generated_at"]
    sorted_rows = sort_rows(payload["rows"])
    selected_row_id = sorted_rows[0]["row_id"] if sorted_rows else None
    hero_metadata_html = render_hero_metadata(payload)
    metrics_cards_html = render_metrics_cards(payload)
    overall_summary_html = render_overall_summary_rows(payload)
    latency_summary_html = render_latency_summary_rows(payload)
    grader_summary_html = render_grader_summary_rows(payload)
    run_summary_html = render_run_summary_rows(payload, share_safe=share_safe)
    curated_rows_html, curated_count_text = render_curated_rows(
        sorted_rows,
        selected_row_id,
    )
    full_rows_html, full_count_text = render_full_rows(
        sorted_rows,
        selected_row_id,
        len(payload["rows"]),
    )
    detail_html = render_detail_panel(
        payload["rows"],
        selected_row_id,
        share_safe=share_safe,
    )
    audio_option_html = "\n".join(
        (
            '<option value="all">All audio conditions</option>'
            if option == "all"
            else f'<option value="{html.escape(option)}">{html.escape(option)}</option>'
        )
        for option in (["all"] + sorted({row["audio_label"] for row in payload["rows"]}))
    )
    repeat_option_html = "\n".join(
        (
            '<option value="all">All repeats</option>'
            if option == "all"
            else f'<option value="{html.escape(option)}">Repeat {html.escape(option)}</option>'
        )
        for option in (["all"] + [str(value) for value in sorted({row["repeat"] for row in payload["rows"]})])
    )
    dimension_option_html = "\n".join(
        (
            '<option value="all">All dimensions</option>'
            if option == "all"
            else f'<option value="{html.escape(option)}">{html.escape(option)}</option>'
        )
        for option in (["all"] + list(DISPLAY_DIMENSIONS.keys()))
    )
    chart_section = ""
    if payload.get("chart_data_url"):
        chart_section = """
      <section class="section">
        <h2>Metric Chart</h2>
        <p class="subtle">Averaged across the three repeats. Turn-taking is intentionally omitted because the batch used <span class="mono">--skip-turn-taking</span>.</p>
        <div class="chart-frame">
          <img alt="Conversation bench audio metrics chart" src="%s">
        </div>
      </section>
""" % payload["chart_data_url"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conversation Bench Audio Batch Review</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 250, 242, 0.94);
      --panel-strong: #fffaf2;
      --ink: #17211f;
      --muted: #5d6964;
      --line: #d6c9b4;
      --accent: #0f5f5a;
      --accent-soft: #def2ee;
      --danger: #9d3b2f;
      --danger-soft: #f6d9d3;
      --warn: #8c5a0f;
      --warn-soft: #f4e6c7;
      --success: #1a6a3b;
      --success-soft: #dff0e3;
      --shadow: 0 18px 40px rgba(41, 33, 12, 0.08);
      --radius: 18px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Avenir", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(15, 95, 90, 0.10), transparent 34%),
        radial-gradient(circle at top left, rgba(157, 59, 47, 0.08), transparent 28%),
        linear-gradient(180deg, #fbf7f0 0%, #f3ecdf 46%, #efe4d3 100%);
    }}

    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}

    .hero {{
      padding: 28px;
      border: 1px solid rgba(214, 201, 180, 0.9);
      border-radius: 28px;
      background:
        linear-gradient(135deg, rgba(255, 250, 242, 0.98), rgba(247, 239, 224, 0.96)),
        linear-gradient(120deg, rgba(15, 95, 90, 0.08), transparent 45%);
      box-shadow: var(--shadow);
    }}

    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}

    h1, h2, h3 {{
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      margin: 0;
      color: #14201c;
    }}

    h1 {{
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.05;
      max-width: 980px;
    }}

    h2 {{
      font-size: 1.5rem;
      margin-bottom: 14px;
    }}

    h3 {{
      font-size: 1.05rem;
      margin-bottom: 10px;
    }}

    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}

    .hero-grid,
    .metrics-grid,
    .detail-meta {{
      display: grid;
      gap: 14px;
    }}

    .hero-grid,
    .metrics-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}

    .hero-grid {{
      margin-top: 18px;
    }}

    .hero-card,
    .section,
    .metric-card,
    .detail-panel {{
      min-width: 0;
      border: 1px solid rgba(214, 201, 180, 0.9);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    .hero-card,
    .section,
    .metric-card,
    .detail-panel {{
      padding: 18px;
    }}

    .mono,
    code,
    pre,
    td.mono {{
      font-family: "SFMono-Regular", "Menlo", "Monaco", "Liberation Mono", monospace;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .label {{
      margin-bottom: 6px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 700;
    }}

    .value {{
      font-size: 0.98rem;
      line-height: 1.45;
      color: var(--ink);
    }}

    .section-stack {{
      display: grid;
      gap: 18px;
      margin-top: 24px;
    }}

    .section {{
      overflow: hidden;
    }}

    .filters {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      align-items: end;
    }}

    .field {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    .field span {{
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--muted);
    }}

    input,
    select {{
      width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--ink);
      font: inherit;
    }}

    .metric-card {{
      background:
        linear-gradient(180deg, rgba(255, 249, 239, 0.98), rgba(252, 244, 232, 0.92));
    }}

    .metric-number {{
      font-size: 1.8rem;
      font-weight: 700;
      margin-top: 4px;
      color: #112623;
    }}

    .metric-note {{
      margin-top: 8px;
      font-size: 0.9rem;
    }}

    .status-banner {{
      margin-top: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(143, 90, 8, 0.32);
      border-radius: 14px;
      background: var(--warn-soft);
      color: var(--warn);
      font-size: 0.92rem;
      line-height: 1.45;
    }}

    .status-banner-error {{
      border-color: rgba(148, 37, 35, 0.25);
      background: rgba(253, 238, 235, 0.95);
      color: #8f221f;
    }}

    .chart-frame {{
      margin-top: 16px;
      border: 1px solid rgba(214, 201, 180, 0.75);
      border-radius: 18px;
      background: rgba(255, 251, 244, 0.96);
      padding: 16px;
    }}

    .chart-frame img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
    }}

    .table-wrap {{
      overflow: auto;
      margin-top: 14px;
      border: 1px solid rgba(214, 201, 180, 0.7);
      border-radius: 14px;
      background: rgba(255, 251, 244, 0.95);
    }}

    .review-scroll {{
      max-height: 360px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    .wide-table table {{
      min-width: 1160px;
    }}

    .review-data-table {{
      table-layout: auto;
      min-width: 1480px;
    }}

    .review-data-table th {{
      white-space: nowrap;
      overflow-wrap: normal;
      word-break: normal;
    }}

    .review-data-table .column-source {{
      width: 180px;
      min-width: 180px;
    }}

    .review-data-table .column-turn {{
      width: 84px;
      min-width: 84px;
    }}

    .review-data-table .column-status {{
      width: 124px;
      min-width: 124px;
    }}

    .review-data-table .column-response {{
      width: 130px;
      min-width: 130px;
    }}

    .review-data-table .column-latency {{
      width: 124px;
      min-width: 124px;
    }}

    .review-data-table .column-failed {{
      width: 190px;
      min-width: 190px;
    }}

    .review-data-table .column-user-prompt {{
      min-width: 320px;
    }}

    .review-data-table .column-assistant-output {{
      min-width: 420px;
    }}

    th,
    td {{
      padding: 11px 12px;
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid rgba(214, 201, 180, 0.7);
      overflow-wrap: anywhere;
      word-break: break-word;
      font-size: 0.92rem;
      line-height: 1.45;
    }}

    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(246, 239, 226, 0.98);
      color: #23312d;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    tbody tr {{
      cursor: pointer;
      transition: background-color 120ms ease;
    }}

    tbody tr:hover {{
      background: rgba(223, 242, 238, 0.55);
    }}

    tbody tr.selected {{
      background: rgba(15, 95, 90, 0.13);
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      white-space: nowrap;
    }}

    .badge-pass {{
      background: var(--success-soft);
      color: var(--success);
    }}

    .badge-fail {{
      background: var(--danger-soft);
      color: var(--danger);
    }}

    .badge-incomplete {{
      background: var(--warn-soft);
      color: var(--warn);
    }}

    .detail-layout {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr);
      align-items: start;
    }}

    .detail-meta {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 16px;
    }}

    .detail-panel {{
      background:
        linear-gradient(180deg, rgba(255, 250, 242, 0.98), rgba(248, 239, 225, 0.94));
    }}

    .stack {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}

    .text-card {{
      border: 1px solid rgba(214, 201, 180, 0.7);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255, 252, 246, 0.98);
      min-width: 0;
    }}

    .scroll-box {{
      margin-top: 8px;
      border-radius: 12px;
      border: 1px solid rgba(214, 201, 180, 0.85);
      background: #fffdf8;
      padding: 12px;
      overflow: auto;
      min-width: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .short-box {{
      max-height: 180px;
    }}

    .tall-box {{
      max-height: 360px;
    }}

    .context-box {{
      max-height: 280px;
    }}

    .reason-box {{
      max-height: 220px;
    }}

    audio {{
      width: 100%;
      margin-top: 8px;
    }}

    .subtle {{
      color: var(--muted);
      font-size: 0.9rem;
    }}

    a {{
      color: var(--accent);
    }}

    @media (max-width: 1040px) {{
      .detail-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 820px) {{
      .page {{
        padding: 16px 14px 28px;
      }}

      .hero {{
        padding: 18px;
        border-radius: 22px;
      }}

      .detail-meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <p class="eyebrow">Conversation Bench Batch Review</p>
      <h1>Audio Condition Comparison for <span class="mono">{payload["metadata"]["model_name"]}</span></h1>
      <p style="margin-top: 12px;">Batch <span class="mono">{payload["metadata"]["batch_name"]}</span>, generated at <span class="mono">{generated_at}</span>. This page compares synthetic audio against two real-speaker conditions across three repeats each, with full turn-level drill-down. Turn-taking is intentionally omitted because the runs used <span class="mono">--skip-turn-taking</span>.</p>
      <div id="js-status" class="status-banner">Interactive mode is inactive. Open this file in a real browser with JavaScript enabled. Finder preview, Slack preview, and some file viewers will show the static report but will not support filters or click-to-detail updates.</div>
      <div class="hero-grid" id="hero-metadata">{hero_metadata_html}</div>
    </section>

    <div class="section-stack">
      <section class="section">
        <h2>Batch Summary</h2>
        <div class="metrics-grid" id="metrics-grid">{metrics_cards_html}</div>
      </section>

{chart_section}

      <section class="section">
        <h2>Global Filters</h2>
        <p class="subtle">Filter by audio condition, repeat, graded dimension, pass/fail state, or free text across prompts, outputs, tools, and judge reasoning.</p>
        <div class="filters" style="margin-top: 16px;">
          <label class="field">
            <span>Audio</span>
            <select id="audio-filter" onchange="if (window.__reviewApplyFilters) window.__reviewApplyFilters()">{audio_option_html}</select>
          </label>
          <label class="field">
            <span>Repeat</span>
            <select id="repeat-filter" onchange="if (window.__reviewApplyFilters) window.__reviewApplyFilters()">{repeat_option_html}</select>
          </label>
          <label class="field">
            <span>Dimension</span>
            <select id="dimension-filter" onchange="if (window.__reviewApplyFilters) window.__reviewApplyFilters()">{dimension_option_html}</select>
          </label>
          <label class="field">
            <span>Status</span>
            <select id="status-filter" onchange="if (window.__reviewApplyFilters) window.__reviewApplyFilters()">
              <option value="all">All</option>
              <option value="fail">Fail</option>
              <option value="pass">Pass</option>
              <option value="incomplete">Incomplete / N/A</option>
            </select>
          </label>
          <label class="field">
            <span>Search</span>
            <input id="search-filter" type="search" placeholder="Audio, turn, prompt, output, reason" oninput="if (window.__reviewApplyFilters) window.__reviewApplyFilters()">
          </label>
        </div>
      </section>

      <section class="section">
        <h2>Overall Summary</h2>
        <p class="subtle">Averages across the three repeats for each audio condition.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Audio</th>
                <th>Pass Rows Avg</th>
                <th>Fail Rows Avg</th>
                <th>Pass Rate Avg</th>
                <th>Empty Resp Avg</th>
                <th>End Session Avg</th>
                <th>Failed Turns By Repeat</th>
              </tr>
            </thead>
            <tbody id="overall-summary-body">{overall_summary_html}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Latency Summary</h2>
        <p class="subtle">Mean and percentile latency metrics averaged by audio condition.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Audio</th>
                <th>Latency Mean</th>
                <th>Latency p50</th>
                <th>Latency p90</th>
                <th>Latency p95</th>
                <th>Latency p99</th>
              </tr>
            </thead>
            <tbody id="latency-summary-body">{latency_summary_html}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Grader Summary</h2>
        <p class="subtle">Turn-taking is omitted because it was skipped during judging.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Audio</th>
                <th>Tool</th>
                <th>Instruction</th>
                <th>KB</th>
                <th>Ambiguity</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody id="grader-summary-body">{grader_summary_html}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Per-Run Summary</h2>
        <p class="subtle">Each row is one repeat. Failures first and direct local artifact links included for drill-down.</p>
        <div class="table-wrap wide-table">
          <table>
            <thead>
              <tr>
                <th>Audio</th>
                <th>Repeat</th>
                <th>Run ID</th>
                <th>Pass Rate</th>
                <th>Pass Rows</th>
                <th>Fail Rows</th>
                <th>Latency p95</th>
                <th>Failed Turns</th>
                <th>Artifacts</th>
              </tr>
            </thead>
            <tbody id="run-summary-body">{run_summary_html}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Curated Failure Review</h2>
        <p class="subtle" id="curated-count">{html.escape(curated_count_text)}</p>
        <div class="table-wrap wide-table review-scroll">
          <table class="review-data-table">
            <thead>
              <tr>
                <th class="column-source">Source</th>
                <th class="column-turn">Turn</th>
                <th class="column-status">Status</th>
                <th class="column-failed">Failed Dims</th>
                <th class="column-user-prompt">User Prompt</th>
                <th class="column-assistant-output">Assistant Output</th>
              </tr>
            </thead>
            <tbody id="curated-body">{curated_rows_html}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>Datapoint Detail</h2>
        <div id="detail-root">{detail_html}</div>
      </section>

      <section class="section">
        <h2>Full Datapoints</h2>
        <p class="subtle" id="full-count">{html.escape(full_count_text)}</p>
        <div class="table-wrap wide-table review-scroll">
          <table class="review-data-table">
            <thead>
              <tr>
                <th class="column-source">Source</th>
                <th class="column-turn">Turn</th>
                <th class="column-status">Overall</th>
                <th class="column-response">Response</th>
                <th class="column-latency">Latency</th>
                <th class="column-failed">Failed Dims</th>
                <th class="column-user-prompt">User Prompt</th>
                <th class="column-assistant-output">Assistant Output</th>
              </tr>
            </thead>
            <tbody id="full-body">{full_rows_html}</tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script id="payload" type="application/json">{payload_json}</script>
  <script>{build_client_script()}</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.expanduser().resolve() if args.batch_dir else find_default_batch_dir()
    payload, flattened_rows = build_batch_payload(batch_dir)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (batch_dir / "review.html").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "row_id",
        "audio_label",
        "repeat",
        "run_id",
        "turn",
        "timestamp",
        "model_name",
        "overall_status",
        "response_status",
        "failed_dimensions_text",
        "failure_count",
        "latency_ms",
        "ttfb_ms",
        "reconnection_count",
        "tool_call_count",
        "tool_use_correct",
        "instruction_following",
        "kb_grounding",
        "ambiguity_handling",
        "state_tracking",
        "user_text",
        "assistant_text",
        "judge_reasoning",
        "tool_calls_text",
        "tool_results_text",
        "golden_assistant_text",
        "golden_tool_calls_text",
        "golden_tool_results_text",
        "golden_context_block",
        "user_audio_path",
        "user_audio_url",
        "user_audio_exists",
        "run_dir",
    ]
    for score_column in SCORE_COLUMNS:
        flattened_rows[score_column] = flattened_rows["scores"].map(
            lambda row_scores: row_scores.get(score_column)
        )

    flattened_rows[csv_columns].to_csv(
        output_path.parent / "review_rows.csv",
        index=False,
        encoding="utf-8",
    )
    (output_path.parent / "review_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    output_path.write_text(build_html(payload, share_safe=False), encoding="utf-8")

    share_safe_payload = build_share_safe_payload(payload)
    share_safe_path = output_path.parent / "review_shared.html"
    share_safe_path.write_text(
        build_html(share_safe_payload, share_safe=True),
        encoding="utf-8",
    )

    print(f"Batch dir: {batch_dir}")
    print(f"Rows: {len(flattened_rows)}")
    print(f"HTML: {output_path}")
    print(f"Share-safe HTML: {share_safe_path}")
    print(f"CSV: {output_path.parent / 'review_rows.csv'}")
    print(f"Payload: {output_path.parent / 'review_payload.json'}")


if __name__ == "__main__":
    main()
