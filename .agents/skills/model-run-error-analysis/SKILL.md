---
name: model-run-error-analysis
description: Analyze benchmark runs to identify dominant error modes per model, shared hard turns, grader or benchmark issues, and representative failing examples. Prefer judged runs, but fall back to transcript + benchmark-contract review when judged artifacts are missing or unreliable.
---

# Model Run Error Analysis

## Overview

Use this skill to produce a model-by-model failure analysis from benchmark runs.

The core requirement is that the final buckets must come from **deep datapoint review**, not just aggregate metrics or heuristic labels.

This skill is designed for:
- multi-model comparison folders produced by [compare_model_runs.py](/Users/minh.hoque/work/github/conversation-bench/scripts/compare_model_runs.py)
- one or more judged run directories containing `openai_judged.jsonl`
- one or more run directories that only have `transcript.jsonl` plus benchmark artifacts, when judging is missing, broken, or incomplete

The output should explain:
- dominant real model errors per model
- grader or strict-matching issues
- benchmark or ground-truth issues
- pipeline or runtime artifacts
- representative failing turns
- shared failures across models
- model-specific unique failures and unique passes

The review packet should include the raw datapoint material needed for inspection:
- input
- assistant output
- tool calls
- tool results
- grader verdicts and reasoning when available
- benchmark golden answer
- required tool call
- model/runtime metadata when relevant

## Review Philosophy

Do not treat every failed judgment as a real model failure.

Before calling something a real model error, check whether it is actually:
- a genuine model mistake
- a grader strictness issue
- a benchmark contract problem
- a runtime or pipeline artifact
- mixed

In conversation-bench style tasks, you must often review all of these together:
- transcript row
- golden answer in `turns.py`
- system prompt in `system.py`
- knowledge base in `data/knowledge_base.txt`
- tool call and tool result payloads

If the system prompt and golden disagree, do not automatically blame the model.

## Turn-Taking Policy

Turn-taking scoring is off by default for this skill.

- Prefer judged inputs that were produced with `--skip-turn-taking`.
- If you need to judge or re-judge runs as part of the analysis workflow, always use `--skip-turn-taking`.
- Do not include turn-taking metrics in the main analysis unless the user explicitly asks for them and you have verified that the run artifacts support valid turn-taking evaluation.
- For `--rehydrate` runs, treat turn-taking as unsupported unless the pipeline produces per-conversation audio artifacts that are known to align with the transcript being judged.

## Execution Defaults

- For rehydrated benchmark runs, default to `--parallel 16` unless the user explicitly asks for a different value or you have evidence that lower concurrency is needed for stability.
- Keep `--skip-turn-taking` as the default for judging unless the user explicitly asks for turn-taking analysis and the artifacts support it.

## Preferred Entrypoint

Use the helper script when judged artifacts are present and sane:

```bash
python3 /Users/minh.hoque/work/github/conversation-bench/.agents/skills/model-run-error-analysis/scripts/deep_error_analysis.py \
  --comparison-dir /absolute/path/to/comparison_dir
```

Or:

```bash
python3 /Users/minh.hoque/work/github/conversation-bench/.agents/skills/model-run-error-analysis/scripts/deep_error_analysis.py \
  --run-dir /absolute/path/to/run_a \
  --run-dir /absolute/path/to/run_b \
  --run-dir /absolute/path/to/run_c
```

If the judged artifacts are missing, incomplete, or clearly inconsistent with the transcript, switch to the manual workflow below instead of forcing the script path.

## Inputs

Provide one of:
- `--comparison-dir <dir>` where the directory contains `run_results.csv`
- one or more `--run-dir <dir>` paths

Preferred judged input:
- `openai_judged.jsonl`

Fallback manual-review inputs:
- `transcript.jsonl`
- `runtime.json`
- benchmark `turns.py`
- benchmark `system.py`
- benchmark `data/knowledge_base.txt`

Optional:
- `--output <path>` to override the report destination

## Workflow

This skill has three phases.

### Phase 0: Sanity-check the artifacts

Do this before trusting any judged output.

1. Confirm what artifacts exist in each run directory:
   - `transcript.jsonl`
   - `runtime.json`
   - `openai_judged.jsonl`
   - `openai_summary.json`
   - `openai_analysis.md`
2. Check transcript coverage:
   - count transcript rows
   - list missing turns
   - note duplicate turn indices
   - note obviously corrupted rows such as `[EMPTY_RESPONSE ...]` or `[MODEL_ENDED_SESSION]`
3. Check whether judged artifacts are aligned with the transcript:
   - do the judged rows cover the same turns as the transcript?
   - did judging fail because a turn is missing?
   - do the judged counts make sense?
4. If artifacts are incomplete or contradictory, explicitly record that and fall back to transcript-first manual review.

Example failure mode:
- a normal-mode realtime run may drop a transcript turn; the stock judge can then fail hard instead of gracefully skipping the missing turn

### Phase 1: Build the review packet

#### Path A: Judged artifacts are available and sane

1. Load each run's `openai_judged.jsonl`.
2. For every turn, extract:
   - `turn`
   - `user_text`
   - `assistant_text`
   - `tool_calls`
   - `tool_results`
   - `scores`
   - `judge_reasoning`
3. Mark a row as failed if any score dimension is `false`.
4. Export failure review rows to JSONL and CSV so the model can inspect raw datapoints directly.
5. Compute exact per-model failure counts:
   - failed rows
   - `MODEL_ENDED_SESSION`
   - `EMPTY_RESPONSE`
   - failed dimensions
   - exclude turn-taking from the main counts unless the user explicitly requested it and the artifacts were validated for turn-taking
6. Generate provisional heuristic clusters from judgment text and response shape:
   - premature end session
   - empty response
   - partial multi-tool execution
   - state memory failure
   - ambiguity handling failure
   - knowledge grounding error
   - tool or action selection error
   - other
7. Compare turn-level failure sets across models:
   - failed by all models
   - unique fail turns per model
   - unique pass turns per model

#### Path B: Judged artifacts are missing, broken, or untrustworthy

1. Load each run's `transcript.jsonl`.
2. Join each transcript row with benchmark source material from:
   - `turns.py`
   - `system.py`
   - `knowledge_base.txt`
3. For every turn, extract:
   - `turn`
   - `user_text`
   - `assistant_text`
   - `tool_calls`
   - `tool_results`
   - `golden_text`
   - `required_function_call`
   - runtime notes such as missing transcript rows or empty responses
4. Build a manual suspicion table using simple review flags:
   - missing transcript row
   - empty response
   - ended session
   - required tool missing
   - required tool arguments mismatched
   - response contamination or appended unrelated content
   - likely contract mismatch between system prompt and golden answer
5. Export the packet to CSV or JSONL even if no judged file exists.

### Phase 2: Deep review and final bucketing

Review actual datapoints, not just the heuristic labels.

For each important bucket, inspect representative turns by reading:
- user input
- assistant output
- tool calls
- tool results
- judged scores and judge reasoning when available
- golden answer
- required tool call
- relevant system prompt instructions
- relevant KB facts

Then classify the turn into one of these top-level buckets:
- `real_model_error`
- `grader_or_strict_matching_issue`
- `reference_or_ground_truth_issue`
- `other_pipeline_or_runtime_issue`
- `mixed_contract_and_model_issue`

Within those, use cause-level sub-buckets such as:
- `premature_end_session`
- `partial_multi_tool_execution`
- `state_memory_failure`
- `tool_selection_failure`
- `error_recovery_failure`
- `constraint_violation`
- `incomplete_correction`
- `tool_arg_string_normalization`
- `optional_tool_detail_omitted`
- `system_vs_golden_conflict`
- `closing_turn_contract_mismatch`
- `response_contamination`
- `missing_transcript_turn`
- `empty_response_execution_failure`

Important:
- use the judged result as evidence, not as the final truth
- re-bucket when the judge-visible failure is only a symptom
- prefer the true underlying mechanism over the surface dimension label

Examples:
- if the model ends the session after one correct tool call in a 3-step workflow, the real error is usually `premature_end_session` or `partial_multi_tool_execution`, not just `instruction_following=false`
- if the model asks for the user's name again and misses a tool call, the real bucket is often `state_memory_failure`, even if the judge only highlights tool use
- if the tool succeeds and the only mismatch is trivial argument wording, that is often `grader_or_strict_matching_issue`
- if the system prompt forbids answering a question but the golden expects the answer, that is often `reference_or_ground_truth_issue`
- if a response contains unrelated carry-over text from another turn, that is usually `other_pipeline_or_runtime_issue`

### Phase 3: Write the deliverables

Write a report under `results/`.

When judged artifacts are good, use the standard names:
- `deep_error_analysis.md`
- `failure_review_rows.jsonl`
- `failure_review_rows.csv`

When the review had to fall back to transcript-first manual analysis, prefer names that make that explicit:
- `manual_error_review.md`
- `manual_error_review.csv`
- `review_metadata.json`

The report should include:
- run path(s) reviewed
- artifact sanity notes
- whether the review used judged rows, manual fallback, or both
- bucket counts
- major takeaways
- representative examples in plain text
- explicit notes on grader issues, benchmark issues, and runtime issues

In the final assistant response, also provide a concise plain-text summary:
- main real error modes per model
- main grader or benchmark issues
- main runtime artifacts
- the path to the saved report and CSV

## Output Expectations

- Put the report under `results/` by default.
- Always return a text summary to the user in the same turn. Do not make the markdown file the only deliverable.
- Use exact dimension counts from judged rows when judged artifacts are usable.
- If judged artifacts are unusable, say so clearly and switch to transcript-first manual review.
- Treat broader error modes as analyst-defined clusters, not as a second judge.
- Cite representative turn numbers in the report.
- For each major bucket, describe in plain language:
  - what went wrong
  - why it belongs in that bucket
  - what evidence was decisive
- Include concrete datapoints to review quickly:
  - turn number
  - user input
  - assistant output
  - judge reasoning when available
  - golden answer when relevant
  - tool calls and tool results when relevant
- Prefer comparing models on:
  - dominant real failure modes
  - grader or contract issues
  - response-pattern failures
  - shared hard turns
  - unique deltas

## Guardrails

- Do not infer error modes from aggregate pass-rate charts alone.
- Do not assume every judged fail is a real model error.
- Use `openai_judged.jsonl` as the preferred source of truth when it is present and sane, but not as the only source of truth.
- When judged artifacts and transcript artifacts disagree, investigate the disagreement explicitly.
- Always read actual `tool_calls` and `tool_results` for tool-failure turns.
- When a model is fluent but wrong, classify by the decisive failure mechanism, not by tone.
- Do not let the heuristic classifier become the final report unreviewed.
- For each top bucket in the final writeup, manually inspect multiple concrete turns before naming the bucket.
- If both a comparison dir and explicit run dirs are given, prefer the explicit run dirs.
- Default to content-only analysis. Do not surface turn-taking scores in summaries, tables, or comparisons unless the user explicitly requested turn-taking and you verified the artifacts are suitable for it.
- Before calling something a benchmark failure, check the system prompt, golden answer, and KB together.
- Before blaming the model for a tool-use mismatch, check whether the mismatch is only strict string matching on a semantically correct tool call.
