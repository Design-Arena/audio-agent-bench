import json
import tempfile
import unittest
from pathlib import Path

from audio_arena.cli import finalize_rehydrated_run_artifacts
from audio_arena.judging.llm_judge import load_transcript


class RehydratedArtifactMergeTests(unittest.TestCase):
    def test_finalize_rehydrated_run_artifacts_merges_sorted_transcript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            turn_2_dir = run_dir / "turn_runs" / "turn_002"
            turn_0_dir = run_dir / "turn_runs" / "turn_000"
            turn_2_dir.mkdir(parents=True)
            turn_0_dir.mkdir(parents=True)

            (turn_2_dir / "transcript.jsonl").write_text(
                json.dumps({"turn": 2, "assistant_text": "a2"}) + "\n",
                encoding="utf-8",
            )
            (turn_0_dir / "transcript.jsonl").write_text(
                json.dumps({"turn": 0, "assistant_text": "a0"}) + "\n",
                encoding="utf-8",
            )

            runtime = finalize_rehydrated_run_artifacts(
                run_dir=run_dir,
                model="test-model",
                target_indices=[2, 0],
                turn_results={
                    2: {"success": True, "turn_run_dir": str(turn_2_dir), "error": None},
                    0: {"success": True, "turn_run_dir": str(turn_0_dir), "error": None},
                },
                parallel=2,
                disable_vad=False,
                real_audio_speaker=None,
            )

            merged = load_transcript(run_dir)
            manifest = json.loads((run_dir / "rehydrated_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(runtime["turns"], 2)
        self.assertEqual([record["turn"] for record in merged], [0, 2])
        self.assertEqual(manifest["turn_artifact_layout"], "per_turn_subdirs")
        self.assertFalse(runtime["turn_taking_supported"])

    def test_finalize_rehydrated_run_artifacts_fails_when_successful_turn_missing_transcript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            turn_1_dir = run_dir / "turn_runs" / "turn_001"
            turn_1_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "missing transcript.jsonl"):
                finalize_rehydrated_run_artifacts(
                    run_dir=run_dir,
                    model="test-model",
                    target_indices=[1],
                    turn_results={
                        1: {"success": True, "turn_run_dir": str(turn_1_dir), "error": None},
                    },
                    parallel=2,
                    disable_vad=False,
                    real_audio_speaker=None,
                )

    def test_finalize_rehydrated_run_artifacts_fails_on_duplicate_turn_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            turn_0_a = run_dir / "turn_runs" / "turn_000"
            turn_0_b = run_dir / "turn_runs" / "turn_001"
            turn_0_a.mkdir(parents=True)
            turn_0_b.mkdir(parents=True)

            duplicate_row = json.dumps({"turn": 0, "assistant_text": "dup"}) + "\n"
            (turn_0_a / "transcript.jsonl").write_text(duplicate_row, encoding="utf-8")
            (turn_0_b / "transcript.jsonl").write_text(duplicate_row, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Duplicate transcript rows"):
                finalize_rehydrated_run_artifacts(
                    run_dir=run_dir,
                    model="test-model",
                    target_indices=[0, 1],
                    turn_results={
                        0: {"success": True, "turn_run_dir": str(turn_0_a), "error": None},
                        1: {"success": True, "turn_run_dir": str(turn_0_b), "error": None},
                    },
                    parallel=2,
                    disable_vad=False,
                    real_audio_speaker=None,
                )

    def test_load_transcript_rejects_duplicate_rehydrated_turn_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "runtime.json").write_text(
                json.dumps({"mode": "rehydrated"}),
                encoding="utf-8",
            )
            (run_dir / "transcript.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"turn": 3, "assistant_text": "a"}),
                        json.dumps({"turn": 3, "assistant_text": "b"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate turn rows found in rehydrated transcript.jsonl"):
                load_transcript(run_dir)


if __name__ == "__main__":
    unittest.main()
