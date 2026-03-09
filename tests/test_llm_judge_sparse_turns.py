import json
import tempfile
import unittest
from pathlib import Path

from audio_arena.judging.llm_judge import build_judge_user_prompt, load_transcript


class LLMJudgeSparseTurnTests(unittest.TestCase):
    def test_load_transcript_sorts_by_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            transcript_path = run_dir / "transcript.jsonl"
            rows = [
                {"turn": 12, "user_text": "u12", "assistant_text": "a12"},
                {"turn": 3, "user_text": "u3", "assistant_text": "a3"},
                {"turn": 74, "user_text": "u74", "assistant_text": "a74"},
            ]
            transcript_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            loaded = load_transcript(run_dir)

            self.assertEqual([record["turn"] for record in loaded], [3, 12, 74])

    def test_prompt_uses_exact_sparse_turn_ids(self):
        prompt = build_judge_user_prompt(
            formatted_turns="formatted",
            turn_numbers=[0, 1, 2, 74],
            cross_turn_realignment=True,
        )

        self.assertIn("[0, 1, 2, 74]", prompt)
        self.assertIn("Do NOT renumber turns.", prompt)
        self.assertNotIn("turns 0-3", prompt)


if __name__ == "__main__":
    unittest.main()
