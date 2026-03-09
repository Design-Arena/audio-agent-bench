import json
import unittest

from pipecat.adapters.schemas.tools_schema import ToolsSchema

from audio_arena.judging.llm_judge import format_turns_for_judge
from audio_arena.pipelines.text import TextPipeline


class DummyBenchmark:
    def __init__(self):
        self.system_instruction = "Base system prompt."
        self.tools_schema = ToolsSchema(standard_tools=[])
        self.turns = [{"input": "target turn"}]


class JudgeAndRehydrationRegressionTests(unittest.TestCase):
    def test_format_turns_for_judge_includes_tool_results(self):
        records = [
            {
                "turn": 0,
                "user_text": "check availability",
                "assistant_text": "The venue is available.",
                "tool_calls": [
                    {
                        "name": "search_venues",
                        "args": {"date": "2025-03-15", "guest_count": 90},
                    }
                ],
                "tool_results": [
                    {
                        "name": "search_venues",
                        "response": {
                            "tool_call_id": "call_123",
                            "result": {"status": "success", "venues": [{"venue_id": "garden_pavilion"}]},
                            "properties": None,
                            "is_duplicate": False,
                        },
                    }
                ],
            }
        ]
        expected_turns = [
            {
                "golden_text": "Let me check availability.",
                "required_function_call": {
                    "name": "search_venues",
                    "args": {"date": "2025-03-15", "guest_count": 90},
                },
                "categories": ["tool_use"],
            }
        ]

        formatted = format_turns_for_judge(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda _: ["instruction_following", "kb_grounding", "tool_use_correct"],
        )

        self.assertIn("**Actual Functions**:", formatted)
        self.assertIn("**Actual Function Results**:", formatted)
        self.assertIn("call_123", formatted)
        self.assertIn("garden_pavilion", formatted)

    def test_text_pipeline_rehydration_keeps_tool_history_in_system_prompt(self):
        pipeline = TextPipeline(DummyBenchmark())
        pipeline._rehydration_turns = [
            {
                "input": "book the event",
                "golden_text": "Your event is booked.",
                "required_function_call": {
                    "name": "book_event",
                    "args": {"name": "Priya Mehta"},
                },
                "function_call_response": {
                    "status": "success",
                    "event_id": "EVT-3001",
                },
            }
        ]

        pipeline._setup_context()
        messages = pipeline.context.get_messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "target turn")
        self.assertIn("[Tool call: book_event", messages[0]["content"])
        self.assertIn("[Tool result:", messages[0]["content"])
        self.assertIn("\"event_id\": \"EVT-3001\"", messages[0]["content"])
        self.assertIn("Assistant: Your event is booked.", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
