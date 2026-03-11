import json
from types import SimpleNamespace
import unittest

from pipecat.adapters.schemas.tools_schema import ToolsSchema

from audio_arena.judging.llm_judge import format_turns_for_judge
from audio_arena.pipelines.openai_realtime import OpenAIRealtimeLLMServiceExplicitToolResult
from audio_arena.pipelines.realtime import RealtimePipeline
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

    def test_openai_rehydration_history_uses_conversation_items_with_assistant_output_text(self):
        items = RealtimePipeline._build_openai_rehydration_history_items(
            [
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
        )

        self.assertEqual(len(items), 4)
        self.assertEqual(items[0].type, "message")
        self.assertEqual(items[0].role, "user")
        self.assertEqual(items[0].content[0].type, "input_text")
        self.assertEqual(items[1].type, "function_call")
        self.assertEqual(items[2].type, "function_call_output")
        self.assertEqual(items[3].type, "message")
        self.assertEqual(items[3].role, "assistant")
        self.assertEqual(items[3].content[0].type, "output_text")
        self.assertEqual(items[3].content[0].text, "Your event is booked.")

    def test_manual_turn_handling_ignores_duplicate_stop_after_first_commit(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._session_properties = SimpleNamespace(
            audio=SimpleNamespace(
                input=SimpleNamespace(turn_detection=False)
            )
        )
        service._awaiting_manual_audio_commit = False
        service._manual_turn_input_committed = True
        service._manual_response_in_flight = False
        service._last_manual_commit_monotonic = 0.0
        service._pending_manual_tool_results = []
        service._awaiting_manual_tool_continuation_start = False

        called = {"send_client_event": 0}

        async def fake_send_client_event(_event):
            called["send_client_event"] += 1

        service.send_client_event = fake_send_client_event

        import asyncio

        asyncio.run(service._handle_user_stopped_speaking(None))

        self.assertEqual(called["send_client_event"], 0)
        self.assertFalse(service._awaiting_manual_audio_commit)


if __name__ == "__main__":
    unittest.main()
