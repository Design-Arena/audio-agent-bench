import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.realtime import events as rt_events

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
    def test_multi_call_tool_responses_match_by_args_not_call_order(self):
        benchmark = DummyBenchmark()
        benchmark.turns = [
            {
                "input": "register me for all the Voice sessions",
                "required_function_call": [
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920101"}},
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920102"}},
                    {"name": "register_for_session", "args": {"name": "Jennifer Smith", "session_id": "920103"}},
                ],
                "function_call_response": [
                    {"status": "success"},
                    {"status": "error", "error_code": "SESSION_FULL"},
                    {"status": "error", "error_code": "SCHEDULE_CONFLICT"},
                ],
            }
        ]
        pipeline = TextPipeline(benchmark)

        first = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920103"},
        )
        second = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920101"},
        )
        third = pipeline._get_turn_tool_response(
            "register_for_session",
            {"name": "Jennifer Smith", "session_id": "920102"},
        )

        self.assertEqual(first, {"status": "error", "error_code": "SCHEDULE_CONFLICT"})
        self.assertEqual(second, {"status": "success"})
        self.assertEqual(third, {"status": "error", "error_code": "SESSION_FULL"})

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

    def test_format_turns_for_judge_includes_tool_use_guidance(self):
        records = [
            {
                "turn": 0,
                "user_text": "change the syrup back to one bottle",
                "assistant_text": "Updated to one bottle.",
                "tool_calls": [],
                "tool_results": [],
            }
        ]
        expected_turns = [
            {
                "golden_text": "Updated to one bottle.",
                "required_function_call": None,
                "categories": ["long_range_memory"],
                "tool_use_guidance": (
                    "Do not require a redundant lookup_item call here; reuse the already-established "
                    "item facts from session state."
                ),
            }
        ]

        formatted = format_turns_for_judge(
            records,
            expected_turns,
            get_relevant_dimensions_fn=lambda _: ["instruction_following", "kb_grounding"],
        )

        self.assertIn("**Tool Use Guidance**:", formatted)
        self.assertIn("reuse the already-established item facts", formatted)

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

    def test_rehydration_reconnect_history_preserves_tool_results_in_prompt(self):
        pipeline = RealtimePipeline(DummyBenchmark())
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
        pipeline.llm = SimpleNamespace(
            _session_properties=SimpleNamespace(instructions="")
        )

        pipeline._seed_rehydration_history_for_reconnects()

        self.assertEqual(
            pipeline._conversation_history[0]["tool_results"],
            [
                {
                    "name": "book_event",
                    "response": {"status": "success", "event_id": "EVT-3001"},
                }
            ],
        )

        pipeline._update_instructions_with_history()

        self.assertIn("[Tool result:", pipeline.llm._session_properties.instructions)
        self.assertIn("\"event_id\": \"EVT-3001\"", pipeline.llm._session_properties.instructions)

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

    def test_retry_current_turn_resets_manual_no_vad_state_before_requeue(self):
        pipeline = RealtimePipeline(DummyBenchmark())
        pipeline.needs_turn_retry = True
        pipeline.turn_idx = 0
        pipeline.current_turn_audio_path = "/tmp/turn.wav"
        pipeline.paced_input = Mock()
        pipeline._reset_openai_manual_turn_state = Mock()
        pipeline._start_turn_watchdog = Mock()
        pipeline._get_audio_duration = Mock(return_value=1.25)

        with patch("audio_arena.pipelines.realtime.asyncio.sleep", new=AsyncMock()):
            import asyncio

            asyncio.run(pipeline._retry_current_turn())

        pipeline._reset_openai_manual_turn_state.assert_called_once_with()
        pipeline.paced_input.enqueue_wav_file.assert_called_once_with("/tmp/turn.wav")
        pipeline._start_turn_watchdog.assert_called_once_with(1.25)
        self.assertFalse(pipeline.needs_turn_retry)

    def test_tool_output_ack_waits_for_response_done_before_continuation(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._pending_response_create = True
        service._waiting_for_response_done_before_response_create = True
        service._pending_tool_output_item_ids = {"tool-item-1"}
        service.send_client_event = AsyncMock()

        evt = SimpleNamespace(
            item=SimpleNamespace(
                id="tool-item-1",
                type="function_call_output",
                call_id="call_1",
            )
        )

        import asyncio

        asyncio.run(service._handle_tool_output_item_event(evt, phase="added"))

        service.send_client_event.assert_not_awaited()
        self.assertTrue(service._pending_response_create)
        self.assertFalse(service._pending_tool_output_item_ids)

        service._waiting_for_response_done_before_response_create = False
        asyncio.run(service._maybe_send_deferred_response_create(trigger="response.done"))

        service.send_client_event.assert_awaited_once()
        sent_event = service.send_client_event.await_args.args[0]
        self.assertIsInstance(sent_event, rt_events.ResponseCreateEvent)
        self.assertFalse(service._pending_response_create)

    def test_function_call_output_marks_deferred_state_before_send(self):
        service = OpenAIRealtimeLLMServiceExplicitToolResult.__new__(
            OpenAIRealtimeLLMServiceExplicitToolResult
        )
        service._context = object()
        service._pending_function_calls = {"call_1": SimpleNamespace(name="get_quote")}
        service._completed_tool_calls = set()
        service._get_last_tool_result = lambda: {"status": "success", "quote": 12750}
        service._pending_response_create = False
        service._waiting_for_response_done_before_response_create = False
        service._pending_tool_output_item_ids = set()
        service.run_function_calls = AsyncMock()

        async def fake_send_client_event(event):
            if isinstance(event, rt_events.ConversationItemCreateEvent):
                self.assertTrue(service._pending_response_create)
                self.assertTrue(service._waiting_for_response_done_before_response_create)
                self.assertIn(event.item.id, service._pending_tool_output_item_ids)

        service.send_client_event = fake_send_client_event

        evt = SimpleNamespace(
            call_id="call_1",
            arguments=json.dumps({"event_type": "wedding"}),
        )

        import asyncio

        asyncio.run(service._handle_evt_function_call_arguments_done(evt))

        self.assertIn("call_1", service._completed_tool_calls)
        service.run_function_calls.assert_awaited_once()
        self.assertTrue(service._pending_response_create)
        self.assertTrue(service._waiting_for_response_done_before_response_create)
        self.assertEqual(len(service._pending_tool_output_item_ids), 1)


if __name__ == "__main__":
    unittest.main()
