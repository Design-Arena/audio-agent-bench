import unittest

from benchmarks.appointment_bench.system import system_instruction as appointment_system_instruction
from benchmarks.assistant_bench.system import system_instruction as assistant_system_instruction
from benchmarks.conversation_bench.system import system_instruction as conversation_system_instruction
from benchmarks.event_bench.system import system_instruction as event_system_instruction
from benchmarks.grocery_bench.system import system_instruction as grocery_system_instruction
from benchmarks.product_bench.system import system_instruction as product_system_instruction


_PROMPTS = {
    "appointment": appointment_system_instruction,
    "assistant": assistant_system_instruction,
    "conversation": conversation_system_instruction,
    "event": event_system_instruction,
    "grocery": grocery_system_instruction,
    "product": product_system_instruction,
}

_PROACTIVE_HEADING = "Act Once You Have Enough Information"
_PROACTIVE_RULE = (
    "call the tool right away instead of asking redundant confirmation questions"
)


class BenchmarkSystemPromptTests(unittest.TestCase):
    def test_all_prompts_include_proactive_tool_use_rule(self):
        for benchmark_name, system_instruction in _PROMPTS.items():
            with self.subTest(benchmark=benchmark_name):
                self.assertIn(_PROACTIVE_HEADING, system_instruction)
                self.assertIn(_PROACTIVE_RULE, system_instruction)


if __name__ == "__main__":
    unittest.main()
