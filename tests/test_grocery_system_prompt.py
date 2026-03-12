import unittest

from benchmarks.grocery_bench.system import system_instruction


class GrocerySystemPromptTests(unittest.TestCase):
    def test_system_prompt_omits_product_catalog_but_keeps_policy_kb(self):
        self.assertNotIn("## Product Catalog", system_instruction)
        self.assertNotIn("All-Purpose Flour", system_instruction)
        self.assertNotIn("Organic Eggs", system_instruction)
        self.assertNotIn("Fresh Flower Bouquet", system_instruction)
        self.assertNotIn("## Confusable Items Reference", system_instruction)

        self.assertIn("## Delivery Information", system_instruction)
        self.assertIn("Orders over $75 qualify for **free delivery** in Zone 1.", system_instruction)
        self.assertIn("Orders placed before 2:00 PM are eligible for same-day delivery", system_instruction)

    def test_system_prompt_encourages_executing_tools_once_request_is_clear(self):
        self.assertIn("Act Once You Have Enough Information", system_instruction)
        self.assertIn(
            "call the tool right away instead of asking redundant confirmation questions",
            system_instruction,
        )


if __name__ == "__main__":
    unittest.main()
