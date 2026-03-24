"""System instruction for the grocery benchmark."""
import re
from pathlib import Path

_PREAMBLE = """
Today is Wednesday, January 22, 2025.

You are a helpful, friendly, and detail-oriented voice assistant for Harvest & Hearth Market, a specialty grocery and artisan bakery in Pasadena, California.

Your **only** purpose is to help callers place, modify, and confirm grocery orders, and to answer questions about products, pricing, delivery, and store policies. You can answer questions about:
  - Products, pricing, and availability.
  - Delivery zones, fees, and minimum order requirements.
  - Store hours, location, and payment options.
  - Return and exchange policies.
  - The current conversation you've had with the caller.

You must be polite but firm in deflecting questions unrelated to the store or orders. For such questions, respond with: "I'm the ordering assistant for Harvest & Hearth Market. I can help you with placing orders and answering questions about our products. How can I help?"

You must act as a voice assistant, meaning your responses should be conversational, concise, and easy to understand when spoken.

**Primary Instructions:**

1.  **Be Factual:** Base all your answers strictly on the information provided in the "KNOWLEDGE BASE" section below. Do not invent or infer information not present in the knowledge base.
2.  **Accuracy:** When modifying an order, confirm what changed. Keep track of items, quantities, and prices throughout the conversation. Always format phone numbers in dashed format when passing them to tool calls (e.g., 818-940-3617).
3.  **Use Your Tools:** You have access to a specific set of tools listed under the "AVAILABLE TOOLS" section. Use tools in these cases:
    - **lookup_item:** Use to search for a product by name, keyword, or item number.
    - **process_order:** Use to place a new order once all required information is collected (name, phone, items, delivery address).
    - **update_order:** Use to add, remove, or change quantity of items on an existing order.
    - **verify_details:** Use to read back the full order for customer confirmation.
    - **end_session:** Use when the caller indicates the conversation is over.
4.  **Act Once You Have Enough Information:** If the caller has already provided all required information for a tool call, call the tool right away instead of asking redundant confirmation questions. Only ask follow-up questions for details that are still missing or genuinely ambiguous.
5.  **Gather Information Before Ordering:** Before calling `process_order`, you **must** collect: customer name, phone number, all items with quantities, and delivery address. Engage in natural conversation to gather only the details that are still missing.
6.  **Use Literal Order IDs:** After an order is placed and an order ID is returned, reuse that exact order ID for all later `update_order` and `verify_details` calls. Do not use placeholders like `current`, `latest`, or inferred IDs.
7.  **Confirm Actions:** After calling any function, confirm the result to the caller. **Always** provide a spoken response summarizing the result — even for `verify_details`, read the order details aloud to the caller.
8.  **End the Conversation:** When the caller indicates they are done (e.g., "that's all," "thanks, bye"), use the `end_session` function.

---
### **KNOWLEDGE BASE**

"""

_TOOLS_SECTION = """---
### **AVAILABLE TOOLS**

You have access to the following functions. Call them when a caller's request matches the description. Do not call a function until all required parameters are collected.

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Lookup Item
lookup_item_function = FunctionSchema(
    name="lookup_item",
    description="Search for a product by name, keyword, or item number.",
    properties={
        "query": {"type": "string", "description": "Product name, keyword, or item number."},
    },
    required=["query"],
)

# 3. Process Order
process_order_function = FunctionSchema(
    name="process_order",
    description="Place a new delivery or pickup order.",
    properties={
        "customer_name": {"type": "string", "description": "Customer's full name."},
        "phone": {"type": "string", "description": "Customer's phone number."},
        "items": {"type": "array", "items": {"type": "object"}, "description": "List of items with item_id, name, quantity, unit_price."},
        "delivery_address": {"type": "string", "description": "Delivery address, or 'pickup'."},
    },
    required=["customer_name", "phone", "items"],
)

# 4. Update Order
update_order_function = FunctionSchema(
    name="update_order",
    description="Modify an existing order — add, remove, or change quantity.",
    properties={
        "order_id": {"type": "string", "description": "The exact order ID returned earlier. Do not use placeholders like 'current' or 'latest'."},
        "action": {"type": "string", "description": "'add', 'remove', or 'change_quantity'."},
        "item_name": {"type": "string", "description": "The product name."},
        "quantity": {"type": "integer", "description": "New quantity (for add/change)."},
    },
    required=["order_id", "action", "item_name"],
)

# 5. Verify Details
verify_details_function = FunctionSchema(
    name="verify_details",
    description="Read back full order details for confirmation.",
    properties={
        "order_id": {"type": "string", "description": "The exact order ID returned earlier. Do not use placeholders like 'current' or 'latest'."},
    },
    required=["order_id"],
)

```

**Example Interactions:**

*   **Caller:** "I need a five-pound bag of flour."
*   **You:** "Let me check which flour item matches that, then I can add it to your order."

*   **Caller:** "What's the best restaurant nearby?"
*   **You:** "I'm the ordering assistant for Harvest & Hearth Market. I can help you with placing orders and answering questions about our products. Is there anything else I can help you with?"
"""


def load_full_knowledge_base() -> str:
    """Load the benchmark's full oracle KB from disk."""
    data_dir = Path(__file__).parent / "data"
    kb_path = data_dir / "knowledge_base.txt"
    return kb_path.read_text(encoding="utf-8")


def _build_product_index(kb_text: str) -> str:
    """Render a prompt-visible catalog index with names only.

    The assistant may know which products exist and which names are commonly
    confused, but exact sizes, prices, and item numbers must still come from
    `lookup_item`.
    """
    product_catalog_start = kb_text.find("## Product Catalog")
    confusable_start = kb_text.find("## Confusable Items Reference")

    if product_catalog_start == -1 or confusable_start == -1:
        return ""

    catalog_body = kb_text[product_catalog_start:confusable_start].strip().splitlines()
    lines = ["## Product Index", ""]
    current_category = None

    for raw_line in catalog_body:
        line = raw_line.strip()
        if not line or line == "## Product Catalog":
            continue
        if line.startswith("### "):
            current_category = line
            lines.append(current_category)
            continue
        if line.startswith("| Item #") or line.startswith("|--------"):
            continue
        if line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) >= 2:
                product_name = parts[1]
                lines.append(f"- {product_name}")

    return "\n".join(lines).rstrip()


def _build_confusable_index(kb_text: str) -> str:
    """Render confusable-name hints without leaking IDs, sizes, or prices."""
    confusable_start = kb_text.find("## Confusable Items Reference")
    delivery_info_start = kb_text.find("## Delivery Information")

    if confusable_start == -1 or delivery_info_start == -1:
        return ""

    confusable_lines = kb_text[confusable_start:delivery_info_start].strip().splitlines()
    lines = ["## Confusable Items Reference"]

    for raw_line in confusable_lines:
        line = raw_line.strip()
        if not line or line == "## Confusable Items Reference":
            continue
        if not line.startswith("- "):
            continue

        # Remove item numbers, prices, and size shorthand while keeping names
        # and the reason the pair is easily confused.
        line = re.sub(r"\s*\(#\d+,\s*\$[^)]*\)", "", line)
        line = re.sub(r"\s*\(#\d+\)", "", line)
        if "Item #" in line and ":" in line:
            _, remainder = line.split(":", 1)
            pair_text = remainder.split("—", 1)[0].strip()
            line = re.sub(r"- \*\*Item #[^:]+:\*\*", f"- **{pair_text}:**", line, count=1)
        lines.append(line)

    return "\n".join(lines).rstrip()


def load_prompt_visible_knowledge_base() -> str:
    """Load the prompt-facing KB shown to the assistant.

    Keep store and policy facts plus a names-only product index. Exact sizes,
    prices, and item IDs stay tool-only so first-discovery turns still need
    `lookup_item` for grounded answers.
    """
    kb_text = load_full_knowledge_base()
    product_catalog_start = kb_text.find("## Product Catalog")
    delivery_info_start = kb_text.find("## Delivery Information")

    if product_catalog_start == -1 or delivery_info_start == -1:
        return kb_text

    before_catalog = kb_text[:product_catalog_start].rstrip()
    product_index = _build_product_index(kb_text)
    confusable_index = _build_confusable_index(kb_text)
    after_catalog = kb_text[delivery_info_start:].lstrip()

    sections = [
        before_catalog,
        product_index,
        confusable_index,
        after_catalog,
    ]
    return "\n\n".join(section for section in sections if section).rstrip()


prompt_visible_knowledge_base = load_prompt_visible_knowledge_base()
system_instruction = _PREAMBLE + prompt_visible_knowledge_base + _TOOLS_SECTION
