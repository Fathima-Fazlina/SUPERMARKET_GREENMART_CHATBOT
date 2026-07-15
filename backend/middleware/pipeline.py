"""
The 'middleware' layer: this sits between the raw incoming request and the
final LLM call. It enriches the request with:
  1. Detected intent (from the classifier)
  2. Retrieved product context (from the vector search) - only when relevant

This keeps routes/chat.py thin — it just calls process_message() and gets
back everything needed to build the final prompt.
"""
from backend.services.classifier_service import detect_intent
from backend.services.retrieval_service import search_products
from prompts.system_prompt import build_system_prompt

BASE_SYSTEM_PROMPT = build_system_prompt()

# Only these intents actually benefit from a product search
PRODUCT_INTENTS = {"product_search", "price_comparison"}

def process_message(user_message: str) -> dict:
    """
    Runs the full enrichment pipeline on an incoming message.
    Returns everything routes/chat.py needs to call the LLM.
    """
    intent = detect_intent(user_message)

    retrieved_context = ""
    if intent in PRODUCT_INTENTS:
        results = search_products(user_message, top_n=5)
        lines = [
            f"- {r['name']} — ${r['price']} — {'in stock' if r['in_stock'] else 'OUT OF STOCK'}"
            for r in results
        ]
        retrieved_context = "\n\n## Relevant products for this question\n" + "\n".join(lines)

    system_prompt = (
        BASE_SYSTEM_PROMPT
        + f"\n\n## Detected customer intent: {intent}"
        + retrieved_context
    )

    return {
        "intent": intent,
        "system_prompt": system_prompt,
    }
