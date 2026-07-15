import json
import os

def load_store_data():
    """Load the store's dataset from data/store_data.json"""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "store_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_products(products):
    lines = []
    for p in products:
        stock = "in stock" if p["in_stock"] else "OUT OF STOCK"
        organic = " (organic)" if p["organic"] else ""
        lines.append(f"- {p['name']}{organic} — ${p['price']} — {p['category']} — {stock}")
    return "\n".join(lines)


def format_promotions(promotions):
    lines = []
    for promo in promotions:
        lines.append(f"- {promo['title']} (valid until {promo['valid_until']}): {promo['details']}")
    return "\n".join(lines)


def format_locations(locations):
    lines = []
    for loc in locations:
        delivery = "offers delivery" if loc["has_delivery"] else "no delivery"
        lines.append(f"- {loc['branch']}: {loc['address']}, phone {loc['phone']} ({delivery})")
    return "\n".join(lines)


def format_faq(faq):
    lines = []
    for item in faq:
        lines.append(f"Q: {item['q']}\nA: {item['a']}")
    return "\n".join(lines)


def build_system_prompt():
    """Builds the full system prompt string from store_data.json"""
    data = load_store_data()

    system_prompt = f"""You are a friendly customer service assistant for {data['store_name']}, a supermarket. {data['tagline']}.

Answer customer questions using ONLY the information provided below. If you don't know the answer from this information, say so honestly and suggest they contact the store directly — never make up prices, stock status, or policies.

Keep answers short, warm, and conversational — this is a chat widget, not an essay.

## Store hours
Monday-Friday: {data['hours']['monday_to_friday']}
Saturday: {data['hours']['saturday']}
Sunday: {data['hours']['sunday']}
Public holidays: {data['hours']['public_holidays']}

## Locations
{format_locations(data['locations'])}

## Products
{format_products(data['products'])}

## Current promotions
{format_promotions(data['promotions'])}

## Policies
Returns: {data['policies']['returns']}
Payment methods: {', '.join(data['policies']['payment_methods'])}
Delivery: {data['policies']['delivery']}

## Loyalty program
{data['loyalty_program']['name']}: {data['loyalty_program']['how_it_works']} {data['loyalty_program']['signup']}

## Frequently asked questions
{format_faq(data['faq'])}
"""
    return system_prompt


# Quick manual test — run this file directly to preview the generated prompt
if __name__ == "__main__":
    print(build_system_prompt())