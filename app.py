import os
import pickle

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from prompts.system_prompt import build_system_prompt
from retrieval.search import search_products


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# -----------------------------
# Load intent classifier
# -----------------------------
with open("classifier/intent_model.pkl", "rb") as f:
    classifier = pickle.load(f)

intent_model = classifier["model"]
vectorizer = classifier["vectorizer"]


# -----------------------------
# Build system prompt
# -----------------------------
SYSTEM_PROMPT = build_system_prompt()


# -----------------------------
# Gemini model
# -----------------------------
model = genai.GenerativeModel(
    model_name="gemini-flash-lite-latest",
    system_instruction=SYSTEM_PROMPT,
)


# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(
    page_title="GreenMart Assistant",
    page_icon="🛒",
)

st.title("🛒 GreenMart Assistant")
st.caption("Ask me about products, prices, promotions, store hours or policies.")


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Intents that require product search
# -----------------------------
PRODUCT_INTENTS = {
    "product_search",
    "price_comparison",
}


# -----------------------------
# Helper function
# -----------------------------
def format_products(products):

    if not products:
        return "No relevant products found."

    lines = []

    for product in products:

        stock = "In Stock" if product["in_stock"] else "Out of Stock"
        organic = "Organic" if product["organic"] else "Regular"

        lines.append(
            f"- {product['name']}\n"
            f"  Category: {product['category']}\n"
            f"  Brand: {product['brand']}\n"
            f"  Price: ${product['price']}\n"
            f"  Availability: {stock}\n"
            f"  Type: {organic}\n"
        )

    return "\n".join(lines)


# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask something about GreenMart...")

if user_input:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------
    # Predict intent
    # -----------------------------
    user_vector = vectorizer.transform([user_input])

    intent = intent_model.predict(user_vector)[0]

    probabilities = intent_model.predict_proba(user_vector)[0]

    confidence = probabilities.max()

    # -----------------------------
    # Product Retrieval
    # -----------------------------
    retrieved_products = []

    if intent in PRODUCT_INTENTS:
        retrieved_products = search_products(
            user_input,
            top_n=5,
        )

    product_context = format_products(retrieved_products)

    # -----------------------------
    # Build enhanced prompt
    # -----------------------------
    enhanced_prompt = f"""
Customer Intent:
{intent}

Classifier Confidence:
{confidence:.2f}

Relevant Products:
{product_context}

Customer Question:
{user_input}

Instructions:

- Answer as GreenMart's customer service assistant.

- If products are provided above, use ONLY those products when answering product questions.

- If no products are found, politely tell the customer.

- For store hours, promotions, policies, delivery and FAQs,
use the store information already provided in your system prompt.

- Never invent products, prices, promotions or policies.

- Keep responses concise, friendly and conversational.
"""

    # -----------------------------
    # Build Gemini history
    # -----------------------------
    gemini_history = []

    for msg in st.session_state.messages[:-1]:

        role = "user" if msg["role"] == "user" else "model"

        gemini_history.append(
            {
                "role": role,
                "parts": [msg["content"]],
            }
        )

    # -----------------------------
    # Generate response
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            chat = model.start_chat(history=gemini_history)

            response = chat.send_message(enhanced_prompt)

            reply = response.text

            st.markdown(reply)

            # Uncomment while testing
            # st.caption(f"Intent: {intent} ({confidence:.2f})")

    # -----------------------------
    # Save assistant reply
    # -----------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )