"""
Wraps calls to the Gemini API.
"""
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_reply(system_prompt: str, history: list, user_message: str) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": str} — our internal format
    Converts to Gemini's format internally.
    """
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history
    ]
    model = genai.GenerativeModel(
        model_name="gemini-flash-lite-latest",
        system_instruction=system_prompt,
    )
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_message)
    return response.text
