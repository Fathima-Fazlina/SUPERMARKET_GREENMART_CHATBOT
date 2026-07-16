"""
Wraps calls to the Gemini API using the google-genai SDK.
"""
import os
from google import genai
from google.genai import types

_client = None

def _get_client():
    """Lazy-initialize the Gemini client once."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

def generate_reply(system_prompt: str, history: list, user_message: str) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": str} — our internal format
    Converts to Gemini's format internally.
    """
    client = _get_client()

    # Convert history to Gemini Content objects
    gemini_history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])]
        )
        for m in history
    ]

    # Add the current user message
    gemini_history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    return response.text
