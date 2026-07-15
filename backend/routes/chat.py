"""
The /api/chat endpoint. Thin by design — the real logic lives in
middleware/pipeline.py and services/.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.middleware.pipeline import process_message
from backend.services.gemini_service import generate_reply

router = APIRouter()

class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    intent: str

@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 1. Run the message through the middleware pipeline
    #    (intent classification + product retrieval if relevant)
    pipeline_result = process_message(request.message)

    # 2. Call the LLM with the enriched system prompt + conversation history
    history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
    reply = generate_reply(
        system_prompt=pipeline_result["system_prompt"],
        history=history_dicts,
        user_message=request.message,
    )

    return ChatResponse(reply=reply, intent=pipeline_result["intent"])
