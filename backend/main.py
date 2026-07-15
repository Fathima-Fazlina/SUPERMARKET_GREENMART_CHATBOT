"""
FastAPI backend entrypoint.
Run: uvicorn backend.main:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()  # must run before importing chat_router, since gemini_service.py
                # reads GEMINI_API_KEY at import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.routes.chat import router as chat_router

load_dotenv()

app = FastAPI(title="GreenMart Chatbot API")

# CORS: lets the frontend (running on a different port/origin) call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict this before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.get("/chat.html")
def read_chat():
    return FileResponse("frontend/chat.html")

@app.get("/index.html")
def read_index():
    return FileResponse("frontend/index.html")

@app.get("/api/health")
def health_check():
    return {"status": "GreenMart chatbot API is running"}
