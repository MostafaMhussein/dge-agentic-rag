"""FastAPI backend with OpenAI compatible chat API."""
import uuid
import time
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag import get_index
from .agents import create_crew
from .tracing import init_phoenix_tracing
from .phoenix_prompts import init_prompts, get_all_prompts
from .memory import ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_phoenix_tracing()
init_prompts()

app = FastAPI(
    title="Abu Dhabi Policy Assistant",
    description="RAG-based Q&A for government policy documents",
    version="1.0.0"
)

app.state.rag_runner = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "policy-assistant"
    messages: List[Message]
    session_id: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Initialize RAG pipeline."""
    logger.info("Starting backend...")
    
    try:
        index = get_index()
        _, _, runner = create_crew(index, nodes=None)
        app.state.rag_runner = runner
        logger.info("Backend initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "rag_ready": app.state.rag_runner is not None,
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "data": [
            {"id": "policy-assistant", "object": "model"},
            {"id": "llama3.1", "object": "model"},
        ]
    }


@app.get("/v1/prompts")
async def list_prompts():
    """List registered prompts."""
    prompts = get_all_prompts()
    return {
        "prompts": [
            {"name": name, "version": data["version"], "description": data["description"]}
            for name, data in prompts.items()
        ]
    }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """Chat endpoint (OpenAI-compatible)."""
    
    if not app.state.rag_runner:
        raise HTTPException(status_code=503, detail="Backend initializing")
    
    user_message = request.messages[-1].content
    session_id = request.session_id or str(uuid.uuid4())
    
    memory = ConversationMemory(session_id)
    history_text = memory.get_history_text(limit=5)
    
    answer, citations, route, contexts = app.state.rag_runner(
        question=user_message,
        history_text=history_text,
    )
    
    memory.add_message("user", user_message)
    memory.add_message("assistant", answer)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(user_message.split()) + len(answer.split()),
        }
    }
