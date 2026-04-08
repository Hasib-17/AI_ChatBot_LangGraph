from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings, configure_logging, ensure_runtime_dirs
from app.graph import build_chat_graph
from app.llm import build_llm
from app.memory import SQLiteChatHistoryStore
from app.schemas import ChatMessageView, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

settings = Settings.from_env()
configure_logging(settings.log_level)
ensure_runtime_dirs(settings)


def message_to_view(message) -> ChatMessageView:
    if isinstance(message, SystemMessage):
        return ChatMessageView(role="system", content=message.content)
    if isinstance(message, HumanMessage):
        return ChatMessageView(role="human", content=message.content)
    if isinstance(message, AIMessage):
        return ChatMessageView(role="ai", content=message.content)
    raise TypeError(f"Unsupported message type: {type(message)!r}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    store = SQLiteChatHistoryStore(settings.database_path)
    llm = build_llm(settings)
    graph = build_chat_graph(store=store, llm=llm, system_prompt=settings.system_prompt)
    app.state.store = store
    app.state.graph = graph
    logger.info("Application initialized")
    yield


app = FastAPI(
    title="Memory-Powered AI Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = app.state.graph.invoke(
            {
                "session_id": request.session_id,
                "chat_history": [],
                "user_message": request.message,
                "assistant_response": "",
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat processing failed for session %s", request.session_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return ChatResponse(
        session_id=result["session_id"],
        reply=result["assistant_response"],
        history=[message_to_view(message) for message in result["chat_history"]],
    )


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
