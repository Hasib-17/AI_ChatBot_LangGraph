from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import ConfigError, Settings, configure_logging, ensure_runtime_dirs
from app.graph import build_chat_graph
from app.memory import SQLiteChatHistoryStore
from app.schemas import ChatMessageView, ChatRequest, ChatResponse, ErrorEnvelope, SessionListItem, SessionsResponse

logger = logging.getLogger(__name__)


def message_to_view(message) -> ChatMessageView:
    if isinstance(message, SystemMessage):
        return ChatMessageView(role="system", content=message.content)
    if isinstance(message, HumanMessage):
        return ChatMessageView(role="human", content=message.content)
    if isinstance(message, AIMessage):
        return ChatMessageView(role="ai", content=message.content)
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorEnvelope(error={"code": code, "message": message}).model_dump(),
    )


def create_app(
    *,
    settings: Settings | None = None,
    graph: Any | None = None,
    store: SQLiteChatHistoryStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    configure_logging(resolved_settings.log_level)
    ensure_runtime_dirs(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            resolved_settings.validate()
        except ConfigError:
            logger.exception("Application startup failed due to invalid configuration")
            raise

        app.state.store = store or SQLiteChatHistoryStore(resolved_settings.database_path)
        if graph is not None:
            app.state.graph = graph
        else:
            from app.llm import build_llm

            app.state.graph = build_chat_graph(
                store=app.state.store,
                llm=build_llm(resolved_settings),
                system_prompt=resolved_settings.system_prompt,
                memory_strategy=resolved_settings.memory_strategy,
                memory_window_size=resolved_settings.memory_window_size,
                max_context_tokens=resolved_settings.max_context_tokens,
            )
        app.state.settings = resolved_settings
        logger.info("Application initialized")
        yield

    app = FastAPI(
        title="Memory-Powered AI Chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            latency_ms = (time.perf_counter() - started_at) * 1000
            status_code = getattr(locals().get("response"), "status_code", 500)
            logger.info(
                "%s %s -> %s in %.2fms",
                request.method,
                request.url.path,
                status_code,
                latency_ms,
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'] if part != 'body')}: {error['msg']}"
            for error in exc.errors()
        )
        return error_response(422, "validation_error", message)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            message = exc.detail.get("message", "Request failed")
            code = exc.detail.get("code", "http_error")
        else:
            message = str(exc.detail)
            code = "http_error"
        return error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
        return error_response(500, "internal_server_error", "Internal server error")

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            422: {"model": ErrorEnvelope},
            500: {"model": ErrorEnvelope},
        },
    )
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            result = await app.state.graph.ainvoke(
                {
                    "session_id": request.session_id,
                    "chat_history": [],
                    "user_message": request.message,
                    "assistant_response": "",
                }
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        except Exception as exc:
            logger.exception("Chat processing failed for session %s", request.session_id)
            raise HTTPException(
                status_code=500,
                detail={"code": "internal_server_error", "message": "Internal server error"},
            ) from exc

        return ChatResponse(
            session_id=result["session_id"],
            reply=result["assistant_response"],
            history=[message_to_view(message) for message in result["chat_history"]],
        )

    @app.get(
        "/sessions/{session_id}/history",
        response_model=List[ChatMessageView],
        responses={
            404: {"model": ErrorEnvelope},
        },
    )
    async def get_history(session_id: str) -> List[ChatMessageView]:
        """
        Retrieve the full message history for a specific session.
        """
        if not app.state.store.session_exists(session_id):
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": f"Session {session_id} not found"},
            )

        history = app.state.store.load_history(session_id)
        return [message_to_view(msg) for msg in history]

    @app.delete(
        "/sessions/{session_id}",
        status_code=204,
        responses={
            404: {"model": ErrorEnvelope},
        },
    )
    async def delete_session(session_id: str):
        """
        Delete all messages and summary state for a specific session.
        """
        deleted = app.state.store.clear_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": f"Session {session_id} not found"},
            )
        return

    @app.get(
        "/sessions",
        response_model=SessionsResponse,
    )
    async def list_sessions(limit: int = 100, offset: int = 0) -> SessionsResponse:
        """
        List all chat sessions with their last active timestamps.
        """
        sessions = app.state.store.list_sessions(limit=limit, offset=offset)
        return SessionsResponse(
            sessions=[
                SessionListItem(session_id=s["session_id"], last_active=s["last_active"])
                for s in sessions
            ]
        )

    return app


settings = Settings.from_env()
app = create_app(settings=settings)


def run() -> None:
    try:
        settings.validate()
    except ConfigError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
