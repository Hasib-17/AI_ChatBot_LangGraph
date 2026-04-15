from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import ConfigError, Settings
from app.main import create_app, run


class StubGraph:
    def invoke(self, state):
        return {
            "session_id": state["session_id"],
            "assistant_response": "stubbed reply",
            "chat_history": [
                SystemMessage(content="system prompt"),
                HumanMessage(content=state["user_message"]),
                AIMessage(content="stubbed reply"),
            ],
        }


class FailingGraph:
    def invoke(self, state):
        raise RuntimeError("model exploded")


def build_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        llm_provider=overrides.get("llm_provider", "groq"),
        openai_api_key=overrides.get("openai_api_key", "test-key"),
        groq_api_key=overrides.get("groq_api_key", "test-key"),
        model_name=overrides.get("model_name", "llama3-8b-8192"),
        system_prompt=overrides.get("system_prompt", "system prompt"),
        database_path=overrides.get("database_path", str(tmp_path / "chat.db")),
        memory_strategy=overrides.get("memory_strategy", "sliding_window"),
        memory_window_size=overrides.get("memory_window_size", 12),
        max_context_tokens=overrides.get("max_context_tokens", 6000),
        api_host=overrides.get("api_host", "127.0.0.1"),
        api_port=overrides.get("api_port", 8000),
        log_level=overrides.get("log_level", "INFO"),
    )


def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _make_request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(_make_request())


def test_chat_request_validation_returns_consistent_error_shape(tmp_path):
    app = create_app(settings=build_settings(tmp_path), graph=StubGraph())

    response = request(
        app,
        "POST",
        "/chat",
        json={"session_id": "   ", "message": "x" * 8001},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "session_id: Value error, must not be blank; message: String should have at most 8000 characters",
        }
    }


def test_chat_model_failure_returns_consistent_error_shape(tmp_path):
    app = create_app(settings=build_settings(tmp_path), graph=FailingGraph())

    response = request(
        app,
        "POST",
        "/chat",
        json={"session_id": "user-1", "message": "hello"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "Internal server error",
        }
    }


def test_chat_success_still_works(tmp_path):
    app = create_app(settings=build_settings(tmp_path), graph=StubGraph())

    response = request(
        app,
        "POST",
        "/chat",
        json={"session_id": "user-1", "message": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "stubbed reply"


def test_request_logging_includes_method_path_status_and_latency(tmp_path, caplog):
    app = create_app(settings=build_settings(tmp_path), graph=StubGraph())

    with caplog.at_level(logging.INFO):
        response = request(app, "GET", "/health")

    assert response.status_code == 200
    assert any("GET /health -> 200 in " in message for message in caplog.messages)


def test_missing_groq_api_key_fails_during_startup(tmp_path):
    app = create_app(
        settings=build_settings(tmp_path, llm_provider="groq", groq_api_key=""),
        graph=StubGraph(),
    )

    with pytest.raises(ConfigError, match="GROQ_API_KEY is required"):
        async def _start() -> None:
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(_start())


def test_run_exits_cleanly_when_required_config_is_missing(tmp_path, monkeypatch):
    from app import main as app_main

    monkeypatch.setattr(
        app_main,
        "settings",
        build_settings(tmp_path, llm_provider="groq", groq_api_key=""),
    )

    with pytest.raises(SystemExit) as exc_info:
        run()

    assert exc_info.value.code == 1
