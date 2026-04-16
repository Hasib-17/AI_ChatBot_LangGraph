from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import ConfigError, Settings
from app.main import create_app, run


class StubGraph:
    def __init__(self, store=None):
        self.store = store

    async def ainvoke(self, state):
        session_id = state["session_id"]
        user_msg = HumanMessage(content=state["user_message"])
        ai_msg = AIMessage(content="stubbed reply")
        
        if self.store:
            self.store.append_messages(session_id, [user_msg, ai_msg])
            history = self.store.load_history(session_id)
        else:
            history = [
                SystemMessage(content="system prompt"),
                user_msg,
                ai_msg,
            ]
            
        return {
            "session_id": session_id,
            "assistant_response": "stubbed reply",
            "chat_history": history,
        }


class FailingGraph:
    async def ainvoke(self, state):
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
    from app.main import run
    
    monkeypatch.setattr(
        "app.main.settings",
        build_settings(tmp_path, llm_provider="groq", groq_api_key=""),
    )

    with pytest.raises(SystemExit) as exc_info:
        run()

    assert exc_info.value.code == 1


def test_session_management_endpoints(tmp_path):
    from app.memory import SQLiteChatHistoryStore

    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    app = create_app(
        settings=build_settings(tmp_path), graph=StubGraph(store=store), store=store
    )

    # 1. Create a session by chatting
    resp1 = request(
        app,
        "POST",
        "/chat",
        json={"session_id": "test-session-123", "message": "hello"},
    )
    assert resp1.status_code == 200

    # 2. GET /sessions should return the session
    resp_sessions = request(app, "GET", "/sessions")
    assert resp_sessions.status_code == 200
    data = resp_sessions.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"] == "test-session-123"

    # 3. GET /sessions/{session_id}/history should return the message history
    resp_history = request(app, "GET", "/sessions/test-session-123/history")
    assert resp_history.status_code == 200
    history = resp_history.json()
    assert len(history) == 2
    assert history[0]["role"] == "human"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "ai"
    assert history[1]["content"] == "stubbed reply"

    # 4. DELETE /sessions/{session_id} should succeed
    resp_delete = request(app, "DELETE", "/sessions/test-session-123")
    assert resp_delete.status_code == 204

    # 5. GET /sessions/{session_id}/history after delete should return 404
    resp_history_after = request(app, "GET", "/sessions/test-session-123/history")
    assert resp_history_after.status_code == 404

    # 6. DELETE a non-existent session returns 404
    resp_delete_again = request(app, "DELETE", "/sessions/test-session-123")
    assert resp_delete_again.status_code == 404

    # 7. Starting a new chat after deletion works cleanly
    resp2 = request(
        app,
        "POST",
        "/chat",
        json={"session_id": "test-session-123", "message": "hello again"},
    )
    assert resp2.status_code == 200

    # history should only contain the new interaction
    resp_history_restart = request(app, "GET", "/sessions/test-session-123/history")
    assert resp_history_restart.status_code == 200
    history_restart = resp_history_restart.json()
    assert len(history_restart) == 2
    assert history_restart[0]["content"] == "hello again"

