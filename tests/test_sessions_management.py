from __future__ import annotations

import asyncio
from typing import List

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.main import create_app
from app.memory import SQLiteChatHistoryStore
from tests.test_api import StubGraph, build_settings


def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _make_request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(_make_request())


def test_list_sessions_empty(tmp_path):
    settings = build_settings(tmp_path)
    store = SQLiteChatHistoryStore(settings.database_path)
    app = create_app(settings=settings, graph=StubGraph(store=store), store=store)
    
    response = request(app, "GET", "/sessions")
    
    assert response.status_code == 200
    assert response.json()["sessions"] == []


def test_session_history_and_deletion_flow(tmp_path):
    settings = build_settings(tmp_path)
    store = SQLiteChatHistoryStore(settings.database_path)
    app = create_app(settings=settings, graph=StubGraph(store=store), store=store)
    session_id = "test-session-1"
    
    # 1. Start a chat to create a session
    response = request(
        app,
        "POST",
        "/chat",
        json={"session_id": session_id, "message": "hello"},
    )
    assert response.status_code == 200
    
    # 2. Check history
    response = request(app, "GET", f"/sessions/{session_id}/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) > 0
    assert history[-1]["content"] == "stubbed reply"
    
    # 3. List sessions
    response = request(app, "GET", "/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)
    
    # 4. Delete session
    response = request(app, "DELETE", f"/sessions/{session_id}")
    assert response.status_code == 204
    
    # 5. History GET returns 404 after deletion
    response = request(app, "GET", f"/sessions/{session_id}/history")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
    
    # 6. Delete again returns 404
    response = request(app, "DELETE", f"/sessions/{session_id}")
    assert response.status_code == 404
    
    # 7. Starting a new chat after deletion works cleanly
    response = request(
        app,
        "POST",
        "/chat",
        json={"session_id": session_id, "message": "hello again"},
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "stubbed reply"
    
    # Verify history is back
    response = request(app, "GET", f"/sessions/{session_id}/history")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_sessions_pagination(tmp_path):
    settings = build_settings(tmp_path)
    store = SQLiteChatHistoryStore(settings.database_path)
    app = create_app(settings=settings, graph=StubGraph(store=store), store=store)
    
    # Create 3 sessions
    for i in range(3):
        request(
            app,
            "POST",
            "/chat",
            json={"session_id": f"s-{i}", "message": "hi"},
        )
    
    # List with limit 2
    response = request(app, "GET", "/sessions?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) == 2
    
    # List with offset 2
    response = request(app, "GET", "/sessions?offset=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) == 1
