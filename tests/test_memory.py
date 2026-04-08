from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.memory import SQLiteChatHistoryStore


def test_sqlite_history_persists_messages(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    session_id = "session-1"
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
    ]

    store.append_messages(session_id, messages)
    loaded = store.load_history(session_id)

    assert [message.content for message in loaded] == ["system", "hello", "hi there"]
    assert [message.type for message in loaded] == ["system", "human", "ai"]
