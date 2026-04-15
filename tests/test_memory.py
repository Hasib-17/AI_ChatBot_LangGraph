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


def test_sqlite_summary_state_persists_and_clears(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    session_id = "session-summary"

    store.upsert_summary_state(
        session_id=session_id,
        summary_text="User likes concise answers.",
        summarized_upto_message_id=42,
    )
    loaded = store.load_summary_state(session_id)
    assert loaded.summary_text == "User likes concise answers."
    assert loaded.summarized_upto_message_id == 42

    store.clear_session(session_id)
    cleared = store.load_summary_state(session_id)
    assert cleared.summary_text == ""
    assert cleared.summarized_upto_message_id == 0
