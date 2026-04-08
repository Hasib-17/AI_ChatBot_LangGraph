from __future__ import annotations

from langchain_core.messages import AIMessage

from app.graph import build_chat_graph
from app.memory import SQLiteChatHistoryStore


class StubLLM:
    def invoke(self, messages):
        human_messages = [message.content for message in messages if message.type == "human"]
        return AIMessage(content=f"echo:{human_messages[-1]}|turns:{len(human_messages)}")


def test_graph_uses_persistent_history(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    graph = build_chat_graph(store=store, llm=StubLLM(), system_prompt="system prompt")

    first = graph.invoke(
        {
            "session_id": "abc",
            "chat_history": [],
            "user_message": "My name is Hasib.",
            "assistant_response": "",
        }
    )
    second = graph.invoke(
        {
            "session_id": "abc",
            "chat_history": [],
            "user_message": "What is my name?",
            "assistant_response": "",
        }
    )

    assert first["assistant_response"] == "echo:My name is Hasib.|turns:1"
    assert second["assistant_response"] == "echo:What is my name?|turns:2"
    assert len(second["chat_history"]) == 5
