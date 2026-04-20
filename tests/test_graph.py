from __future__ import annotations

from langchain_core.messages import AIMessage

from app.context_window import estimate_context_tokens
from app.graph import build_chat_graph
from app.memory import SQLiteChatHistoryStore


class StubLLM:
    def invoke(self, messages):
        human_messages = [message.content for message in messages if message.type == "human"]
        return AIMessage(content=f"echo:{human_messages[-1]}|turns:{len(human_messages)}")


class CapturingLLM:
    def __init__(self):
        self.calls = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        human_messages = [message.content for message in messages if message.type == "human"]
        return AIMessage(content=f"echo:{human_messages[-1]}|turns:{len(human_messages)}")


class ToolRoutingLLM:
    def __init__(self):
        self.calls = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        last_human = next(
            (message.content for message in reversed(messages) if message.type == "human"),
            "",
        )
        tool_messages = [message for message in messages if message.type == "tool"]

        if "time" in last_human.lower() and not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "current_utc_datetime",
                        "args": {},
                        "id": "call-utc-time",
                        "type": "tool_call",
                    }
                ],
            )

        if tool_messages:
            return AIMessage(content=f"UTC time is {tool_messages[-1].content}")

        return AIMessage(content=f"direct:{last_human}")


class FailingToolRoutingLLM(ToolRoutingLLM):
    def invoke(self, messages):
        self.calls.append(list(messages))
        last_human = next(
            (message.content for message in reversed(messages) if message.type == "human"),
            "",
        )
        tool_messages = [message for message in messages if message.type == "tool"]

        if "time" in last_human.lower() and not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "broken_time_tool",
                        "args": {},
                        "id": "call-broken-time",
                        "type": "tool_call",
                    }
                ],
            )

        if tool_messages:
            return AIMessage(content=f"fallback:{tool_messages[-1].content}")

        return AIMessage(content=f"direct:{last_human}")


def broken_time_tool():
    raise RuntimeError("boom")


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


def test_graph_uses_tool_for_time_questions(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    llm = ToolRoutingLLM()
    graph = build_chat_graph(store=store, llm=llm, system_prompt="system prompt")

    result = graph.invoke(
        {
            "session_id": "tool-session",
            "chat_history": [],
            "user_message": "What time is it?",
            "assistant_response": "",
        }
    )

    assert result["assistant_response"].startswith("UTC time is ")
    assert len(llm.calls) == 2
    history = store.load_history("tool-session")
    assert any(message.type == "tool" for message in history)
    assert len(result["chat_history"]) == 5


def test_graph_answers_directly_for_non_tool_questions(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    llm = ToolRoutingLLM()
    graph = build_chat_graph(store=store, llm=llm, system_prompt="system prompt")

    result = graph.invoke(
        {
            "session_id": "direct-session",
            "chat_history": [],
            "user_message": "Explain recursion",
            "assistant_response": "",
        }
    )

    assert result["assistant_response"] == "direct:Explain recursion"
    assert len(llm.calls) == 1
    history = store.load_history("direct-session")
    assert all(message.type != "tool" for message in history)


def test_graph_degrades_gracefully_when_tool_fails(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    llm = FailingToolRoutingLLM()
    graph = build_chat_graph(
        store=store,
        llm=llm,
        system_prompt="system prompt",
        tools={"broken_time_tool": broken_time_tool},
    )

    result = graph.invoke(
        {
            "session_id": "broken-tool-session",
            "chat_history": [],
            "user_message": "What time is it?",
            "assistant_response": "",
        }
    )

    assert result["assistant_response"].startswith("fallback:Tool 'broken_time_tool' failed: boom")
    assert len(llm.calls) == 2
    assert any(
        message.type == "tool" and "failed" in message.content.lower()
        for message in store.load_history("broken-tool-session")
    )
    assert any(
        message.type == "tool" and "failed" in message.content.lower()
        for message in llm.calls[-1]
    )


def test_sliding_window_drops_oldest_pair_at_boundary(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    llm = CapturingLLM()
    graph = build_chat_graph(
        store=store,
        llm=llm,
        system_prompt="system prompt",
        memory_strategy="sliding_window",
        memory_window_size=2,
        max_context_tokens=6000,
    )

    for turn in range(1, 5):
        graph.invoke(
            {
                "session_id": "window-boundary",
                "chat_history": [],
                "user_message": f"turn-{turn}",
                "assistant_response": "",
            }
        )

    last_call = llm.calls[-1]
    human_messages = [message.content for message in last_call if message.type == "human"]
    assert last_call[0].type == "system"
    assert human_messages == ["turn-2", "turn-3", "turn-4"]


def test_summary_window_persists_summary_across_restart(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    first_llm = CapturingLLM()
    first_graph = build_chat_graph(
        store=store,
        llm=first_llm,
        system_prompt="system prompt",
        memory_strategy="summary_window",
        memory_window_size=1,
        max_context_tokens=6000,
    )

    for turn in range(1, 4):
        first_graph.invoke(
            {
                "session_id": "persisted-summary",
                "chat_history": [],
                "user_message": f"turn-{turn}",
                "assistant_response": "",
            }
        )

    summary_state = store.load_summary_state("persisted-summary")
    assert summary_state.summary_text.strip() != ""
    assert summary_state.summarized_upto_message_id > 0

    restarted_llm = CapturingLLM()
    restarted_graph = build_chat_graph(
        store=store,
        llm=restarted_llm,
        system_prompt="system prompt",
        memory_strategy="summary_window",
        memory_window_size=1,
        max_context_tokens=6000,
    )
    restarted_graph.invoke(
        {
            "session_id": "persisted-summary",
            "chat_history": [],
            "user_message": "turn-4",
            "assistant_response": "",
        }
    )

    restart_call = restarted_llm.calls[-1]
    assert restart_call[0].type == "system"
    assert any(
        message.type == "system" and message.content.startswith("Conversation summary:")
        for message in restart_call[1:]
    )


def test_hundred_turns_stay_within_token_cap(tmp_path):
    store = SQLiteChatHistoryStore(str(tmp_path / "chat.db"))
    llm = CapturingLLM()
    max_context_tokens = 180
    graph = build_chat_graph(
        store=store,
        llm=llm,
        system_prompt="system prompt",
        memory_strategy="summary_window",
        memory_window_size=3,
        max_context_tokens=max_context_tokens,
    )

    for turn in range(1, 101):
        graph.invoke(
            {
                "session_id": "long-session",
                "chat_history": [],
                "user_message": f"turn-{turn} {'x' * 90}",
                "assistant_response": "",
            }
        )

    assert len(llm.calls) == 100
    assert all(
        estimate_context_tokens(messages) <= max_context_tokens for messages in llm.calls
    )
