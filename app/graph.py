from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.memory import SQLiteChatHistoryStore
from app.state import AgentState

logger = logging.getLogger(__name__)


def build_chat_graph(store: SQLiteChatHistoryStore, llm, system_prompt: str):
    def process_message(state: AgentState) -> AgentState:
        session_id = state["session_id"]
        user_message = state["user_message"].strip()

        if not user_message:
            raise ValueError("user_message must not be empty")

        history = store.load_history(session_id)
        new_messages = []

        if not history:
            system_message = SystemMessage(content=system_prompt)
            history.append(system_message)
            new_messages.append(system_message)

        human_message = HumanMessage(content=user_message)
        history.append(human_message)
        new_messages.append(human_message)

        logger.info(
            "Invoking model for session %s with %s total messages",
            session_id,
            len(history),
        )
        response = llm.invoke(history)
        assistant_message = AIMessage(content=response.content)
        history.append(assistant_message)
        new_messages.append(assistant_message)

        store.append_messages(session_id, new_messages)

        return {
            "session_id": session_id,
            "chat_history": history,
            "user_message": user_message,
            "assistant_response": assistant_message.content,
        }

    graph = StateGraph(AgentState)
    graph.add_node("process_message", process_message)
    graph.add_edge(START, "process_message")
    graph.add_edge("process_message", END)
    return graph.compile()
