from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.context_window import ConversationContextManager
from app.memory import SQLiteChatHistoryStore
from app.state import AgentState

logger = logging.getLogger(__name__)


def build_chat_graph(
    store: SQLiteChatHistoryStore,
    llm,
    system_prompt: str,
    memory_strategy: str = "sliding_window",
    memory_window_size: int = 12,
    max_context_tokens: int = 6000,
):
    context_manager = ConversationContextManager(
        store=store,
        system_prompt=system_prompt,
        memory_strategy=memory_strategy,
        memory_window_size=memory_window_size,
        max_context_tokens=max_context_tokens,
    )

    def process_message(state: AgentState) -> AgentState:
        session_id = state["session_id"]
        user_message = state["user_message"].strip()

        if not user_message:
            raise ValueError("user_message must not be empty")

        history_with_ids = store.load_history_with_ids(session_id)
        context = context_manager.build_context(
            session_id=session_id,
            history_with_ids=history_with_ids,
            user_message=user_message,
        )
        new_messages = []

        if context.persist_system_message:
            new_messages.append(SystemMessage(content=system_prompt))
        new_messages.append(HumanMessage(content=user_message))

        logger.info(
            "Invoking model for session %s with %s messages (~%s/%s tokens)",
            session_id,
            len(context.messages_for_model),
            context.token_estimate,
            max_context_tokens,
        )
        response = llm.invoke(context.messages_for_model)
        assistant_message = AIMessage(content=response.content)
        new_messages.append(assistant_message)

        store.append_messages(session_id, new_messages)
        response_history = [*context.messages_for_model, assistant_message]

        return {
            "session_id": session_id,
            "chat_history": response_history,
            "user_message": user_message,
            "assistant_response": assistant_message.content,
        }

    graph = StateGraph(AgentState)
    graph.add_node("process_message", process_message)
    graph.add_edge(START, "process_message")
    graph.add_edge("process_message", END)
    return graph.compile()
