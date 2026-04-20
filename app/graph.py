from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.context_window import ConversationContextManager
from app.memory import SQLiteChatHistoryStore
from app.state import AgentState
from app.tools import DEFAULT_TOOL_REGISTRY

logger = logging.getLogger(__name__)


def _tool_guidance_message(tool_registry: dict[str, Any]) -> SystemMessage:
    tool_lines = []
    for tool_name, tool_object in tool_registry.items():
        description = getattr(tool_object, "description", "") or "No description provided."
        tool_lines.append(f"- {tool_name}: {description}")

    guidance = "Use a tool only when it directly helps answer the user."
    if tool_lines:
        guidance = "Available tools:\n" + "\n".join(tool_lines) + f"\n{guidance}"
    return SystemMessage(content=guidance)


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call is missing a tool name")
    return name


def _tool_call_id(tool_call: dict[str, Any]) -> str:
    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("Tool call is missing an id")
    return tool_call_id


def build_chat_graph(
    store: SQLiteChatHistoryStore,
    llm,
    system_prompt: str,
    memory_strategy: str = "sliding_window",
    memory_window_size: int = 12,
    max_context_tokens: int = 6000,
    tools: dict[str, Any] | None = None,
):
    tool_registry = dict(tools or DEFAULT_TOOL_REGISTRY)
    context_manager = ConversationContextManager(
        store=store,
        system_prompt=system_prompt,
        memory_strategy=memory_strategy,
        memory_window_size=memory_window_size,
        max_context_tokens=max_context_tokens,
    )
    tool_llm = llm.bind_tools(list(tool_registry.values())) if tool_registry and hasattr(llm, "bind_tools") else llm

    def route_message(state: AgentState) -> AgentState:
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
        router_messages = list(context.messages_for_model)
        logger.info(
            "Invoking router for session %s with %s messages (~%s/%s tokens)",
            session_id,
            len(router_messages),
            context.token_estimate,
            max_context_tokens,
        )
        router_response = tool_llm.invoke(router_messages)
        return {
            "session_id": session_id,
            "model_messages": context.messages_for_model,
            "persist_system_message": context.persist_system_message,
            "user_message": user_message,
            "router_message": router_response,
            "route": "tool" if getattr(router_response, "tool_calls", None) else "direct",
        }

    def execute_tool(state: AgentState) -> AgentState:
        router_message = state["router_message"]
        tool_calls = list(getattr(router_message, "tool_calls", None) or [])
        tool_messages: list[BaseMessage] = []

        for tool_call in tool_calls:
            tool_name = _tool_call_name(tool_call)
            tool_call_id = _tool_call_id(tool_call)
            tool_object = tool_registry.get(tool_name)
            if tool_object is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' is unavailable.",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
                continue

            arguments = tool_call.get("args") or {}
            try:
                if hasattr(tool_object, "invoke"):
                    result = tool_object.invoke(arguments)
                else:
                    result = tool_object(**arguments)
            except Exception as exc:  # pragma: no cover - defensive fallback
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' failed: {exc}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=result if isinstance(result, str) else str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        return {
            **state,
            "tool_messages": tool_messages,
        }

    def respond(state: AgentState) -> AgentState:
        session_id = state["session_id"]
        user_message = state["user_message"].strip()
        model_messages = list(state.get("model_messages", []))
        router_message = state["router_message"]
        tool_messages = list(state.get("tool_messages", []))
        persist_messages: list[BaseMessage] = []

        if state.get("persist_system_message"):
            persist_messages.append(SystemMessage(content=system_prompt))

        persist_messages.append(HumanMessage(content=user_message))

        if state.get("route") == "tool":
            final_messages = [*model_messages, router_message, *tool_messages]
            final_response = llm.invoke(final_messages)
            assistant_message = AIMessage(content=final_response.content)
            persist_messages.extend([router_message, *tool_messages, assistant_message])
            response_history = [*model_messages, router_message, *tool_messages, assistant_message]
        else:
            assistant_message = router_message
            persist_messages.append(assistant_message)
            response_history = [*model_messages, assistant_message]

        store.append_messages(session_id, persist_messages)
        return {
            "session_id": session_id,
            "chat_history": response_history,
            "user_message": user_message,
            "assistant_response": assistant_message.content,
        }

    graph = StateGraph(AgentState)
    graph.add_node("route_message", route_message)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("respond", respond)
    graph.add_edge(START, "route_message")
    graph.add_conditional_edges(
        "route_message",
        lambda state: "execute_tool" if state.get("route") == "tool" else "respond",
        {
            "execute_tool": "execute_tool",
            "respond": "respond",
        },
    )
    graph.add_edge("execute_tool", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
