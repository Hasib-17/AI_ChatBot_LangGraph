from __future__ import annotations

from typing import List, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage


class AgentState(TypedDict):
    session_id: str
    chat_history: List[BaseMessage]
    user_message: str
    assistant_response: str
    route: NotRequired[Literal["direct", "tool"]]
    model_messages: NotRequired[List[BaseMessage]]
    router_message: NotRequired[AIMessage]
    tool_messages: NotRequired[List[BaseMessage]]
    persist_system_message: NotRequired[bool]
    fallback_note: NotRequired[str]
