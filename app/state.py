from __future__ import annotations

from typing import List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    chat_history: List[BaseMessage]
    user_message: str
    assistant_response: str
