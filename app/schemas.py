from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1)


class ChatMessageView(BaseModel):
    role: Literal["system", "human", "ai"]
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessageView]
