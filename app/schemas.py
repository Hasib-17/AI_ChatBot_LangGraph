from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=8000)

    @field_validator("session_id", "message")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ChatMessageView(BaseModel):
    role: Literal["system", "human", "ai"]
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessageView]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
