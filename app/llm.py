from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the chatbot.")

    return ChatOpenAI(
        model=settings.model_name,
        temperature=0,
        api_key=settings.openai_api_key,
    )
