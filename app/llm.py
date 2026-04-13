from __future__ import annotations

from app.config import Settings


def build_llm(settings: Settings):
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model_name,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.model_name,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
