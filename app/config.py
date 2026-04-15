from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "groq"
    openai_api_key: str = ""
    groq_api_key: str = ""
    model_name: str = "llama3-8b-8192"
    system_prompt: str = (
        "You are a helpful AI assistant with persistent memory. Use the full "
        "conversation history to answer consistently and contextually."
    )
    database_path: str = "data/chat_memory.db"
    memory_strategy: str = "sliding_window"
    memory_window_size: int = 12
    max_context_tokens: int = 6000
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "groq").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            model_name=os.getenv(
                "MODEL_NAME",
                os.getenv(
                    "OPENAI_MODEL",
                    "llama3-8b-8192",
                ),
            ),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a helpful AI assistant with persistent memory. Use the "
                "full conversation history to answer consistently and contextually.",
            ),
            database_path=os.getenv("DATABASE_PATH", "data/chat_memory.db"),
            memory_strategy=os.getenv("MEMORY_STRATEGY", "sliding_window").lower(),
            memory_window_size=int(os.getenv("MEMORY_WINDOW_SIZE", "12")),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "6000")),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        if self.llm_provider not in {"openai", "groq"}:
            raise ConfigError("LLM_PROVIDER must be either 'openai' or 'groq'.")
        if self.llm_provider == "openai" and not self.openai_api_key.strip():
            raise ConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        if self.llm_provider == "groq" and not self.groq_api_key.strip():
            raise ConfigError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
        if not self.model_name.strip():
            raise ConfigError("MODEL_NAME must not be empty.")
        if not self.database_path.strip():
            raise ConfigError("DATABASE_PATH must not be empty.")
        if self.memory_strategy not in {"sliding_window", "summary_window"}:
            raise ConfigError(
                "MEMORY_STRATEGY must be either 'sliding_window' or 'summary_window'."
            )
        if self.memory_window_size < 1:
            raise ConfigError("MEMORY_WINDOW_SIZE must be at least 1.")
        if self.max_context_tokens < 128:
            raise ConfigError("MAX_CONTEXT_TOKENS must be at least 128.")
        if not self.api_host.strip():
            raise ConfigError("API_HOST must not be empty.")
        if not 1 <= self.api_port <= 65535:
            raise ConfigError("API_PORT must be between 1 and 65535.")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
