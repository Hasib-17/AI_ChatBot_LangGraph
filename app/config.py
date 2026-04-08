from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful AI assistant with persistent memory. Use the full "
        "conversation history to answer consistently and contextually."
    )
    database_path: str = "data/chat_memory.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a helpful AI assistant with persistent memory. Use the "
                "full conversation history to answer consistently and contextually.",
            ),
            database_path=os.getenv("DATABASE_PATH", "data/chat_memory.db"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
