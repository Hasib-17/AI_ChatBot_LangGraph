from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


ROLE_TO_CLASS = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
}


def message_to_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def role_to_message(role: str, content: str) -> BaseMessage:
    message_cls = ROLE_TO_CLASS.get(role)
    if message_cls is None:
        raise ValueError(f"Unsupported role in storage: {role}")
    return message_cls(content=content)


class SQLiteChatHistoryStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id
                ON chat_messages (session_id, id)
                """
            )
        logger.info("SQLite chat history store ready at %s", self.database_path)

    def load_history(self, session_id: str) -> List[BaseMessage]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        history = [role_to_message(row["role"], row["content"]) for row in rows]
        logger.debug("Loaded %s messages for session %s", len(history), session_id)
        return history

    def append_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if not messages:
            return

        payload = [
            (session_id, message_to_role(message), message.content)
            for message in messages
        ]
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                payload,
            )
        logger.debug("Persisted %s messages for session %s", len(messages), session_id)

    def clear_session(self, session_id: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?",
                (session_id,),
            )
        logger.info("Cleared chat history for session %s", session_id)
