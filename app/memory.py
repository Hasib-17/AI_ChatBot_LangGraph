from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


ROLE_TO_CLASS = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
}


def message_to_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, ToolMessage):
        return "tool"
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def message_to_metadata(message: BaseMessage) -> str:
    metadata: dict[str, object] = {}
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        metadata["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        metadata["tool_call_id"] = message.tool_call_id
        if message.name:
            metadata["name"] = message.name
    return json.dumps(metadata, sort_keys=True)


def role_to_message(role: str, content: str, metadata_json: str | None = None) -> BaseMessage:
    message_cls = ROLE_TO_CLASS.get(role)
    if message_cls is None:
        raise ValueError(f"Unsupported role in storage: {role}")

    metadata = json.loads(metadata_json or "{}")
    if message_cls is AIMessage:
        return AIMessage(content=content, tool_calls=metadata.get("tool_calls", []))
    if message_cls is ToolMessage:
        tool_call_id = metadata.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Stored tool message is missing tool_call_id")
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=metadata.get("name"))
    return message_cls(content=content)


@dataclass(frozen=True)
class StoredMessage:
    id: int
    message: BaseMessage


@dataclass(frozen=True)
class SummaryState:
    summary_text: str
    summarized_upto_message_id: int


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
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
            }
            if "metadata" not in columns:
                conn.execute(
                    "ALTER TABLE chat_messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id
                ON chat_messages (session_id, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    summarized_upto_message_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        logger.info("SQLite chat history store ready at %s", self.database_path)

    def load_history(self, session_id: str) -> List[BaseMessage]:
        return [stored.message for stored in self.load_history_with_ids(session_id)]

    def load_history_with_ids(self, session_id: str) -> List[StoredMessage]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, metadata
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        history = [
            StoredMessage(
                id=int(row["id"]),
                message=role_to_message(row["role"], row["content"], row["metadata"]),
            )
            for row in rows
        ]
        logger.debug("Loaded %s messages for session %s", len(history), session_id)
        return history

    def append_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if not messages:
            return

        payload = [
            (session_id, message_to_role(message), message.content, message_to_metadata(message))
            for message in messages
        ]
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO chat_messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                payload,
            )
        logger.debug("Persisted %s messages for session %s", len(messages), session_id)

    def load_summary_state(self, session_id: str) -> SummaryState:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT summary_text, summarized_upto_message_id
                FROM chat_summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return SummaryState(summary_text="", summarized_upto_message_id=0)
        return SummaryState(
            summary_text=row["summary_text"],
            summarized_upto_message_id=int(row["summarized_upto_message_id"]),
        )

    def upsert_summary_state(
        self, session_id: str, summary_text: str, summarized_upto_message_id: int
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_summaries (
                    session_id,
                    summary_text,
                    summarized_upto_message_id,
                    updated_at
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    summarized_upto_message_id = excluded.summarized_upto_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, summary_text, summarized_upto_message_id),
            )
        logger.debug("Persisted summary state for session %s", session_id)

    def clear_session(self, session_id: str) -> bool:
        with self.connection() as conn:
            cursor1 = conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?",
                (session_id,),
            )
            cursor2 = conn.execute(
                "DELETE FROM chat_summaries WHERE session_id = ?",
                (session_id,),
            )
            found = cursor1.rowcount > 0 or cursor2.rowcount > 0

        if found:
            logger.info("Cleared chat history for session %s", session_id)
        return found

    def session_exists(self, session_id: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chat_messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if row:
                return True
            row = conn.execute(
                "SELECT 1 FROM chat_summaries WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None

    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, MAX(last_active) as last_active
                FROM (
                    SELECT session_id, created_at as last_active FROM chat_messages
                    UNION ALL
                    SELECT session_id, updated_at as last_active FROM chat_summaries
                )
                GROUP BY session_id
                ORDER BY last_active DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]
