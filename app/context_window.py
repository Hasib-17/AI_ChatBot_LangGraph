from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.memory import SQLiteChatHistoryStore, StoredMessage


def estimate_context_tokens(messages: Sequence[BaseMessage]) -> int:
    total_chars = sum(len(message.content) for message in messages)
    return max(1, total_chars // 4)


@dataclass(frozen=True)
class ContextBuildResult:
    messages_for_model: List[BaseMessage]
    persist_system_message: bool
    token_estimate: int


class ConversationContextManager:
    def __init__(
        self,
        *,
        store: SQLiteChatHistoryStore,
        system_prompt: str,
        memory_strategy: str,
        memory_window_size: int,
        max_context_tokens: int,
    ) -> None:
        self.store = store
        self.system_prompt = system_prompt
        self.memory_strategy = memory_strategy
        self.memory_window_size = memory_window_size
        self.max_context_tokens = max_context_tokens

    def build_context(
        self,
        *,
        session_id: str,
        history_with_ids: Sequence[StoredMessage],
        user_message: str,
    ) -> ContextBuildResult:
        if not user_message.strip():
            raise ValueError("user_message must not be empty")

        persist_system_message = len(history_with_ids) == 0
        system_message = SystemMessage(content=self.system_prompt)

        non_system_history = [
            record
            for record in history_with_ids
            if not isinstance(record.message, SystemMessage)
        ]
        recent_limit = self.memory_window_size * 2
        older_history = non_system_history[:-recent_limit] if recent_limit else non_system_history
        recent_history = non_system_history[-recent_limit:] if recent_limit else []

        summary_text = ""
        summarized_upto_message_id = 0
        summary_changed = False

        if self.memory_strategy == "summary_window":
            summary_state = self.store.load_summary_state(session_id)
            summary_text = summary_state.summary_text
            summarized_upto_message_id = summary_state.summarized_upto_message_id
            latest_older_id = older_history[-1].id if older_history else 0
            if latest_older_id > summarized_upto_message_id:
                new_records = [
                    record
                    for record in older_history
                    if record.id > summarized_upto_message_id
                ]
                if new_records:
                    summary_text = self._merge_summary(
                        summary_text, self._summarize_records(new_records)
                    )
                    summary_changed = True
                summarized_upto_message_id = latest_older_id
                summary_changed = True

        recent_history, summary_text, summarized_upto_message_id, trim_changed = self._trim_to_token_cap(
            recent_history=recent_history,
            summary_text=summary_text,
            summarized_upto_message_id=summarized_upto_message_id,
            system_message=system_message,
            user_message=user_message,
        )
        summary_changed = summary_changed or trim_changed

        if self.memory_strategy == "summary_window" and summary_changed:
            self.store.upsert_summary_state(
                session_id=session_id,
                summary_text=summary_text,
                summarized_upto_message_id=summarized_upto_message_id,
            )

        messages_for_model = self._compose_context(
            system_message=system_message,
            summary_text=summary_text,
            recent_history=recent_history,
            user_message=user_message,
        )
        token_estimate = estimate_context_tokens(messages_for_model)
        return ContextBuildResult(
            messages_for_model=messages_for_model,
            persist_system_message=persist_system_message,
            token_estimate=token_estimate,
        )

    def _compose_context(
        self,
        *,
        system_message: SystemMessage,
        summary_text: str,
        recent_history: Sequence[StoredMessage],
        user_message: str,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [system_message]
        if self.memory_strategy == "summary_window" and summary_text.strip():
            messages.append(SystemMessage(content=f"Conversation summary:\n{summary_text.strip()}"))
        messages.extend(record.message for record in recent_history)
        messages.append(HumanMessage(content=user_message))
        return messages

    def _trim_to_token_cap(
        self,
        *,
        recent_history: Sequence[StoredMessage],
        summary_text: str,
        summarized_upto_message_id: int,
        system_message: SystemMessage,
        user_message: str,
    ) -> Tuple[List[StoredMessage], str, int, bool]:
        mutable_recent = list(recent_history)
        mutable_summary = summary_text
        mutable_summarized_upto = summarized_upto_message_id
        summary_changed = False

        while True:
            context = self._compose_context(
                system_message=system_message,
                summary_text=mutable_summary,
                recent_history=mutable_recent,
                user_message=user_message,
            )
            if estimate_context_tokens(context) <= self.max_context_tokens:
                break

            if mutable_recent:
                dropped = self._pop_oldest_turn(mutable_recent)
                if self.memory_strategy == "summary_window":
                    mutable_summary = self._merge_summary(
                        mutable_summary, self._summarize_records(dropped)
                    )
                    mutable_summarized_upto = max(
                        mutable_summarized_upto, dropped[-1].id
                    )
                    summary_changed = True
                continue

            if self.memory_strategy == "summary_window" and mutable_summary:
                truncated_summary = self._truncate_summary(mutable_summary)
                if truncated_summary == mutable_summary:
                    break
                mutable_summary = truncated_summary
                summary_changed = True
                continue

            break

        return mutable_recent, mutable_summary, mutable_summarized_upto, summary_changed

    def _pop_oldest_turn(self, recent_history: List[StoredMessage]) -> List[StoredMessage]:
        if (
            len(recent_history) >= 2
            and isinstance(recent_history[0].message, HumanMessage)
            and isinstance(recent_history[1].message, AIMessage)
        ):
            dropped = recent_history[:2]
            del recent_history[:2]
            return dropped
        return [recent_history.pop(0)]

    def _summarize_records(self, records: Sequence[StoredMessage]) -> str:
        lines: List[str] = []
        for record in records:
            if isinstance(record.message, HumanMessage):
                speaker = "User"
            elif isinstance(record.message, AIMessage):
                speaker = "Assistant"
            else:
                speaker = "Message"
            text = record.message.content.replace("\n", " ").strip()
            if len(text) > 180:
                text = f"{text[:177]}..."
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def _merge_summary(self, existing_summary: str, new_summary: str) -> str:
        if not new_summary.strip():
            return existing_summary
        if existing_summary.strip():
            merged = f"{existing_summary.strip()}\n{new_summary.strip()}"
        else:
            merged = new_summary.strip()

        max_summary_chars = self.max_context_tokens * 3
        if len(merged) > max_summary_chars:
            tail = merged[-max_summary_chars:]
            merged = f"[Earlier summary truncated]\n{tail}"
        return merged

    def _truncate_summary(self, summary_text: str) -> str:
        if len(summary_text) <= 500:
            return summary_text
        reduced = summary_text[len(summary_text) // 3 :]
        return f"[Earlier summary truncated]\n{reduced.strip()}"
