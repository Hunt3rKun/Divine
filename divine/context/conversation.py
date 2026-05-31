"""Short-lived task conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from divine.context.types import Message


@dataclass
class ConversationMemory:
    task_id: str
    max_recent_messages: int = 8
    summary: str | None = None
    messages: list[Message] = field(default_factory=list)

    def append(self, role: str, content: object) -> None:
        self.messages.append(Message(role=role, content=content))

    def recent_messages(self) -> list[Message]:
        return self.messages[-self.max_recent_messages :]

    def needs_compaction(self) -> bool:
        return len(self.messages) > self.max_recent_messages

    def compact(self, summary: str) -> None:
        self.summary = summary
        self.messages = self.recent_messages()

    def to_context_payload(self) -> Mapping[str, object]:
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "recent_messages": [message.as_dict() for message in self.recent_messages()],
            "total_messages": len(self.messages),
        }
