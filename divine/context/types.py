"""Provider-neutral request primitives for context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Message:
    role: str
    content: Any

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[Message | Mapping[str, Any]]
    model: str | None = None
    system: Any | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    agent: str | None = None
    prompt_trace: Mapping[str, Any] | None = None
    trace_metadata: Mapping[str, Any] | None = None

    def normalized_messages(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in self.messages:
            if isinstance(message, Message):
                items.append(message.as_dict())
            else:
                items.append(dict(message))
        return items
