"""Common LLM request, response, and usage types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Message:
    """Normalized chat message."""

    role: str
    content: Any

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    """Provider-neutral token accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, usage: Any) -> "TokenUsage":
        """Normalize token usage from SDK response objects or dictionaries."""

        if usage is None:
            return cls()

        raw = _to_plain_data(usage)
        prompt_tokens = _pick_int(raw, "prompt_tokens", "input_tokens")
        completion_tokens = _pick_int(raw, "completion_tokens", "output_tokens")
        total_tokens = _pick_int(raw, "total_tokens")

        prompt_details = _pick_mapping(raw, "prompt_tokens_details", "input_tokens_details")
        completion_details = _pick_mapping(
            raw,
            "completion_tokens_details",
            "output_tokens_details",
        )

        cached_tokens = _pick_int(prompt_details, "cached_tokens", "cache_read_input_tokens")
        cached_tokens += _pick_int(raw, "cache_read_input_tokens")
        cached_tokens += _pick_int(raw, "prompt_cache_hit_tokens")
        cache_miss_tokens = _pick_int(raw, "prompt_cache_miss_tokens")
        reasoning_tokens = _pick_int(completion_details, "reasoning_tokens")
        reasoning_tokens += _pick_int(raw, "reasoning_tokens")

        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_miss_tokens=cache_miss_tokens,
            reasoning_tokens=reasoning_tokens,
            raw=raw if isinstance(raw, dict) else {"value": raw},
        )

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            raw={"items": [self.raw, other.raw]},
        )


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral generation request."""

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


@dataclass
class LLMResponse:
    """Provider-neutral generation response."""

    provider: str
    model: str
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    reasoning_content: str | None = None
    raw: Any = None


def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain_data(value.model_dump())
    if hasattr(value, "to_dict"):
        return _to_plain_data(value.to_dict())

    fields = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        if isinstance(item, (str, int, float, bool, type(None), Mapping, list)):
            fields[key] = _to_plain_data(item)
    return fields


def _pick_mapping(data: Any, *keys: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _pick_int(data: Any, *keys: str) -> int:
    if not isinstance(data, Mapping):
        return 0
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0
