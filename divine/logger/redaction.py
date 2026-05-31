"""Helpers for summarizing and redacting trace payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "secret",
    "session",
    "token",
)


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_data(value: Any, key: str | None = None) -> Any:
    if key and is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): redact_data(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_data(item) for item in value]
    return value


def summarize_mapping(
    data: Mapping[str, Any],
    *,
    preview_chars: int,
    large_value_threshold_chars: int,
) -> dict[str, dict[str, Any]]:
    return {
        key: summarize_value(
            value,
            key=key,
            preview_chars=preview_chars,
            large_value_threshold_chars=large_value_threshold_chars,
        )
        for key, value in data.items()
    }


def summarize_value(
    value: Any,
    *,
    key: str,
    preview_chars: int,
    large_value_threshold_chars: int,
) -> dict[str, Any]:
    text = stringify(value)
    summary: dict[str, Any] = {
        "type": type(value).__name__,
        "length": len(text),
        "large": len(text) > large_value_threshold_chars,
    }

    if is_sensitive_key(key):
        summary["redacted"] = True
        return summary

    summary["sha256"] = sha256_text(text)
    summary["preview"] = text[:preview_chars]
    if len(text) > preview_chars:
        summary["preview_truncated"] = True
    return summary


def summarize_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    preview_chars: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        text = stringify(content)
        items.append(
            {
                "role": message.get("role"),
                "content_length": len(text),
                "content_sha256": sha256_text(text),
                "content_preview": text[:preview_chars],
            }
        )
    return items


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
