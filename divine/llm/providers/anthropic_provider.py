"""Anthropic provider adapter."""

from __future__ import annotations

from typing import Any

from divine.llm.config import LLMSettings
from divine.llm.types import LLMRequest, LLMResponse, TokenUsage
from divine.llm.providers.base import LLMProvider, import_or_raise


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        anthropic = import_or_raise("anthropic", "anthropic")
        kwargs: dict[str, Any] = {
            "api_key": self._require_api_key("providers.anthropic.api_key"),
        }
        if settings.timeout:
            kwargs["timeout"] = settings.timeout
        self._client = anthropic.Anthropic(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self._model_for(request)
        payload: dict[str, Any] = {
            "model": model,
            "messages": request.normalized_messages(),
            "max_tokens": self._max_tokens_for(request),
        }
        if request.system:
            payload["system"] = request.system
        temperature = self._temperature_for(request)
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(dict(request.extra))

        response = self._client.messages.create(**payload)
        return LLMResponse(
            provider=self.provider_name,
            model=getattr(response, "model", model) or model,
            content=_extract_text(response),
            usage=TokenUsage.from_raw(getattr(response, "usage", None)),
            finish_reason=getattr(response, "stop_reason", None),
            raw=response,
        )


def _extract_text(response: Any) -> str:
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)
