"""Alibaba DashScope provider adapter."""

from __future__ import annotations

from typing import Any

from divine.llm.config import LLMSettings
from divine.llm.providers.base import LLMProvider, import_or_raise
from divine.llm.types import LLMRequest, LLMResponse, TokenUsage


class DashScopeProvider(LLMProvider):
    provider_name = "dashscope"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self._dashscope = import_or_raise("dashscope", "dashscope")
        self._dashscope.api_key = self._require_api_key("providers.dashscope.api_key")
        if settings.base_url:
            self._dashscope.base_http_api_url = settings.base_url

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self._model_for(request)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _messages_with_system(request),
            "result_format": "message",
            "max_tokens": self._max_tokens_for(request),
        }
        temperature = self._temperature_for(request)
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(dict(request.extra))

        response = self._dashscope.Generation.call(**payload)
        raw = _response_to_mapping(response)
        message = (
            raw.get("output", {})
            .get("choices", [{}])[0]
            .get("message", {})
        )
        usage = raw.get("usage") or raw.get("output", {}).get("usage")
        return LLMResponse(
            provider=self.provider_name,
            model=model,
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content"),
            usage=TokenUsage.from_raw(usage),
            finish_reason=raw.get("output", {}).get("finish_reason"),
            raw=response,
        )


def _messages_with_system(request: LLMRequest) -> list[dict[str, Any]]:
    messages = request.normalized_messages()
    if request.system:
        messages = [{"role": "system", "content": request.system}, *messages]
    return messages


def _response_to_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return {
        key: getattr(response, key)
        for key in ("status_code", "request_id", "code", "message", "output", "usage")
        if hasattr(response, key)
    }
