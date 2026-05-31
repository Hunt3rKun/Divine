"""ZhipuAI provider adapter."""

from __future__ import annotations

from typing import Any

from divine.llm.config import LLMSettings
from divine.llm.providers.base import LLMProvider, import_or_raise
from divine.llm.types import LLMRequest, LLMResponse, TokenUsage


class ZhipuProvider(LLMProvider):
    provider_name = "zhipu"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        zhipuai = import_or_raise("zhipuai", "zhipuai")
        kwargs: dict[str, Any] = {
            "api_key": self._require_api_key("providers.zhipu.api_key"),
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = zhipuai.ZhipuAI(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self._model_for(request)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _messages_with_system(request),
            "max_tokens": self._max_tokens_for(request),
        }
        temperature = self._temperature_for(request)
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(dict(request.extra))

        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message
        return LLMResponse(
            provider=self.provider_name,
            model=getattr(response, "model", model) or model,
            content=getattr(message, "content", None) or "",
            reasoning_content=getattr(message, "reasoning_content", None),
            usage=TokenUsage.from_raw(getattr(response, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )


def _messages_with_system(request: LLMRequest) -> list[dict[str, Any]]:
    messages = request.normalized_messages()
    if request.system:
        messages = [{"role": "system", "content": request.system}, *messages]
    return messages
