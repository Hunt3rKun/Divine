"""OpenAI and OpenAI-compatible provider adapters."""

from __future__ import annotations

from typing import Any

from divine.llm.config import LLMSettings
from divine.llm.errors import LLMConfigurationError
from divine.llm.types import LLMRequest, LLMResponse, TokenUsage
from divine.llm.providers.base import LLMProvider, import_or_raise


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        openai = import_or_raise("openai", "openai")
        kwargs: dict[str, Any] = {
            "api_key": self._require_api_key("providers.openai.api_key"),
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        if settings.timeout:
            kwargs["timeout"] = settings.timeout
        self._client = openai.OpenAI(**kwargs)

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self._model_for(request)
        response = self._client.responses.create(**self._build_payload(request, model))
        content = _extract_response_text(response)
        return LLMResponse(
            provider=self.provider_name,
            model=getattr(response, "model", model) or model,
            content=content,
            usage=TokenUsage.from_raw(getattr(response, "usage", None)),
            finish_reason=getattr(response, "status", None),
            raw=response,
        )

    def _build_payload(self, request: LLMRequest, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": _messages_to_responses_input(request),
            "max_output_tokens": self._max_tokens_for(request),
        }
        temperature = self._temperature_for(request)
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(dict(request.extra))
        return payload


class OpenAICompatibleProvider(LLMProvider):
    """Adapter for providers that expose the OpenAI chat completions protocol."""

    provider_name = "openai_compatible"

    def __init__(self, settings: LLMSettings, provider_name: str | None = None) -> None:
        super().__init__(settings)
        if provider_name:
            self.provider_name = provider_name
        openai = import_or_raise("openai", "openai")
        kwargs: dict[str, Any] = {
            "api_key": self._require_api_key(f"providers.{self.provider_name}.api_key"),
        }
        if not settings.base_url:
            raise LLMConfigurationError(
                f"{self.provider_name} requires 'providers.{self.provider_name}.base_url' in config."
            )
        kwargs["base_url"] = settings.base_url
        if settings.timeout:
            kwargs["timeout"] = settings.timeout
        self._client = openai.OpenAI(**kwargs)

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


def _messages_to_responses_input(request: LLMRequest) -> list[dict[str, Any]]:
    messages = request.normalized_messages()
    if request.system:
        messages = [{"role": "system", "content": request.system}, *messages]
    return messages


def _messages_with_system(request: LLMRequest) -> list[dict[str, Any]]:
    messages = request.normalized_messages()
    if request.system:
        messages = [{"role": "system", "content": request.system}, *messages]
    return messages


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)
