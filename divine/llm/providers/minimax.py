from collections.abc import AsyncIterator

import openai

from divine.config import ProviderConfig
from divine.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage

_MINIMAX_BASE_URL = "https://api.minimax.chat/v1"


class MiniMaxProvider(LLMProvider):
    """Provider for MiniMax models via OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        base_url = config.base_url or _MINIMAX_BASE_URL
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=base_url,
            timeout=config.timeout,
        )

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "minimax-text-01")
        response = await self._client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "minimax-text-01")
        stream = await self._client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
