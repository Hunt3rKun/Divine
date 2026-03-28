from collections.abc import AsyncIterator

import openai

from divine.config import ProviderConfig
from divine.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage


class OpenAIProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        kwargs: dict = {"api_key": config.api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "gpt-4o")
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
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "gpt-4o")
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
