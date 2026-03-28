from collections.abc import AsyncIterator

import anthropic

from divine.config import ProviderConfig
from divine.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage


class AnthropicProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        kwargs: dict = {"api_key": config.api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _split_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str, list[dict]]:
        """Separate system prompt from conversation messages."""
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), chat_messages

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "claude-sonnet-4-20250514")
        max_tokens = kwargs.pop("max_tokens", 8192)
        system_prompt, chat_messages = self._split_messages(messages)

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            **kwargs,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await self._client.messages.create(**create_kwargs)

        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return LLMResponse(
            content=content,
            model=response.model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "claude-sonnet-4-20250514")
        max_tokens = kwargs.pop("max_tokens", 8192)
        system_prompt, chat_messages = self._split_messages(messages)

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            **kwargs,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        async with self._client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
