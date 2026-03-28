from collections.abc import AsyncIterator

from divine.config import ProviderConfig
from divine.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage


class ZhipuProvider(LLMProvider):
    """Provider for ZhipuAI (GLM models). Uses the synchronous zhipuai SDK."""

    def __init__(self, config: ProviderConfig):
        try:
            import zhipuai
        except ImportError as e:
            raise ImportError(
                "zhipuai package is required for ZhipuProvider. "
                "Install it with: pip install zhipuai"
            ) from e
        self._zhipuai = zhipuai
        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = zhipuai.ZhipuAI(**kwargs)

    def _to_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        import asyncio

        model = kwargs.pop("model", "glm-4")

        def _sync_call():
            return self._client.chat.completions.create(
                model=model,
                messages=self._to_messages(messages),
                **kwargs,
            )

        response = await asyncio.get_event_loop().run_in_executor(None, _sync_call)
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
        import asyncio

        model = kwargs.pop("model", "glm-4")

        def _sync_stream():
            return self._client.chat.completions.create(
                model=model,
                messages=self._to_messages(messages),
                stream=True,
                **kwargs,
            )

        stream = await asyncio.get_event_loop().run_in_executor(None, _sync_stream)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
