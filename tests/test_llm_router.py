from unittest.mock import AsyncMock

import pytest

from divine.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage
from divine.llm.router import ROUTE_MAP, LLMRouter
from divine.config import LLMConfig, ProviderConfig


class TestRouteMap:
    def test_openai_models(self):
        assert "gpt-" in ROUTE_MAP
        assert "o4" in ROUTE_MAP

    def test_anthropic_models(self):
        assert "claude-" in ROUTE_MAP

    def test_zhipu_models(self):
        assert "glm-" in ROUTE_MAP

    def test_minimax_models(self):
        assert "minimax" in ROUTE_MAP


class TestLLMRouter:
    def _make_config(self) -> LLMConfig:
        return LLMConfig(
            providers={
                "openai": ProviderConfig(api_key="sk-test"),
                "anthropic": ProviderConfig(api_key="sk-ant-test"),
            }
        )

    def test_route_openai(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("gpt-4o") == "openai"

    def test_route_openai_o4(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("o4-mini") == "openai"

    def test_route_anthropic(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_route_glm(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("glm-4") == "zhipu"

    def test_route_minimax(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("minimax-text-01") == "minimax"

    def test_route_unknown_falls_back(self):
        router = LLMRouter(self._make_config())
        assert router._resolve_provider("some-unknown-model") == "openai_compat"

    def test_get_provider_returns_configured(self):
        router = LLMRouter(self._make_config())
        provider = router.get_provider("gpt-4o")
        from divine.llm.providers.openai import OpenAIProvider
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_raises_when_no_provider_and_no_fallback(self):
        router = LLMRouter(self._make_config())
        # Remove providers and fallback so nothing is available
        router._providers.clear()
        router._fallback = None
        with pytest.raises(ValueError, match="No provider available"):
            router.get_provider("unknown-model")

    def test_get_provider_uses_fallback_for_unconfigured(self):
        config = LLMConfig(
            providers={
                "openai_compat": ProviderConfig(api_key="test", base_url="http://localhost:11434/v1"),
            }
        )
        router = LLMRouter(config)
        # zhipu is not configured, should fall back to openai_compat
        provider = router.get_provider("glm-4")
        from divine.llm.providers.openai_compat import OpenAICompatProvider
        assert isinstance(provider, OpenAICompatProvider)

    @pytest.mark.asyncio
    async def test_chat_routes_to_correct_provider(self):
        config = self._make_config()
        router = LLMRouter(config)
        mock_response = LLMResponse(
            content="test response",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.chat.return_value = mock_response
        router._providers["openai"] = mock_provider

        messages = [LLMMessage(role="user", content="hello")]
        result = await router.chat("gpt-4o", messages)

        assert result.content == "test response"
        mock_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_attaches_cost(self):
        config = LLMConfig(
            providers={"openai": ProviderConfig(api_key="sk-test")},
            pricing={"gpt-4o": {"input": 2.5, "output": 10.0}},
        )
        router = LLMRouter(config)
        mock_response = LLMResponse(
            content="hello",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000),
        )
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.chat.return_value = mock_response
        router._providers["openai"] = mock_provider

        messages = [LLMMessage(role="user", content="hi")]
        result = await router.chat("gpt-4o", messages)

        # 1M input @ $2.5 + 1M output @ $10 = $12.5
        assert result.cost == pytest.approx(12.5)
