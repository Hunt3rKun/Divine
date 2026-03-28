from typing import Optional

from loguru import logger

from divine.config import LLMConfig, ProviderConfig
from divine.llm.base import LLMMessage, LLMProvider, LLMResponse
from divine.llm.providers import PROVIDER_CLASSES
from divine.llm.utils.cost import CostCalculator
from divine.llm.utils.retry import RetryConfig, RetryHandler

ROUTE_MAP = {
    "gpt-": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "glm-": "zhipu",
    "minimax": "minimax",
}


class LLMRouter:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._providers: dict[str, LLMProvider] = {}
        self._retry = RetryHandler(RetryConfig())
        self._cost_calc = CostCalculator(config.pricing)
        self._fallback: Optional[LLMProvider] = None
        self._init_providers()

    def _init_providers(self) -> None:
        for name, pconf in self._config.providers.items():
            if name in PROVIDER_CLASSES:
                self._providers[name] = PROVIDER_CLASSES[name](pconf)
                logger.debug(f"Initialized provider: {name}")
        compat_config = self._config.providers.get("openai_compat")
        if compat_config:
            self._fallback = PROVIDER_CLASSES["openai_compat"](compat_config)
            logger.debug("Initialized fallback openai_compat provider")

    def _resolve_provider(self, model: str) -> str:
        for prefix, provider_name in ROUTE_MAP.items():
            if model.startswith(prefix):
                return provider_name
        return "openai_compat"

    def get_provider(self, model: str) -> LLMProvider:
        name = self._resolve_provider(model)
        provider = self._providers.get(name)
        if provider:
            return provider
        if self._fallback:
            logger.debug(f"No provider '{name}' configured, falling back to openai_compat")
            return self._fallback
        raise ValueError(f"No provider available for model '{model}'")

    async def chat(self, model: str, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        provider = self.get_provider(model)

        async def _call():
            return await provider.chat(messages, model=model, **kwargs)

        response = await self._retry.execute(_call)
        response.cost = self._cost_calc.calculate(model, response.usage)
        return response

    async def chat_stream(self, model: str, messages: list[LLMMessage], **kwargs):
        provider = self.get_provider(model)
        async for chunk in provider.chat_stream(messages, model=model, **kwargs):
            yield chunk
