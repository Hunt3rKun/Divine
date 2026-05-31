"""Provider-neutral LLM client."""

from __future__ import annotations

from collections.abc import Callable

from divine.llm.config import LLMSettings
from divine.llm.errors import LLMProviderNotFoundError
from divine.llm.providers.anthropic_provider import AnthropicProvider
from divine.llm.providers.dashscope_provider import DashScopeProvider
from divine.llm.providers.openai_provider import OpenAICompatibleProvider, OpenAIProvider
from divine.llm.providers.zhipu_provider import ZhipuProvider
from divine.llm.types import LLMRequest, LLMResponse, TokenUsage
from divine.logger import get_logger
from divine.logger.trace import LLMTraceRecorder


ProviderFactory = Callable[[LLMSettings], object]


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": lambda settings: OpenAICompatibleProvider(settings, provider_name="deepseek"),
    "dashscope": DashScopeProvider,
    "zhipu": ZhipuProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


class LLMClient:
    """Thin orchestration layer around a concrete provider adapter."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._log = get_logger("llm")
        factory = PROVIDER_FACTORIES.get(settings.provider)
        if not factory:
            known = ", ".join(sorted(PROVIDER_FACTORIES))
            raise LLMProviderNotFoundError(f"Unknown LLM provider '{settings.provider}'. Known providers: {known}")
        self._provider = factory(settings)
        self.total_usage = TokenUsage()
        self._trace_recorder = LLMTraceRecorder()
        self._log.info(
            "LLM client initialized provider={} model={} base_url_configured={} api_key_configured={}",
            settings.provider,
            settings.model,
            bool(settings.base_url),
            bool(settings.api_key),
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.settings.model
        trace_context = self._trace_recorder.start_request(request, provider=self.settings.provider, model=model)
        self._log.debug(
            "LLM request started trace_id={} agent={} template_id={} provider={} model={} message_count={} max_tokens={} metadata={}",
            trace_context.trace_id,
            trace_context.agent,
            trace_context.template_id,
            self.settings.provider,
            model,
            len(request.messages),
            request.max_tokens or self.settings.default_max_tokens,
            dict(trace_context.metadata),
        )
        try:
            response = self._provider.generate(request)  # type: ignore[attr-defined]
        except Exception as exc:
            artifact_path = self._trace_recorder.record_failure(trace_context, request, exc)
            self._log.exception(
                "LLM request failed trace_id={} agent={} template_id={} provider={} model={} artifact_path={} metadata={}",
                trace_context.trace_id,
                trace_context.agent,
                trace_context.template_id,
                self.settings.provider,
                model,
                artifact_path,
                dict(trace_context.metadata),
            )
            raise

        self.total_usage = self.total_usage.add(response.usage)
        artifact_path = self._trace_recorder.record_success(trace_context, request, response)
        self._log.info(
            "LLM request completed trace_id={} agent={} template_id={} provider={} model={} prompt_tokens={} completion_tokens={} total_tokens={} reasoning_tokens={} cached_tokens={} cache_miss_tokens={} artifact_path={} metadata={}",
            trace_context.trace_id,
            trace_context.agent,
            trace_context.template_id,
            response.provider,
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            response.usage.reasoning_tokens,
            response.usage.cached_tokens,
            response.usage.cache_miss_tokens,
            artifact_path,
            dict(trace_context.metadata),
        )
        return response


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    config_path: str | None = None,
    settings: LLMSettings | None = None,
) -> LLMClient:
    return LLMClient(settings or LLMSettings.from_file(config_path or "config/llm.json", provider=provider, model=model))
