"""Model catalog and defaults.

Defaults are intentionally centralized so model switching does not require
touching provider implementations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    label: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    notes: str = ""


MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    "openai": [
        ModelInfo("openai", "gpt-5.5", "GPT-5.5", 1_000_000, 128_000, "OpenAI flagship default."),
        ModelInfo("openai", "gpt-5.4", "GPT-5.4", 1_000_000, 128_000),
        ModelInfo("openai", "gpt-5.4-mini", "GPT-5.4 mini", 400_000, 128_000),
        ModelInfo("openai", "gpt-5.4-nano", "GPT-5.4 nano", 400_000, 128_000),
        ModelInfo("openai", "gpt-4.1", "GPT-4.1", 1_047_576, 32_768),
    ],
    "anthropic": [
        ModelInfo("anthropic", "claude-opus-4-1-20250805", "Claude Opus 4.1", 200_000, 32_000),
        ModelInfo("anthropic", "claude-opus-4-1", "Claude Opus 4.1 alias", 200_000, 32_000),
        ModelInfo("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4", 200_000, 64_000),
        ModelInfo("anthropic", "claude-sonnet-4-0", "Claude Sonnet 4 alias", 200_000, 64_000),
    ],
    "deepseek": [
        ModelInfo("deepseek", "deepseek-v4-flash", "DeepSeek V4 Flash", 1_000_000, 384_000),
        ModelInfo("deepseek", "deepseek-v4-pro", "DeepSeek V4 Pro", 1_000_000, 384_000),
        ModelInfo("deepseek", "deepseek-chat", "DeepSeek Chat compatibility", 128_000, 8_000, "Deprecated after 2026-07-24."),
        ModelInfo("deepseek", "deepseek-reasoner", "DeepSeek Reasoner compatibility", 128_000, 64_000, "Deprecated after 2026-07-24."),
    ],
    "dashscope": [
        ModelInfo("dashscope", "qwen3-max-2026-01-23", "Qwen3 Max snapshot"),
        ModelInfo("dashscope", "qwen3-max", "Qwen3 Max"),
        ModelInfo("dashscope", "qwen3.6-plus", "Qwen3.6 Plus"),
        ModelInfo("dashscope", "qwen3.5-plus", "Qwen3.5 Plus"),
        ModelInfo("dashscope", "qwen3.5-flash", "Qwen3.5 Flash"),
    ],
    "zhipu": [
        ModelInfo("zhipu", "glm-5.1", "GLM-5.1"),
        ModelInfo("zhipu", "glm-4.7", "GLM-4.7"),
        ModelInfo("zhipu", "glm-4.6", "GLM-4.6"),
    ],
    "openai_compatible": [
        ModelInfo("openai_compatible", "moonshot-v1-128k", "Moonshot v1 128k"),
    ],
}

DEFAULT_MODELS: dict[str, str] = {
    provider: models[0].model for provider, models in MODEL_CATALOG.items()
}


def list_models(provider: str | None = None) -> list[ModelInfo]:
    if provider is None:
        return [model for models in MODEL_CATALOG.values() for model in models]
    return list(MODEL_CATALOG.get(provider, []))


def get_default_model(provider: str) -> str:
    return DEFAULT_MODELS[provider]
