"""LLM configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from divine.llm.catalog import DEFAULT_MODELS
from divine.llm.errors import LLMConfigurationError


DEFAULT_CONFIG_PATH = Path("config/llm.json")

@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    default_max_tokens: int = 4096
    default_temperature: float | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> "LLMSettings":
        config_path = Path(path)
        if not config_path.exists():
            raise LLMConfigurationError(
                f"LLM config file not found: {config_path}. Copy config/llm.example.json to config/llm.json."
            )

        data = _load_json_mapping(config_path)
        resolved_provider = str(provider or data.get("provider") or "openai")
        if resolved_provider not in DEFAULT_MODELS:
            known = ", ".join(sorted(DEFAULT_MODELS))
            raise LLMConfigurationError(f"Unknown LLM provider '{resolved_provider}'. Known providers: {known}")

        provider_configs = data.get("providers", {})
        if not isinstance(provider_configs, Mapping):
            raise LLMConfigurationError("'providers' must be an object in LLM config.")

        provider_config = provider_configs.get(resolved_provider, {})
        if not isinstance(provider_config, Mapping):
            raise LLMConfigurationError(f"Provider config for '{resolved_provider}' must be an object.")

        generation_config = data.get("generation", {})
        if not isinstance(generation_config, Mapping):
            raise LLMConfigurationError("'generation' must be an object in LLM config.")

        timeout = _optional_float(data.get("timeout"))
        max_tokens = _optional_int(generation_config.get("max_tokens")) or 4096
        temperature = _optional_float(generation_config.get("temperature"))
        extra = provider_config.get("extra", {})
        if not isinstance(extra, Mapping):
            raise LLMConfigurationError(f"Provider extra config for '{resolved_provider}' must be an object.")

        return cls(
            provider=resolved_provider,
            model=str(model or provider_config.get("model") or DEFAULT_MODELS[resolved_provider]),
            api_key=_optional_str(provider_config.get("api_key")),
            base_url=_optional_str(provider_config.get("base_url")),
            timeout=timeout,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            extra=dict(extra),
        )


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError(f"Invalid JSON in LLM config: {path}") from exc

    if not isinstance(data, dict):
        raise LLMConfigurationError("LLM config root must be a JSON object.")
    return data


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
