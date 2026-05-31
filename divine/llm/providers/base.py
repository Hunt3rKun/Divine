"""Provider adapter base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from divine.llm.config import LLMSettings
from divine.llm.errors import LLMConfigurationError, LLMDependencyError
from divine.llm.types import LLMRequest, LLMResponse


class LLMProvider(ABC):
    provider_name: str

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a single non-streaming response."""

    def _model_for(self, request: LLMRequest) -> str:
        return request.model or self.settings.model

    def _max_tokens_for(self, request: LLMRequest) -> int:
        return request.max_tokens or self.settings.default_max_tokens

    def _temperature_for(self, request: LLMRequest) -> float | None:
        if request.temperature is not None:
            return request.temperature
        return self.settings.default_temperature

    def _require_api_key(self, config_key: str) -> str:
        if not self.settings.api_key:
            raise LLMConfigurationError(f"{self.provider_name} requires '{config_key}' in the LLM config file.")
        return self.settings.api_key


def import_or_raise(module_name: str, package_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise LLMDependencyError(
            f"Missing SDK dependency '{package_name}'. Install project dependencies or run: pip install {package_name}"
        ) from exc
