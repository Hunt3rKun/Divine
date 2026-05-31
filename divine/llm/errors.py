"""LLM error types."""


class LLMError(Exception):
    """Base error for LLM provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when provider configuration is missing or invalid."""


class LLMProviderNotFoundError(LLMConfigurationError):
    """Raised when a provider name is unknown."""


class LLMDependencyError(LLMConfigurationError):
    """Raised when a provider SDK is not installed."""
