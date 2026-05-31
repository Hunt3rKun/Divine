"""Application logging foundation based on Loguru."""

from loguru import logger

from divine.logger.config import (
    DEFAULT_LOGGING_CONFIG_PATH,
    LLMTraceSettings,
    LoggingSettings,
    configure_logging,
    get_logger,
    get_logging_settings,
)

__all__ = [
    "DEFAULT_LOGGING_CONFIG_PATH",
    "LLMTraceSettings",
    "LoggingSettings",
    "configure_logging",
    "get_logger",
    "get_logging_settings",
    "logger",
]
