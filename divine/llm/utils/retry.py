import asyncio
import random
from dataclasses import dataclass

from loguru import logger


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


def _is_retryable(e: Exception) -> bool:
    """Check if an exception is retryable (network errors + rate limits)."""
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    # Handle SDK-specific rate limit / server errors by class name
    # to avoid hard dependency on openai/anthropic packages
    cls_name = type(e).__name__
    if cls_name in ("RateLimitError", "APIStatusError", "InternalServerError"):
        return True
    # Check for HTTP status code on the exception
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status and isinstance(status, int) and status in (429, 500, 502, 503, 504):
        return True
    return False


class RetryHandler:
    def __init__(self, config: RetryConfig = None):
        self._config = config or RetryConfig()

    async def execute(self, fn, *args, **kwargs):
        last_error = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e):
                    raise
                last_error = e
                if attempt == self._config.max_attempts:
                    raise
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt}/{self._config.max_attempts} "
                    f"after {delay:.1f}s: {type(e).__name__}: {str(e)[:100]}"
                )
                await asyncio.sleep(delay)
        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        delay = self._config.base_delay * (self._config.exponential_base ** (attempt - 1))
        delay = min(delay, self._config.max_delay)
        delay *= (0.5 + random.random() * 0.5)
        return delay
