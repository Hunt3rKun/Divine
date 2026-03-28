import asyncio
import random
from dataclasses import dataclass

from loguru import logger


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


class RetryHandler:
    def __init__(self, config: RetryConfig = None):
        self._config = config or RetryConfig()

    async def execute(self, fn, *args, **kwargs):
        last_error = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt == self._config.max_attempts:
                    raise
                delay = self._calculate_delay(attempt)
                logger.warning(f"Retry {attempt}/{self._config.max_attempts} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            except Exception:
                raise
        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        delay = self._config.base_delay * (self._config.exponential_base ** (attempt - 1))
        delay = min(delay, self._config.max_delay)
        delay *= (0.5 + random.random() * 0.5)
        return delay
