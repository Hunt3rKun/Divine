"""
重试处理器
"""
import time
import asyncio
import random
import logging
from typing import Callable, TypeVar, Any, Coroutine
from functools import wraps

from .llm_config import RetryConfig
from .models import LLMError, RetryableError, NonRetryableError


logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryHandler:
    """
    重试处理器

    重试策略：
    - 可重试：网络超时、速率限制(429)、服务端错误(5xx)
    - 不重试：参数错误(4xx 非429)、认证失败(401/403)
    - 采用指数退避 + 抖动
    """

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        执行函数，自动处理重试（同步版本）

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            LLMError: 重试耗尽或不可重试的错误
        """
        last_error = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)

            except NonRetryableError:
                # 不可重试，直接抛出
                raise

            except RetryableError as e:
                last_error = e

                if attempt == self.config.max_attempts:
                    logger.error(f"重试耗尽，共尝试 {attempt} 次: {e}")
                    raise

                # 计算退避延迟
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"第 {attempt} 次调用失败，{delay:.2f}s 后重试: {e}"
                )
                time.sleep(delay)

            except Exception as e:
                # 未知错误，包装后抛出
                raise RetryableError(str(e), raw_error=e)

        # 理论上不会到达这里
        raise last_error

    async def execute_async(self, func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs) -> T:
        """
        执行异步函数，自动处理重试（异步版本）

        Args:
            func: 要执行的异步函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            LLMError: 重试耗尽或不可重试的错误
        """
        last_error = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)

            except NonRetryableError:
                raise

            except RetryableError as e:
                last_error = e

                if attempt == self.config.max_attempts:
                    logger.error(f"重试耗尽，共尝试 {attempt} 次: {e}")
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"第 {attempt} 次调用失败，{delay:.2f}s 后重试: {e}"
                )
                await asyncio.sleep(delay)

            except Exception as e:
                raise RetryableError(str(e), raw_error=e)

        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        """
        计算退避延迟（指数退避 + 抖动）

        delay = min(base_delay * (exponential_base ^ attempt) + jitter, max_delay)
        """
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        # 添加抖动（±20%）
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        delay += jitter

        # 限制最大延迟
        return min(delay, self.config.max_delay)


def with_retry(config: RetryConfig = None):
    """
    重试装饰器（同步版本）

    使用方式:
        @with_retry(RetryConfig(max_attempts=5))
        def my_function():
            ...
    """
    handler = RetryHandler(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return handler.execute(func, *args, **kwargs)
        return wrapper

    return decorator


def with_retry_async(config: RetryConfig = None):
    """
    重试装饰器（异步版本）

    使用方式:
        @with_retry_async(RetryConfig(max_attempts=5))
        async def my_async_function():
            ...
    """
    handler = RetryHandler(config)

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await handler.execute_async(func, *args, **kwargs)
        return wrapper

    return decorator


# HTTP 状态码分类
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


def classify_http_error(status_code: int, message: str = "") -> LLMError:
    """
    根据 HTTP 状态码分类错误

    Args:
        status_code: HTTP 状态码
        message: 错误信息

    Returns:
        对应的错误类型
    """
    if status_code in RETRYABLE_STATUS_CODES:
        return RetryableError(f"HTTP {status_code}: {message}")
    elif status_code in NON_RETRYABLE_STATUS_CODES:
        return NonRetryableError(f"HTTP {status_code}: {message}")
    elif status_code >= 500:
        return RetryableError(f"HTTP {status_code}: {message}")
    else:
        return NonRetryableError(f"HTTP {status_code}: {message}")