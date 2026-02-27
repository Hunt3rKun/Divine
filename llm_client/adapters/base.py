from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Type
import asyncio

from logger_config import log
from ..llm_config import ProviderConfig

from ..models import Message, LLMResponse, LLMChunk, Tool, FinishReason
from ..models import RetryableError, NonRetryableError



class ProviderAdapter(ABC):
    """
    厂商适配器抽象基类
    所有厂商适配器必须继承此类并实现相关方法
    仅支持异步调用方式
    """

    # 子类应设置此属性用于日志标识
    provider_name: str = "Provider"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None
        self._client_lock = asyncio.Lock()

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        model: str,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        thinking: Optional[dict] = None,
        **kwargs
    ) -> Union[LLMResponse, AsyncIterator[LLMChunk]]:
        """
        执行 LLM 调用（异步）

        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            tools: 工具定义列表（可选）
            thinking: 思考配置（可选），如 {"type": "enabled", "budget_tokens": 10000}
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            非流式返回 LLMResponse，流式返回 AsyncIterator[LLMChunk]
        """
        pass

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """解析非流式响应为统一格式"""
        pass

    # ==================== 工具方法 ====================

    def _init_client(self):
        """初始化异步客户端（懒加载）"""
        pass

    @property
    def client(self):
        """获取异步客户端（懒加载）"""
        if self._client is None:
            self._init_client()
        return self._client

    # ==================== 通用错误处理 ====================

    def _handle_api_error(
        self,
        error: Exception,
        exception_types: Optional[Dict[str, Type[Exception]]] = None
    ) -> None:
        """
        通用 API 错误处理

        处理常见的 API 错误类型并转换为 RetryableError 或 NonRetryableError。
        子类可传入特定的异常类型映射。

        Args:
            error: 原始异常
            exception_types: 异常类型字典，包含以下可选键：
                - AuthenticationError
                - BadRequestError
                - RateLimitError
                - APIConnectionError
                - APIStatusError

        Raises:
            RetryableError: 可重试的错误（速率限制、连接错误、服务端错误）
            NonRetryableError: 不可重试的错误（认证失败、参数错误）
        """
        provider = self.provider_name

        if exception_types is None:
            # 无法获取异常类型，默认可重试
            log.error(f"[{provider}] 未知错误 | {type(error).__name__}: {error}")
            raise RetryableError(str(error), raw_error=error)

        # 认证错误 - 不可重试
        auth_error = exception_types.get("AuthenticationError")
        if auth_error and isinstance(error, auth_error):
            log.error(f"[{provider}] 认证失败 | {error}")
            raise NonRetryableError(f"认证失败: {error}", raw_error=error)

        # 请求参数错误 - 不可重试
        bad_request = exception_types.get("BadRequestError")
        if bad_request and isinstance(error, bad_request):
            log.error(f"[{provider}] 请求参数错误 | {error}")
            raise NonRetryableError(f"请求参数错误: {error}", raw_error=error)

        # 速率限制 - 可重试
        rate_limit = exception_types.get("RateLimitError")
        if rate_limit and isinstance(error, rate_limit):
            log.error(f"[{provider}] 速率限制 | {error}")
            raise RetryableError(f"速率限制: {error}", raw_error=error)

        # 连接错误 - 可重试
        conn_error = exception_types.get("APIConnectionError")
        if conn_error and isinstance(error, conn_error):
            log.error(f"[{provider}] 连接错误 | {error}")
            raise RetryableError(f"连接错误: {error}", raw_error=error)

        # API 状态错误 - 根据状态码决定
        status_error = exception_types.get("APIStatusError")
        if status_error and isinstance(error, status_error):
            if hasattr(error, 'status_code') and error.status_code >= 500:
                log.error(f"[{provider}] 服务端错误 | status={error.status_code} | {error}")
                raise RetryableError(f"服务端错误: {error}", raw_error=error)
            else:
                status = getattr(error, 'status_code', 'unknown')
                log.error(f"[{provider}] API错误 | status={status} | {error}")
                raise NonRetryableError(f"API 错误: {error}", raw_error=error)

        # 未匹配的错误，默认可重试
        log.error(f"[{provider}] 未知错误 | {type(error).__name__}: {error}")
        raise RetryableError(str(error), raw_error=error)

    def _handle_error_with_string_fallback(self, error: Exception) -> None:
        """
        使用字符串匹配作为错误处理的后备方案

        当无法导入特定 SDK 异常类型时使用。

        Args:
            error: 原始异常

        Raises:
            RetryableError: 可重试的错误
            NonRetryableError: 不可重试的错误
        """
        provider = self.provider_name
        error_str = str(error).lower()

        # 认证错误
        if "unauthorized" in error_str or "invalid api key" in error_str or "401" in error_str:
            log.error(f"[{provider}] 认证失败 | {error}")
            raise NonRetryableError(f"认证失败，请检查 API Key: {error}", raw_error=error)

        # 请求参数错误
        if "400" in error_str:
            log.error(f"[{provider}] 请求参数错误 | {error}")
            raise NonRetryableError(f"请求参数错误: {error}", raw_error=error)

        # 权限不足
        if "403" in error_str or "forbidden" in error_str:
            log.error(f"[{provider}] 权限不足 | {error}")
            raise NonRetryableError(f"权限不足: {error}", raw_error=error)

        # 资源不存在
        if "404" in error_str or "not found" in error_str:
            log.error(f"[{provider}] 资源不存在 | {error}")
            raise NonRetryableError(f"资源不存在: {error}", raw_error=error)

        # 速率限制
        if "rate limit" in error_str or "429" in error_str or "too many" in error_str:
            log.error(f"[{provider}] 速率限制 | {error}")
            raise RetryableError(f"请求过于频繁: {error}", raw_error=error)

        # 服务端错误
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            log.error(f"[{provider}] 服务端错误 | {error}")
            raise RetryableError(f"服务端错误: {error}", raw_error=error)

        # 超时
        if "timeout" in error_str or "timed out" in error_str:
            log.error(f"[{provider}] 请求超时 | {error}")
            raise RetryableError(f"请求超时: {error}", raw_error=error)

        # 连接错误
        if "connection" in error_str:
            log.error(f"[{provider}] 连接错误 | {error}")
            raise RetryableError(f"连接错误: {error}", raw_error=error)

        # 默认可重试
        log.error(f"[{provider}] 未知错误 | {type(error).__name__}: {error}")
        raise RetryableError(str(error), raw_error=error)

    # ==================== 通用解析方法 ====================

    def _parse_finish_reason_common(
        self,
        reason: Optional[str],
        mapping: Optional[Dict[str, FinishReason]] = None
    ) -> FinishReason:
        """
        通用的结束原因解析

        Args:
            reason: 原始结束原因字符串
            mapping: 自定义映射表，如果为 None 则使用默认映射

        Returns:
            FinishReason 枚举值
        """
        if reason is None:
            return FinishReason.STOP

        default_mapping = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }

        if mapping:
            default_mapping.update(mapping)

        return default_mapping.get(reason, FinishReason.STOP)
