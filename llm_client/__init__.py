"""
LLM 客户端模块

统一的 LLM 调用接口，支持多厂商、流式/非流式、token统计、费用计算、错误重试

使用示例:
    from llm import LLMClient, Message

    # 快速开始（从环境变量读取配置）
    client = LLMClient.from_env()

    # 非流式调用
    response = client.chat([
        Message(role="system", content="你是一个助手"),
        Message(role="user", content="你好")
    ], model="gpt-4o")

    print(response.content)
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Cost: ${response.cost:.6f}")

    # 流式调用
    for chunk in client.chat_stream([
        {"role": "user", "content": "写一首诗"}
    ], model="claude-sonnet-4-20250514"):
        print(chunk.delta, end="", flush=True)

    # 获取累计统计
    stats = client.get_stats()
    print(f"\\n总费用: ${stats.total_cost:.4f}")
"""

from .client import LLMClient
from .llm_config import LLMConfig, ProviderConfig, RetryConfig
from .models import (
    Message,
    LLMResponse,
    LLMChunk,
    LLMStats,
    TokenUsage,
    LLMError,
    RetryableError,
    NonRetryableError,
)

from .cost import CostCalculator
from .retry import RetryHandler
from .adapters import ProviderAdapter, OpenAIAdapter, ClaudeAdapter, ZhipuAdapter

__version__ = "0.1.0"

__all__ = [
    # 核心类
    "LLMClient",

    # 数据模型
    "Message",
    "LLMResponse",
    "LLMChunk",
    "LLMStats",
    "TokenUsage",

    # 异常
    "LLMError",
    "RetryableError",
    "NonRetryableError",

    # 配置
    "LLMConfig",
    "ProviderConfig",
    "RetryConfig",

    # 工具类
    "CostCalculator",
    "RetryHandler",

    # 适配器
    "ProviderAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "ZhipuAdapter",
]