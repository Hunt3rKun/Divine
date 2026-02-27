"""
LLM 厂商适配器
"""
from .base import ProviderAdapter
from .openai import OpenAIAdapter
from .claude import ClaudeAdapter
from .zhipu import ZhipuAdapter

__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "ZhipuAdapter",
    "get_adapter_class"
]



# 适配器注册表
ADAPTER_REGISTRY = {
    "openai": OpenAIAdapter,
    "claude": ClaudeAdapter,
    "anthropic": ClaudeAdapter,  # 别名
    "zhipu": ZhipuAdapter,
    "glm": ZhipuAdapter,         # 别名
}


def get_adapter_class(provider: str):
    """根据厂商名称获取适配器类"""
    return ADAPTER_REGISTRY.get(provider.lower())


def register_adapter(name: str, adapter_class):
    """注册新的适配器"""
    ADAPTER_REGISTRY[name.lower()] = adapter_class