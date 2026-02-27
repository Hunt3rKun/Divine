import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Type, Tuple

from config.config import LLM_REQUEST_TIMEOUT, DEFAULT_PRICING


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 30.0  # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数退避底数

@dataclass
class ProviderConfig:
    """单个厂商配置"""
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 60.0
    extra: Dict = field(default_factory=dict)  # 厂商特定配置

@dataclass
class LLMConfig:
    """LLM 客户端总配置"""
    default_provider: str = "openai"
    retry: RetryConfig = field(default_factory=RetryConfig)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """从字典创建配置"""
        config = cls()

        if "default_provider" in data:
            config.default_provider = data["default_provider"]

        # 解析重试配置
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])

        # 解析厂商配置
        if "providers" in data:
            for name, provider_data in data["providers"].items():
                # 支持环境变量替换
                api_key = provider_data.get("api_key", "")
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, "")

                config.providers[name] = ProviderConfig(
                    api_key=api_key,
                    base_url=provider_data.get("base_url"),
                    timeout=provider_data.get("timeout", 60.0),
                    extra=provider_data.get("extra", {})
                )

        # 解析价格表
        if "pricing" in data:
            config.pricing = data["pricing"]

        return config

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量创建默认配置"""
        config = cls()

        # OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            config.providers["openai"] = ProviderConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
                timeout=LLM_REQUEST_TIMEOUT

            )
        # Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            config.providers["claude"] = ProviderConfig(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ.get("ANTHROPIC_API_BASE_URL"),
                timeout=LLM_REQUEST_TIMEOUT
            )
        # Zhipu
        if os.environ.get("ZHIPU_API_KEY"):
            config.providers["zhipu"] = ProviderConfig(
                api_key=os.environ["ZHIPU_API_KEY"],
                base_url=os.environ.get("ZHIPU_API_BASE_URL"),
                timeout=LLM_REQUEST_TIMEOUT
            )
        config.pricing = DEFAULT_PRICING


        return config
