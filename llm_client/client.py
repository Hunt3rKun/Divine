
from typing import Optional, Dict, Union, List, AsyncIterator

from .cost import CostCalculator
from .retry import RetryHandler
from .adapters import get_adapter_class, ProviderAdapter
from .llm_config import LLMConfig
from .models import LLMStats, Message, LLMResponse, Tool, LLMChunk
from logger_config import log


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._adapters: Dict[str, ProviderAdapter] = {}
        self._stats = LLMStats()
        self._cost_calculator = CostCalculator(config.pricing)
        self._retry_handler = RetryHandler(config.retry)

        # 初始化已配置的适配器
        self._init_adapters()
        log.debug(f"[LLMClient] 初始化完成 | adapters={list(self._adapters.keys())}")

    @classmethod
    def from_env(cls) -> "LLMClient":
        """从环境变量创建客户端"""
        config = LLMConfig.from_env()
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LLMClient":
        """从字典创建客户端"""
        config = LLMConfig.from_dict(config_dict)
        return cls(config)


    def _init_adapters(self):
        """初始化适配器"""
        for provider_name, provider_config in self.config.providers.items():
            adapter_class = get_adapter_class(provider_name)
            if adapter_class:
                self._adapters[provider_name] = adapter_class(provider_config)
                log.debug(f"[LLMClient] 适配器初始化成功 | {provider_name}")
            else:
                log.warning(f"[LLMClient] 未找到适配器 | {provider_name}")


    def _get_adapter(self, provider: Optional[str] = None) -> ProviderAdapter:
        """获取适配器"""
        provider = provider or self.config.default_provider

        if provider not in self._adapters:
            raise ValueError(f"未配置厂商: {provider}，可用: {list(self._adapters.keys())}")

        return self._adapters[provider]

    def _infer_provider(self, model: str) -> str:
        """根据模型名推断厂商"""
        model_lower = model.lower()

        if model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"
        elif model_lower.startswith("claude"):
            return "claude"
        elif model_lower.startswith("glm"):
            return "zhipu"

        return self.config.default_provider


    def _normalize_messages(
        self,
        messages: List[Union[Message, dict]]
    ) -> List[Message]:
        """标准化消息格式"""
        result = []
        for i, msg in enumerate(messages):
            if isinstance(msg, Message):
                result.append(msg)
            elif isinstance(msg, dict):
                if "role" not in msg:
                    raise ValueError(f"消息 #{i} 缺少必需字段: role")
                result.append(Message(
                    role=msg["role"],
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    name=msg.get("name")
                ))
            else:
                raise ValueError(f"不支持的消息格式: {type(msg)}")
        return result


    def _normalize_tools(
        self,
        tools: List[Union[Tool, dict]]
    ) -> List[Tool]:
        """标准化工具格式"""
        result = []
        for tool in tools:
            if isinstance(tool, Tool):
                result.append(tool)
            elif isinstance(tool, dict):
                result.append(Tool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {})
                ))
            else:
                raise ValueError(f"不支持的工具格式: {type(tool)}")
        return result


    async def chat(
        self,
        messages: List[Union[Message, dict]],
        model: str,
        provider: Optional[str] = None,
        tools: Optional[List[Union[Tool, dict]]] = None,
        thinking: Optional[dict] = None,
        **kwargs
    ) -> LLMResponse:
        """
        非流式调用（异步）
        Args:
            messages: 消息列表（Message 对象或字典）
            model: 模型名称
            provider: 厂商名称（可选，不指定则自动推断）
            tools: 工具定义列表（可选）
            thinking: 思考配置（可选），如 {"type": "enabled", "budget_tokens": 10000}
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            LLMResponse 对象
        """
        log.info(f"[LLMClient] chat开始 | model={model} | messages={len(messages)} | tools={len(tools) if tools else 0}")
        messages = self._normalize_messages(messages)   # 将消息标准化为 Message 对象列表
        tools = self._normalize_tools(tools) if tools else None  # 将工具标准化为 Tool 对象列表
        provider = provider or self._infer_provider(model)
        adapter = self._get_adapter(provider)

        response: LLMResponse = await self._retry_handler.execute_async(
            adapter.complete,
            messages=messages,
            model=model,
            stream=False,
            tools=tools,
            thinking=thinking,
            **kwargs
        )

        response.cost = self._cost_calculator.calculate(response.model, response.usage)
        self._stats.update(response.usage, response.cost)

        log.debug(
            f"[LLMClient] chat调用完成 | model={response.model} | "
            f"total_tokens={response.usage.total_tokens} | cost=${response.cost:.6f} | "
            f"finish_reason={response.finish_reason.value}"
        )
        return response

    async def chat_stream(
        self,
        messages: List[Union[Message, dict]],
        model: str,
        provider: Optional[str] = None,
        tools: Optional[List[Union[Tool, dict]]] = None,
        thinking: Optional[dict] = None,
        **kwargs
    ) -> AsyncIterator[LLMChunk]:
        """
        流式调用（异步）

        Args:
            messages: 消息列表
            model: 模型名称
            provider: 厂商名称（可选）
            tools: 工具定义列表（可选）
            thinking: 思考配置（可选）
            **kwargs: 其他参数

        Returns:
            异步迭代器，yield LLMChunk 片段
        """
        log.info(f"[LLMClient] chat_stream | model={model} | messages={len(messages)} | tools={len(tools) if tools else 0}")
        messages = self._normalize_messages(messages)
        tools = self._normalize_tools(tools) if tools else None
        provider = provider or self._infer_provider(model)
        adapter = self._get_adapter(provider)

        stream = await self._retry_handler.execute_async(
            adapter.complete,
            messages=messages,
            model=model,
            stream=True,
            tools=tools,
            thinking=thinking,
            **kwargs
        )

        final_usage = None
        final_model = model

        async for chunk in stream:
            yield chunk

            if chunk.is_final and chunk.usage:
                final_usage = chunk.usage
            if chunk.model:
                final_model = chunk.model

        if final_usage:
            cost = self._cost_calculator.calculate(final_model, final_usage)
            self._stats.update(final_usage, cost)
            log.debug(
                f"[LLMClient] 流式调用完成 | model={final_model} | "
                f"total_tokens={final_usage.total_tokens} | cost=${cost:.6f}"
            )

    def get_stats(self) -> LLMStats:
        """获取累计统计"""
        return self._stats

    def reset_stats(self):
        """重置统计"""
        self._stats.reset()

