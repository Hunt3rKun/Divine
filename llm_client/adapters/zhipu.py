from typing import List, Union, Any, Optional, AsyncIterator
import json

from traitlets import Dict

from logger_config import log
from .base import ProviderAdapter
from ..llm_config import ProviderConfig
from ..models import (
    Message, LLMResponse, LLMChunk, TokenUsage, Tool, ToolCall,
    FinishReason, ThinkingContent,
    RetryableError, NonRetryableError
)



class ZhipuAdapter(ProviderAdapter):
    """
    智谱AI适配器（仅异步）
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"

    # 支持思维链的模型（GLM-4.5 及以上）
    THINKING_MODELS = {"glm-4.5", "glm-4.6", "glm-4.7"}

    # 强制思考的模型（即使不传 thinking 参数也会返回 reasoning_content）
    FORCED_THINKING_MODELS = {"glm-4.7"}

    # 支持流式工具调用的模型
    TOOL_STREAM_MODELS = {"glm-4.7", "glm-4.6"}

    provider_name = "Zhipu"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    def _init_client(self):
        """初始化异步 OpenAI 客户端"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise NonRetryableError("请安装 openai 包: pip install openai")

        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or self.DEFAULT_BASE_URL,
            timeout=self.config.timeout or 200.0,
        )

    def _supports_thinking(self, model: str) -> bool:
        """判断模型是否支持思维链（GLM-4.5及以上）"""
        model_lower = model.lower()
        return any(model_lower.startswith(tm) for tm in self.THINKING_MODELS)

    def _is_forced_thinking(self, model: str) -> bool:
        """判断模型是否强制思考（GLM-4.7）"""
        model_lower = model.lower()
        return any(model_lower.startswith(tm) for tm in self.FORCED_THINKING_MODELS)

    def _supports_tool_stream(self, model: str) -> bool:
        """判断模型是否支持流式工具调用"""
        model_lower = model.lower()
        return any(model_lower.startswith(tm) for tm in self.TOOL_STREAM_MODELS)

    def _build_request_kwargs(
        self,
        messages: List[Message],
        model: str,
        stream: bool,
        tools: Optional[List[Tool]],
        thinking: Optional[dict],
        **kwargs
    ) -> dict:
        """构建请求参数"""
        converted_messages = self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": converted_messages,
            "stream": stream,
        }

        # 思维链配置（GLM-4.5 及以上支持）
        # 使用 extra_body 传递智谱特有参数
        extra_body = {}

        if self._supports_thinking(model):
            if thinking:
                # 转换 thinking 参数
                thinking_config = {}

                if "type" in thinking:
                    thinking_config["type"] = thinking["type"]
                elif "budget_tokens" in thinking:
                    # 兼容 Claude 格式
                    thinking_config["type"] = "enabled"
                else:
                    thinking_config["type"] = "enabled"

                if "clear_thinking" in thinking:
                    thinking_config["clear_thinking"] = thinking["clear_thinking"]

                extra_body["thinking"] = thinking_config
            elif self._is_forced_thinking(model):
                # GLM-4.7 强制思考，显式启用以确保返回 reasoning_content
                extra_body["thinking"] = {"type": "enabled"}

        # 采样参数
        if "do_sample" in kwargs:
            extra_body["do_sample"] = kwargs.pop("do_sample")

        if "temperature" in kwargs:
            temp = kwargs.pop("temperature")
            request_kwargs["temperature"] = round(max(0, min(1, temp)), 2)

        if "top_p" in kwargs:
            top_p = kwargs.pop("top_p")
            request_kwargs["top_p"] = round(max(0.01, min(1, top_p)), 2)

        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            request_kwargs["max_tokens"] = max(1, min(131072, max_tokens))

        # 停止词（智谱只支持1个）
        if "stop" in kwargs:
            stop = kwargs.pop("stop")
            if isinstance(stop, list) and len(stop) > 0:
                request_kwargs["stop"] = stop[:1]
            elif isinstance(stop, str):
                request_kwargs["stop"] = [stop]

        # JSON 输出模式
        if "response_format" in kwargs:
            request_kwargs["response_format"] = kwargs.pop("response_format")

        # 请求追踪
        if "request_id" in kwargs:
            extra_body["request_id"] = kwargs.pop("request_id")
        if "user_id" in kwargs:
            extra_body["user_id"] = kwargs.pop("user_id")

        # 工具定义
        if tools:
            request_kwargs["tools"] = [self._convert_tool(t) for t in tools]

            if "tool_choice" in kwargs:
                request_kwargs["tool_choice"] = kwargs.pop("tool_choice")

            # 流式工具调用（GLM-4.7/4.6）
            if stream and self._supports_tool_stream(model):
                extra_body["tool_stream"] = kwargs.pop("tool_stream", True)

        # 添加 extra_body
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """转换消息格式"""
        result = []

        for msg in messages:
            msg_dict  = {"role": msg.role}

            if msg.content is not None:
                msg_dict["content"] = msg.content

            # 处理 assistant 的工具调用
            if msg.role == "assistant" and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_raw
                        }
                    }
                    for tc in msg.tool_calls
                ]

            # 处理 tool 消息
            if msg.role == "tool":
                msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    msg_dict["name"] = msg.name

            result.append(msg_dict)

        return result

    def _convert_tool(self, tool: Tool) -> dict:
        """转换工具定义"""
        name = tool.name
        if len(name) > 64:
            log.warning(f"[Zhipu] 工具名称被截断 | 原名: {name} | 截断后: {name[:64]}")
            name = name[:64]
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }

    # ==================== 异步方法 ====================

    async def complete(
        self,
        messages: List[Message],
        model: str,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        thinking: Optional[dict] = None,
        **kwargs
    ) -> Union[LLMResponse, AsyncIterator[LLMChunk]]:
        """执行智谱AI API 调用（异步）"""
        request_kwargs = self._build_request_kwargs(
            messages, model, stream, tools, thinking, **kwargs
        )

        try:
            response = await self.client.chat.completions.create(**request_kwargs)

            if stream:
                return self._handle_stream(response, model)
            else:
                return self._parse_response(response)
        except Exception as e:
            self._handle_error(e)
            raise

    async def _handle_stream(self, response: Any, model: str) -> AsyncIterator[LLMChunk]:
        """处理流式响应（异步）"""
        tool_calls_buffer: dict = {}

        async for chunk in response:
            result = self._process_stream_chunk(chunk, tool_calls_buffer)
            if result:
                yield result

    # ==================== 响应解析 ====================

    def _parse_response(self, response: Any) -> LLMResponse:
        """解析非流式响应"""
        choice = response.choices[0]
        message = choice.message
        usage_data = response.usage

        # 解析内容
        content = message.content or ""

        # 解析思维链内容（reasoning_content）
        # 智谱返回的 reasoning_content 在 message 对象上
        thinking = None
        reasoning_content = getattr(message, 'reasoning_content', None)
        if reasoning_content:
            thinking = ThinkingContent(content=reasoning_content, tokens=0)
            log.debug(f"[Zhipu] 解析到思维链内容 | content: \n{reasoning_content}")

        log.debug(f"[Zhipu] 解析到响应内容 | content: \n{content}")


        # 解析工具调用
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                args_raw = tc.function.arguments or "{}"
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(ToolCall(
                    id=tc.id or "",
                    name=tc.function.name or "",
                    arguments=args,
                    arguments_raw=args_raw
                ))
            log.debug(f"[Zhipu] 解析到工具调用 | tool_calls: {tool_calls}")

        # 解析结束原因
        finish_reason = self._parse_finish_reason(choice.finish_reason)

        # 解析 token 用量
        usage = TokenUsage(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
        )

        # 缓存 token（如果有）
        if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
            cached = getattr(usage_data.prompt_tokens_details, 'cached_tokens', None)
            if cached:
                usage.cache_read_tokens = cached

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            thinking=thinking,
            raw_response=response
        )

    def _process_stream_chunk(self, chunk: Any, tool_calls_buffer: dict) -> Optional[LLMChunk]:
        """处理流式响应片段"""
        if not chunk.choices:
            # 可能是最后的 usage 信息
            if hasattr(chunk, 'usage') and chunk.usage:
                return LLMChunk(
                    delta="",
                    is_final=True,
                    usage=TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens
                    ),
                    model=chunk.model
                )
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        # 文本增量
        content = delta.content or "" if hasattr(delta, 'content') else ""

        # 思维链增量（reasoning_content）
        thinking_delta = getattr(delta, 'reasoning_content', None)
        if thinking_delta == "":
            thinking_delta = None

        # 工具调用增量
        tool_calls_delta = None
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            tool_calls_delta = []
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}

                if tc_delta.id:
                    tool_calls_buffer[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_buffer[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

                tool_calls_delta.append({
                    "index": idx,
                    "id": tc_delta.id,
                    "name": tc_delta.function.name if tc_delta.function else None,
                    "arguments_delta": tc_delta.function.arguments if tc_delta.function else None
                })

        # 结束标志
        is_final = choice.finish_reason is not None
        finish_reason = self._parse_finish_reason(choice.finish_reason) if choice.finish_reason else None

        # 流式 usage
        usage = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = TokenUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens
            )

        return LLMChunk(
            delta=content,
            is_final=is_final,
            usage=usage,
            model=chunk.model,
            tool_calls_delta=tool_calls_delta,
            finish_reason=finish_reason,
            thinking_delta=thinking_delta
        )

    def _parse_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """
        解析结束原因

        智谱 API finish_reason 值：
        - stop: 自然结束或触发stop词
        - tool_calls: 模型命中函数
        - length: 达到token长度限制
        - sensitive: 内容被安全审核接口拦截
        - network_error: 模型推理异常
        """
        zhipu_mapping = {
            "sensitive": FinishReason.CONTENT_FILTER,
            "network_error": FinishReason.STOP,
        }
        return self._parse_finish_reason_common(reason, zhipu_mapping)

    # ==================== 错误处理 ====================

    def _handle_error(self, error: Exception):
        """处理错误（优先使用 SDK 异常类型，字符串匹配作为 fallback）"""
        # 优先尝试使用 OpenAI SDK 异常类型（Zhipu 使用 openai 包）
        try:
            from openai import (
                APIConnectionError, RateLimitError, APIStatusError,
                AuthenticationError, BadRequestError
            )
            exception_types = {
                "AuthenticationError": AuthenticationError,
                "BadRequestError": BadRequestError,
                "RateLimitError": RateLimitError,
                "APIConnectionError": APIConnectionError,
                "APIStatusError": APIStatusError,
            }
            # 使用基类的统一错误处理
            self._handle_api_error(error, exception_types)
        except ImportError:
            # 无法导入 SDK 异常，使用字符串匹配 fallback
            self._handle_error_with_string_fallback(error)
