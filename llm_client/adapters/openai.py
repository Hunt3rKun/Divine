"""
OpenAI 适配器
"""
from typing import List, Union, Any, Optional, AsyncIterator, Dict

from logger_config import log
from .base import ProviderAdapter
from ..llm_config import ProviderConfig

from ..models import (
    Message, LLMResponse, LLMChunk, TokenUsage, Tool, ToolCall,
    FinishReason, ThinkingContent,
    RetryableError, NonRetryableError
)


class OpenAIAdapter(ProviderAdapter):
    """OpenAI API 适配器（仅异步）"""

    # o1/o3 系列模型（思考模型）
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini"}

    provider_name = "OpenAI"

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
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        log.debug(f"[OpenAI] 异步客户端初始化成功 | base_url={self.config.base_url}")

    def _is_reasoning_model(self, model: str) -> bool:
        """判断是否为思考模型"""
        model_lower = model.lower()
        return any(model_lower.startswith(rm) for rm in self.REASONING_MODELS)

    def _build_request_kwargs(
            self,
            messages: List[Message],
            model: str,
            stream: bool,
            tools: Optional[List[Tool]],
            **kwargs
    ) -> tuple[dict[str, str | list[dict] | bool], bool]:
        """构建请求参数"""
        is_reasoning = self._is_reasoning_model(model)

        # 转换消息格式
        converted_messages = self._convert_messages(messages, is_reasoning)

        # 构建请求参数
        request_kwargs = {
            "model": model,
            "messages": converted_messages,
            "stream": stream,
        }

        # 思考模型特殊处理
        if is_reasoning:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("presence_penalty", None)
            kwargs.pop("frequency_penalty", None)

            if "reasoning_effort" in kwargs:
                request_kwargs["reasoning_effort"] = kwargs.pop("reasoning_effort")

            if "max_tokens" in kwargs:
                request_kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        request_kwargs.update(kwargs)

        # 添加工具定义（思考模型部分支持）
        if tools and not is_reasoning:
            request_kwargs["tools"] = [t.to_openai_format() for t in tools]

        return request_kwargs, is_reasoning

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
        """执行 OpenAI API 调用（异步）"""
        log.info(f"[OpenAI] API调用 | model={model} | messages={len(messages)} | stream={stream} | tools={len(tools) if tools else 0}")
        request_kwargs, is_reasoning = self._build_request_kwargs(
            messages, model, stream, tools, **kwargs
        )
        log.debug(f"OpenAI Request kwargs: {request_kwargs}")

        try:
            response = await self.client.chat.completions.create(**request_kwargs)

            if stream:
                return self._handle_stream(response, model, is_reasoning)
            else:
                return self._parse_response(response, is_reasoning)

        except Exception as e:
            self._handle_error(e)
            raise

    async def _handle_stream(self, response: Any, model: str, is_reasoning: bool = False) -> AsyncIterator[LLMChunk]:
        """处理流式响应（异步）"""
        actual_model = model
        tool_calls_buffer: dict = {}

        async for chunk in response:
            result = self._process_stream_chunk(chunk, actual_model, tool_calls_buffer)
            if result:
                if result.model:
                    actual_model = result.model
                yield result

    def _convert_messages(self, messages: List[Message], is_reasoning: bool = False) -> List[dict]:
        """转换消息格式为 OpenAI 格式"""
        result = []

        for msg in messages:
            # 思考模型不支持 system role，转换为 user message
            if is_reasoning and msg.role == "system":
                result.append({
                    "role": "user",
                    "content": f"[System Instructions]\n{msg.content}"
                })
                continue

            msg_dict : Dict[str,Any] = {"role": msg.role}

            if msg.content is not None:
                msg_dict["content"] = msg.content

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

            if msg.role == "tool":
                msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    msg_dict["name"] = msg.name

            result.append(msg_dict)

        return result

    def _parse_response(self, raw_response: Any, is_reasoning: bool = False) -> LLMResponse:
        """解析非流式响应"""
        choice = raw_response.choices[0]
        usage_data = raw_response.usage
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [ToolCall.from_openai(tc) for tc in message.tool_calls]

        finish_reason = self._parse_finish_reason(choice.finish_reason)

        reasoning_tokens = 0
        if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
            reasoning_tokens = getattr(usage_data.completion_tokens_details, 'reasoning_tokens', 0) or 0

        usage = TokenUsage(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
            reasoning_tokens=reasoning_tokens
        )

        thinking = None
        if is_reasoning and reasoning_tokens > 0:
            thinking = ThinkingContent(content="", tokens=reasoning_tokens)

        return LLMResponse(
            content=message.content or "",
            model=raw_response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            thinking=thinking,
            raw_response=raw_response
        )

    def _process_stream_chunk(self, chunk: Any, actual_model: str, tool_calls_buffer: dict) -> Optional[LLMChunk]:
        """处理单个流式片段"""
        if not chunk.choices:
            # 可能是 usage 信息
            if hasattr(chunk, 'usage') and chunk.usage:
                reasoning_tokens = 0
                if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                    reasoning_tokens = getattr(chunk.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

                return LLMChunk(
                    delta="",
                    is_final=True,
                    usage=TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        reasoning_tokens=reasoning_tokens
                    ),
                    model=actual_model
                )
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        model = chunk.model if chunk.model else actual_model
        content = delta.content or ""

        # 处理工具调用增量
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

        is_final = choice.finish_reason is not None
        finish_reason = self._parse_finish_reason(choice.finish_reason) if choice.finish_reason else None

        usage = None
        if hasattr(chunk, 'usage') and chunk.usage:
            reasoning_tokens = 0
            if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                reasoning_tokens = getattr(chunk.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

            usage = TokenUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                reasoning_tokens=reasoning_tokens
            )

        return LLMChunk(
            delta=content,
            is_final=is_final,
            usage=usage,
            model=model,
            tool_calls_delta=tool_calls_delta,
            finish_reason=finish_reason
        )

    def _parse_finish_reason(self, reason: str) -> FinishReason:
        """解析结束原因"""
        return self._parse_finish_reason_common(reason)

    def _handle_error(self, error: Exception):
        """处理 OpenAI 错误"""
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
        except ImportError:
            exception_types = None

        self._handle_api_error(error, exception_types)
