"""
Anthropic Claude 适配器
"""
from typing import List, Union, Any, Optional, AsyncIterator

from logger_config import log
from .base import ProviderAdapter
from ..llm_config import ProviderConfig
from ..models import (
    Message, LLMResponse, LLMChunk, TokenUsage, Tool, ToolCall,
    FinishReason, ThinkingContent,
    RetryableError, NonRetryableError
)



class ClaudeAdapter(ProviderAdapter):
    """Anthropic Claude API 适配器（仅异步）"""

    DEFAULT_MAX_TOKENS = 40960

    # 支持 extended thinking 的模型
    THINKING_MODELS = {"claude-3-7-sonnet", "claude-sonnet-4"}

    provider_name = "Claude"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    def _init_client(self):
        """初始化异步 Anthropic 客户端"""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise NonRetryableError("请安装 anthropic 包: pip install anthropic")

        self._client = AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        log.debug("[Claude] 异步客户端初始化成功")

    def _supports_thinking(self, model: str) -> bool:
        """判断模型是否支持 extended thinking"""
        model_lower = model.lower()
        return any(tm in model_lower for tm in self.THINKING_MODELS)

    def _build_request_kwargs(
        self,
        messages: List[Message],
        model: str,
        tools: Optional[List[Tool]],
        thinking: Optional[dict],
        **kwargs
    ) -> dict:
        """构建请求参数"""
        # 分离 system 消息并转换消息格式
        system_prompt, api_messages = self._prepare_messages(messages)

        # Claude 必须指定 max_tokens
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = self.DEFAULT_MAX_TOKENS

        # 构建请求参数
        request_kwargs = {
            "model": model,
            "system": system_prompt,
            "messages": api_messages,
            **kwargs
        }

        # 添加工具定义
        if tools:
            request_kwargs["tools"] = [t.to_anthropic_format() for t in tools]

        # 添加 thinking 配置
        if thinking and self._supports_thinking(model):
            request_kwargs["thinking"] = thinking
            if thinking.get("budget_tokens", 0) > 0:
                budget = thinking["budget_tokens"]
                current_max = request_kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
                request_kwargs["max_tokens"] = max(current_max, budget + 1000)

        return request_kwargs

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
        """执行 Claude API 调用（异步）"""
        log.info(f"[Claude] API调用 | model={model} | messages={len(messages)} | stream={stream} | tools={len(tools) if tools else 0}")
        request_kwargs = self._build_request_kwargs(
            messages, model, tools, thinking, **kwargs
        )

        try:
            if stream:
                return self._handle_stream(request_kwargs)
            else:
                response = await self.client.messages.create(**request_kwargs)
                return self._parse_response(response)

        except Exception as e:
            self._handle_error(e)

    async def _handle_stream(self, request_kwargs: dict) -> AsyncIterator[LLMChunk]:
        """处理流式响应（异步）"""
        async with self.client.messages.stream(**request_kwargs) as stream:
            actual_model = request_kwargs.get("model", "")
            current_tool_call = None
            current_block_type = None

            async for event in stream:
                chunk = self._process_stream_event(event, actual_model, current_tool_call, current_block_type)
                if chunk:
                    if chunk.get("update_model"):
                        actual_model = chunk["update_model"]
                    if chunk.get("update_tool_call"):
                        current_tool_call = chunk["update_tool_call"]
                    if chunk.get("update_block_type") is not None:
                        current_block_type = chunk["update_block_type"]
                    if chunk.get("chunk"):
                        yield chunk["chunk"]

            # 获取最终的 usage
            final_message = await stream.get_final_message()
            if final_message:
                yield self._build_final_chunk(final_message)

    # ==================== 共用方法 ====================

    def _prepare_messages(self, messages: List[Message]) -> tuple:
        """准备消息，分离 system prompt 并转换格式"""
        system_prompt = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content
                        }
                    ]
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })

        return system_prompt, api_messages

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """解析非流式响应"""
        content = ""
        tool_calls = []
        thinking_content = ""

        for block in raw_response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall.from_anthropic(block))
            elif block.type == "thinking":
                thinking_content += block.thinking

        finish_reason = self._parse_finish_reason(raw_response.stop_reason)

        usage = TokenUsage(
            input_tokens=raw_response.usage.input_tokens,
            output_tokens=raw_response.usage.output_tokens
        )

        if hasattr(raw_response.usage, 'cache_read_input_tokens'):
            usage.cache_read_tokens = raw_response.usage.cache_read_input_tokens or 0
        if hasattr(raw_response.usage, 'cache_creation_input_tokens'):
            usage.cache_creation_tokens = raw_response.usage.cache_creation_input_tokens or 0

        thinking = None
        if thinking_content:
            thinking = ThinkingContent(content=thinking_content, tokens=0)

        return LLMResponse(
            content=content,
            model=raw_response.model,
            usage=usage,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            thinking=thinking,
            raw_response=raw_response
        )

    def _process_stream_event(self, event: Any, actual_model: str, current_tool_call: Optional[dict], current_block_type: Optional[str]) -> Optional[dict]:
        """处理单个流式事件"""
        result = {}

        if not hasattr(event, 'type'):
            return None

        if event.type == "message_start":
            if hasattr(event, 'message'):
                result["update_model"] = event.message.model
            return result

        if event.type == "content_block_start":
            if hasattr(event, 'content_block'):
                block = event.content_block
                result["update_block_type"] = block.type

                if block.type == "tool_use":
                    result["update_tool_call"] = {
                        "id": block.id,
                        "name": block.name,
                        "arguments": ""
                    }
            return result

        if event.type == "content_block_delta":
            delta_text = ""
            thinking_delta = None
            tool_calls_delta = None

            if hasattr(event.delta, 'text'):
                delta_text = event.delta.text
            elif hasattr(event.delta, 'thinking'):
                thinking_delta = event.delta.thinking
            elif hasattr(event.delta, 'partial_json'):
                if current_tool_call:
                    current_tool_call["arguments"] += event.delta.partial_json
                    tool_calls_delta = [{
                        "id": current_tool_call["id"],
                        "name": current_tool_call["name"],
                        "arguments_delta": event.delta.partial_json
                    }]
                    result["update_tool_call"] = current_tool_call

            result["chunk"] = LLMChunk(
                delta=delta_text,
                is_final=False,
                model=actual_model,
                tool_calls_delta=tool_calls_delta,
                thinking_delta=thinking_delta
            )
            return result

        if event.type == "content_block_stop":
            result["update_tool_call"] = None
            result["update_block_type"] = None
            return result

        if event.type == "message_delta":
            finish_reason = None
            if hasattr(event.delta, 'stop_reason'):
                finish_reason = self._parse_finish_reason(event.delta.stop_reason)

            usage = None
            if hasattr(event, 'usage'):
                usage = TokenUsage(
                    input_tokens=getattr(event.usage, 'input_tokens', 0),
                    output_tokens=event.usage.output_tokens
                )

            result["chunk"] = LLMChunk(
                delta="",
                is_final=True,
                usage=usage,
                model=actual_model,
                finish_reason=finish_reason
            )
            return result

        return None

    def _build_final_chunk(self, final_message: Any) -> LLMChunk:
        """构建最终的 chunk"""
        usage = TokenUsage(
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens
        )
        if hasattr(final_message.usage, 'cache_read_input_tokens'):
            usage.cache_read_tokens = final_message.usage.cache_read_input_tokens or 0
        if hasattr(final_message.usage, 'cache_creation_input_tokens'):
            usage.cache_creation_tokens = final_message.usage.cache_creation_input_tokens or 0

        return LLMChunk(
            delta="",
            is_final=True,
            usage=usage,
            model=final_message.model,
            finish_reason=self._parse_finish_reason(final_message.stop_reason)
        )

    def _parse_finish_reason(self, reason: str) -> FinishReason:
        """解析结束原因"""
        claude_mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "max_tokens": FinishReason.LENGTH,
        }
        return self._parse_finish_reason_common(reason, claude_mapping)

    def _handle_error(self, error: Exception):
        """处理 Anthropic 错误"""
        try:
            from anthropic import (
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
