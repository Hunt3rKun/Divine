"""
LLM 客户端数据模型定义
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator, List, Dict
from enum import Enum
import json

from logger_config import log


class ErrorType(Enum):
    """LLM调用错误类型枚举"""
    RETRYABLE = "retryable"      # 可重试：网络、限流、服务端错误
    NON_RETRYABLE = "non_retryable"  # 不可重试：参数错误、认证失败


class ToolErrorType(Enum):
    """工具执行错误类型枚举"""
    NONE = "none"                    # 无错误
    TIMEOUT = "timeout"              # 执行超时
    PERMISSION = "permission"        # 权限不足
    NETWORK = "network"              # 网络错误（连接失败、DNS解析失败等）
    NOT_FOUND = "not_found"          # 资源未找到（命令、文件、URL等）
    INVALID_ARGS = "invalid_args"    # 参数无效
    BLOCKED = "blocked"              # 被安全策略阻止
    EXECUTION = "execution"          # 执行异常（运行时错误）
    UNKNOWN = "unknown"              # 未知错误


class FinishReason(Enum):
    """结束原因"""
    STOP = "stop"                # 正常结束
    TOOL_CALLS = "tool_calls"    # 需要调用工具
    LENGTH = "length"            # 达到长度限制
    CONTENT_FILTER = "content_filter"  # 内容过滤


@dataclass
class ThinkingContent:
    """思考内容（用于思考模型）"""
    content: str                  # 思考内容
    tokens: int = 0               # 思考消耗的 token 数


@dataclass
class TokenUsage:
    """Token 使用统计"""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0     # 思考 token
    cache_read_tokens: int = 0    # 缓存读取 token
    cache_creation_tokens: int = 0  # 缓存创建 token

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """支持累加"""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens
        )


@dataclass
class ToolCall:
    """工具调用"""
    id: str                     # 调用 ID
    name: str                   # 函数名
    arguments: Dict[str, Any]   # 参数（已解析的字典）
    arguments_raw: str = ""     # 原始参数字符串（JSON）

    @classmethod
    def from_openai(cls, tool_call) -> "ToolCall":
        """从 OpenAI 格式创建"""
        args_raw = tool_call.function.arguments
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError as e:
            log.error(f"[ToolCall] JSON解析失败 | tool={tool_call.function.name} | raw_args={args_raw} | error={e}")
            args = {}

        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
            arguments_raw=args_raw
        )

    @classmethod
    def from_anthropic(cls, tool_use_block) -> "ToolCall":
        """从 Anthropic 格式创建"""
        return cls(
            id=tool_use_block.id,
            name=tool_use_block.name,
            arguments=tool_use_block.input or {},
            arguments_raw=json.dumps(tool_use_block.input or {})
        )


@dataclass
class ToolResult:
    """工具调用结果"""
    tool_call_id: str           # 对应的 ToolCall.id
    content: str                # 返回内容
    is_error: bool = False      # 是否为错误结果
    error_type: ToolErrorType = ToolErrorType.NONE  # 错误类型（细粒度分类）
    error_details: Optional[Dict[str, Any]] = None  # 错误详情（调试信息）
    metadata: Optional[Dict[str, Any]] = None       # 执行元数据（耗时、资源等）

    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        retryable_types = {
            ToolErrorType.TIMEOUT,
            ToolErrorType.NETWORK,
        }
        return self.is_error and self.error_type in retryable_types

    def is_blocked(self) -> bool:
        """判断是否被安全策略阻止"""
        return self.error_type == ToolErrorType.BLOCKED


@dataclass
class Tool:
    """工具定义"""
    name: str                           # 函数名
    description: str                    # 函数描述
    parameters: Dict[str, Any]          # JSON Schema 格式的参数定义

    def to_openai_format(self) -> Dict:
        """转为 OpenAI 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_anthropic_format(self) -> Dict:
        """转为 Anthropic 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


@dataclass
class LLMResponse:
    """LLM 响应结果"""
    content: str                                    # 回复内容
    model: str                                      # 实际使用的模型
    usage: TokenUsage                               # token 统计
    cost: float = 0.0                               # 本次费用
    tool_calls: Optional[List[ToolCall]] = None     # 工具调用列表
    finish_reason: FinishReason = FinishReason.STOP # 结束原因
    thinking: Optional[ThinkingContent] = None      # 思考内容（思考模型）
    raw_response: Optional[Any] = None              # 原始响应（调试用）

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def has_thinking(self) -> bool:
        """是否包含思考内容"""
        return self.thinking is not None and len(self.thinking.content) > 0


@dataclass
class LLMChunk:
    """流式响应片段"""
    delta: str                                      # 增量内容
    is_final: bool = False                          # 是否最后一片
    usage: Optional[TokenUsage] = None              # 仅最后一片有值
    model: Optional[str] = None                     # 模型名称
    tool_calls_delta: Optional[List[Dict]] = None   # 工具调用增量（流式）
    finish_reason: Optional[FinishReason] = None    # 结束原因
    thinking_delta: Optional[str] = None            # 思考内容增量


@dataclass
class LLMStats:
    """累计统计信息"""
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    total_cost: float = 0.0
    request_count: int = 0

    def update(self, usage: TokenUsage, cost: float):
        """更新统计"""
        self.total_usage = self.total_usage + usage
        self.total_cost += cost
        self.request_count += 1

    def reset(self):
        """重置统计"""
        self.total_usage = TokenUsage()
        self.total_cost = 0.0
        self.request_count = 0


@dataclass
class Message:
    """消息结构"""
    role: str                                       # system / user / assistant / tool
    content: Optional[str] = None                   # 消息内容
    tool_calls: Optional[List[ToolCall]] = None     # assistant 的工具调用
    tool_call_id: Optional[str] = None              # tool 角色：对应的调用 ID
    name: Optional[str] = None                      # tool 角色：工具名称

    def to_dict(self) -> dict:
        """转为字典（通用格式）"""
        result: Dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_raw
                    }
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.name:
            result["name"] = self.name

        return result

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str = None) -> "Message":
        """创建工具结果消息"""
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name
        )

    @classmethod
    def from_response(cls, response: "LLMResponse") -> "Message":
        """从 LLMResponse 创建 assistant 消息"""
        return cls(
            role="assistant",
            content=response.content if response.content else None,
            tool_calls=response.tool_calls
        )


class LLMError(Exception):
    """LLM 调用异常基类"""
    def __init__(self, message: str, error_type: ErrorType, raw_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.raw_error = raw_error


class RetryableError(LLMError):
    """可重试错误"""
    def __init__(self, message: str, raw_error: Optional[Exception] = None):
        super().__init__(message, ErrorType.RETRYABLE, raw_error)


class NonRetryableError(LLMError):
    """不可重试错误"""
    def __init__(self, message: str, raw_error: Optional[Exception] = None):
        super().__init__(message, ErrorType.NON_RETRYABLE, raw_error)