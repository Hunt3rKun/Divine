from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    cost: float = 0.0
    raw_response: Any = None


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse: ...

    @abstractmethod
    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]: ...
