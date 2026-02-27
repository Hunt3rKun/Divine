import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

from llm_client import LLMClient, LLMResponse, Message
from llm_client.models import Tool
from logger_config import log
from prompts.manager import PromptManager


class AgentRole(Enum):
    """Agent 角色枚举"""
    PLANNER = "planner"           # 规划器：负责任务分解和策略制定
    EXECUTOR = "executor"         # 执行器：负责执行具体任务
    REFLECTOR = "reflector"       # 反思器：负责结果分析和经验总结



@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str                                     # Agent 名称
    role: AgentRole                               # Agent 角色
    model: str = "glm-4.7"                         # 默认使用的模型
    max_iterations: int = 30                      # 最大迭代次数
    temperature: float = 0.7                      # 生成温度
    timeout: float = 180.0                        # 单次调用超时（秒）
    response_format: str = "json_object"            # 响应格式
    enable_thinking: bool = False                 # 是否启用思考模型
    verbose: bool = True                           # 是否输出详细日志





class Agent(ABC):
    def __init__(
            self,
            config: AgentConfig,
            llm_client: Optional[LLMClient] = None,
            tools: Optional[List[Tool]] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        self.config = config
        self.llm_client = llm_client or LLMClient.from_env()
        self.prompt_manager = prompt_manager or PromptManager()
        self.tools = tools or []


    @property
    def name(self) -> str:
        return self.config.name


    @abstractmethod
    def _build_messages(self,system_prompt:str) -> List[Message]:
        """构建消息列表的抽象方法"""
        pass


    @abstractmethod
    async def parse_response(self, response):
        """解析LLM响应的抽象方法"""
        pass

    async def _call_llm(self, messages: List[Message]) -> LLMResponse:
        kwargs = {
            "temperature": self.config.temperature,
            "response_format":{ "type": self.config.response_format }
        }

        thinking = None
        if self.config.enable_thinking:
            thinking = {
                "type": "enabled",
            }

        response = await self.llm_client.chat(
            messages=messages,
            model=self.config.model,
            provider=None,
            tools=self.tools if self.tools else None,
            thinking=thinking,
            **kwargs,
        )
        return response



    def _extract_json(self,content: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应中提取 JSON
        尝试多种方式提取：
        1. 直接解析整个内容
        2. 从 markdown 代码块中提取（支持多种格式）
        3. 从大括号包围的内容中提取（使用括号匹配）
        Args:
            content: LLM 响应内容
        Returns:
            解析后的 JSON 字典，失败返回 None
        """
        if not content:
            return None

        content = content.strip()

        # 方式1：直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 方式2：从 markdown 代码块提取（支持多种格式）
        # 模式1: ```json ... ``` 或 ``` ... ```
        json_block_patterns = [
            r"```json\s*\n([\s\S]*?)\n```",  # ```json\n...\n```
            r"```json\s*([\s\S]*?)```",  # ```json...``` (无换行)
            r"```\s*\n([\s\S]*?)\n```",  # ```\n...\n```
            r"```([\s\S]*?)```",  # ```...```
        ]

        for pattern in json_block_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                block = match.group(1).strip()
                if block.startswith('{'):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue

        # 方式3：使用括号匹配从内容中提取 JSON 对象
        # 找到第一个 { 并使用括号计数匹配到对应的 }
        start_idx = content.find('{')
        if start_idx != -1:
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(content[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = content[start_idx:i + 1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError as e:
                                log.debug(f"括号匹配提取的JSON解析失败: {e}")
                                break

        # 方式4：回退到贪婪正则匹配（最后手段）
        brace_pattern = r"\{[\s\S]*\}"
        matches = list(re.finditer(brace_pattern, content))
        for match in matches:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return None


    def on_start(self) -> None:
        """开始执行前的钩子"""
        if self.config.verbose:
            log.debug(f"Agent [{self.config.name}] 开始执行")

    def on_iteration(self,iteration: int) -> None:
        """每次迭代开始时的钩子"""
        if self.config.verbose:
            log.debug(f"Agent [{self.config.name}] 迭代 {iteration}/{self.config.max_iterations}")


    def on_complete(self) -> None:
        """执行完成时的钩子"""
        if self.config.verbose:
            log.info(
                f"Agent [{self.config.name}] 完成执行，"
            )

    def on_error(self, error: Exception) -> None:
        """发生错误时的钩子"""
        log.error(f"Agent [{self.config.name}] 执行出错: {error}")