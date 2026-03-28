from unittest.mock import AsyncMock
import pytest

from divine.codeact.executor import CodeActExecutor
from divine.codeact.sandbox import Sandbox
from divine.llm.base import LLMMessage, LLMResponse, TokenUsage
from divine.llm.router import LLMRouter
from divine.blackboard import Blackboard
from divine.prompts.engine import PromptEngine
from divine.models.task import TaskNode
from divine.models.common import PentestPhase, ExecutorType


def make_task() -> TaskNode:
    return TaskNode(
        id="recon_1", description="扫描端口",
        phase=PentestPhase.RECON, executor_type=ExecutorType.RECON,
    )


class TestCodeActExecutor:
    def _setup_executor(self, llm_responses: list[str]) -> CodeActExecutor:
        router = AsyncMock(spec=LLMRouter)
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            content = llm_responses[min(call_count, len(llm_responses) - 1)]
            call_count += 1
            return LLMResponse(
                content=content, model=model,
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            )

        router.chat = mock_chat

        bb = Blackboard()
        sandbox = Sandbox(timeout=10)
        engine = PromptEngine()

        executor = CodeActExecutor(
            router=router, sandbox=sandbox,
            blackboard=bb, prompt_engine=engine,
            model="test-model",
        )
        return executor

    async def test_single_code_execution(self):
        executor = self._setup_executor([
            "让我扫描端口\n```python\nprint('scanning...')\n```",
            "扫描完成，未发现开放端口。",
        ])
        result = await executor.execute_task(make_task(), context={})
        assert result is not None
        assert "iterations" in result

    async def test_no_code_means_done(self):
        executor = self._setup_executor([
            "任务分析完成，无需执行代码。",
        ])
        result = await executor.execute_task(make_task(), context={})
        assert result is not None
        assert result["iterations"] == 1

    async def test_max_iterations_respected(self):
        executor = self._setup_executor([
            "```python\nprint('loop')\n```",
        ] * 20)
        executor._max_iterations = 3
        result = await executor.execute_task(make_task(), context={})
        assert result["iterations"] == 3

    def test_extract_code(self):
        executor = self._setup_executor([])
        code = executor._extract_code("here is code\n```python\nx = 1\n```\ndone")
        assert code == "x = 1"

    def test_extract_code_no_block(self):
        executor = self._setup_executor([])
        code = executor._extract_code("no code here")
        assert code is None
