import re
from typing import Optional

from loguru import logger

from divine.blackboard.blackboard import Blackboard
from divine.codeact.sandbox import Sandbox, ExecutionResult
from divine.codeact.stdlib import create_stdlib
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.models.task import TaskNode
from divine.prompts.engine import PromptEngine


class CodeActExecutor:
    def __init__(self, router: LLMRouter, sandbox: Sandbox,
                 blackboard: Blackboard, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._sandbox = sandbox
        self._blackboard = blackboard
        self._prompt_engine = prompt_engine
        self._model = model
        self._max_iterations = 10

        # Build stdlib docs for prompt
        stdlib = create_stdlib(blackboard)
        self._stdlib_docs = self._build_stdlib_docs(stdlib)

    def _build_stdlib_docs(self, stdlib: dict) -> str:
        lines = []
        for name, fn in stdlib.items():
            doc = getattr(fn, "__doc__", "") or ""
            first_line = doc.strip().split("\n")[0] if doc.strip() else "No description"
            lines.append(f"- {name}: {first_line}")
        return "\n".join(lines)

    async def execute_task(self, task: TaskNode, context: dict) -> dict:
        # Reset sandbox and inject fresh stdlib for each task
        self._sandbox.reset()
        stdlib = create_stdlib(self._blackboard)
        self._sandbox.setup(stdlib)

        system_prompt = self._prompt_engine.build_executor_system_prompt(
            executor_type=task.executor_type,
            task=task,
            context=context,
            stdlib_docs=self._stdlib_docs,
        )

        conversation = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"开始执行任务: {task.description}"),
        ]

        iterations = 0
        for i in range(self._max_iterations):
            iterations = i + 1
            response = await self._router.chat(self._model, conversation)
            content = response.content

            code = self._extract_code(content)
            if not code:
                logger.info(f"Task {task.id}: LLM 未返回代码，视为完成 (iteration {iterations})")
                break

            conversation.append(LLMMessage(role="assistant", content=content))

            result = await self._sandbox.execute(code)
            observation = self._prompt_engine.build_observation_prompt(result)
            conversation.append(LLMMessage(role="user", content=observation))

            logger.debug(f"Task {task.id} iteration {iterations}: success={result.success}")
        else:
            logger.warning(f"Task {task.id}: 达到最大迭代次数 {self._max_iterations}")

        return {"iterations": iterations}

    def _extract_code(self, content: str) -> Optional[str]:
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
