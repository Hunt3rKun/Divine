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

TASK_COMPLETE_MARKER = "TASK_COMPLETE"


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

        # Snapshot blackboard state before task to diff later
        bb_before = self._snapshot_blackboard()

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

        logger.info(f"Task {task.id}: 开始执行 type={task.executor_type.value} phase={task.phase.value}")
        logger.debug(f"Task {task.id}: system_prompt length={len(system_prompt)}")

        iterations = 0
        finish_reason = "max_iterations"
        last_stdout = ""

        for i in range(self._max_iterations):
            iterations = i + 1
            response = await self._router.chat(self._model, conversation)
            content = response.content

            logger.info(f"Task {task.id} [{iterations}/{self._max_iterations}] LLM response length={len(content)}")
            logger.debug(f"Task {task.id} [{iterations}] LLM 完整回复:\n{content}")

            # Check for TASK_COMPLETE marker in LLM response
            if TASK_COMPLETE_MARKER in content:
                logger.info(f"Task {task.id}: LLM 标记任务完成 (iteration {iterations})")
                finish_reason = "task_complete"
                # Still execute any final code if present
                code = self._extract_code(content)
                if code:
                    logger.debug(f"Task {task.id}: 最终代码:\n{code}")
                    result = await self._sandbox.execute(code)
                    logger.info(f"Task {task.id}: 最终代码执行 success={result.success}")
                    if result.stdout:
                        logger.info(f"Task {task.id}: 最终 stdout:\n{result.stdout[:1000]}")
                    if result.stderr:
                        logger.warning(f"Task {task.id}: 最终 stderr:\n{result.stderr[:1000]}")
                break

            code = self._extract_code(content)
            if not code:
                logger.info(f"Task {task.id}: LLM 未返回代码，视为完成 (iteration {iterations})")
                finish_reason = "no_code"
                break

            logger.debug(f"Task {task.id} [{iterations}] 提取到代码:\n{code}")
            conversation.append(LLMMessage(role="assistant", content=content))

            result = await self._sandbox.execute(code)
            observation = self._prompt_engine.build_observation_prompt(result)
            conversation.append(LLMMessage(role="user", content=observation))

            last_stdout = result.stdout[:500] if result.stdout else ""

            logger.info(
                f"Task {task.id} [{iterations}/{self._max_iterations}] "
                f"success={result.success} "
                f"stdout={result.stdout[:500].strip()!r}"
            )
            if result.stderr:
                logger.warning(f"Task {task.id} [{iterations}] stderr:\n{result.stderr[:500].strip()}")
        else:
            logger.warning(f"Task {task.id}: 达到最大迭代次数 {self._max_iterations}")

        # Collect what this task wrote to blackboard
        bb_diff = self._diff_blackboard(bb_before)

        return {
            "iterations": iterations,
            "finish_reason": finish_reason,
            "bb_writes": bb_diff,
            "last_stdout": last_stdout,
        }

    def _snapshot_blackboard(self) -> dict[str, set[str]]:
        """Snapshot current blackboard keys per section."""
        snapshot = {}
        for section in ["hosts", "ports", "findings", "credentials"]:
            data = self._blackboard.read(section)
            snapshot[section] = set(data.keys()) if isinstance(data, dict) else set()
        return snapshot

    def _diff_blackboard(self, before: dict[str, set[str]]) -> dict[str, list[str]]:
        """Return new keys written to blackboard since snapshot."""
        diff = {}
        for section in before:
            data = self._blackboard.read(section)
            current_keys = set(data.keys()) if isinstance(data, dict) else set()
            new_keys = current_keys - before[section]
            if new_keys:
                diff[section] = list(new_keys)
        return diff

    def _extract_code(self, content: str) -> Optional[str]:
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
