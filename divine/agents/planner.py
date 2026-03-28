import json
import re
from loguru import logger
from divine.config import DivineConfig
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine

OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "command": {"enum": ["add_node", "remove_node", "update_node"]},
            "node_id": {"type": "string"},
            "node_data": {"type": "object", "properties": {
                "id": {"type": "string"}, "description": {"type": "string"},
                "phase": {"enum": ["recon", "scan", "exploit", "post_exploit"]},
                "executor_type": {"enum": ["recon", "web", "host", "service"]},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer"},
            }},
            "updates": {"type": "object"},
        },
    },
}


class Planner:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._prompt_engine = prompt_engine
        self._model = model

    async def init_plan(self, goal: str, config: DivineConfig) -> list[dict]:
        prompt = self._prompt_engine.build_init_plan_prompt(
            goal=goal, targets=config.targets, output_schema=OUTPUT_SCHEMA,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=f"请为以下目标制定渗透测试计划: {goal}"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_operations(response.content)

    async def replan(self, blackboard_summary: dict, dag_stats: dict,
                     reflections: list[dict]) -> list[dict]:
        prompt = self._prompt_engine.build_replan_prompt(
            blackboard_summary=blackboard_summary, dag_stats=dag_stats,
            reflections=reflections, output_schema=OUTPUT_SCHEMA,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="根据当前进度，是否需要调整攻击计划？"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_operations(response.content)

    async def should_terminate(self, blackboard_summary: dict,
                               dag_stats: dict) -> tuple[bool, str]:
        prompt = self._prompt_engine.build_terminate_check_prompt(
            blackboard_summary=blackboard_summary, dag_stats=dag_stats, goal="",
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="判断目标是否已达成。"),
        ]
        response = await self._router.chat(self._model, messages)
        try:
            data = self._extract_json_object(response.content)
            return data.get("terminate", False), data.get("reason", "")
        except Exception:
            return False, "无法解析终止判断"

    def _parse_operations(self, content: str) -> list[dict]:
        """Multi-method JSON array extraction: direct parse, markdown block, bracket match."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        pattern = r"```(?:json)?\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        bracket_match = re.search(r"\[.*\]", content, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        logger.warning(f"Planner: 无法解析 operations: {content[:200]}")
        return []

    def _extract_json_object(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        pattern = r"```(?:json)?\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group())
        raise ValueError(f"Cannot extract JSON from: {content[:200]}")
