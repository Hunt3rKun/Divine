import json
import re
from dataclasses import dataclass, field
from loguru import logger
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine


@dataclass
class Reflection:
    insights: list[str] = field(default_factory=list)
    suggested_tasks: list[dict] = field(default_factory=list)
    risk_assessment: str = ""
    progress_summary: str = ""


class Reflector:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._prompt_engine = prompt_engine
        self._model = model

    async def analyze(self, blackboard_summary: dict, recent_results: list[dict],
                      dag_stats: dict) -> Reflection:
        prompt = self._prompt_engine.build_analyze_prompt(
            blackboard_summary=blackboard_summary,
            recent_results=recent_results, dag_stats=dag_stats,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="分析最近一轮执行结果。"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_reflection(response.content)

    def _parse_reflection(self, content: str) -> Reflection:
        try:
            data = self._extract_json(content)
            return Reflection(
                insights=data.get("insights", []),
                suggested_tasks=data.get("suggested_tasks", []),
                risk_assessment=data.get("risk_assessment", ""),
                progress_summary=data.get("progress_summary", ""),
            )
        except Exception:
            logger.warning("Reflector: 无法解析反思结果，使用原始文本")
            return Reflection(insights=[content], progress_summary=content[:200])

    def _extract_json(self, content: str) -> dict:
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
        raise ValueError("Cannot extract JSON")
