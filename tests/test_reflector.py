from unittest.mock import AsyncMock
import json
import pytest
from divine.agents.reflection import Reflector, Reflection
from divine.llm.base import LLMResponse, TokenUsage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine


class TestReflector:
    def _make_reflector(self, llm_content: str) -> Reflector:
        router = AsyncMock(spec=LLMRouter)
        router.chat.return_value = LLMResponse(
            content=llm_content, model="test",
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        )
        engine = PromptEngine()
        return Reflector(router=router, prompt_engine=engine, model="test")

    async def test_analyze(self):
        reflection_data = {
            "insights": ["发现 SSH 弱密码", "80 端口运行 Apache"],
            "suggested_tasks": [
                {"description": "尝试 SSH 暴破", "phase": "exploit",
                 "executor_type": "service", "reason": "弱密码"},
            ],
            "risk_assessment": "中等风险，攻击面有限",
            "progress_summary": "完成初步侦察，发现 2 个服务",
        }
        reflector = self._make_reflector(json.dumps(reflection_data))
        result = await reflector.analyze(blackboard_summary={}, recent_results=[], dag_stats={})
        assert isinstance(result, Reflection)
        assert len(result.insights) == 2
        assert len(result.suggested_tasks) == 1
        assert "中等" in result.risk_assessment
