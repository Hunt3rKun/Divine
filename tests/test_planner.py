from unittest.mock import AsyncMock
import json
import pytest
from divine.agents.planner import Planner
from divine.llm.base import LLMResponse, TokenUsage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine
from divine.config import DivineConfig


class TestPlanner:
    def _make_planner(self, llm_content: str) -> Planner:
        router = AsyncMock(spec=LLMRouter)
        router.chat.return_value = LLMResponse(
            content=llm_content, model="test",
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        )
        engine = PromptEngine()
        return Planner(router=router, prompt_engine=engine, model="test")

    async def test_init_plan(self):
        ops = [{"command": "add_node", "node_data": {
            "id": "recon_1", "description": "端口扫描",
            "phase": "recon", "executor_type": "recon", "dependencies": [],
        }}]
        planner = self._make_planner(json.dumps(ops))
        config = DivineConfig(targets=["192.168.1.1"], goal="获取控制权")
        result = await planner.init_plan(goal="获取控制权", config=config)
        assert len(result) == 1
        assert result[0]["command"] == "add_node"

    async def test_replan_no_changes(self):
        planner = self._make_planner("[]")
        result = await planner.replan(blackboard_summary={}, dag_stats={}, reflections=[])
        assert result == []

    async def test_replan_with_new_task(self):
        ops = [{"command": "add_node", "node_data": {
            "id": "web_1", "description": "SQL 注入", "phase": "exploit",
            "executor_type": "web", "dependencies": ["recon_1"],
        }}]
        planner = self._make_planner(json.dumps(ops))
        result = await planner.replan(blackboard_summary={}, dag_stats={}, reflections=[])
        assert len(result) == 1

    async def test_should_terminate_true(self):
        planner = self._make_planner(json.dumps({"terminate": True, "reason": "目标已达成"}))
        should_stop, reason = await planner.should_terminate(blackboard_summary={}, dag_stats={})
        assert should_stop
        assert "达成" in reason

    async def test_should_terminate_false(self):
        planner = self._make_planner(json.dumps({"terminate": False, "reason": "还有任务未完成"}))
        should_stop, reason = await planner.should_terminate(blackboard_summary={}, dag_stats={})
        assert not should_stop

    def test_parse_operations_json(self):
        router = AsyncMock(spec=LLMRouter)
        planner = Planner(router=router, prompt_engine=PromptEngine(), model="test")
        ops = planner._parse_operations('[{"command": "add_node", "node_data": {"id": "t1"}}]')
        assert len(ops) == 1

    def test_parse_operations_markdown_wrapped(self):
        router = AsyncMock(spec=LLMRouter)
        planner = Planner(router=router, prompt_engine=PromptEngine(), model="test")
        content = '```json\n[{"command": "add_node", "node_data": {"id": "t1"}}]\n```'
        ops = planner._parse_operations(content)
        assert len(ops) == 1
