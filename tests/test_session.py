from unittest.mock import AsyncMock
import pytest
from divine.session import Session
from divine.config import DivineConfig, LLMConfig, ProviderConfig
from divine.models.task import TaskStatus
from divine.agents.reflection import Reflection


def make_config() -> DivineConfig:
    return DivineConfig(
        targets=["192.168.1.1"],
        goal="获取目标控制权",
        max_rounds=3,
        concurrency=1,
        llm=LLMConfig(providers={"openai": ProviderConfig(api_key="test")}),
    )


class TestSession:
    async def test_session_init(self):
        """Session should initialize all components"""
        config = make_config()
        session = Session(config)
        assert session._dag is not None
        assert session._blackboard is not None
        assert session._planner is not None
        assert session._reflector is not None

    async def test_session_run_basic_flow(self):
        """Test basic flow: init_plan -> run_round -> reflect -> replan -> terminate"""
        config = make_config()
        session = Session(config)

        # Mock planner
        init_ops = [{"command": "add_node", "node_data": {
            "id": "t1", "description": "扫描", "phase": "recon",
            "executor_type": "recon", "dependencies": [],
        }}]
        session._planner.init_plan = AsyncMock(return_value=init_ops)
        session._planner.replan = AsyncMock(return_value=[])
        session._planner.should_terminate = AsyncMock(return_value=(True, "目标达成"))

        # Mock executor
        session._executor.execute_task = AsyncMock(return_value={"status": "done"})

        # Mock reflector
        session._reflector.analyze = AsyncMock(return_value=Reflection(
            insights=["test"], suggested_tasks=[], risk_assessment="low", progress_summary="done",
        ))

        await session.run()

        session._planner.init_plan.assert_called_once()
        assert session._dag.get_task("t1").status == TaskStatus.COMPLETED

    async def test_session_max_rounds_terminates(self):
        """max_rounds should be respected as hard limit"""
        config = make_config()
        config.max_rounds = 2
        session = Session(config)

        init_ops = [
            {"command": "add_node", "node_data": {
                "id": f"t{i}", "description": f"Task {i}", "phase": "recon",
                "executor_type": "recon", "dependencies": [],
            }} for i in range(10)
        ]
        session._planner.init_plan = AsyncMock(return_value=init_ops)
        session._planner.replan = AsyncMock(return_value=[])
        session._planner.should_terminate = AsyncMock(return_value=(False, ""))
        session._executor.execute_task = AsyncMock(return_value={})
        session._reflector.analyze = AsyncMock(return_value=Reflection())

        await session.run()
        # Should stop after max_rounds, not infinite loop
