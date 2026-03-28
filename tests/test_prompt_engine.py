from divine.prompts.engine import PromptEngine
from divine.models.common import ExecutorType
from divine.models.task import TaskNode, PentestPhase


class TestPromptEngine:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_build_init_plan_prompt(self):
        prompt = self.engine.build_init_plan_prompt(
            goal="获取目标 Web 服务器控制权",
            targets=["192.168.1.100"],
            output_schema={"type": "object"},
        )
        assert "获取目标 Web 服务器控制权" in prompt
        assert "192.168.1.100" in prompt

    def test_build_replan_prompt(self):
        prompt = self.engine.build_replan_prompt(
            blackboard_summary={"hosts": {"count": 1}},
            dag_stats={"total": 3, "completed": 1},
            reflections=[{"insights": ["发现 SSH 开放"]}],
            output_schema={},
        )
        assert prompt

    def test_build_terminate_check_prompt(self):
        prompt = self.engine.build_terminate_check_prompt(
            blackboard_summary={"findings": {"count": 5}},
            dag_stats={"total": 3, "completed": 3},
            goal="获取 root shell",
        )
        assert "获取 root shell" in prompt

    def test_build_executor_system_prompt(self):
        task = TaskNode(
            id="recon_1",
            description="端口扫描",
            phase=PentestPhase.RECON,
            executor_type=ExecutorType.RECON,
        )
        prompt = self.engine.build_executor_system_prompt(
            executor_type=ExecutorType.RECON,
            task=task,
            context={"hosts": {}},
            stdlib_docs="run_command(cmd): 执行 shell 命令",
        )
        assert "run_command" in prompt
        assert "端口扫描" in prompt

    def test_build_executor_prompt_different_types(self):
        task = TaskNode(
            id="web_1",
            description="SQL 注入测试",
            phase=PentestPhase.EXPLOIT,
            executor_type=ExecutorType.WEB,
        )
        prompt = self.engine.build_executor_system_prompt(
            executor_type=ExecutorType.WEB,
            task=task,
            context={},
            stdlib_docs="",
        )
        assert prompt

    def test_build_observation_prompt(self):
        from divine.codeact.sandbox import ExecutionResult

        result = ExecutionResult(
            success=True,
            stdout="PORT  STATE SERVICE\n22/tcp open ssh",
            execution_time=1.2,
        )
        prompt = self.engine.build_observation_prompt(result)
        assert "22/tcp" in prompt

    def test_build_analyze_prompt(self):
        prompt = self.engine.build_analyze_prompt(
            blackboard_summary={"findings": {"count": 3}},
            recent_results=[{"task_id": "t1", "status": "completed"}],
            dag_stats={"total": 5, "completed": 2},
        )
        assert prompt
