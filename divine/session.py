from dataclasses import asdict
from loguru import logger
from divine.blackboard import Blackboard
from divine.codeact.executor import CodeActExecutor
from divine.codeact.sandbox import Sandbox
from divine.config import DivineConfig
from divine.dag import TaskDAG, DAGScheduler
from divine.agents.planner import Planner
from divine.agents.reflection import Reflector
from divine.llm.router import LLMRouter
from divine.models.task import TaskNode
from divine.prompts.engine import PromptEngine


class Session:
    def __init__(self, config: DivineConfig):
        self._config = config
        self._blackboard = Blackboard(db_path=config.db_path)
        self._router = LLMRouter(config.llm)
        self._prompt_engine = PromptEngine()
        self._dag = TaskDAG()
        self._scheduler = DAGScheduler(self._dag, concurrency=config.concurrency)
        self._sandbox = Sandbox(timeout=config.code_execution_timeout)
        self._executor = CodeActExecutor(
            router=self._router, sandbox=self._sandbox,
            blackboard=self._blackboard, prompt_engine=self._prompt_engine,
            model=config.executor_model,
        )
        self._planner = Planner(
            router=self._router, prompt_engine=self._prompt_engine,
            model=config.planner_model,
        )
        self._reflector = Reflector(
            router=self._router, prompt_engine=self._prompt_engine,
            model=config.reflector_model,
        )

    async def run(self) -> None:
        logger.info(f"Session 启动: 目标={self._config.targets}, 最大轮次={self._config.max_rounds}")

        # 1. 初始规划
        operations = await self._planner.init_plan(
            goal=self._config.goal, config=self._config,
        )
        await self._dag.apply_operations(operations)
        logger.info(f"初始规划完成: {self._dag.stats}")

        # 2. 主循环
        for round_num in range(1, self._config.max_rounds + 1):
            logger.info(f"=== 第 {round_num} 轮 ===")

            # 2a. 调度并执行就绪任务
            completed = await self._scheduler.run_round(execute_fn=self._execute_task)
            logger.info(f"本轮完成 {len(completed)} 个任务: {completed}")

            if not completed and self._dag.is_finished:
                logger.info("所有任务已完成")
                break

            # 2b. 反思
            reflection = await self._reflector.analyze(
                blackboard_summary=self._blackboard.summary(),
                recent_results=self._get_recent_results(completed),
                dag_stats=self._dag.stats,
            )
            self._blackboard.write(
                "reflections", f"round_{round_num}",
                value=asdict(reflection), source="reflector",
            )

            # 2c. 重规划
            operations = await self._planner.replan(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
                reflections=[asdict(reflection)],
            )
            if operations:
                await self._dag.apply_operations(operations)
                logger.info(f"重规划: 应用了 {len(operations)} 个操作")

            # 2d. 终止检查
            should_stop, reason = await self._planner.should_terminate(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
            )
            if should_stop:
                logger.info(f"目标达成: {reason}")
                break

        logger.info(f"Session 结束: {self._dag.stats}")

    async def _execute_task(self, task: TaskNode) -> dict:
        context = self._blackboard.summary(
            sections=["hosts", "ports", "credentials", "findings"],
        )
        return await self._executor.execute_task(task, context)

    def _get_recent_results(self, task_ids: list[str]) -> list[dict]:
        results = []
        for tid in task_ids:
            task = self._dag.get_task(tid)
            if task:
                results.append({
                    "task_id": tid,
                    "description": task.description,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                })
        return results
