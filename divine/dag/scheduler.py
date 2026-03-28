import asyncio
from typing import Awaitable, Callable

from divine.dag.task_dag import TaskDAG
from divine.models.task import TaskNode, TaskStatus


class DAGScheduler:
    def __init__(self, dag: TaskDAG, concurrency: int = 3):
        self._dag = dag
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run_round(
        self, execute_fn: Callable[[TaskNode], Awaitable[dict]]
    ) -> list[str]:
        ready = self._dag.get_ready_tasks()
        if not ready:
            return []

        completed_ids = []

        async def _run_one(task: TaskNode) -> None:
            async with self._semaphore:
                await self._dag.update_status(task.id, TaskStatus.RUNNING)
                try:
                    result = await execute_fn(task)
                    await self._dag.update_status(
                        task.id, TaskStatus.COMPLETED, result=result
                    )
                except Exception as e:
                    await self._dag.update_status(
                        task.id, TaskStatus.FAILED, error=str(e)
                    )
                    await self._dag.propagate_failure(task.id)
                completed_ids.append(task.id)

        await asyncio.gather(*[_run_one(t) for t in ready])
        return completed_ids
