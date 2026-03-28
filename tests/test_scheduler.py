import asyncio
import pytest
from divine.dag.task_dag import TaskDAG
from divine.dag.scheduler import DAGScheduler
from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


def make_task(id: str, deps: list[str] = None, priority: int = 0) -> TaskNode:
    return TaskNode(
        id=id,
        description=f"Task {id}",
        phase=PentestPhase.RECON,
        executor_type=ExecutorType.RECON,
        dependencies=deps or [],
        priority=priority,
    )


async def success_fn(task: TaskNode) -> dict:
    return {"output": f"done:{task.id}"}


async def failing_fn(task: TaskNode) -> dict:
    raise RuntimeError(f"Task {task.id} failed")


# ---------------------------------------------------------------------------

class TestSchedulerBasic:
    @pytest.mark.asyncio
    async def test_run_round_basic(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        scheduler = DAGScheduler(dag, concurrency=3)

        completed = await scheduler.run_round(success_fn)

        assert set(completed) == {"t1", "t2"}
        assert dag.get_task("t1").status == TaskStatus.COMPLETED
        assert dag.get_task("t2").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_round_respects_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        scheduler = DAGScheduler(dag, concurrency=3)

        # First round: only t1 is ready
        round1 = await scheduler.run_round(success_fn)
        assert round1 == ["t1"]
        assert dag.get_task("t1").status == TaskStatus.COMPLETED
        assert dag.get_task("t2").status == TaskStatus.PENDING

        # Second round: t2 is now ready
        round2 = await scheduler.run_round(success_fn)
        assert round2 == ["t2"]
        assert dag.get_task("t2").status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_round_handles_failure(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        scheduler = DAGScheduler(dag, concurrency=3)

        completed = await scheduler.run_round(failing_fn)

        assert "t1" in completed
        assert dag.get_task("t1").status == TaskStatus.FAILED
        assert dag.get_task("t1").error == "Task t1 failed"
        # Descendant t2 should be skipped
        assert dag.get_task("t2").status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_round_no_ready_tasks(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        # Mark t1 as already running so it's not PENDING
        await dag.update_status("t1", TaskStatus.RUNNING)
        scheduler = DAGScheduler(dag, concurrency=3)

        completed = await scheduler.run_round(success_fn)
        assert completed == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """With concurrency=1, max simultaneous executions must be 1."""
        dag = TaskDAG()
        for i in range(3):
            await dag.add_task(make_task(f"t{i}"))

        active_at_once = []
        current_active = 0

        async def counting_fn(task: TaskNode) -> dict:
            nonlocal current_active
            current_active += 1
            active_at_once.append(current_active)
            await asyncio.sleep(0.01)
            current_active -= 1
            return {}

        scheduler = DAGScheduler(dag, concurrency=1)
        await scheduler.run_round(counting_fn)

        assert max(active_at_once) == 1

    @pytest.mark.asyncio
    async def test_result_stored_on_task(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        scheduler = DAGScheduler(dag, concurrency=1)

        async def fn_with_result(task: TaskNode) -> dict:
            return {"key": "value"}

        await scheduler.run_round(fn_with_result)
        assert dag.get_task("t1").result == {"key": "value"}
