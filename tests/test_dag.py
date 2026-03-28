import pytest
from divine.dag.task_dag import TaskDAG
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


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestTaskDAGBasic:
    @pytest.mark.asyncio
    async def test_add_task(self):
        dag = TaskDAG()
        t = make_task("t1")
        await dag.add_task(t)
        assert dag.get_task("t1") is t

    @pytest.mark.asyncio
    async def test_add_task_with_deps(self):
        dag = TaskDAG()
        t1 = make_task("t1")
        t2 = make_task("t2", deps=["t1"])
        await dag.add_task(t1)
        await dag.add_task(t2)
        tasks = dag.get_all_tasks()
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_remove_task(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.remove_task("t1")
        assert dag.get_task("t1") is None

    @pytest.mark.asyncio
    async def test_get_all_tasks(self):
        dag = TaskDAG()
        for i in range(3):
            await dag.add_task(make_task(f"t{i}"))
        assert len(dag.get_all_tasks()) == 3

    @pytest.mark.asyncio
    async def test_cycle_detection_raises(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        # t3 depends on t2, and t1 would need to depend on t3 to create cycle.
        # Simulate a direct cycle: t1 → t2, add t3 dep=[t2] then try t1 dep=[t3]
        await dag.add_task(make_task("t3", deps=["t2"]))
        with pytest.raises(ValueError, match="cycle"):
            # Create a node that would make t1 → t2 → t3 → t1
            t1_bad = make_task("t1_cycle", deps=["t3"])
            # Manually inject dependency pointing back to create cycle
            # Instead test: add t4 dep=[t3], then try to add t1 dep=[t4]
            # Actually the simplest is to try to re-add t1 with dep on t3
            # We can't re-add t1, so let's try adding a node that introduces a cycle
            # by depending on a node that transitively depends on a not-yet-existing node
            # The real test: add t_new dep=[t3], which is fine. For a real cycle,
            # we need a node already in the graph to depend on a new one.
            # The only way to get a cycle here is if the DAG allows self-loops or
            # we manipulate the graph. Let's just verify a direct self-dep raises.
            await dag.add_task(make_task("t_self", deps=["t_self"]))


# ---------------------------------------------------------------------------
# Ready tasks
# ---------------------------------------------------------------------------

class TestTaskDAGReadyTasks:
    @pytest.mark.asyncio
    async def test_get_ready_no_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        ready = dag.get_ready_tasks()
        assert {t.id for t in ready} == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_get_ready_with_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        ready = dag.get_ready_tasks()
        assert [t.id for t in ready] == ["t1"]

    @pytest.mark.asyncio
    async def test_ready_after_dep_completed(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        ready = dag.get_ready_tasks()
        assert [t.id for t in ready] == ["t2"]

    @pytest.mark.asyncio
    async def test_ready_sorted_by_priority(self):
        dag = TaskDAG()
        await dag.add_task(make_task("low", priority=1))
        await dag.add_task(make_task("high", priority=10))
        await dag.add_task(make_task("mid", priority=5))
        ready = dag.get_ready_tasks()
        assert [t.id for t in ready] == ["high", "mid", "low"]


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------

class TestTaskDAGFailurePropagation:
    @pytest.mark.asyncio
    async def test_propagate_failure_marks_descendants_skipped(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.add_task(make_task("t3", deps=["t2"]))
        await dag.add_task(make_task("t4"))  # not a descendant

        await dag.update_status("t1", TaskStatus.FAILED)
        skipped = await dag.propagate_failure("t1")

        assert set(skipped) == {"t2", "t3"}
        assert dag.get_task("t2").status == TaskStatus.SKIPPED
        assert dag.get_task("t3").status == TaskStatus.SKIPPED
        assert dag.get_task("t4").status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_propagate_failure_nonexistent_task(self):
        dag = TaskDAG()
        skipped = await dag.propagate_failure("nonexistent")
        assert skipped == []

    @pytest.mark.asyncio
    async def test_get_descendants(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.add_task(make_task("t3", deps=["t2"]))
        assert dag.get_descendants("t1") == {"t2", "t3"}
        assert dag.get_descendants("t3") == set()


# ---------------------------------------------------------------------------
# Properties: is_finished, stats
# ---------------------------------------------------------------------------

class TestTaskDAGProperties:
    @pytest.mark.asyncio
    async def test_is_finished_false_when_pending(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        assert dag.is_finished is False

    @pytest.mark.asyncio
    async def test_is_finished_true_when_all_terminal(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        await dag.update_status("t2", TaskStatus.SKIPPED)
        assert dag.is_finished is True

    @pytest.mark.asyncio
    async def test_is_finished_true_empty_dag(self):
        dag = TaskDAG()
        # Empty DAG: all() on empty is True
        assert dag.is_finished is True

    @pytest.mark.asyncio
    async def test_stats(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.add_task(make_task("t3"))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        await dag.update_status("t2", TaskStatus.FAILED)
        s = dag.stats
        assert s["total"] == 3
        assert s["completed"] == 1
        assert s["failed"] == 1
        assert s["pending"] == 1

    @pytest.mark.asyncio
    async def test_get_failed_tasks(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.update_status("t1", TaskStatus.FAILED)
        failed = dag.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].id == "t1"


# ---------------------------------------------------------------------------
# apply_operations
# ---------------------------------------------------------------------------

class TestTaskDAGApplyOperations:
    @pytest.mark.asyncio
    async def test_apply_add_node(self):
        dag = TaskDAG()
        ops = [
            {
                "command": "add_node",
                "node_data": {
                    "id": "op_t1",
                    "description": "Added via op",
                    "phase": "recon",
                    "executor_type": "recon",
                    "dependencies": [],
                    "priority": 5,
                },
            }
        ]
        await dag.apply_operations(ops)
        task = dag.get_task("op_t1")
        assert task is not None
        assert task.priority == 5

    @pytest.mark.asyncio
    async def test_apply_remove_node(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        ops = [{"command": "remove_node", "node_id": "t1"}]
        await dag.apply_operations(ops)
        assert dag.get_task("t1") is None

    @pytest.mark.asyncio
    async def test_apply_update_node(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1", priority=1))
        ops = [{"command": "update_node", "node_id": "t1", "updates": {"priority": 99}}]
        await dag.apply_operations(ops)
        assert dag.get_task("t1").priority == 99

    @pytest.mark.asyncio
    async def test_apply_multiple_operations(self):
        dag = TaskDAG()
        ops = [
            {
                "command": "add_node",
                "node_data": {"id": "a", "phase": "recon", "executor_type": "recon"},
            },
            {
                "command": "add_node",
                "node_data": {
                    "id": "b",
                    "phase": "recon",
                    "executor_type": "recon",
                    "dependencies": ["a"],
                },
            },
            {"command": "update_node", "node_id": "a", "updates": {"priority": 7}},
        ]
        await dag.apply_operations(ops)
        assert dag.get_task("a").priority == 7
        assert dag.get_task("b") is not None
