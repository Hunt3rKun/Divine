"""Main orchestration loop for planner, executor, evaluator, and blackboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from divine.agents import (
    EvaluatorAgent,
    ExecutionRouter,
    ExecutorAgent,
    PlannerAgent,
    create_executor_agents,
)
from divine.blackboard import PlannerResult, SharedBlackboard, TaskContext, TaskNode
from divine.logger import get_logger
from divine.tools import ToolAdapter


StopReason = Literal[
    "max_iterations",
    "max_consecutive_failures",
    "no_executable_nodes",
    "route_blocked",
    "executor_missing",
    "executor_failed",
    "evaluator_failed",
    "planner_failed",
    "planner_terminated",
]


@dataclass
class RunResult:
    blackboard: SharedBlackboard
    iterations: int
    stop_reason: StopReason
    planner_results: list[PlannerResult] = field(default_factory=list)


class Orchestrator:
    """Coordinates the framework loop without doing agent-specific reasoning."""

    def __init__(
        self,
        *,
        planner: PlannerAgent | None = None,
        router: ExecutionRouter | None = None,
        evaluator: EvaluatorAgent | None = None,
        executors: dict[str, ExecutorAgent] | None = None,
        llm_client: Any | None = None,
        tools: ToolAdapter | None = None,
    ) -> None:
        if llm_client is None and (planner is None or evaluator is None or executors is None):
            raise ValueError("llm_client is required when planner, evaluator, or executors are not provided.")
        self.planner = planner or PlannerAgent(llm_client)
        self.router = router or ExecutionRouter()
        self.evaluator = evaluator or EvaluatorAgent(llm_client)
        self.executors = executors or create_executor_agents(tools, llm_client=llm_client)
        self._log = get_logger("orchestrator")

    def run(self, context: TaskContext) -> RunResult:
        blackboard = SharedBlackboard(context=context)
        self._log.info(
            "Run started task_id={} target={} max_iterations={} max_consecutive_failures={}",
            context.task_id,
            context.target,
            context.max_iterations,
            context.max_consecutive_failures,
        )
        try:
            initial_result = self.planner.generate_initial_dag(context, blackboard)
        except Exception as exc:
            blackboard.record_event(
                "PlannerFailed",
                "Initial planner failed",
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            self._log.exception(
                "Run stopped task_id={} stop_reason=planner_failed stage=initial_planning",
                context.task_id,
            )
            return RunResult(
                blackboard=blackboard,
                iterations=0,
                stop_reason="planner_failed",
                planner_results=[],
            )
        planner_results = [initial_result]
        self._log.info(
            "Initial planning completed task_id={} status={} strategy={} added_nodes={}",
            context.task_id,
            planner_results[0].status,
            planner_results[0].strategy,
            planner_results[0].added_nodes,
        )
        if initial_result.should_terminate:
            self._log.info(
                "Run stopped task_id={} stop_reason=planner_terminated iterations=0",
                context.task_id,
            )
            return RunResult(
                blackboard=blackboard,
                iterations=0,
                stop_reason="planner_terminated",
                planner_results=planner_results,
            )
        if not blackboard.graph.nodes:
            blackboard.record_event(
                "PlannerFailed",
                "Initial planner produced an empty DAG",
                {"status": initial_result.status, "strategy": initial_result.strategy},
            )
            self._log.error(
                "Run stopped task_id={} stop_reason=planner_failed stage=initial_planning reason=empty_dag",
                context.task_id,
            )
            return RunResult(
                blackboard=blackboard,
                iterations=0,
                stop_reason="planner_failed",
                planner_results=planner_results,
            )
        consecutive_failures = 0

        for iteration in range(1, context.max_iterations + 1):
            self._log.debug(
                "Iteration started task_id={} iteration={} node_count={} consecutive_failures={}",
                context.task_id,
                iteration,
                len(blackboard.graph.nodes),
                consecutive_failures,
            )
            node = self._select_executable_node(blackboard)
            if node is None:
                self._log.info(
                    "Run stopped task_id={} stop_reason=no_executable_nodes iterations={} node_count={}",
                    context.task_id,
                    iteration - 1,
                    len(blackboard.graph.nodes),
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration - 1,
                    stop_reason="no_executable_nodes",
                    planner_results=planner_results,
                )

            route = self.router.route(node)
            self._log.info(
                "Node routed task_id={} iteration={} node_id={} task_type={} selected_agent={} blocked={} reason={}",
                context.task_id,
                iteration,
                node.node_id,
                node.task_type,
                route.selected_agent,
                route.blocked,
                route.block_reason or route.routing_reason,
            )
            if route.blocked:
                node.status = "blocked"
                node.metadata["blocked_reason"] = route.block_reason or route.routing_reason
                blackboard.record_event(
                    "TaskAssigned",
                    f"Routing blocked for {node.node_id}",
                    {
                        "node_id": node.node_id,
                        "selected_agent": route.selected_agent,
                        "block_reason": route.block_reason,
                    },
                )
                self._log.warning(
                    "Run stopped task_id={} stop_reason=route_blocked iterations={} node_id={} block_reason={}",
                    context.task_id,
                    iteration - 1,
                    node.node_id,
                    route.block_reason,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration - 1,
                    stop_reason="route_blocked",
                    planner_results=planner_results,
                )

            executor = self.executors.get(route.selected_agent)
            if executor is None:
                node.status = "blocked"
                node.metadata["blocked_reason"] = f"Missing executor {route.selected_agent}"
                blackboard.record_event(
                    "TaskAssigned",
                    f"Missing executor for {node.node_id}",
                    {"node_id": node.node_id, "selected_agent": route.selected_agent},
                )
                self._log.error(
                    "Run stopped task_id={} stop_reason=executor_missing iterations={} node_id={} selected_agent={}",
                    context.task_id,
                    iteration - 1,
                    node.node_id,
                    route.selected_agent,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration - 1,
                    stop_reason="executor_missing",
                    planner_results=planner_results,
                )

            blackboard.record_event(
                "TaskAssigned",
                f"Assigned {node.node_id} to {route.selected_agent}",
                {
                    "node_id": node.node_id,
                    "selected_agent": route.selected_agent,
                    "routing_reason": route.routing_reason,
                },
            )
            try:
                execution_result = executor.execute(node, blackboard)
            except Exception as exc:
                node.status = "blocked"
                node.metadata["blocked_reason"] = f"Executor failed: {exc}"
                blackboard.record_event(
                    "ExecutionFailed",
                    f"Executor failed for {node.node_id}",
                    {
                        "node_id": node.node_id,
                        "selected_agent": route.selected_agent,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                self._log.exception(
                    "Run stopped task_id={} stop_reason=executor_failed iteration={} node_id={} executor={}",
                    context.task_id,
                    iteration,
                    node.node_id,
                    route.selected_agent,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration,
                    stop_reason="executor_failed",
                    planner_results=planner_results,
                )
            self._log.info(
                "Execution completed task_id={} iteration={} node_id={} execution_id={} executor={} status={} evidence_refs={} errors={}",
                context.task_id,
                iteration,
                node.node_id,
                execution_result.execution_id,
                execution_result.executor,
                execution_result.status,
                execution_result.evidence_refs,
                execution_result.errors,
            )
            try:
                feedback = self.evaluator.audit(
                    node=node,
                    execution_result=execution_result,
                    blackboard=blackboard,
                )
            except Exception as exc:
                node.status = "blocked"
                node.metadata["blocked_reason"] = f"Evaluator failed: {exc}"
                blackboard.record_event(
                    "AuditFailed",
                    f"Evaluator failed for {node.node_id}",
                    {"node_id": node.node_id, "error": str(exc), "error_type": type(exc).__name__},
                )
                self._log.exception(
                    "Run stopped task_id={} stop_reason=evaluator_failed iteration={} node_id={}",
                    context.task_id,
                    iteration,
                    node.node_id,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration,
                    stop_reason="evaluator_failed",
                    planner_results=planner_results,
                )
            try:
                planner_result = self.planner.evolve_dag(feedback, blackboard)
            except Exception as exc:
                node.status = "blocked"
                node.metadata["blocked_reason"] = f"Planner failed: {exc}"
                blackboard.record_event(
                    "PlannerFailed",
                    f"Planner failed for {node.node_id}",
                    {"node_id": node.node_id, "error": str(exc), "error_type": type(exc).__name__},
                )
                self._log.exception(
                    "Run stopped task_id={} stop_reason=planner_failed iteration={} node_id={} feedback_id={}",
                    context.task_id,
                    iteration,
                    node.node_id,
                    feedback.feedback_id,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration,
                    stop_reason="planner_failed",
                    planner_results=planner_results,
                )
            planner_results.append(planner_result)
            self._log.info(
                "Planning evolved task_id={} iteration={} strategy={} status={} added_nodes={} updated_nodes={} pruned_nodes={} should_terminate={}",
                context.task_id,
                iteration,
                planner_result.strategy,
                planner_result.status,
                planner_result.added_nodes,
                planner_result.updated_nodes,
                planner_result.pruned_nodes,
                planner_result.should_terminate,
            )

            if planner_result.should_terminate:
                self._log.info(
                    "Run stopped task_id={} stop_reason=planner_terminated iterations={}",
                    context.task_id,
                    iteration,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration,
                    stop_reason="planner_terminated",
                    planner_results=planner_results,
                )

            if feedback.task_judgement.status in {"failed", "blocked", "uncertain"}:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if consecutive_failures >= context.max_consecutive_failures:
                self._log.warning(
                    "Run stopped task_id={} stop_reason=max_consecutive_failures iterations={} consecutive_failures={}",
                    context.task_id,
                    iteration,
                    consecutive_failures,
                )
                return RunResult(
                    blackboard=blackboard,
                    iterations=iteration,
                    stop_reason="max_consecutive_failures",
                    planner_results=planner_results,
                )

        self._log.info(
            "Run stopped task_id={} stop_reason=max_iterations iterations={}",
            context.task_id,
            context.max_iterations,
        )
        return RunResult(
            blackboard=blackboard,
            iterations=context.max_iterations,
            stop_reason="max_iterations",
            planner_results=planner_results,
        )

    def _select_executable_node(self, blackboard: SharedBlackboard) -> TaskNode | None:
        nodes = blackboard.graph.executable_nodes()
        if not nodes:
            return None
        return sorted(nodes, key=lambda node: node.node_id)[0]
