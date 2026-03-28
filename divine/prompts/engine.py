from pathlib import Path

import jinja2

from divine.models.common import ExecutorType
from divine.models.task import TaskNode

DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"


class PromptEngine:
    def __init__(self, template_dir: Path = None):
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                str(template_dir or DEFAULT_TEMPLATE_DIR)
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _render(self, template_path: str, **kwargs) -> str:
        template = self._env.get_template(f"{template_path}.jinja2")
        return template.render(**kwargs)

    def build_init_plan_prompt(
        self, goal: str, targets: list[str], output_schema: dict
    ) -> str:
        return self._render(
            "planner/init_plan",
            goal=goal,
            targets=targets,
            output_schema=output_schema,
        )

    def build_replan_prompt(
        self,
        blackboard_summary: dict,
        dag_stats: dict,
        reflections: list[dict],
        output_schema: dict,
    ) -> str:
        return self._render(
            "planner/replan",
            blackboard_summary=blackboard_summary,
            dag_stats=dag_stats,
            reflections=reflections,
            output_schema=output_schema,
        )

    def build_terminate_check_prompt(
        self, blackboard_summary: dict, dag_stats: dict, goal: str
    ) -> str:
        return self._render(
            "planner/terminate_check",
            blackboard_summary=blackboard_summary,
            dag_stats=dag_stats,
            goal=goal,
        )

    def build_executor_system_prompt(
        self,
        executor_type: ExecutorType,
        task: TaskNode,
        context: dict,
        stdlib_docs: str,
    ) -> str:
        base = self._render("executor/base_system", stdlib_docs=stdlib_docs)
        specific = self._render(
            f"executor/{executor_type.value}_system", task=task, context=context
        )
        return f"{base}\n\n{specific}"

    def build_observation_prompt(self, result) -> str:
        return self._render("executor/observation", result=result)

    def build_analyze_prompt(
        self,
        blackboard_summary: dict,
        recent_results: list[dict],
        dag_stats: dict,
    ) -> str:
        return self._render(
            "reflector/analyze",
            blackboard_summary=blackboard_summary,
            recent_results=recent_results,
            dag_stats=dag_stats,
        )
