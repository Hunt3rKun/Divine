import json
import os
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent.planner_types import PlannerContext, PlanningMode


class PromptManager:
    def __init__(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        # 创建Jinja2环境
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self.planner_template = self.env.get_template("planner/planner_template.jinja2")
        self.branch_replan_template = self.env.get_template("planner/branch_replan_template.jinja2")
        # self.executor_template = self.env.get_template("executor/executor_template.jinja2")
        # self.reflector_template = self.env.get_template("reflector/reflector_template.jinja2")


    def build_planner_prompt(
            self,
            planner_context: PlannerContext,
            planning_mode: PlanningMode,
            intelligence_summary_str: str = "",
            failed_tasks_summary_str:str = "",
            retrieved_experience: str = "",
            graph_summary: str = "",
    ) -> str:
        """构建规划器提示语"""

        input_variables :Dict[str,Any] = {
            "goal": planner_context.final_goal,
            "use_ctf_optimizations":True

        }

        if planning_mode == PlanningMode.DYNAMIC:
            dynamic_planning ={
                "intelligence_summary": intelligence_summary_str,
                "failed_tasks_summary": failed_tasks_summary_str,
                "graph_summary": graph_summary,
                "retrieved_experience": retrieved_experience,
            }
            input_variables["dynamic_planning"] = dynamic_planning

        if planner_context.planning_history:
            input_variables["planning_history"] = self._render_planning_history_section(planner_context)


        return self.planner_template.render(**input_variables)




    def build_branch_replan_prompt(
            self,
            original_branch_goal,
            failure_reason,
            dead_end_tasks
    ):
        # 组装输入变量
        input_variables = {
            "original_branch_goal": original_branch_goal,
            "failure_reason": failure_reason,
            "dead_end_tasks": json.dumps(dead_end_tasks, indent=2, ensure_ascii=False),
        }


        return self.branch_replan_template.render(**input_variables)








    def _render_planning_history_section(self,planner_context: PlannerContext):
        """渲染规划历史部分"""
        # TODO: 可以根据需要调整格式
        history_entries = []
        for planning in planner_context.planning_history:
            entry = f"时间: {planning.timestamp:.2f}s\n"
            entry += f"模式: {planning.mode.value}\n"
            entry += f"策略: {planning.strategy}\n"
            entry += f"结果总结: {planning.outcome_summary}\n"
            entry += "-" * 20 + "\n"
            history_entries.append(entry)
        return "\n".join(history_entries)










