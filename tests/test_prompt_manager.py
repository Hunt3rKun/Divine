from prompts.manager import PromptManager
from agent.planner_types import PlannerContext, PlanningMode


def test_prompt_manager():
    pm = PromptManager()
    ctx = PlannerContext(planning_mode=PlanningMode.INITIAL, final_goal="test goal")

    prompt = pm.build_planner_prompt(planner_context=ctx, planning_mode=PlanningMode.INITIAL)

    assert isinstance(prompt, str)
    assert prompt.strip()
