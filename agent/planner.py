import json
from typing import Optional, List

from agent.agent import Agent, AgentConfig
from agent.planner_types import (
    PlanningMode,
    PlanningThought,
    PlanningResult,
    PlannerContext, GraphOperation,
)
from llm_client import Message, LLMClient
from logger_config import log
from prompts.manager import PromptManager


class Planner(Agent):

    def __init__(
            self,
            config: AgentConfig,
            llm_client: Optional[LLMClient] = None,
            prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client, None, prompt_manager)
        self.planner_context: Optional[PlannerContext] = None

    def _sanitize_graph_operations(self, ops: List[GraphOperation]) -> List[GraphOperation]:
        """
        净化图操作指令：去重 ADD_NODE，保留其他操作。
        单次遍历完成，提高效率。
        """
        sanitized: List[GraphOperation] = []
        seen_add_ids: set = set()

        for op in ops:
            cmd = op.command

            if cmd == "ADD_NODE":
                node_id = op.node_data.get("id")
                if not node_id or node_id is None:
                    continue
                if node_id in seen_add_ids:
                    continue
                seen_add_ids.add(node_id)
                sanitized.append(op)

            elif cmd in {"DELETE_NODE", "DEPRECATE_NODE", "UPDATE_NODE"}:
                node_id = op.node_id
                if not node_id:
                    continue
                if cmd == "UPDATE_NODE" and not op.updates:
                    continue
                sanitized.append(op)

            else:
                # 其他未知的或自定义的指令，直接保留
                sanitized.append(op)

        return sanitized

    def _build_messages(self,planner_prompt:str) -> List[Message]:

        return [Message(
            role="user",
            content=planner_prompt,
        )]


    def parse_response(self, response) -> PlanningResult:
        """解析LLM响应为规划结果"""
        content = response.content
        log.debug(f"[{self.name}] 解析响应 | content={content}")

        # 尝试提取JSON
        json_data = self._extract_json(content)

        if json_data is None:
            log.warning(f"[{self.name}] 无法提取JSON | raw_content: \n{content}")
            # 返回空结果
            return PlanningResult(
                thought=PlanningThought(),
                graph_operations=[],
                raw_response=content,
            )

        result = PlanningResult.from_dict(json_data, raw_response=content)

        return result




    async def init_plan(self,goal: str):
        log.info(f"[{self.name}] 开始规划 | mode={PlanningMode.INITIAL.value} | goal={goal}")
        self.planner_context = PlannerContext(
            planning_mode=PlanningMode.INITIAL,
            final_goal=goal,
        )

        self.on_start()

        init_planner_prompt = self.prompt_manager.build_planner_prompt(
            planning_mode=PlanningMode.INITIAL,
            planner_context=self.planner_context
        )

        log.debug(f"[{self.name}] 初始规划提示词:\n{init_planner_prompt}\n")

        try:
            response = await self._call_llm(self._build_messages(init_planner_prompt))
            planning_result = self.parse_response(response)

            if not planning_result.graph_operations:
                log.error(f"[{self.name}] 未生成任何图操作，规划失败。")
                raise ValueError("未生成任何图操作")
            return planning_result
        except ValueError:
            log.error(f"[{self.name}] 规划过程中出现错误，使用兜底计划。")
            return PlanningResult(
                thought=PlanningThought(),
                graph_operations=[GraphOperation.from_dict({
                    "command": "ADD_NODE",
                    "node_data": {
                        "id": "subtask_1",
                        "description": f"执行初步信息收集以理解目标: {goal}",
                        "dependencies": [],
                        "priority": 1,
                    },
                })],
                raw_response=None,
            )


    async def dynamic_plan(
            self,
            intelligence_summary,
            graph_manager
    ):
        failed_tasks_summary = ""
        if graph_manager:
            failed_nodes = graph_manager.get_failed_nodes()
            if failed_nodes:
                failed_tasks_list = []
                for node_id, data in failed_nodes.items():
                    failed_tasks_list.append(
                        f"- Task ID: {node_id}, Status: {data.get('status')}, Description: {data.get('description')}"
                    )
                failed_tasks_summary = (
                        "\n### 高优先级：失败/阻塞的任务\n你必须优先处理以下失败或阻塞的任务。请为它们设计诊断或替代方案。\n"
                        + "\n".join(failed_tasks_list)
                )

        log.debug(f"[{self.name}] 开始动态规划 | failed_tasks_summary=\n{failed_tasks_summary}\n")

        intelligence_summary_str = json.dumps(intelligence_summary, ensure_ascii=False, indent=2)


        # TODO: 添加检索的专家经验
        retrieved_experience = ""

        dynamic_planner_prompt = self.prompt_manager.build_planner_prompt(
            planning_mode=PlanningMode.DYNAMIC,
            planner_context=self.planner_context,
            intelligence_summary_str=intelligence_summary_str,
            failed_tasks_summary_str=failed_tasks_summary,
            retrieved_experience_str = retrieved_experience
        )

        log.debug(f"[{self.name}] 动态规划提示词:\n{dynamic_planner_prompt}\n")

        try:
            response = await self._call_llm(self._build_messages(dynamic_planner_prompt))
            planning_result = self.parse_response(response)

            if not planning_result.graph_operations:
                log.error(f"[{self.name}] 未生成任何图操作，动态规划失败。")
                raise ValueError("未生成任何图操作")
            return planning_result
        except (ValueError,Exception):
            log.error(f"[{self.name}] 动态规划过程中出现错误")
            return PlanningResult(
                thought=PlanningThought(),
                graph_operations=[],
                raw_response=None,
            )


    async def branch_replan(
            self,
            graph_manager,
            failed_branch_root_id: str,
            failure_reason: str
    ):
        log.info(f"[{self.name}] 分支重规划 | failed_branch_root_id={failed_branch_root_id}")
        try:
            original_branch_goal = graph_manager.graph.nodes[failed_branch_root_id].get("description", "[目标描述丢失]")
            descendants = graph_manager.get_descendants(failed_branch_root_id)
            dead_end_tasks = list({failed_branch_root_id}.union(descendants))
        except Exception as e:
            log.error(f"[{self.name}] 获取失败分支信息上下文时出错，{e}")
            return PlanningResult(
                thought=PlanningThought(),
                graph_operations=[],
                raw_response=None,
            )

        sanitized_ops = []
        seen_add_ids = set()

        branch_replan_prompt = self.prompt_manager.build_branch_replan_prompt(
            original_branch_goal=original_branch_goal,
            failure_reason=failure_reason,
            dead_end_tasks=dead_end_tasks,
        )

        log.debug(f"[{self.name}] 分支重规划提示词:\n{branch_replan_prompt}\n")

        try:
            response = await self._call_llm(self._build_messages(branch_replan_prompt))
            planning_result = self.parse_response(response)

            if not planning_result.graph_operations:
                log.error(f"[{self.name}] 未生成任何图操作，分支重规划失败。")
                raise ValueError("未生成任何图操作")
            ops = planning_result.graph_operations
            try:
                dead_branch_ids = {failed_branch_root_id}.union(descendants)
            except Exception:
                dead_branch_ids = {failed_branch_root_id}
            for op in ops:
                cmd = op.command
                if cmd == "ADD_NODE":
                    node_data = op.node_data
                    node_id = node_data.get("id")
                    if not node_id or node_id == "None":
                        continue
                    if node_id in seen_add_ids:
                        continue

                    # 关键修复：清理新节点的依赖关系，移除对dead_branch的依赖
                    dependencies = node_data.get("dependencies", [])
                    if dependencies:
                        # 过滤掉指向dead_branch的依赖
                        clean_dependencies = [dep for dep in dependencies if dep not in dead_branch_ids]

                        # 如果所有依赖都被移除了，需要找到一个合适的替代依赖
                        if not clean_dependencies and dependencies:
                            # 尝试找到dead_branch父节点的其他健康子节点
                            try:
                                # 获取失败分支的父节点
                                parents_of_failed = list(graph_manager.graph.predecessors(failed_branch_root_id))
                                if parents_of_failed:
                                    parent = parents_of_failed[0]
                                    # 获取父节点的其他已完成的子节点
                                    siblings = [succ for succ in graph_manager.graph.successors(parent)
                                                if graph_manager.graph.nodes[succ].get('status') == 'completed'
                                                and succ not in dead_branch_ids]
                                    if siblings:
                                        clean_dependencies = [siblings[-1]]  # 使用最后一个完成的兄弟节点
                                    else:
                                        # 如果没有兄弟节点，直接依赖父节点
                                        clean_dependencies = [parent]
                            except Exception as e:
                                clean_dependencies = []

                        node_data["dependencies"] = clean_dependencies

                    # 若要添加的节点已存在，则交由执行层处理为更新；此处仍保留，但避免重复
                    seen_add_ids.add(node_id)
                    sanitized_ops.append(op)
                elif cmd in {"DELETE_NODE", "DEPRECATE_NODE", "UPDATE_NODE"}:
                    node_id = op.node_id
                    if not node_id:
                        continue
                    # 跳过对失败分支及其后代的更新，改为 DEPRECATE（由执行层统一处理为状态变更）
                    if node_id in dead_branch_ids and cmd == "UPDATE_NODE":
                        sanitized_ops.append(
                            {
                                "command": "DEPRECATE_NODE",
                                "node_id": node_id,
                                "reason": f"Branch '{failed_branch_root_id}' failed: {failure_reason}",
                            }
                        )
                        continue
                    if cmd == "UPDATE_NODE" and not op.updates:
                        continue
                    sanitized_ops.append(op)
                else:
                    sanitized_ops.append(op)

            planning_result.graph_operations = sanitized_ops
            return planning_result

        except (ValueError,Exception):
            log.error(f"[{self.name}] 分支重规划过程中出现错误")
            return PlanningResult(
                thought=PlanningThought(),
                graph_operations=[],
                raw_response=None,
            )
