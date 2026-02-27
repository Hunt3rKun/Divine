from typing import Dict, Any


class AgentRunner:

    def _aggregate_intelligence(self,completed_reflections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        汇总多个反思器输出为情报摘要.

        将多个子任务的反思结果汇总为统一的情报摘要，优先处理已达成目标的状态
        和目标产物类型的artifacts。

        Args:
            completed_reflections: 已完成的反思结果字典，key为子任务ID，value为反思输出

        Returns:
            Dict[str, Any]: 汇总的情报摘要，包含findings、audit_result、artifacts等字段
        """
        all_findings = []
        all_artifacts = []
        all_insights = []

        # 检查是否有任何反思结果标记了 GOAL_ACHIEVED
        goal_achieved = False
        aggregated_completion_check = f'汇总了 {len(completed_reflections)} 个任务的审计结果'
        for subtask_id, reflection in completed_reflections.items():
            # 从 reflection 提取所需字段
            audit_result = reflection.get('audit_result', {})
            if audit_result.get('status') == 'GOAL_ACHIEVED':
                goal_achievement_reason = audit_result.get('completion_check', 'Unknown reason for GOAL_ACHIEVED')
                goal_achieved = True
                aggregated_completion_check = goal_achievement_reason

            # 提取 key_findings
            findings = reflection.get('key_findings', [])
            all_findings.extend(findings)

            # 提取 validated_nodes
            nodes = reflection.get('validated_nodes', [])
            all_artifacts.extend(nodes)

            # 提取 insight
            insight = reflection.get('insight')
            if insight:
                all_insights.append(insight)

        # 构建汇总的情报摘要
        aggregated_status = 'GOAL_ACHIEVED' if goal_achieved else 'AGGREGATED'
        intelligence_summary = {
            'findings': all_findings,
            'audit_result': {
                'status': aggregated_status,
                'completion_check': aggregated_completion_check
            },
            'artifacts': all_artifacts,
            'insight': {
                'type': 'aggregated',
                'insights': all_insights
            }
        }

        return intelligence_summary
