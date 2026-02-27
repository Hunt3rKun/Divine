import pytest
from agent.agent import AgentConfig, AgentRole
from agent.graph_manager import GraphManager
from agent.planner import Planner
from agent.runner import AgentRunner
from llm_client import LLMClient

llm_client = LLMClient.from_env()
agent_config = AgentConfig(
        name="Planner",
        role=AgentRole.PLANNER,
        model="glm-4.7",
        temperature=0.7,
        max_iterations=3,
        verbose=True,
    )


planner = Planner(
        config=agent_config,
        llm_client=llm_client,
        prompt_manager=None,
    )

@pytest.mark.asyncio
async def test_init_plan():

    res = await planner.init_plan("对这个CTF的Web题目http://172.17.84.187:80进行自由探索，题目名称：Trading Platform - Authorization，直到获取flag。题目描述：Every time you see a value that looks like an ID, you should try to fuzz it with similar values!")
    print("\n" + "="*50)
    print("init_plan 返回结果:")
    print("="*50)
    print(res)
    print("="*50 + "\n")
    assert res is not None



@pytest.mark.asyncio
async def test_dynamic_plan():
    rel = { "web_root_aaaa":
        {
  "audit_result": {
    "status": "completed",
    "completion_check": "子任务目标已完全满足。执行日志显示成功识别了4个功能端点（/, /password/{username}, /orders, /order/{order_id}/receipt），并收集了ID参数：user_id (值: 10032, 位置: Body/隐藏字段) 和 order_id (值: 300123等, 位置: URL/Body)。",
    "is_strategic_failure": False,
    "logic_issues": [],
    "methodology_issues": []
  },
  "attack_intelligence": {
    "confirmed_vulnerabilities": [],
    "protection_mechanisms": [
      "Session 使用 JWT (JSON Web Token) 进行管理",
      "登录流程采用两步验证机制（先用户名后密码）"
    ],
    "actionable_insights": [
      "测试账号信息泄露：HTML注释中包含 test:test，可直接用于登录。",
      "隐藏字段泄露敏感信息：登录表单中包含隐藏的 user_id=10032。",
      "潜在 IDOR 漏洞：端点 /order/{order_id}/receipt 接受任意 order_id 参数，且测试不存在的ID (999999) 也返回200状态码，建议进一步验证响应内容是否存在越权访问。"
    ]
  },
  "key_findings": [
    {
      "type": "credential",
      "severity": "high",
      "title": "测试账号信息泄露",
      "description": "在首页 HTML 注释中发现测试账号 test:test。",
      "evidence": "<!-- TODO: Delete the testing account (test:test). -->"
    },
    {
      "type": "system_property",
      "severity": "medium",
      "title": "用户ID泄露",
      "description": "登录流程中通过隐藏字段泄露了 user_id=10032。",
      "evidence": "<input type=\"text\" id=\"user_id\" name=\"user_id\" value=\"10032\" hidden>"
    },
    {
      "type": "vulnerability",
      "severity": "high",
      "title": "潜在的IDOR漏洞",
      "description": "订单收据端点 /order/{order_id}/receipt 可能存在不安全的直接对象引用，允许访问任意订单数据。",
      "evidence": "测试 order_id=999999 返回 HTTP 200，且端点模式为 /order/{order_id}/receipt。"
    }
  ],
  "key_facts": [
    "目标系统为 Trading Platform，采用两步登录验证。",
    "测试账号为 test:test，登录后获得 JWT session。",
    "已确认 user_id=10032，order_id 范围包含 300123, 300214, 300327, 300481。",
    "已确认功能端点：/, /password/{username}, /orders, /order/{order_id}/receipt。"
  ],
  "exploitation_priority": {
    "should_exploit_now": False,
    "priority_target": None,
    "reason": "尚未确认可立即利用的高危漏洞（如RCE或SQL注入）。虽然发现潜在的IDOR漏洞，但需要先验证其有效性（即确认能否访问其他用户数据），因此暂不进行大规模利用。"
  },
  "strategic_correction": {
    "is_required": False,
    "current_issue": None,
    "counterfactual": None,
    "recommended_pivot": None,
    "abandon_tasks": []
  },
  "goal_assessment": {
    "is_main_goal_achieved": False,
    "confidence": 0.0,
    "achievement_evidence": None,
    "remaining_gap": "尚未获取最终目标产物（如flag或管理员权限），也未成功利用漏洞获取敏感数据。",
    "needs_more_tasks": True,
    "suggested_next_actions": [
      "验证 /order/{order_id}/receipt 端点的 IDOR 漏洞，尝试枚举 order_id 获取其他用户数据。",
      "分析 JWT session 的签名机制，尝试伪造 token 提权。",
      "探索其他未发现的端点（如 /admin, /api 等）。"
    ]
  }
}}
    runner = AgentRunner()
    intelligence_summary = runner._aggregate_intelligence(completed_reflections=rel)

    graph_manager = GraphManager("web_root_aaaa","Initial goal")

    res = await planner.dynamic_plan(
        intelligence_summary=intelligence_summary,
        graph_manager=graph_manager,

    )
    print("\n" + "="*50)
    print("dynamic_plan 返回结果:")
    print("="*50)
    print(res)
    print("="*50 + "\n")
    assert res is not None