# Divine - 多智能体自动化渗透测试框架

> 用于本地授权安全测试学习的多智能体自动化渗透测试框架

## 项目概述

- **语言**: Python 3.11+
- **用途**: 本地授权渗透测试 + 安全学习
- **LLM 后端**: 可插拔多模型（OpenAI / Anthropic / DeepSeek / DashScope / Zhipu / OpenAI 兼容兜底）
- **入口**: `python -m divine` 或 `divine engage --config targets.yaml`

## 三大核心设计

1. **动态任务图（Dynamic DAG）** — 任务以依赖感知的节点 + 多类型边（dependency/hypothesis/validation/alternative）表达，由规划器按策略演化（扩展/重生/分支重规划/剪枝）
2. **共享黑板（SharedBlackboard）** — 内存中统一管理任务图、情报库、产物、执行结果、审计反馈与事件日志，全程留痕、统一 ID 生成
3. **结构化执行（Structured Tool-Calling）** — 执行器通过受约束的 JSON 工具协议采集证据，每条结论绑定 `evidence_refs` 产物引用（区别于自由生成代码）

## 架构图

```
              ┌──────────────────┐
              │   CLI (Typer)    │  divine engage / python -m divine
              └────────┬─────────┘
                       │  DivineConfig.from_yaml → TaskContext
              ┌────────▼─────────┐
              │   Orchestrator   │  同步编排主循环（唯一控制流）
              └────────┬─────────┘
        ┌──────────────┼───────────────────────────┐
        │ 1.初始/演化   │ 2.选点+路由   3.执行       4.审计
        ▼              ▼              ▼             ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐  ┌───────────┐
   │ Planner │   │  Router  │   │ Executor │  │ Evaluator │
   │  Agent  │   │(能力路由) │   │Recon/Web │  │ (语义审计) │
   └────┬────┘   └────┬─────┘   │  /Host   │  └─────┬─────┘
        │             │         └────┬─────┘        │
        │             │              │ ToolAdapter  │
        │             │              ▼ (结构化工具)  │
        │      ┌──────▼──────────────▼──────────────▼──────┐
        └─────►│            SharedBlackboard               │
               │  DynamicTaskGraph · IntelligenceStore     │
               │  artifacts · execution_results            │
               │  audit_feedback · event_log               │
               └───────────────────┬───────────────────────┘
                                   │ AuditFeedback
                                   └──► Planner.evolve_dag()  (闭环)

   LLMClient(多 provider) / PromptRenderer(Jinja2) / Logger+LLMTrace 横向服务全部 Agent
```

## 主循环流程

```python
# Orchestrator.run(context) —— 同步循环
planner.generate_initial_dag(context, blackboard)        # 初始任务图

for iteration in range(1, context.max_iterations + 1):
    node = select_executable_node(blackboard)            # 依赖就绪节点（无则 no_executable_nodes 终止）
    route = router.route(node)                            # 阻断 → route_blocked
    result = executors[route.selected_agent].execute(node, blackboard)
    feedback = evaluator.audit(node=node, execution_result=result, blackboard=blackboard)
    plan = planner.evolve_dag(feedback, blackboard)       # expand / regenerate_node / replan_branch
    if plan.should_terminate:                             # → planner_terminated
        break
    # 失败计数达阈值 → max_consecutive_failures
# 跑满 → max_iterations
```

**终止原因（StopReason）**：`planner_terminated` / `no_executable_nodes` / `route_blocked` / `executor_missing` / `executor_failed` / `evaluator_failed` / `planner_failed` / `max_consecutive_failures` / `max_iterations`。

## 核心组件职责

| 组件 | 职责 | 关键约束 |
|------|------|----------|
| **Orchestrator** | 选点→路由→执行→审计→演化→终止 | 唯一控制流拥有者，只编排不推理；同步执行 |
| **PlannerAgent** | `generate_initial_dag` / `evolve_dag` | 仅输出受白名单约束的操作（create_node/update_node_status），单决策最多 3 个新节点；按失败层级选策略 |
| **ExecutionRouter** | 依 `task_type`/`assigned_executor` 选执行器 | `CapabilityRegistry` 注册能力；不支持/不匹配则阻断回交规划器 |
| **ExecutorAgent**(Recon/Web/Host) | LLM 动作循环：`tool_call`↔`final_result` | 通过 `ToolAdapter` 采证，产出 `ExecutionResult` + 候选事实 + 证据引用 |
| **EvaluatorAgent** | LLM 语义审计 → `AuditFeedback` | 校验 evidence_refs 有效性；输出任务判定/确认事实/失败归因/规划建议 |
| **SharedBlackboard** | 任务图/情报/产物/结果/反馈/事件集中管理 | `next_id` 统一发号；`record_event` 留痕；只增不隐式删 |
| **LLMClient** | provider 工厂路由 + token 计量 + 调用追踪 | 配置驱动；`create_llm_client` 由 `config/llm.json` 装配 |
| **PromptRenderer** | Jinja2 模板渲染 + 变量填充追踪 | 模板按 agent 角色组织；`RenderedPrompt.as_trace()` 可观测 |
| **ContextBuilder** | 缓存感知上下文装配/token 预算/稳定前缀哈希 | 面向 prompt 缓存与成本（独立工具，非主链路强依赖） |
| **ToolAdapter** | 纯标准库探测工具集 | 执行器与环境交互的唯一受约束通道 |
| **Logger/LLMTrace** | 结构化日志 + 脱敏 + 全量 LLM 追踪 | 由 `config/logging.json` 控制 |

## 目录结构

```
divine/
├── __init__.py / __main__.py / cli.py / config.py   # 入口与配置
├── orchestrator/
│   ├── __init__.py
│   └── core.py            # Orchestrator + RunResult + StopReason
├── blackboard/
│   ├── models.py          # TaskContext/TaskNode/TaskEdge/Fact/Artifact/ExecutionResult/AuditFeedback/PlannerResult ...
│   ├── store.py           # SharedBlackboard / DynamicTaskGraph / IntelligenceStore
│   └── __init__.py
├── agents/
│   ├── planner.py         # PlannerAgent（初始/演化 DAG，策略与操作校验）
│   ├── executor.py        # ExecutorAgent + ReconAgent/WebAgent/HostAgent + ExecutionRouter/CapabilityRegistry
│   ├── evaluator.py       # EvaluatorAgent（语义审计）
│   └── __init__.py
├── llm/
│   ├── client.py          # LLMClient + PROVIDER_FACTORIES + create_llm_client
│   ├── config.py          # LLMSettings.from_file(config/llm.json)
│   ├── catalog.py          # 模型目录 + DEFAULT_MODELS
│   ├── types.py           # Message/LLMRequest/LLMResponse/TokenUsage
│   ├── errors.py
│   └── providers/         # anthropic / openai(+compat) / dashscope / zhipu / base
├── prompts/
│   ├── renderer.py        # PromptRenderer / RenderedPrompt（模板目录用 __file__）
│   └── templates/         # shared/ missions/ runtime/ agents/{planner,executor,evaluator,router}
├── context/               # builder / cache_policy / conversation / segments / token_budget
├── tools/
│   └── adapter.py         # ToolAdapter / ToolResult（tcp_connect_check/http_probe/https_probe/path_probe/host_info）
└── logger/                # config / redaction / trace
config/                    # llm.example.json / logging.example.json（真实 *.json 被 gitignore）
tests/                     # 64 用例
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 编排模式 | Orchestrator 中心化、同步 | 流程透明、可调试、易复盘 |
| 任务管理 | 自实现 DynamicTaskGraph | 贴合演化策略（扩展/重生/重规划/剪枝），无需重型图库 |
| 状态管理 | SharedBlackboard（内存，集中） | 读写快、全程留痕、统一发号 |
| 执行范式 | 结构化工具协议（JSON tool-calling） | 证据可绑定、可审计，安全可控 |
| 闭环终止 | 规划器判定 + 多重兜底（迭代/连续失败/无可执行节点） | 多保险防失控 |
| LLM 适配 | provider 工厂 + OpenAI 兼容兜底 | 配置即切换，完全可控 |
| Prompt 管理 | Jinja2 模板 + 渲染追踪 | 模板分离、可观测 |
| 可观测性 | Loguru 结构化日志 + 全量 LLM 追踪落盘 | 成本核算与回放 |
| 安全隔离 | 工具受约束 + 不做强隔离，靠部署兜底 | 简化开发，靠授权范围与 Docker 约束 |

## 开发命令

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest -q                      # 64 passed

# 配置（首次）
cp config/llm.example.json config/llm.json          # 填入 API Key
cp config/logging.example.json config/logging.json

# 运行 CLI
python -m divine version
divine engage --config targets.yaml
```

## 约定与注意

- **改动须保持 `pytest -q` 全绿**；新增 provider/executor/模板均有对应测试守护（模板覆盖率测试会校验所有 `.j2` 被使用）。
- **真实凭据/目标不入库**：`config/llm.json`、`config/logging.json`、`targets.yaml`、`logs/`、`artifacts/` 已被 `.gitignore` 忽略。
- **智能体输出均为结构化 JSON**：解析失败/越权操作会抛错，不要放宽校验白名单（`ALLOWED_PLANNER_*`、工具目录、状态枚举）。
- **仅用于授权测试与教学**，严禁对未授权目标运行。
