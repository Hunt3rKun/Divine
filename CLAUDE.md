# Divine - 多智能体自动化渗透测试工具

> 用于本地授权安全测试学习的多智能体自动化渗透测试框架

## 项目概述

- **语言**: Python 3.11+
- **用途**: 本地授权渗透测试 + 安全学习
- **LLM 后端**: 可插拔多模型（Anthropic / OpenAI / 智谱 / MiniMax / OpenAI 兼容格式兜底）
- **入口**: `python -m divine` 或 `divine engage --config targets.yaml`

## 三大核心设计

1. **DAG 任务编排** — NetworkX 有向无环图管理任务依赖，支持并行执行和动态插入
2. **Blackboard 黑板架构** — 内存字典 + SQLite 写透，统一共享状态管理
3. **CodeAct 执行范式** — Agent 生成 Python 代码与环境交互，比预定义工具封装更灵活

## 架构图

```
              ┌──────────────────┐
              │   CLI (Typer)    │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     Session      │  中心化编排主循环
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     Planner      │  LLM 攻击规划，输出 graph operations
              └────────┬─────────┘
                       │ apply_operations
              ┌────────▼─────────┐
              │    TaskDAG       │  NetworkX DiGraph，节点=TaskNode
              └────────┬─────────┘
                       │ get_ready_tasks
              ┌────────▼─────────┐
              │  DAGScheduler    │  asyncio 并行，Semaphore 限流
              └──┬───┬───┬───┬───┘
     ┌───────────┘   │   │   └───────────┐
     ▼               ▼   ▼               ▼
  Recon          Web       Host       Service
  CodeAct        CodeAct   CodeAct    CodeAct
     │              │        │            │
     └──────────────┴────┬───┴────────────┘
                         │
                ┌────────▼────────┐
                │ CodeAct Sandbox │  子进程执行 + 超时保护
                └────────┬────────┘
                         │ stdlib: bb_write()
                ┌────────▼────────┐
                │   Blackboard    │  sections: hosts, ports, findings,
                │ (内存+SQLite)   │  credentials, tasks, reflections
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │   Reflector     │  分析执行结果，输出 Reflection
                └────────┬────────┘
                         │ 写入 blackboard["reflections"]
                         └──► Planner.replan() → 修改 DAG (闭环)
```

## 主循环流程

```python
# Session.run()
operations = await planner.init_plan(goal, config)
dag.apply_operations(operations)

for round_num in range(max_rounds):      # 硬性兜底
    completed = await scheduler.run_round(execute_fn)
    if not completed and dag.is_finished:
        break
    reflection = await reflector.analyze(bb.summary(), results, dag.stats)
    bb.write("reflections", round_num, reflection)
    operations = await planner.replan(bb.summary(), dag.stats, reflections)
    dag.apply_operations(operations)
    if await planner.should_terminate(bb.summary(), dag.stats):
        break
```

## 核心组件职责

| 组件 | 职责 | 关键约束 |
|------|------|----------|
| **Session** | 组装所有组件、驱动主循环 | 唯一的组装点和控制流拥有者 |
| **Planner** | init_plan / replan / should_terminate | 只输出 operations，不直接操作 DAG |
| **TaskDAG** | NetworkX DAG 管理，环检测，失败传播 | asyncio.Lock 保护写操作 |
| **DAGScheduler** | Semaphore 限流并行调度 | 通过 execute_fn 回调解耦 |
| **Blackboard** | 内存读写 + SQLite 写透 + asyncio.Event 订阅 | 预定义 SECTIONS |
| **CodeActExecutor** | 驱动 生成代码→执行→观察 循环 | max_iterations 上限 |
| **Sandbox** | 执行 LLM 生成的 Python 代码 | 持久命名空间，任务间 reset |
| **stdlib** | Sandbox 标准库（run_command, bb_read/write 等） | LLM 代码与系统交互的唯一通道 |
| **Reflector** | 分析执行结果，输出 Reflection dataclass | 只建议不修改 DAG |
| **PromptEngine** | Jinja2 模板 + 每个 Agent 专用 build 方法 | 模板按 agent 角色组织 |
| **LLMRouter** | 模型名前缀路由到 provider | openai_compat 兜底 |
| **ReportGenerator** | 从黑板汇总 + LLM 生成叙述 + Jinja2 渲染 | 数据全部来自 Blackboard |

## 目录结构

```
divine/
├── __init__.py / __main__.py / cli.py / config.py / session.py
├── models/
│   ├── common.py          # AgentRole, PentestPhase, ExecutorType
│   ├── task.py            # TaskNode, TaskStatus
│   └── finding.py         # Finding, Severity, FindingType
├── blackboard/
│   ├── models.py          # BlackboardEntry, SECTIONS
│   └── blackboard.py      # Blackboard 核心
├── dag/
│   ├── task_dag.py        # TaskDAG (NetworkX DiGraph)
│   └── scheduler.py       # DAGScheduler
├── llm/
│   ├── base.py            # LLMProvider ABC, LLMMessage, LLMResponse, TokenUsage
│   ├── router.py          # LLMRouter + ROUTE_MAP
│   ├── providers/
│   │   ├── openai.py / anthropic.py / zhipu.py / minimax.py / openai_compat.py
│   └── utils/
│       ├── retry.py       # RetryHandler (指数退避+抖动)
│       └── cost.py        # CostCalculator
├── codeact/
│   ├── sandbox.py         # Sandbox, ExecutionResult
│   ├── executor.py        # CodeActExecutor
│   └── stdlib.py          # create_stdlib()
├── agents/
│   ├── planner.py         # Planner
│   └── reflection.py      # Reflector, Reflection
├── prompts/
│   ├── engine.py          # PromptEngine (专用 build 方法)
│   └── templates/         # planner/ executor/ reflector/ 模板
└── reporting/
    ├── generator.py       # ReportGenerator
    └── templates/         # report.jinja2
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 编排模式 | Session 中心化 | 流程透明、可调试 |
| 闭环终止 | LLM 判定 + max_rounds 兜底 | 双保险 |
| 任务管理 | NetworkX DAG | 环检测、拓扑排序开箱即用 |
| 状态管理 | Blackboard (内存+SQLite) | 读快、持久可靠 |
| 执行范式 | CodeAct | 生成代码比预定义工具灵活 |
| LLM 适配 | 各 provider 原生 SDK + OpenAI 兼容兜底 | 完全控制 |
| Prompt 管理 | Jinja2 + 专用 build 方法 | 模板分离、接口清晰 |
| CLI | Typer | 现代、类型注解驱动 |
| 并发 | asyncio + Semaphore | 简单可靠 |
| 安全隔离 | 不做严格隔离，靠 Docker 部署兜底 | 简化开发 |

## 开发命令

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest tests/ -v

# 运行 CLI
python -m divine version
python -m divine engage --config targets.yaml
```
