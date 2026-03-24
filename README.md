# Divine

基于大语言模型的自动化任务规划与执行框架。

## 核心特性

- **多模型支持**: 支持 OpenAI GPT、Anthropic Claude、智谱 GLM 等多种 LLM 提供商
- **智能规划**: 支持初始规划、动态规划、分支重规划三种模式
- **图结构管理**: 基于 NetworkX 的有向图管理任务依赖关系
- **成本追踪**: 内置 Token 用量统计与成本计算
- **流式输出**: 支持 LLM 流式响应
- **重试机制**: 内置自动重试与错误恢复

## 项目架构

```
Divine/
├── agent/               # Agent 核心模块
│   ├── agent.py         # Agent 基类
│   ├── planner.py       # 规划器 (Planner)
│   ├── graph_manager.py # 任务图管理器
│   ├── runner.py        # 执行运行器
│   └── communication/   # 通信模块
├── llm_client/          # LLM 客户端
│   ├── client.py        # 统一客户端
│   ├── adapters/       # 模型适配器 (OpenAI/Claude/Zhipu)
│   ├── models.py        # 数据模型
│   ├── retry.py         # 重试机制
│   └── cost.py          # 成本计算
├── prompts/             # 提示词管理
│   └── templates/       # Jinja2 模板
├── config/              # 配置管理
└── main.py              # 入口文件
```

## 核心概念

### Agent 角色

| 角色 | 职责 |
|------|------|
| **Planner** | 任务分解与策略制定 |
| **Executor** | 执行具体任务 |
| **Reflector** | 结果分析与经验总结 |

### 规划模式

- **INITIAL**: 初始规划，根据目标生成任务分解图
- **DYNAMIC**: 动态规划，根据执行状态动态调整计划
- **BRANCH_REPLAN**: 分支重规划，当任务失败时重新规划

### 任务图

使用有向图管理任务依赖关系，支持：
- 节点状态追踪 (pending/in_progress/completed/failed)
- 依赖关系管理
- 失败路径回溯

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件：

```env
# OpenAI
OPENAI_API_KEY=sk-xxx

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# 智谱 GLM
ZHIPU_API_KEY=xxx
```

### 基本使用

```python
import asyncio
from llm_client import LLMClient, Message

client = LLMClient.from_env()

messages = [
    Message(role="system", content="你是一个智能助手。"),
    Message(role="user", content="你好")
]

async def main():
    response = await client.chat(
        messages=messages,
        model="glm-4.7",
        provider="zhipu"
    )
    print(response.content)

asyncio.run(main())
```

## 支持的模型

| 提供商 | 模型 |
|--------|------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | claude-sonnet-4, claaude-3-5-sonnet, claude-3-5-haiku, claude-3-opus |
| 智谱 | glm-4.7, glm-4.6, glm-4.5 |

## License

MIT
