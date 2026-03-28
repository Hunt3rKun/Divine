# Divine 重构设计文档

> 多智能体自动化渗透测试框架 — 全量重写设计

## 1. 背景与目标

### 1.1 为什么重构

当前代码存在以下问题：
- 目录结构平铺混乱，无统一包结构
- `executor_types.py` 760+ 行过度设计（协作协议、微任务状态机等），但 executor 本身未实现
- Blackboard、Reflection、CodeAct 等核心模块缺失
- 无 CLI 入口，无 Session 编排

### 1.2 重构策略

**一次性重写**：创建全新的 `divine/` 包结构，从零开始按新架构实现。旧代码仅作参考，不做渐进迁移。

### 1.3 保留的设计思路

- 国产模型支持（智谱 GLM）
- Jinja2 模板引擎用于 prompt 管理
- NetworkX 用于 DAG 管理
- Agent 基础架构（Planner 的规划模式）

---

## 2. 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 编排模式 | Session 中心化编排 | 流程透明、可调试，适合分阶段推进的渗透测试 |
| 闭环终止 | Planner LLM 判定 + 硬性上限（max_rounds/max_tasks/timeout） | 双保险 |
| Blackboard 订阅 | asyncio.Event/Condition | 符合 asyncio 风格，流控简单 |
| LLM 适配 | 全部重写，各 provider 用原生 SDK + OpenAI 兼容格式兜底 | 完全控制，避免 SDK 间差异 |
| DAG 实现 | NetworkX DiGraph | 环检测、拓扑排序等开箱即用 |
| CodeAct 隔离 | 不做严格隔离，子进程 + 超时保护 | 整个系统将部署在 Docker 中 |
| Prompt 管理 | Jinja2 + 每个 Agent 专用 build 方法 | 模板分离 + 接口清晰 |
| CLI | Typer | 现代、简洁、类型注解驱动 |
| 数据模型 | 精简，砍掉所有过度设计 | YAGNI，需要时再加 |

---

## 3. 目录结构

```
divine/
├── __init__.py
├── __main__.py                # python -m divine 入口
├── cli.py                     # Typer CLI
├── config.py                  # DivineConfig dataclass + YAML 加载
├── session.py                 # 中心化编排主循环
├── models/
│   ├── __init__.py
│   ├── common.py              # AgentRole, PentestPhase, ExecutorType
│   ├── task.py                # TaskNode, TaskStatus
│   └── finding.py             # Finding, Severity, FindingType
├── blackboard/
│   ├── __init__.py
│   ├── models.py              # BlackboardEntry, SECTIONS
│   └── blackboard.py          # 读写、订阅、审计、摘要
├── dag/
│   ├── __init__.py
│   ├── task_dag.py            # NetworkX DAG 管理
│   └── scheduler.py           # 并行调度器
├── llm/
│   ├── __init__.py
│   ├── base.py                # LLMProvider, LLMResponse, LLMMessage, TokenUsage
│   ├── router.py              # 模型名路由
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── openai.py          # OpenAI 原生 SDK
│   │   ├── anthropic.py       # Anthropic 原生 SDK
│   │   ├── zhipu.py           # 智谱原生 SDK
│   │   ├── minimax.py         # MiniMax 原生 SDK
│   │   └── openai_compat.py   # OpenAI 兼容格式兜底
│   └── utils/
│       ├── __init__.py
│       ├── retry.py           # 指数退避重试
│       └── cost.py            # 成本计算
├── codeact/
│   ├── __init__.py
│   ├── sandbox.py             # 代码执行环境
│   ├── executor.py            # 生成->执行->观察循环
│   └── stdlib.py              # sandbox 标准库函数
├── agents/
│   ├── __init__.py
│   ├── planner.py             # 攻击规划
│   └── reflection.py          # 反思分析
├── prompts/
│   ├── __init__.py
│   ├── engine.py              # PromptEngine + 各 Agent 专用 build 方法
│   └── templates/
│       ├── planner/
│       │   ├── init_plan.jinja2
│       │   ├── replan.jinja2
│       │   └── terminate_check.jinja2
│       ├── executor/
│       │   ├── base_system.jinja2
│       │   ├── recon_system.jinja2
│       │   ├── web_system.jinja2
│       │   ├── host_system.jinja2
│       │   ├── service_system.jinja2
│       │   └── observation.jinja2
│       └── reflector/
│           └── analyze.jinja2
└── reporting/
    ├── __init__.py
    ├── generator.py           # 报告生成
    └── templates/
        └── report.jinja2
```

---

## 4. 组件详细设计

### 4.1 配置与入口

**`config.py`** — `DivineConfig` dataclass：

```python
@dataclass
class DivineConfig:
    targets: list[str]           # 目标列表
    goal: str                    # 最终目标描述
    llm: LLMConfig               # 模型配置
    max_rounds: int = 20         # 闭环最大轮次
    max_tasks: int = 50          # DAG 最大任务数
    timeout: int = 3600          # 全局超时（秒）
    concurrency: int = 3         # Scheduler 并发数
    code_execution_timeout: int = 60  # 单次代码执行超时
    planner_model: str = "claude-sonnet-4-20250514"
    reflector_model: str = "claude-sonnet-4-20250514"
    executor_model: str = "claude-sonnet-4-20250514"
    db_path: str = ":memory:"   # Blackboard SQLite 路径
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "DivineConfig": ...
```

**`cli.py`** — Typer 入口：

```python
app = typer.Typer()

@app.command()
def engage(config: Path = typer.Option(..., help="目标配置文件")):
    """启动渗透测试"""
    cfg = DivineConfig.from_yaml(config)
    asyncio.run(Session(cfg).run())
```

**`__main__.py`**：

```python
from divine.cli import app
app()
```

### 4.2 数据模型（`models/`）

**`common.py`**：

```python
class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REFLECTOR = "reflector"

class PentestPhase(str, Enum):
    RECON = "recon"
    SCAN = "scan"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post_exploit"

class ExecutorType(str, Enum):
    RECON = "recon"
    WEB = "web"
    HOST = "host"
    SERVICE = "service"
```

**`task.py`**：

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"          # 依赖失败时跳过

@dataclass
class TaskNode:
    id: str
    description: str
    phase: PentestPhase
    executor_type: ExecutorType
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
```

**`finding.py`**：

```python
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class FindingType(str, Enum):
    ASSET = "asset"
    CREDENTIAL = "credential"
    VULNERABILITY = "vulnerability"
    KNOWLEDGE = "knowledge"

@dataclass
class Finding:
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    type: FindingType = FindingType.KNOWLEDGE
    severity: Severity = Severity.INFO
    title: str = ""
    detail: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    source_task: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
```

### 4.3 Blackboard（`blackboard/`）

**`models.py`**：

```python
SECTIONS = ["hosts", "ports", "findings", "credentials", "tasks", "reflections"]

@dataclass
class BlackboardEntry:
    section: str
    key: str
    value: Any
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
```

**`blackboard.py`**：

```python
class Blackboard:
    def __init__(self, db_path: str = ":memory:"):
        self._memory: dict[str, dict[str, BlackboardEntry]] = {s: {} for s in SECTIONS}
        self._db = sqlite3.connect(db_path)
        self._events: dict[str, asyncio.Event] = {s: asyncio.Event() for s in SECTIONS}
        self._init_db()

    def read(self, section: str, key: str = None) -> Any:
        """读取 section 下某个 key，或整个 section"""

    def write(self, section: str, key: str, value: Any, source: str = "") -> None:
        """写入内存 + SQLite 写透 + set event"""

    def query(self, section: str, filter_fn: Callable = None) -> list[BlackboardEntry]:
        """条件查询某个 section"""

    async def wait_for(self, section: str) -> None:
        """await 等待某个 section 有新写入"""

    def clear_event(self, section: str) -> None:
        """消费后重置 event"""

    def summary(self, sections: list[str] = None) -> dict:
        """生成指定 sections 的摘要，用于构建 LLM prompt 上下文"""

    def audit_log(self, limit: int = 100) -> list[dict]:
        """从 SQLite 读取最近的写入记录"""
```

设计要点：
- 内存字典 + SQLite 写透：读取走内存，写入同步落盘
- 预定义 SECTIONS，write 时校验合法性
- 每个 section 一个 asyncio.Event，write 时 set，消费者 await wait_for
- `summary()` 为 Planner/Reflector 提供压缩后的上下文摘要
- `version` 字段预留乐观锁，当前不强制校验

### 4.4 DAG 任务编排（`dag/`）

**`task_dag.py`**：

```python
class TaskDAG:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()

    async def add_task(self, task: TaskNode) -> None:
        """添加任务节点 + 依赖边 + 环检测，不合法则回滚"""

    async def remove_task(self, task_id: str) -> None:
        """移除节点及所有边"""

    async def update_status(self, task_id: str, status: TaskStatus,
                            result: dict = None, error: str = None) -> None:
        """更新任务状态"""

    def get_ready_tasks(self) -> list[TaskNode]:
        """依赖已完成 + 自身 PENDING 的任务，按 priority 排序"""

    def get_task(self, task_id: str) -> TaskNode: ...
    def get_all_tasks(self) -> list[TaskNode]: ...
    def get_failed_tasks(self) -> list[TaskNode]: ...
    def get_descendants(self, task_id: str) -> set[str]: ...

    async def propagate_failure(self, task_id: str) -> list[str]:
        """失败任务的所有后代标记为 SKIPPED"""

    async def apply_operations(self, operations: list[dict]) -> None:
        """批量应用 Planner 输出的 graph operations"""

    @property
    def is_finished(self) -> bool:
        """所有任务处于终态"""

    @property
    def stats(self) -> dict:
        """返回 {total, pending, running, completed, failed, skipped}"""
```

**`scheduler.py`**：

```python
class DAGScheduler:
    def __init__(self, dag: TaskDAG, concurrency: int = 3):
        self._dag = dag
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run_round(self, execute_fn: Callable[[TaskNode], Awaitable]) -> list[str]:
        """
        调度一轮：
        1. 获取就绪任务
        2. Semaphore 限流并行执行
        3. 更新 DAG 状态
        4. 失败传播
        5. 返回本轮完成的任务 ID
        """
```

设计要点：
- asyncio.Lock 保护 DAG 写操作
- 添加节点时立即环检测，不合法则回滚
- 失败传播：后代自动 SKIPPED
- `apply_operations` 是 Planner -> DAG 的唯一桥梁
- Scheduler 通过 `execute_fn` 回调解耦，不持有 executor 引用
- `run_round` 跑一轮就返回，对齐 Session 中心化编排

### 4.5 LLM 适配层（`llm/`）

**`base.py`**：

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    cost: float = 0.0
    raw_response: Any = None

@dataclass
class LLMMessage:
    role: str          # system / user / assistant
    content: str

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse: ...

    @abstractmethod
    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]: ...
```

**`router.py`**：

```python
ROUTE_MAP = {
    "gpt-": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "glm-": "zhipu",
    "minimax": "minimax",
}

class LLMRouter:
    def __init__(self, config: LLMConfig):
        self._providers: dict[str, LLMProvider] = {}
        self._fallback: Optional[LLMProvider] = None   # openai_compat

    def get_provider(self, model: str) -> LLMProvider:
        """模型名前缀匹配 -> provider，找不到走兜底"""

    async def chat(self, model: str, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """路由 + 调用 + 重试 + 成本计算"""
```

**Providers**：

| Provider | SDK | 说明 |
|----------|-----|------|
| `openai.py` | `openai` | 适配当前主流模型 |
| `anthropic.py` | `anthropic` | system prompt 独立参数，extended thinking |
| `zhipu.py` | `zhipuai` | 智谱 GLM 系列 |
| `minimax.py` | MiniMax SDK | MiniMax 适配 |
| `openai_compat.py` | `openai` + 自定义 `base_url` | 兜底任何 OpenAI 兼容 API |

**`utils/retry.py`**：指数退避 + 抖动，区分可重试/不可重试错误。

**`utils/cost.py`**：按模型名查定价表计算费用，支持配置覆盖。

设计要点：
- 每个 provider 用原生 SDK
- openai_compat 兜底未来扩展（Ollama 等）
- Router 是上层唯一出口
- LLMMessage/LLMResponse 是自有模型，不暴露 SDK 内部类型

### 4.6 CodeAct 执行引擎（`codeact/`）

**`sandbox.py`**：

```python
@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time: float = 0.0

class Sandbox:
    def __init__(self, timeout: int = 60):
        self._timeout = timeout
        self._globals: dict = {}    # 持久化命名空间，跨迭代保留

    def setup(self, stdlib: dict) -> None:
        """注入标准库函数"""

    async def execute(self, code: str) -> ExecutionResult:
        """子进程执行，超时保护"""

    def reset(self) -> None:
        """重置命名空间，任务间隔离"""
```

**`stdlib.py`**：

```python
def create_stdlib(blackboard: Blackboard) -> dict:
    return {
        "run_command": run_command,       # shell 命令执行
        "http_request": http_request,     # HTTP 请求
        "bb_read": blackboard.read,       # 黑板读取
        "bb_write": blackboard.write,     # 黑板写入
        "parse_nmap": parse_nmap,         # nmap 输出解析
        "parse_url": parse_url,           # URL 解析
        "b64encode": b64encode,
        "b64decode": b64decode,
    }
```

**`executor.py`**：

```python
class CodeActExecutor:
    def __init__(self, router: LLMRouter, sandbox: Sandbox,
                 blackboard: Blackboard, prompt_engine: PromptEngine):
        self._max_iterations = 10

    async def execute_task(self, task: TaskNode, context: dict) -> dict:
        """
        CodeAct 循环：
        1. 构建 conversation = [system_prompt, task_description + context]
        2. 循环（最多 max_iterations 次）：
           a. LLM 生成回复
           b. 提取代码块
           c. 有代码 -> sandbox 执行 -> 结果作为 observation 追加到 conversation
           d. 无代码 -> 视为任务完成
           e. 检查完成标记
        3. reset sandbox
        4. 返回提取的 findings
        """
```

设计要点：
- Sandbox 持久命名空间，同任务多次迭代共享变量，任务间 reset
- stdlib 是 LLM 生成代码与系统交互的唯一通道
- 不做严格隔离，安全靠 Docker 部署环境兜底
- LLM 返回无代码块视为完成信号

### 4.7 Agents（`agents/`）

**`planner.py`**：

```python
class Planner:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine, model: str): ...

    async def init_plan(self, goal: str, config: DivineConfig) -> list[dict]:
        """初始规划：目标 -> graph operations 列表"""

    async def replan(self, blackboard_summary: dict, dag_stats: dict,
                     reflections: list[dict]) -> list[dict]:
        """动态重规划：输出 DAG 调整操作，空列表表示无需调整"""

    async def should_terminate(self, blackboard_summary: dict,
                               dag_stats: dict) -> tuple[bool, str]:
        """LLM 判断目标是否达成"""

    def _parse_operations(self, content: str) -> list[dict]:
        """从 LLM JSON 响应解析 graph operations"""
```

**`reflection.py`**：

```python
@dataclass
class Reflection:
    insights: list[str]
    suggested_tasks: list[dict]
    risk_assessment: str
    progress_summary: str

class Reflector:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine, model: str): ...

    async def analyze(self, blackboard_summary: dict,
                      recent_results: list[dict],
                      dag_stats: dict) -> Reflection:
        """
        分析最近一轮执行结果：
        - 识别失败模式
        - 发现高价值线索
        - 评估整体进度
        - 建议新任务方向（不直接修改 DAG）
        """
```

设计要点：
- Planner 三个方法职责分离：init_plan / replan / should_terminate
- Planner 只输出 operations 列表，不直接操作 DAG
- Reflector 只输出建议，Planner 在 replan 时决定是否采纳
- 不同 Agent 可配置不同模型

### 4.8 Prompt 模板引擎（`prompts/`）

**`engine.py`**：

```python
class PromptEngine:
    def __init__(self, template_dir: Path = None):
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir or DEFAULT_TEMPLATE_DIR),
            trim_blocks=True, lstrip_blocks=True,
        )

    def _render(self, template_path: str, **kwargs) -> str:
        """内部渲染方法"""

    # ---- Planner 专用 ----
    def build_init_plan_prompt(self, goal, targets, output_schema) -> str: ...
    def build_replan_prompt(self, blackboard_summary, dag_stats, reflections, output_schema) -> str: ...
    def build_terminate_check_prompt(self, blackboard_summary, dag_stats, goal) -> str: ...

    # ---- Executor 专用 ----
    def build_executor_system_prompt(self, executor_type, task, context, stdlib_docs) -> str:
        """base_system + executor_type 专用模板拼接"""
    def build_observation_prompt(self, result) -> str: ...

    # ---- Reflector 专用 ----
    def build_analyze_prompt(self, blackboard_summary, recent_results, dag_stats) -> str: ...
```

模板目录与方法对应关系：

| 方法 | 模板 |
|------|------|
| `build_init_plan_prompt` | `planner/init_plan.jinja2` |
| `build_replan_prompt` | `planner/replan.jinja2` |
| `build_terminate_check_prompt` | `planner/terminate_check.jinja2` |
| `build_executor_system_prompt` | `executor/base_system.jinja2` + `executor/{type}_system.jinja2` |
| `build_observation_prompt` | `executor/observation.jinja2` |
| `build_analyze_prompt` | `reflector/analyze.jinja2` |

设计要点：
- 每个 Agent 有专用 build 方法，参数类型明确
- Executor 模板分两层：base（CodeAct 通用指令）+ 专用（领域知识）
- `_render` 为内部方法，外部只通过专用方法调用

### 4.9 Session 主循环（`session.py`）

```python
class Session:
    def __init__(self, config: DivineConfig):
        # 组装所有组件
        self._blackboard = Blackboard(db_path=config.db_path)
        self._router = LLMRouter(config.llm)
        self._prompt_engine = PromptEngine()
        self._dag = TaskDAG()
        self._scheduler = DAGScheduler(self._dag, concurrency=config.concurrency)
        self._sandbox = Sandbox(timeout=config.code_execution_timeout)
        self._executor = CodeActExecutor(self._router, self._sandbox,
                                         self._blackboard, self._prompt_engine)
        self._planner = Planner(self._router, self._prompt_engine, model=config.planner_model)
        self._reflector = Reflector(self._router, self._prompt_engine, model=config.reflector_model)

    async def run(self) -> None:
        # 1. 初始规划
        operations = await self._planner.init_plan(goal=self._config.goal, config=self._config)
        await self._dag.apply_operations(operations)

        # 2. 主循环
        for round_num in range(1, self._config.max_rounds + 1):
            # 2a. 调度并执行就绪任务
            completed = await self._scheduler.run_round(execute_fn=self._execute_task)
            if not completed and self._dag.is_finished:
                break

            # 2b. 反思
            reflection = await self._reflector.analyze(
                blackboard_summary=self._blackboard.summary(),
                recent_results=self._get_recent_results(completed),
                dag_stats=self._dag.stats,
            )
            self._blackboard.write("reflections", f"round_{round_num}",
                                   value=asdict(reflection), source="reflector")

            # 2c. 重规划
            operations = await self._planner.replan(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
                reflections=[asdict(reflection)],
            )
            if operations:
                await self._dag.apply_operations(operations)

            # 2d. 终止检查
            should_stop, reason = await self._planner.should_terminate(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
            )
            if should_stop:
                break

        # 3. 生成报告
        await self._generate_report()

    async def _execute_task(self, task: TaskNode) -> dict:
        """Scheduler 的执行回调"""
        context = self._blackboard.summary(sections=["hosts", "ports", "credentials", "findings"])
        return await self._executor.execute_task(task, context)
```

设计要点：
- Session 是唯一的组装点，所有组件在 __init__ 中创建
- 主循环每轮四步：调度 -> 反思 -> 重规划 -> 终止检查
- 双重终止：LLM 判定 + max_rounds 硬性兜底 + DAG is_finished
- 单任务异常在 Scheduler 中捕获，不影响主循环

### 4.10 Reporting（`reporting/`）

```python
class ReportGenerator:
    def __init__(self, router: LLMRouter, blackboard: Blackboard): ...

    async def generate(self, output_path: Path) -> None:
        """
        1. 从黑板收集 findings/credentials/hosts/reflections
        2. LLM 生成叙述性摘要
        3. Jinja2 渲染为最终报告
        """

    def _collect_data(self) -> dict: ...
    async def _generate_narrative(self, data: dict) -> dict: ...
    def _render(self, data: dict, narrative: dict) -> str: ...
```

设计要点：
- 数据全部来自 Blackboard
- LLM 生成执行摘要和攻击路径叙述
- 模板与生成逻辑分离

---

## 5. 数据流

```
Session.run()
  │
  ├─ Planner.init_plan(goal) ──► DAG.apply_operations()
  │
  └─ for round in range(max_rounds):
       │
       ├─ Scheduler.run_round(execute_fn)
       │    ├─ DAG.get_ready_tasks()
       │    ├─ CodeActExecutor.execute_task(task, context)
       │    │    ├─ PromptEngine.build_executor_system_prompt()
       │    │    ├─ LLMRouter.chat() ──► LLM 生成代码
       │    │    ├─ Sandbox.execute(code) ──► 执行结果
       │    │    ├─ stdlib: bb_write() ──► Blackboard
       │    │    └─ 循环直到完成或达到 max_iterations
       │    └─ DAG.update_status() / propagate_failure()
       │
       ├─ Reflector.analyze(blackboard_summary, results, stats)
       │    └─ Blackboard.write("reflections", ...)
       │
       ├─ Planner.replan(summary, stats, reflections)
       │    └─ DAG.apply_operations()
       │
       └─ Planner.should_terminate() ──► break if done
```

---

## 6. 实施顺序

1. 项目骨架 + `pyproject.toml` + `config.py` + `cli.py` + `__main__.py`
2. 数据模型（`models/`：TaskNode, Finding, 枚举）
3. Blackboard + 测试
4. TaskDAG + Scheduler + 测试
5. LLM Provider 层（base + router + 4 个 provider + openai_compat + retry + cost）
6. Prompt Engine + 模板骨架
7. CodeAct Sandbox + stdlib + Executor + 测试
8. Planner + Reflector
9. Session 主循环集成
10. Prompt 模板内容编写
11. Report Generator
12. 端到端测试
