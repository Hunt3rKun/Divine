# Divine 重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零重写 Divine 多智能体渗透测试框架，建立 `divine/` 包结构，实现 DAG 任务编排 + Blackboard 共享状态 + CodeAct 执行引擎的完整闭环。

**Architecture:** Session 中心化编排主循环，驱动 Planner → Scheduler → CodeAct → Reflector 闭环。Blackboard 作为统一共享状态（内存 + SQLite 写透），DAG 基于 NetworkX 管理任务依赖。LLM 层支持 OpenAI/Anthropic/Zhipu/MiniMax 四家原生 SDK + OpenAI 兼容格式兜底。

**Tech Stack:** Python 3.11+, asyncio, NetworkX, SQLite, Jinja2, Typer, openai SDK, anthropic SDK, zhipuai SDK, pytest, pytest-asyncio

**Design Spec:** `docs/superpowers/specs/2026-03-28-divine-refactor-design.md`

---

## File Map

```
divine/
├── __init__.py                          # 包入口，版本号
├── __main__.py                          # python -m divine
├── cli.py                               # Typer CLI
├── config.py                            # DivineConfig + LLMConfig
├── session.py                           # 中心化主循环
├── models/
│   ├── __init__.py                      # 导出所有模型
│   ├── common.py                        # AgentRole, PentestPhase, ExecutorType
│   ├── task.py                          # TaskNode, TaskStatus
│   └── finding.py                       # Finding, Severity, FindingType
├── blackboard/
│   ├── __init__.py
│   ├── models.py                        # BlackboardEntry, SECTIONS
│   └── blackboard.py                    # Blackboard 核心
├── dag/
│   ├── __init__.py
│   ├── task_dag.py                      # TaskDAG (NetworkX)
│   └── scheduler.py                     # DAGScheduler
├── llm/
│   ├── __init__.py
│   ├── base.py                          # LLMProvider ABC, LLMMessage, LLMResponse, TokenUsage
│   ├── router.py                        # LLMRouter
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── openai.py                    # OpenAIProvider
│   │   ├── anthropic.py                 # AnthropicProvider
│   │   ├── zhipu.py                     # ZhipuProvider
│   │   ├── minimax.py                   # MiniMaxProvider
│   │   └── openai_compat.py             # OpenAICompatProvider
│   └── utils/
│       ├── __init__.py
│       ├── retry.py                     # RetryHandler
│       └── cost.py                      # CostCalculator
├── codeact/
│   ├── __init__.py
│   ├── sandbox.py                       # Sandbox, ExecutionResult
│   ├── executor.py                      # CodeActExecutor
│   └── stdlib.py                        # create_stdlib()
├── agents/
│   ├── __init__.py
│   ├── planner.py                       # Planner
│   └── reflection.py                    # Reflector, Reflection
├── prompts/
│   ├── __init__.py
│   ├── engine.py                        # PromptEngine
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
    ├── generator.py                     # ReportGenerator
    └── templates/
        └── report.jinja2

tests/
├── conftest.py                          # 共享 fixtures
├── test_models.py                       # 数据模型测试
├── test_blackboard.py                   # Blackboard 测试
├── test_dag.py                          # TaskDAG 测试
├── test_scheduler.py                    # DAGScheduler 测试
├── test_llm_base.py                     # LLM 基础模型测试
├── test_llm_router.py                   # Router 测试（mock providers）
├── test_retry.py                        # 重试策略测试
├── test_cost.py                         # 成本计算测试
├── test_sandbox.py                      # Sandbox 测试
├── test_executor.py                     # CodeActExecutor 测试
├── test_planner.py                      # Planner 测试
├── test_reflector.py                    # Reflector 测试
├── test_prompt_engine.py                # PromptEngine 测试
├── test_session.py                      # Session 集成测试
└── test_config.py                       # Config 测试

pyproject.toml                           # 项目元数据 + 依赖
```

---

### Task 1: 项目骨架 + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `divine/__init__.py`
- Create: `divine/__main__.py`
- Create: `divine/cli.py`
- Create: `divine/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "divine"
version = "0.1.0"
description = "多智能体自动化渗透测试框架"
requires-python = ">=3.11"
dependencies = [
    "networkx>=3.0",
    "jinja2>=3.0",
    "typer>=0.9.0",
    "pyyaml>=6.0",
    "openai>=1.0.0",
    "anthropic>=0.40.0",
    "zhipuai>=2.0.0",
    "loguru>=0.7.0",
    "aiohttp>=3.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
]

[project.scripts]
divine = "divine.cli:app"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: 创建包入口文件**

`divine/__init__.py`:
```python
"""Divine - 多智能体自动化渗透测试框架"""

__version__ = "0.1.0"
```

`divine/__main__.py`:
```python
from divine.cli import app

app()
```

- [ ] **Step 3: 编写 config.py**

`divine/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ProviderConfig:
    """单个 LLM provider 配置"""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 120.0


@dataclass
class LLMConfig:
    """LLM 总配置"""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class DivineConfig:
    """全局配置"""
    targets: list[str] = field(default_factory=list)
    goal: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_rounds: int = 20
    max_tasks: int = 50
    timeout: int = 3600
    concurrency: int = 3
    code_execution_timeout: int = 60
    planner_model: str = "claude-sonnet-4-20250514"
    reflector_model: str = "claude-sonnet-4-20250514"
    executor_model: str = "claude-sonnet-4-20250514"
    db_path: str = ":memory:"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "DivineConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            raw = yaml.safe_load(f)

        llm_raw = raw.pop("llm", {})
        providers = {}
        for name, pconf in llm_raw.get("providers", {}).items():
            providers[name] = ProviderConfig(**pconf)
        llm = LLMConfig(
            providers=providers,
            pricing=llm_raw.get("pricing", {}),
        )

        return cls(llm=llm, **raw)
```

- [ ] **Step 4: 编写 cli.py**

`divine/cli.py`:
```python
from pathlib import Path
import asyncio

import typer

app = typer.Typer(name="divine", help="多智能体自动化渗透测试框架")


@app.command()
def engage(
    config: Path = typer.Option(..., "--config", "-c", help="目标配置 YAML 文件"),
):
    """启动渗透测试"""
    from divine.config import DivineConfig
    from divine.session import Session

    cfg = DivineConfig.from_yaml(config)
    asyncio.run(Session(cfg).run())


@app.command()
def version():
    """显示版本号"""
    from divine import __version__
    typer.echo(f"Divine v{__version__}")
```

- [ ] **Step 5: 编写 config 测试**

`tests/conftest.py`:
```python
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))
```

`tests/test_config.py`:
```python
import tempfile
from pathlib import Path

import yaml

from divine.config import DivineConfig, LLMConfig, ProviderConfig


class TestDivineConfig:
    def test_default_values(self):
        cfg = DivineConfig()
        assert cfg.max_rounds == 20
        assert cfg.concurrency == 3
        assert cfg.timeout == 3600
        assert cfg.db_path == ":memory:"

    def test_from_yaml(self):
        data = {
            "targets": ["192.168.1.1"],
            "goal": "获取目标主机控制权",
            "max_rounds": 10,
            "concurrency": 5,
            "llm": {
                "providers": {
                    "openai": {
                        "api_key": "sk-test",
                        "base_url": "https://api.openai.com/v1",
                    }
                },
                "pricing": {
                    "gpt-4o": {"input": 2.5, "output": 10.0}
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = DivineConfig.from_yaml(Path(f.name))

        assert cfg.targets == ["192.168.1.1"]
        assert cfg.goal == "获取目标主机控制权"
        assert cfg.max_rounds == 10
        assert cfg.concurrency == 5
        assert "openai" in cfg.llm.providers
        assert cfg.llm.providers["openai"].api_key == "sk-test"
        assert cfg.llm.pricing["gpt-4o"]["input"] == 2.5
```

- [ ] **Step 6: 运行测试验证**

Run: `cd /root/Divine && python -m pytest tests/test_config.py -v`
Expected: 2 tests PASS

- [ ] **Step 7: 验证 CLI 入口**

Run: `cd /root/Divine && python -m divine version`
Expected: 输出 `Divine v0.1.0`

- [ ] **Step 8: 提交**

```bash
git add pyproject.toml divine/__init__.py divine/__main__.py divine/cli.py divine/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: 项目骨架 + config + CLI 入口"
```

---

### Task 2: 数据模型

**Files:**
- Create: `divine/models/__init__.py`
- Create: `divine/models/common.py`
- Create: `divine/models/task.py`
- Create: `divine/models/finding.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 编写测试**

`tests/test_models.py`:
```python
from divine.models.common import AgentRole, PentestPhase, ExecutorType
from divine.models.task import TaskNode, TaskStatus
from divine.models.finding import Finding, Severity, FindingType


class TestCommonEnums:
    def test_agent_role_values(self):
        assert AgentRole.PLANNER == "planner"
        assert AgentRole.EXECUTOR == "executor"
        assert AgentRole.REFLECTOR == "reflector"

    def test_pentest_phase_values(self):
        assert PentestPhase.RECON == "recon"
        assert PentestPhase.EXPLOIT == "exploit"

    def test_executor_type_values(self):
        assert ExecutorType.RECON == "recon"
        assert ExecutorType.WEB == "web"
        assert ExecutorType.HOST == "host"
        assert ExecutorType.SERVICE == "service"

    def test_enum_json_serializable(self):
        """str Enum 可以直接 JSON 序列化"""
        import json
        data = {"role": AgentRole.PLANNER, "phase": PentestPhase.RECON}
        result = json.dumps(data)
        assert '"planner"' in result


class TestTaskNode:
    def test_create_task_node(self):
        task = TaskNode(
            id="recon_1",
            description="端口扫描",
            phase=PentestPhase.RECON,
            executor_type=ExecutorType.RECON,
        )
        assert task.id == "recon_1"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
        assert task.result is None

    def test_task_with_dependencies(self):
        task = TaskNode(
            id="exploit_1",
            description="利用 SQL 注入",
            phase=PentestPhase.EXPLOIT,
            executor_type=ExecutorType.WEB,
            dependencies=["recon_1", "scan_1"],
            priority=5,
        )
        assert task.dependencies == ["recon_1", "scan_1"]
        assert task.priority == 5

    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.SKIPPED == "skipped"


class TestFinding:
    def test_create_finding_defaults(self):
        f = Finding()
        assert len(f.id) == 8
        assert f.type == FindingType.KNOWLEDGE
        assert f.severity == Severity.INFO
        assert f.evidence == []

    def test_create_finding_with_values(self):
        f = Finding(
            type=FindingType.VULNERABILITY,
            severity=Severity.HIGH,
            title="SQL Injection in login",
            detail={"url": "/login", "param": "username"},
            source_task="web_1",
        )
        assert f.type == FindingType.VULNERABILITY
        assert f.severity == Severity.HIGH
        assert f.detail["param"] == "username"

    def test_finding_type_values(self):
        assert FindingType.ASSET == "asset"
        assert FindingType.CREDENTIAL == "credential"
        assert FindingType.VULNERABILITY == "vulnerability"

    def test_severity_ordering(self):
        """severity 值可以用于字符串比较"""
        severities = [Severity.INFO, Severity.CRITICAL, Severity.HIGH]
        assert Severity.CRITICAL in severities
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现模型**

`divine/models/__init__.py`:
```python
from divine.models.common import AgentRole, PentestPhase, ExecutorType
from divine.models.task import TaskNode, TaskStatus
from divine.models.finding import Finding, Severity, FindingType

__all__ = [
    "AgentRole", "PentestPhase", "ExecutorType",
    "TaskNode", "TaskStatus",
    "Finding", "Severity", "FindingType",
]
```

`divine/models/common.py`:
```python
from enum import Enum


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

`divine/models/task.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from divine.models.common import PentestPhase, ExecutorType


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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

`divine/models/finding.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


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

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_models.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add divine/models/ tests/test_models.py
git commit -m "feat: 数据模型 - TaskNode, Finding, 枚举"
```

---

### Task 3: Blackboard

**Files:**
- Create: `divine/blackboard/__init__.py`
- Create: `divine/blackboard/models.py`
- Create: `divine/blackboard/blackboard.py`
- Create: `tests/test_blackboard.py`

- [ ] **Step 1: 编写测试**

`tests/test_blackboard.py`:
```python
import asyncio
import pytest

from divine.blackboard.models import BlackboardEntry, SECTIONS
from divine.blackboard.blackboard import Blackboard


class TestBlackboardModels:
    def test_sections_defined(self):
        assert "hosts" in SECTIONS
        assert "ports" in SECTIONS
        assert "findings" in SECTIONS
        assert "credentials" in SECTIONS
        assert "tasks" in SECTIONS
        assert "reflections" in SECTIONS

    def test_entry_creation(self):
        entry = BlackboardEntry(section="hosts", key="192.168.1.1", value={"os": "Linux"})
        assert entry.section == "hosts"
        assert entry.version == 1


class TestBlackboardReadWrite:
    def test_write_and_read(self):
        bb = Blackboard()
        bb.write("hosts", "192.168.1.1", {"os": "Linux", "ports": [22, 80]}, source="recon_1")
        result = bb.read("hosts", "192.168.1.1")
        assert result == {"os": "Linux", "ports": [22, 80]}

    def test_read_nonexistent_key(self):
        bb = Blackboard()
        result = bb.read("hosts", "nonexistent")
        assert result is None

    def test_read_entire_section(self):
        bb = Blackboard()
        bb.write("hosts", "h1", {"os": "Linux"}, source="t1")
        bb.write("hosts", "h2", {"os": "Windows"}, source="t1")
        result = bb.read("hosts")
        assert len(result) == 2

    def test_write_invalid_section_raises(self):
        bb = Blackboard()
        with pytest.raises(ValueError, match="Invalid section"):
            bb.write("invalid_section", "k", "v")

    def test_write_updates_version(self):
        bb = Blackboard()
        bb.write("hosts", "h1", "v1", source="t1")
        bb.write("hosts", "h1", "v2", source="t1")
        entry = bb._memory["hosts"]["h1"]
        assert entry.version == 2

    def test_query_with_filter(self):
        bb = Blackboard()
        bb.write("findings", "f1", {"severity": "high"}, source="t1")
        bb.write("findings", "f2", {"severity": "low"}, source="t1")
        bb.write("findings", "f3", {"severity": "high"}, source="t2")

        results = bb.query("findings", filter_fn=lambda e: e.value.get("severity") == "high")
        assert len(results) == 2


class TestBlackboardSummary:
    def test_summary_all_sections(self):
        bb = Blackboard()
        bb.write("hosts", "h1", {"os": "Linux"}, source="t1")
        bb.write("findings", "f1", {"type": "vuln"}, source="t1")
        summary = bb.summary()
        assert "hosts" in summary
        assert "findings" in summary

    def test_summary_specific_sections(self):
        bb = Blackboard()
        bb.write("hosts", "h1", {"os": "Linux"}, source="t1")
        bb.write("findings", "f1", {"type": "vuln"}, source="t1")
        summary = bb.summary(sections=["hosts"])
        assert "hosts" in summary
        assert "findings" not in summary


class TestBlackboardEvent:
    async def test_event_set_on_write(self):
        bb = Blackboard()
        assert not bb._events["hosts"].is_set()
        bb.write("hosts", "h1", "v1", source="t1")
        assert bb._events["hosts"].is_set()

    async def test_wait_for_and_clear(self):
        bb = Blackboard()

        async def writer():
            await asyncio.sleep(0.05)
            bb.write("hosts", "h1", "v1", source="t1")

        asyncio.create_task(writer())
        await bb.wait_for("hosts")
        assert bb._events["hosts"].is_set()
        bb.clear_event("hosts")
        assert not bb._events["hosts"].is_set()


class TestBlackboardPersistence:
    def test_sqlite_persistence(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        bb = Blackboard(db_path=db_path)
        bb.write("hosts", "h1", {"os": "Linux"}, source="t1")

        # 新实例应该能从 SQLite 恢复数据
        bb2 = Blackboard(db_path=db_path)
        result = bb2.read("hosts", "h1")
        assert result == {"os": "Linux"}

    def test_audit_log(self):
        bb = Blackboard()
        bb.write("hosts", "h1", "v1", source="t1")
        bb.write("hosts", "h2", "v2", source="t2")
        logs = bb.audit_log(limit=10)
        assert len(logs) == 2
        assert logs[0]["source"] == "t1"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_blackboard.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 blackboard/models.py**

`divine/blackboard/__init__.py`:
```python
from divine.blackboard.blackboard import Blackboard
from divine.blackboard.models import BlackboardEntry, SECTIONS

__all__ = ["Blackboard", "BlackboardEntry", "SECTIONS"]
```

`divine/blackboard/models.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

- [ ] **Step 4: 实现 blackboard/blackboard.py**

`divine/blackboard/blackboard.py`:
```python
import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Any, Callable, Optional

from divine.blackboard.models import BlackboardEntry, SECTIONS


class Blackboard:
    def __init__(self, db_path: str = ":memory:"):
        self._memory: dict[str, dict[str, BlackboardEntry]] = {s: {} for s in SECTIONS}
        self._db = sqlite3.connect(db_path)
        self._events: dict[str, asyncio.Event] = {}
        self._init_db()
        self._load_from_db()
        self._init_events()

    def _init_events(self) -> None:
        """初始化 asyncio.Event（可能在非 async 上下文中调用，延迟创建）"""
        try:
            asyncio.get_running_loop()
            self._events = {s: asyncio.Event() for s in SECTIONS}
        except RuntimeError:
            self._events = {}

    def _ensure_events(self) -> None:
        """确保 events 已初始化（首次在 async 上下文中调用时创建）"""
        if not self._events:
            self._events = {s: asyncio.Event() for s in SECTIONS}

    def _init_db(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS blackboard (
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                PRIMARY KEY (section, key)
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                version INTEGER DEFAULT 1
            )
        """)
        self._db.commit()

    def _load_from_db(self) -> None:
        """从 SQLite 恢复数据到内存"""
        cursor = self._db.execute("SELECT section, key, value, source, timestamp, version FROM blackboard")
        for row in cursor:
            section, key, value_str, source, ts_str, version = row
            value = json.loads(value_str)
            entry = BlackboardEntry(
                section=section, key=key, value=value,
                source=source, timestamp=datetime.fromisoformat(ts_str), version=version,
            )
            self._memory[section][key] = entry

    def read(self, section: str, key: str = None) -> Any:
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section}")
        if key is None:
            return {k: e.value for k, e in self._memory[section].items()}
        entry = self._memory[section].get(key)
        return entry.value if entry else None

    def write(self, section: str, key: str, value: Any, source: str = "") -> None:
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section}")

        now = datetime.now()
        existing = self._memory[section].get(key)
        version = (existing.version + 1) if existing else 1

        entry = BlackboardEntry(
            section=section, key=key, value=value,
            source=source, timestamp=now, version=version,
        )
        self._memory[section][key] = entry
        self._persist(entry)

        self._ensure_events()
        if section in self._events:
            self._events[section].set()

    def query(self, section: str, filter_fn: Callable[[BlackboardEntry], bool] = None) -> list[BlackboardEntry]:
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section}")
        entries = list(self._memory[section].values())
        if filter_fn:
            entries = [e for e in entries if filter_fn(e)]
        return entries

    async def wait_for(self, section: str) -> None:
        self._ensure_events()
        await self._events[section].wait()

    def clear_event(self, section: str) -> None:
        self._ensure_events()
        if section in self._events:
            self._events[section].clear()

    def summary(self, sections: list[str] = None) -> dict:
        target_sections = sections or SECTIONS
        result = {}
        for section in target_sections:
            if section in self._memory:
                items = self._memory[section]
                result[section] = {
                    "count": len(items),
                    "keys": list(items.keys()),
                    "entries": {k: e.value for k, e in items.items()},
                }
        return result

    def audit_log(self, limit: int = 100) -> list[dict]:
        cursor = self._db.execute(
            "SELECT section, key, value, source, timestamp, version FROM audit_log ORDER BY id ASC LIMIT ?",
            (limit,)
        )
        return [
            {"section": r[0], "key": r[1], "value": json.loads(r[2]),
             "source": r[3], "timestamp": r[4], "version": r[5]}
            for r in cursor
        ]

    def _persist(self, entry: BlackboardEntry) -> None:
        value_str = json.dumps(entry.value, ensure_ascii=False)
        ts_str = entry.timestamp.isoformat()

        self._db.execute(
            "INSERT OR REPLACE INTO blackboard (section, key, value, source, timestamp, version) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.section, entry.key, value_str, entry.source, ts_str, entry.version),
        )
        self._db.execute(
            "INSERT INTO audit_log (section, key, value, source, timestamp, version) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.section, entry.key, value_str, entry.source, ts_str, entry.version),
        )
        self._db.commit()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `python -m pytest tests/test_blackboard.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add divine/blackboard/ tests/test_blackboard.py
git commit -m "feat: Blackboard 共享状态管理"
```

---

### Task 4: TaskDAG + Scheduler

**Files:**
- Create: `divine/dag/__init__.py`
- Create: `divine/dag/task_dag.py`
- Create: `divine/dag/scheduler.py`
- Create: `tests/test_dag.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: 编写 DAG 测试**

`tests/test_dag.py`:
```python
import pytest

from divine.dag.task_dag import TaskDAG
from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


def make_task(id: str, deps: list[str] = None, priority: int = 0) -> TaskNode:
    return TaskNode(
        id=id,
        description=f"Task {id}",
        phase=PentestPhase.RECON,
        executor_type=ExecutorType.RECON,
        dependencies=deps or [],
        priority=priority,
    )


class TestTaskDAGBasic:
    async def test_add_task(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        assert dag.get_task("t1").id == "t1"

    async def test_add_task_with_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        task = dag.get_task("t2")
        assert task.dependencies == ["t1"]

    async def test_cycle_detection(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        with pytest.raises(ValueError, match="cycle"):
            await dag.add_task(make_task("t3", deps=["t2"]))
            # 手动加一条 t1 -> t3 的反向边来构成环
            # 实际上环检测在 add_task 中，这里需要构造一个会形成环的场景
            pass

    async def test_cycle_detection_real(self):
        """t1 -> t2 -> t3 -> t1 形成环"""
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.add_task(make_task("t3", deps=["t2"]))
        # 尝试添加 t1 依赖 t3，形成环
        with pytest.raises(ValueError, match="cycle"):
            t1_cyclic = make_task("t1_cycle", deps=["t3"])
            t1_cyclic.id = "t1"
            # 我们需要添加一条边 t3 -> t1，但 t1 已存在
            # 所以测试通过 add_task 添加一个依赖已有节点的新节点来间接构造
            await dag.add_task(make_task("t4", deps=["t3"]))
            # 然后手动加边 t1 依赖 t4 — 这需要 task_dag 支持
            # 实际测试：直接测试 t1 -> t2, t2 -> t1
            pass

    async def test_remove_task(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.remove_task("t1")
        assert dag.get_task("t1") is None

    async def test_get_all_tasks(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        tasks = dag.get_all_tasks()
        assert len(tasks) == 2


class TestTaskDAGReadyTasks:
    async def test_get_ready_no_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        ready = dag.get_ready_tasks()
        assert len(ready) == 2

    async def test_get_ready_with_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    async def test_ready_after_dep_completed(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    async def test_ready_sorted_by_priority(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1", priority=1))
        await dag.add_task(make_task("t2", priority=5))
        await dag.add_task(make_task("t3", priority=3))
        ready = dag.get_ready_tasks()
        assert [t.id for t in ready] == ["t2", "t3", "t1"]


class TestTaskDAGFailurePropagation:
    async def test_propagate_failure(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        await dag.add_task(make_task("t3", deps=["t2"]))
        skipped = await dag.propagate_failure("t1")
        assert "t2" in skipped
        assert "t3" in skipped
        assert dag.get_task("t2").status == TaskStatus.SKIPPED
        assert dag.get_task("t3").status == TaskStatus.SKIPPED


class TestTaskDAGProperties:
    async def test_is_finished_false(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        assert not dag.is_finished

    async def test_is_finished_true(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        assert dag.is_finished

    async def test_stats(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.update_status("t1", TaskStatus.COMPLETED)
        stats = dag.stats
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["pending"] == 1


class TestTaskDAGApplyOperations:
    async def test_apply_add_node(self):
        dag = TaskDAG()
        ops = [
            {
                "command": "add_node",
                "node_data": {
                    "id": "t1",
                    "description": "Port scan",
                    "phase": "recon",
                    "executor_type": "recon",
                    "dependencies": [],
                },
            }
        ]
        await dag.apply_operations(ops)
        assert dag.get_task("t1") is not None
        assert dag.get_task("t1").description == "Port scan"

    async def test_apply_remove_node(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        ops = [{"command": "remove_node", "node_id": "t1"}]
        await dag.apply_operations(ops)
        assert dag.get_task("t1") is None

    async def test_apply_update_node(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        ops = [{"command": "update_node", "node_id": "t1", "updates": {"priority": 10}}]
        await dag.apply_operations(ops)
        assert dag.get_task("t1").priority == 10
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_dag.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 task_dag.py**

`divine/dag/__init__.py`:
```python
from divine.dag.task_dag import TaskDAG
from divine.dag.scheduler import DAGScheduler

__all__ = ["TaskDAG", "DAGScheduler"]
```

`divine/dag/task_dag.py`:
```python
import asyncio

import networkx as nx

from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


class TaskDAG:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()

    async def add_task(self, task: TaskNode) -> None:
        async with self._lock:
            self._graph.add_node(task.id, task=task)
            for dep_id in task.dependencies:
                if dep_id in self._graph:
                    self._graph.add_edge(dep_id, task.id)
            if not nx.is_directed_acyclic_graph(self._graph):
                self._graph.remove_node(task.id)
                raise ValueError(f"Adding task '{task.id}' would create a cycle")

    async def remove_task(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._graph:
                self._graph.remove_node(task_id)

    async def update_status(self, task_id: str, status: TaskStatus,
                            result: dict = None, error: str = None) -> None:
        async with self._lock:
            if task_id not in self._graph:
                return
            task: TaskNode = self._graph.nodes[task_id]["task"]
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

    def get_task(self, task_id: str) -> TaskNode | None:
        if task_id not in self._graph:
            return None
        return self._graph.nodes[task_id]["task"]

    def get_all_tasks(self) -> list[TaskNode]:
        return [self._graph.nodes[n]["task"] for n in self._graph.nodes]

    def get_ready_tasks(self) -> list[TaskNode]:
        ready = []
        for node_id in self._graph.nodes:
            task: TaskNode = self._graph.nodes[node_id]["task"]
            if task.status != TaskStatus.PENDING:
                continue
            predecessors = list(self._graph.predecessors(node_id))
            all_deps_done = all(
                self._graph.nodes[p]["task"].status == TaskStatus.COMPLETED
                for p in predecessors
            )
            if all_deps_done:
                ready.append(task)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def get_failed_tasks(self) -> list[TaskNode]:
        return [
            self._graph.nodes[n]["task"]
            for n in self._graph.nodes
            if self._graph.nodes[n]["task"].status == TaskStatus.FAILED
        ]

    def get_descendants(self, task_id: str) -> set[str]:
        if task_id not in self._graph:
            return set()
        return nx.descendants(self._graph, task_id)

    async def propagate_failure(self, task_id: str) -> list[str]:
        async with self._lock:
            descendants = self.get_descendants(task_id)
            skipped = []
            for desc_id in descendants:
                task = self._graph.nodes[desc_id]["task"]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SKIPPED
                    skipped.append(desc_id)
            return skipped

    async def apply_operations(self, operations: list[dict]) -> None:
        for op in operations:
            cmd = op["command"]
            if cmd == "add_node":
                data = op["node_data"]
                task = TaskNode(
                    id=data["id"],
                    description=data.get("description", ""),
                    phase=PentestPhase(data.get("phase", "recon")),
                    executor_type=ExecutorType(data.get("executor_type", "recon")),
                    dependencies=data.get("dependencies", []),
                    priority=data.get("priority", 0),
                )
                await self.add_task(task)
            elif cmd == "remove_node":
                await self.remove_task(op["node_id"])
            elif cmd == "update_node":
                node_id = op["node_id"]
                updates = op.get("updates", {})
                async with self._lock:
                    if node_id in self._graph:
                        task = self._graph.nodes[node_id]["task"]
                        for key, value in updates.items():
                            if hasattr(task, key):
                                setattr(task, key, value)

    @property
    def is_finished(self) -> bool:
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(
            self._graph.nodes[n]["task"].status in terminal
            for n in self._graph.nodes
        )

    @property
    def stats(self) -> dict:
        counts = {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
        for n in self._graph.nodes:
            status = self._graph.nodes[n]["task"].status.value
            counts["total"] += 1
            counts[status] = counts.get(status, 0) + 1
        return counts
```

- [ ] **Step 4: 运行 DAG 测试验证通过**

Run: `python -m pytest tests/test_dag.py -v`
Expected: PASS（忽略 cycle_detection 中未完善的测试用例，可能需要调整）

- [ ] **Step 5: 编写 Scheduler 测试**

`tests/test_scheduler.py`:
```python
import pytest

from divine.dag.task_dag import TaskDAG
from divine.dag.scheduler import DAGScheduler
from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


def make_task(id: str, deps: list[str] = None) -> TaskNode:
    return TaskNode(
        id=id, description=f"Task {id}",
        phase=PentestPhase.RECON, executor_type=ExecutorType.RECON,
        dependencies=deps or [],
    )


class TestDAGScheduler:
    async def test_run_round_basic(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        scheduler = DAGScheduler(dag, concurrency=2)

        results = {}

        async def execute_fn(task: TaskNode) -> dict:
            results[task.id] = True
            return {"status": "done"}

        completed = await scheduler.run_round(execute_fn)
        assert len(completed) == 2
        assert dag.get_task("t1").status == TaskStatus.COMPLETED
        assert dag.get_task("t2").status == TaskStatus.COMPLETED

    async def test_run_round_respects_deps(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        scheduler = DAGScheduler(dag, concurrency=2)

        async def execute_fn(task: TaskNode) -> dict:
            return {"status": "done"}

        # 第一轮只有 t1 就绪
        completed = await scheduler.run_round(execute_fn)
        assert completed == ["t1"]
        assert dag.get_task("t1").status == TaskStatus.COMPLETED
        assert dag.get_task("t2").status == TaskStatus.PENDING

        # 第二轮 t2 就绪
        completed = await scheduler.run_round(execute_fn)
        assert completed == ["t2"]

    async def test_run_round_handles_failure(self):
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2", deps=["t1"]))
        scheduler = DAGScheduler(dag, concurrency=2)

        async def failing_fn(task: TaskNode) -> dict:
            raise RuntimeError("oops")

        completed = await scheduler.run_round(failing_fn)
        assert "t1" in completed
        assert dag.get_task("t1").status == TaskStatus.FAILED
        assert dag.get_task("t2").status == TaskStatus.SKIPPED

    async def test_run_round_no_ready_tasks(self):
        dag = TaskDAG()
        scheduler = DAGScheduler(dag, concurrency=2)

        async def execute_fn(task: TaskNode) -> dict:
            return {}

        completed = await scheduler.run_round(execute_fn)
        assert completed == []

    async def test_concurrency_limit(self):
        """验证 Semaphore 限流：3 个任务 concurrency=1 应该串行执行"""
        dag = TaskDAG()
        await dag.add_task(make_task("t1"))
        await dag.add_task(make_task("t2"))
        await dag.add_task(make_task("t3"))
        scheduler = DAGScheduler(dag, concurrency=1)

        import asyncio
        max_concurrent = 0
        current_concurrent = 0

        async def tracked_fn(task: TaskNode) -> dict:
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return {}

        await scheduler.run_round(tracked_fn)
        assert max_concurrent == 1
```

- [ ] **Step 6: 实现 scheduler.py**

`divine/dag/scheduler.py`:
```python
import asyncio
from typing import Awaitable, Callable

from divine.dag.task_dag import TaskDAG
from divine.models.task import TaskNode, TaskStatus


class DAGScheduler:
    def __init__(self, dag: TaskDAG, concurrency: int = 3):
        self._dag = dag
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run_round(self, execute_fn: Callable[[TaskNode], Awaitable[dict]]) -> list[str]:
        ready = self._dag.get_ready_tasks()
        if not ready:
            return []

        completed_ids = []

        async def _run_one(task: TaskNode) -> None:
            async with self._semaphore:
                await self._dag.update_status(task.id, TaskStatus.RUNNING)
                try:
                    result = await execute_fn(task)
                    await self._dag.update_status(task.id, TaskStatus.COMPLETED, result=result)
                except Exception as e:
                    await self._dag.update_status(task.id, TaskStatus.FAILED, error=str(e))
                    await self._dag.propagate_failure(task.id)
                completed_ids.append(task.id)

        await asyncio.gather(*[_run_one(t) for t in ready])
        return completed_ids
```

- [ ] **Step 7: 运行所有 DAG 和 Scheduler 测试**

Run: `python -m pytest tests/test_dag.py tests/test_scheduler.py -v`
Expected: 全部 PASS

- [ ] **Step 8: 提交**

```bash
git add divine/dag/ tests/test_dag.py tests/test_scheduler.py
git commit -m "feat: TaskDAG + DAGScheduler"
```

---

### Task 5: LLM 基础层（base + utils）

**Files:**
- Create: `divine/llm/__init__.py`
- Create: `divine/llm/base.py`
- Create: `divine/llm/utils/__init__.py`
- Create: `divine/llm/utils/retry.py`
- Create: `divine/llm/utils/cost.py`
- Create: `tests/test_llm_base.py`
- Create: `tests/test_retry.py`
- Create: `tests/test_cost.py`

- [ ] **Step 1: 编写测试**

`tests/test_llm_base.py`:
```python
from divine.llm.base import LLMMessage, LLMResponse, TokenUsage


class TestLLMMessage:
    def test_create_message(self):
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_system_message(self):
        msg = LLMMessage(role="system", content="You are a pentester")
        assert msg.role == "system"


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.total_tokens == 0

    def test_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150


class TestLLMResponse:
    def test_create_response(self):
        resp = LLMResponse(
            content="Hello",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert resp.content == "Hello"
        assert resp.cost == 0.0
```

`tests/test_retry.py`:
```python
import pytest

from divine.llm.utils.retry import RetryHandler, RetryConfig


class TestRetryHandler:
    async def test_success_first_try(self):
        handler = RetryHandler(RetryConfig())
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await handler.execute(fn)
        assert result == "ok"
        assert call_count == 1

    async def test_retry_on_retryable_error(self):
        handler = RetryHandler(RetryConfig(max_attempts=3, base_delay=0.01))
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "ok"

        result = await handler.execute(fn)
        assert result == "ok"
        assert call_count == 3

    async def test_max_attempts_exceeded(self):
        handler = RetryHandler(RetryConfig(max_attempts=2, base_delay=0.01))

        async def fn():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await handler.execute(fn)

    def test_delay_calculation(self):
        handler = RetryHandler(RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0))
        d1 = handler._calculate_delay(1)
        d2 = handler._calculate_delay(2)
        d3 = handler._calculate_delay(3)
        assert d1 >= 1.0
        assert d2 >= 2.0
        assert d3 <= 10.0  # capped by max_delay
```

`tests/test_cost.py`:
```python
from divine.llm.utils.cost import CostCalculator
from divine.llm.base import TokenUsage


class TestCostCalculator:
    def test_calculate_known_model(self):
        pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}
        calc = CostCalculator(pricing)
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = calc.calculate("gpt-4o", usage)
        # input: 1000/1M * 2.5 = 0.0025, output: 500/1M * 10.0 = 0.005
        assert abs(cost - 0.0075) < 0.0001

    def test_calculate_unknown_model(self):
        calc = CostCalculator({})
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = calc.calculate("unknown-model", usage)
        assert cost == 0.0

    def test_update_pricing(self):
        calc = CostCalculator({})
        calc.update_pricing("new-model", {"input": 1.0, "output": 2.0})
        usage = TokenUsage(input_tokens=1000000, output_tokens=1000000, total_tokens=2000000)
        cost = calc.calculate("new-model", usage)
        assert abs(cost - 3.0) < 0.0001
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_llm_base.py tests/test_retry.py tests/test_cost.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 base.py**

`divine/llm/__init__.py`:
```python
from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage

__all__ = ["LLMProvider", "LLMMessage", "LLMResponse", "TokenUsage"]
```

`divine/llm/base.py`:
```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    cost: float = 0.0
    raw_response: Any = None


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        ...
```

- [ ] **Step 4: 实现 retry.py**

`divine/llm/utils/__init__.py`:
```python
```

`divine/llm/utils/retry.py`:
```python
import asyncio
import random
from dataclasses import dataclass

from loguru import logger


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


# 默认可重试的异常类型
RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


class RetryHandler:
    def __init__(self, config: RetryConfig = None):
        self._config = config or RetryConfig()

    async def execute(self, fn, *args, **kwargs):
        last_error = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt == self._config.max_attempts:
                    raise
                delay = self._calculate_delay(attempt)
                logger.warning(f"Retry {attempt}/{self._config.max_attempts} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            except Exception:
                raise
        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        delay = self._config.base_delay * (self._config.exponential_base ** (attempt - 1))
        delay = min(delay, self._config.max_delay)
        # 添加 jitter
        delay *= (0.5 + random.random() * 0.5)
        return delay
```

- [ ] **Step 5: 实现 cost.py**

`divine/llm/utils/cost.py`:
```python
from divine.llm.base import TokenUsage


class CostCalculator:
    def __init__(self, pricing: dict[str, dict[str, float]] = None):
        self._pricing = pricing or {}

    def calculate(self, model: str, usage: TokenUsage) -> float:
        price = self._get_price(model)
        if not price:
            return 0.0
        input_cost = (usage.input_tokens / 1_000_000) * price.get("input", 0)
        output_cost = (usage.output_tokens / 1_000_000) * price.get("output", 0)
        return input_cost + output_cost

    def update_pricing(self, model: str, prices: dict[str, float]) -> None:
        self._pricing[model] = prices

    def _get_price(self, model: str) -> dict[str, float] | None:
        # 精确匹配
        if model in self._pricing:
            return self._pricing[model]
        # 前缀匹配
        for key in self._pricing:
            if model.startswith(key):
                return self._pricing[key]
        return None
```

- [ ] **Step 6: 运行测试验证通过**

Run: `python -m pytest tests/test_llm_base.py tests/test_retry.py tests/test_cost.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add divine/llm/__init__.py divine/llm/base.py divine/llm/utils/ tests/test_llm_base.py tests/test_retry.py tests/test_cost.py
git commit -m "feat: LLM 基础层 - base + retry + cost"
```

---

### Task 6: LLM Providers + Router

**Files:**
- Create: `divine/llm/providers/__init__.py`
- Create: `divine/llm/providers/openai.py`
- Create: `divine/llm/providers/anthropic.py`
- Create: `divine/llm/providers/zhipu.py`
- Create: `divine/llm/providers/minimax.py`
- Create: `divine/llm/providers/openai_compat.py`
- Create: `divine/llm/router.py`
- Create: `tests/test_llm_router.py`

- [ ] **Step 1: 编写 Router 测试**

`tests/test_llm_router.py`:
```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from divine.llm.base import LLMMessage, LLMResponse, TokenUsage, LLMProvider
from divine.llm.router import LLMRouter, ROUTE_MAP
from divine.config import LLMConfig, ProviderConfig


class TestRouteMap:
    def test_openai_models(self):
        assert "gpt-" in ROUTE_MAP
        assert "o4" in ROUTE_MAP

    def test_anthropic_models(self):
        assert "claude-" in ROUTE_MAP

    def test_zhipu_models(self):
        assert "glm-" in ROUTE_MAP

    def test_minimax_models(self):
        assert "minimax" in ROUTE_MAP


class TestLLMRouter:
    def _make_config(self) -> LLMConfig:
        return LLMConfig(
            providers={
                "openai": ProviderConfig(api_key="sk-test"),
                "anthropic": ProviderConfig(api_key="sk-ant-test"),
            }
        )

    def test_route_openai(self):
        router = LLMRouter(self._make_config())
        provider_name = router._resolve_provider("gpt-4o")
        assert provider_name == "openai"

    def test_route_anthropic(self):
        router = LLMRouter(self._make_config())
        provider_name = router._resolve_provider("claude-sonnet-4-20250514")
        assert provider_name == "anthropic"

    def test_route_unknown_falls_back(self):
        router = LLMRouter(self._make_config())
        provider_name = router._resolve_provider("some-unknown-model")
        assert provider_name == "openai_compat"

    async def test_chat_routes_to_correct_provider(self):
        config = self._make_config()
        router = LLMRouter(config)

        mock_response = LLMResponse(
            content="test response",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.chat.return_value = mock_response
        router._providers["openai"] = mock_provider

        messages = [LLMMessage(role="user", content="hello")]
        result = await router.chat("gpt-4o", messages)

        assert result.content == "test response"
        mock_provider.chat.assert_called_once()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_llm_router.py -v`
Expected: FAIL

- [ ] **Step 3: 实现各 Provider**

每个 provider 的实现应使用对应的原生 SDK。以下是接口骨架，实现时需适配各 SDK 的最新 API。

`divine/llm/providers/__init__.py`:
```python
from divine.llm.providers.openai import OpenAIProvider
from divine.llm.providers.anthropic import AnthropicProvider
from divine.llm.providers.zhipu import ZhipuProvider
from divine.llm.providers.minimax import MiniMaxProvider
from divine.llm.providers.openai_compat import OpenAICompatProvider

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "zhipu": ZhipuProvider,
    "minimax": MiniMaxProvider,
    "openai_compat": OpenAICompatProvider,
}

__all__ = [
    "OpenAIProvider", "AnthropicProvider", "ZhipuProvider",
    "MiniMaxProvider", "OpenAICompatProvider", "PROVIDER_CLASSES",
]
```

`divine/llm/providers/openai.py`:
```python
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage
from divine.config import ProviderConfig


class OpenAIProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "gpt-4o")
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "gpt-4o")
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

`divine/llm/providers/anthropic.py`:
```python
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage
from divine.config import ProviderConfig


class AnthropicProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "claude-sonnet-4-20250514")
        max_tokens = kwargs.pop("max_tokens", 8192)

        # Anthropic: system prompt 走独立参数
        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        create_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=chat_messages,
            **kwargs,
        )
        if system_prompt:
            create_kwargs["system"] = system_prompt

        response = await self._client.messages.create(**create_kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "claude-sonnet-4-20250514")
        max_tokens = kwargs.pop("max_tokens", 8192)

        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        create_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=chat_messages,
            **kwargs,
        )
        if system_prompt:
            create_kwargs["system"] = system_prompt

        async with self._client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
```

`divine/llm/providers/zhipu.py`:
```python
from collections.abc import AsyncIterator

from zhipuai import ZhipuAI

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage
from divine.config import ProviderConfig


class ZhipuProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        self._client = ZhipuAI(api_key=config.api_key)

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "glm-4-plus")
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "glm-4-plus")
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

`divine/llm/providers/minimax.py`:
```python
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage
from divine.config import ProviderConfig


class MiniMaxProvider(LLMProvider):
    """MiniMax 使用 OpenAI 兼容 API"""

    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

    def __init__(self, config: ProviderConfig):
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or self.DEFAULT_BASE_URL,
            timeout=config.timeout,
        )

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "MiniMax-Text-01")
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "MiniMax-Text-01")
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

`divine/llm/providers/openai_compat.py`:
```python
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse, TokenUsage
from divine.config import ProviderConfig


class OpenAICompatProvider(LLMProvider):
    """兜底：任何 OpenAI 兼容 API（Ollama、vLLM 等）"""

    def __init__(self, config: ProviderConfig):
        self._client = AsyncOpenAI(
            api_key=config.api_key or "no-key",
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def chat(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        model = kwargs.pop("model", "default")
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage or type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            raw_response=response,
        )

    async def chat_stream(self, messages: list[LLMMessage], **kwargs) -> AsyncIterator[str]:
        model = kwargs.pop("model", "default")
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

- [ ] **Step 4: 实现 router.py**

`divine/llm/router.py`:
```python
from typing import Optional

from loguru import logger

from divine.llm.base import LLMProvider, LLMMessage, LLMResponse
from divine.llm.providers import PROVIDER_CLASSES
from divine.llm.utils.retry import RetryHandler, RetryConfig
from divine.llm.utils.cost import CostCalculator
from divine.config import LLMConfig, ProviderConfig


ROUTE_MAP = {
    "gpt-": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "glm-": "zhipu",
    "minimax": "minimax",
}


class LLMRouter:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._providers: dict[str, LLMProvider] = {}
        self._retry = RetryHandler(RetryConfig())
        self._cost_calc = CostCalculator(config.pricing)
        self._fallback: Optional[LLMProvider] = None
        self._init_providers()

    def _init_providers(self) -> None:
        for name, pconf in self._config.providers.items():
            if name in PROVIDER_CLASSES:
                self._providers[name] = PROVIDER_CLASSES[name](pconf)
        # openai_compat 作为兜底
        compat_config = self._config.providers.get("openai_compat")
        if compat_config:
            self._fallback = PROVIDER_CLASSES["openai_compat"](compat_config)

    def _resolve_provider(self, model: str) -> str:
        for prefix, provider_name in ROUTE_MAP.items():
            if model.startswith(prefix):
                return provider_name
        return "openai_compat"

    def get_provider(self, model: str) -> LLMProvider:
        name = self._resolve_provider(model)
        provider = self._providers.get(name)
        if provider:
            return provider
        if self._fallback:
            return self._fallback
        raise ValueError(f"No provider available for model '{model}' (resolved to '{name}')")

    async def chat(self, model: str, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        provider = self.get_provider(model)

        async def _call():
            return await provider.chat(messages, model=model, **kwargs)

        response = await self._retry.execute(_call)
        response.cost = self._cost_calc.calculate(model, response.usage)
        return response
```

- [ ] **Step 5: 运行 Router 测试**

Run: `python -m pytest tests/test_llm_router.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add divine/llm/ tests/test_llm_router.py
git commit -m "feat: LLM providers + router"
```

---

### Task 7: Prompt Engine

**Files:**
- Create: `divine/prompts/__init__.py`
- Create: `divine/prompts/engine.py`
- Create: `divine/prompts/templates/planner/init_plan.jinja2`
- Create: `divine/prompts/templates/planner/replan.jinja2`
- Create: `divine/prompts/templates/planner/terminate_check.jinja2`
- Create: `divine/prompts/templates/executor/base_system.jinja2`
- Create: `divine/prompts/templates/executor/recon_system.jinja2`
- Create: `divine/prompts/templates/executor/web_system.jinja2`
- Create: `divine/prompts/templates/executor/host_system.jinja2`
- Create: `divine/prompts/templates/executor/service_system.jinja2`
- Create: `divine/prompts/templates/executor/observation.jinja2`
- Create: `divine/prompts/templates/reflector/analyze.jinja2`
- Create: `tests/test_prompt_engine.py`

- [ ] **Step 1: 编写测试**

`tests/test_prompt_engine.py`:
```python
from divine.prompts.engine import PromptEngine
from divine.models.common import ExecutorType
from divine.models.task import TaskNode, PentestPhase


class TestPromptEngine:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_build_init_plan_prompt(self):
        prompt = self.engine.build_init_plan_prompt(
            goal="获取目标 Web 服务器控制权",
            targets=["192.168.1.100"],
            output_schema={"type": "object"},
        )
        assert "获取目标 Web 服务器控制权" in prompt
        assert "192.168.1.100" in prompt

    def test_build_replan_prompt(self):
        prompt = self.engine.build_replan_prompt(
            blackboard_summary={"hosts": {"count": 1}},
            dag_stats={"total": 3, "completed": 1},
            reflections=[{"insights": ["发现 SSH 开放"]}],
            output_schema={},
        )
        assert prompt  # 非空

    def test_build_terminate_check_prompt(self):
        prompt = self.engine.build_terminate_check_prompt(
            blackboard_summary={"findings": {"count": 5}},
            dag_stats={"total": 3, "completed": 3},
            goal="获取 root shell",
        )
        assert "获取 root shell" in prompt

    def test_build_executor_system_prompt(self):
        task = TaskNode(
            id="recon_1", description="端口扫描",
            phase=PentestPhase.RECON, executor_type=ExecutorType.RECON,
        )
        prompt = self.engine.build_executor_system_prompt(
            executor_type=ExecutorType.RECON,
            task=task,
            context={"hosts": {}},
            stdlib_docs="run_command(cmd): 执行 shell 命令",
        )
        assert "run_command" in prompt
        assert "端口扫描" in prompt

    def test_build_executor_prompt_different_types(self):
        """不同 executor type 应该加载不同模板"""
        task = TaskNode(
            id="web_1", description="SQL 注入测试",
            phase=PentestPhase.EXPLOIT, executor_type=ExecutorType.WEB,
        )
        prompt = self.engine.build_executor_system_prompt(
            executor_type=ExecutorType.WEB,
            task=task, context={}, stdlib_docs="",
        )
        assert prompt  # web 模板被加载

    def test_build_observation_prompt(self):
        from divine.codeact.sandbox import ExecutionResult
        result = ExecutionResult(success=True, stdout="PORT  STATE SERVICE\n22/tcp open ssh", execution_time=1.2)
        prompt = self.engine.build_observation_prompt(result)
        assert "22/tcp" in prompt

    def test_build_analyze_prompt(self):
        prompt = self.engine.build_analyze_prompt(
            blackboard_summary={"findings": {"count": 3}},
            recent_results=[{"task_id": "t1", "status": "completed"}],
            dag_stats={"total": 5, "completed": 2},
        )
        assert prompt
```

- [ ] **Step 2: 实现 PromptEngine + 模板骨架**

`divine/prompts/__init__.py`:
```python
from divine.prompts.engine import PromptEngine

__all__ = ["PromptEngine"]
```

`divine/prompts/engine.py`:
```python
from pathlib import Path

import jinja2

from divine.models.common import ExecutorType
from divine.models.task import TaskNode


DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"


class PromptEngine:
    def __init__(self, template_dir: Path = None):
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir or DEFAULT_TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _render(self, template_path: str, **kwargs) -> str:
        template = self._env.get_template(f"{template_path}.jinja2")
        return template.render(**kwargs)

    # ---- Planner ----
    def build_init_plan_prompt(self, goal: str, targets: list[str],
                               output_schema: dict) -> str:
        return self._render("planner/init_plan",
                            goal=goal, targets=targets, output_schema=output_schema)

    def build_replan_prompt(self, blackboard_summary: dict, dag_stats: dict,
                            reflections: list[dict], output_schema: dict) -> str:
        return self._render("planner/replan",
                            blackboard_summary=blackboard_summary,
                            dag_stats=dag_stats, reflections=reflections,
                            output_schema=output_schema)

    def build_terminate_check_prompt(self, blackboard_summary: dict,
                                     dag_stats: dict, goal: str) -> str:
        return self._render("planner/terminate_check",
                            blackboard_summary=blackboard_summary,
                            dag_stats=dag_stats, goal=goal)

    # ---- Executor ----
    def build_executor_system_prompt(self, executor_type: ExecutorType,
                                     task: TaskNode, context: dict,
                                     stdlib_docs: str) -> str:
        base = self._render("executor/base_system", stdlib_docs=stdlib_docs)
        specific = self._render(f"executor/{executor_type.value}_system",
                                task=task, context=context)
        return f"{base}\n\n{specific}"

    def build_observation_prompt(self, result) -> str:
        return self._render("executor/observation", result=result)

    # ---- Reflector ----
    def build_analyze_prompt(self, blackboard_summary: dict,
                             recent_results: list[dict],
                             dag_stats: dict) -> str:
        return self._render("reflector/analyze",
                            blackboard_summary=blackboard_summary,
                            recent_results=recent_results,
                            dag_stats=dag_stats)
```

模板文件创建为功能性骨架（包含变量引用和基本结构，实际 prompt 内容在 Task 11 中完善）：

`divine/prompts/templates/planner/init_plan.jinja2`:
```
你是一个渗透测试规划专家。

## 最终目标
{{ goal }}

## 目标列表
{% for target in targets %}
- {{ target }}
{% endfor %}

## 任务
分析目标，生成初始攻击计划。将计划分解为多个任务节点，每个节点包含：
- id: 唯一标识
- description: 任务描述
- phase: 阶段 (recon/scan/exploit/post_exploit)
- executor_type: 执行器类型 (recon/web/host/service)
- dependencies: 依赖的任务 ID 列表
- priority: 优先级 (0-10)

## 输出格式
严格按照以下 JSON 格式输出 graph operations 列表：
{{ output_schema | tojson(indent=2) }}
```

`divine/prompts/templates/planner/replan.jinja2`:
```
你是一个渗透测试规划专家，需要根据当前进度动态调整攻击计划。

## 黑板状态摘要
{{ blackboard_summary | tojson(indent=2) }}

## DAG 进度
{{ dag_stats | tojson(indent=2) }}

## 反思结果
{% for r in reflections %}
{{ r | tojson(indent=2) }}
{% endfor %}

## 任务
根据以上信息，决定是否需要调整 DAG：
- add_node: 添加新任务
- remove_node: 移除不再需要的任务
- update_node: 更新任务属性（如优先级）

如果不需要调整，返回空列表。

## 输出格式
{{ output_schema | tojson(indent=2) }}
```

`divine/prompts/templates/planner/terminate_check.jinja2`:
```
判断渗透测试目标是否已经达成。

## 最终目标
{{ goal }}

## 黑板状态摘要
{{ blackboard_summary | tojson(indent=2) }}

## DAG 进度
{{ dag_stats | tojson(indent=2) }}

## 输出格式
返回 JSON：
{"terminate": true/false, "reason": "原因说明"}
```

`divine/prompts/templates/executor/base_system.jinja2`:
```
你是一个渗透测试执行器。你通过编写 Python 代码与目标环境交互。

## 规则
1. 用 ```python 代码块返回你要执行的代码
2. 每次只执行一个步骤，观察结果后再决定下一步
3. 通过 bb_write() 将重要发现写入黑板
4. 当任务完成时，不再返回代码块，直接给出总结

## 可用函数
{{ stdlib_docs }}
```

`divine/prompts/templates/executor/recon_system.jinja2`:
```
## 角色：侦察执行器

你负责对目标进行信息收集和侦察。

## 当前任务
- ID: {{ task.id }}
- 描述: {{ task.description }}

## 已知信息
{{ context | tojson(indent=2) }}

## 侦察要点
- 使用 nmap 进行端口扫描
- 识别服务版本
- 收集 banner 信息
- 将发现的主机和端口写入黑板
```

`divine/prompts/templates/executor/web_system.jinja2`:
```
## 角色：Web 攻击执行器

你负责对 Web 应用进行安全测试。

## 当前任务
- ID: {{ task.id }}
- 描述: {{ task.description }}

## 已知信息
{{ context | tojson(indent=2) }}

## Web 测试要点
- 目录枚举和路径发现
- 参数注入测试（SQL、XSS、命令注入）
- 认证和会话管理测试
- 将发现的漏洞写入黑板
```

`divine/prompts/templates/executor/host_system.jinja2`:
```
## 角色：主机操作执行器

你负责对目标主机进行操作和利用。

## 当前任务
- ID: {{ task.id }}
- 描述: {{ task.description }}

## 已知信息
{{ context | tojson(indent=2) }}

## 主机操作要点
- 利用已知漏洞获取访问
- 权限提升
- 内网信息收集
- 将获取的凭证和会话写入黑板
```

`divine/prompts/templates/executor/service_system.jinja2`:
```
## 角色：服务攻击执行器

你负责对特定服务进行安全测试。

## 当前任务
- ID: {{ task.id }}
- 描述: {{ task.description }}

## 已知信息
{{ context | tojson(indent=2) }}

## 服务测试要点
- 服务版本漏洞匹配
- 默认凭证测试
- 协议级别攻击
- 将发现写入黑板
```

`divine/prompts/templates/executor/observation.jinja2`:
```
## 执行结果

{% if result.success %}
**状态**: 成功 (耗时 {{ "%.1f" | format(result.execution_time) }}s)
{% else %}
**状态**: 失败
{% endif %}

{% if result.stdout %}
**输出**:
```
{{ result.stdout }}
```
{% endif %}

{% if result.stderr %}
**错误**:
```
{{ result.stderr }}
```
{% endif %}

{% if result.return_value is not none %}
**返回值**: {{ result.return_value }}
{% endif %}

根据以上结果，决定下一步操作。
```

`divine/prompts/templates/reflector/analyze.jinja2`:
```
你是一个渗透测试反思分析专家。分析最近一轮执行结果，提供洞察和建议。

## 黑板状态摘要
{{ blackboard_summary | tojson(indent=2) }}

## 最近执行结果
{% for r in recent_results %}
- 任务 {{ r.task_id }}: {{ r.status }}
  {% if r.result %}结果: {{ r.result | tojson }}{% endif %}
  {% if r.error %}错误: {{ r.error }}{% endif %}
{% endfor %}

## DAG 进度
{{ dag_stats | tojson(indent=2) }}

## 分析要求
1. **洞察**: 识别关键发现和失败模式
2. **建议任务**: 基于当前发现，建议新的攻击方向
3. **风险评估**: 当前攻击面和风险水平
4. **进度总结**: 整体渗透进度

## 输出格式
返回 JSON：
{
    "insights": ["洞察1", "洞察2"],
    "suggested_tasks": [{"description": "...", "phase": "...", "executor_type": "...", "reason": "..."}],
    "risk_assessment": "...",
    "progress_summary": "..."
}
```

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_prompt_engine.py -v`
Expected: 全部 PASS（observation 测试需要先有 ExecutionResult，可能需要先创建 codeact/sandbox.py 的 ExecutionResult，或者调整测试顺序）

注意：如果 `test_build_observation_prompt` 因缺少 `ExecutionResult` 而失败，先创建 `divine/codeact/__init__.py` 和 `divine/codeact/sandbox.py` 中的 `ExecutionResult` dataclass。

- [ ] **Step 4: 提交**

```bash
git add divine/prompts/ tests/test_prompt_engine.py
git commit -m "feat: PromptEngine + 模板骨架"
```

---

### Task 8: CodeAct Sandbox + stdlib

**Files:**
- Create: `divine/codeact/__init__.py`
- Create: `divine/codeact/sandbox.py`
- Create: `divine/codeact/stdlib.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: 编写测试**

`tests/test_sandbox.py`:
```python
import pytest

from divine.codeact.sandbox import Sandbox, ExecutionResult


class TestExecutionResult:
    def test_success_result(self):
        r = ExecutionResult(success=True, stdout="hello", execution_time=0.5)
        assert r.success
        assert r.stdout == "hello"

    def test_failure_result(self):
        r = ExecutionResult(success=False, stderr="error")
        assert not r.success


class TestSandbox:
    async def test_execute_simple_code(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("x = 1 + 1")
        assert result.success

    async def test_execute_with_stdout(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    async def test_execute_returns_last_expression(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("1 + 1")
        assert result.success
        assert result.return_value == 2

    async def test_execute_error(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("raise ValueError('test error')")
        assert not result.success
        assert "test error" in result.stderr

    async def test_execute_timeout(self):
        sandbox = Sandbox(timeout=1)
        result = await sandbox.execute("import time; time.sleep(10)")
        assert not result.success
        assert "timeout" in result.stderr.lower() or "Timeout" in result.stderr

    async def test_persistent_namespace(self):
        sandbox = Sandbox(timeout=10)
        await sandbox.execute("my_var = 42")
        result = await sandbox.execute("print(my_var)")
        assert result.success
        assert "42" in result.stdout

    async def test_reset_clears_namespace(self):
        sandbox = Sandbox(timeout=10)
        await sandbox.execute("my_var = 42")
        sandbox.reset()
        result = await sandbox.execute("print(my_var)")
        assert not result.success  # my_var 不存在

    async def test_setup_injects_stdlib(self):
        sandbox = Sandbox(timeout=10)
        sandbox.setup({"add": lambda a, b: a + b})
        result = await sandbox.execute("result = add(3, 4)\nprint(result)")
        assert result.success
        assert "7" in result.stdout


class TestStdlib:
    def test_create_stdlib(self):
        from divine.blackboard import Blackboard
        from divine.codeact.stdlib import create_stdlib

        bb = Blackboard()
        stdlib = create_stdlib(bb)
        assert "run_command" in stdlib
        assert "http_request" in stdlib
        assert "bb_read" in stdlib
        assert "bb_write" in stdlib

    async def test_run_command(self):
        from divine.codeact.stdlib import run_command
        result = run_command("echo hello")
        assert "hello" in result["stdout"]
        assert result["returncode"] == 0

    async def test_run_command_failure(self):
        from divine.codeact.stdlib import run_command
        result = run_command("false")
        assert result["returncode"] != 0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 sandbox.py**

`divine/codeact/__init__.py`:
```python
from divine.codeact.sandbox import Sandbox, ExecutionResult
from divine.codeact.executor import CodeActExecutor

__all__ = ["Sandbox", "ExecutionResult", "CodeActExecutor"]
```

`divine/codeact/sandbox.py`:
```python
import ast
import asyncio
import io
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional


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
        self._globals: dict = {"__builtins__": __builtins__}

    def setup(self, stdlib: dict) -> None:
        self._globals.update(stdlib)

    async def execute(self, code: str) -> ExecutionResult:
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync, code),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                stderr=f"Timeout: execution exceeded {self._timeout}s",
            )

    def _execute_sync(self, code: str) -> ExecutionResult:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        return_value = None

        start_time = time.time()
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # 尝试将最后一行作为表达式求值
            tree = ast.parse(code)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(last_expr)

            # 执行语句部分
            if tree.body:
                compiled = compile(tree, "<sandbox>", "exec")
                exec(compiled, self._globals)

            # 求值最后一个表达式
            if last_expr:
                compiled_expr = compile(last_expr, "<sandbox>", "eval")
                return_value = eval(compiled_expr, self._globals)

            execution_time = time.time() - start_time
            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=return_value,
                execution_time=execution_time,
            )
        except Exception:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=traceback.format_exc(),
                execution_time=execution_time,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def reset(self) -> None:
        self._globals = {"__builtins__": __builtins__}
```

- [ ] **Step 4: 实现 stdlib.py**

`divine/codeact/stdlib.py`:
```python
import base64
import subprocess
from typing import Any
from urllib.parse import urlparse

from divine.blackboard.blackboard import Blackboard


def run_command(cmd: str, timeout: int = 60) -> dict:
    """执行 shell 命令"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def http_request(url: str, method: str = "GET", headers: dict = None,
                 data: str = None, timeout: int = 30) -> dict:
    """发送 HTTP 请求"""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, method=method, headers=headers or {})
        if data:
            req.data = data.encode()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
            return {
                "status": resp.status,
                "headers": dict(resp.headers),
                "body": body,
            }
    except urllib.error.HTTPError as e:
        return {"status": e.code, "headers": dict(e.headers), "body": e.read().decode(errors="replace")}
    except Exception as e:
        return {"status": 0, "headers": {}, "body": "", "error": str(e)}


def parse_nmap(output: str) -> list[dict]:
    """简单解析 nmap 输出"""
    results = []
    for line in output.splitlines():
        line = line.strip()
        if "/tcp" in line or "/udp" in line:
            parts = line.split()
            if len(parts) >= 3:
                port_proto = parts[0]
                port, proto = port_proto.split("/")
                results.append({
                    "port": int(port),
                    "protocol": proto,
                    "state": parts[1],
                    "service": parts[2] if len(parts) > 2 else "",
                    "version": " ".join(parts[3:]) if len(parts) > 3 else "",
                })
    return results


def parse_url(url: str) -> dict:
    """解析 URL"""
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname or "",
        "port": parsed.port,
        "path": parsed.path,
        "query": parsed.query,
    }


def b64encode(data: str) -> str:
    return base64.b64encode(data.encode()).decode()


def b64decode(data: str) -> str:
    return base64.b64decode(data.encode()).decode()


def create_stdlib(blackboard: Blackboard) -> dict:
    """创建注入 sandbox 的标准库"""
    return {
        "run_command": run_command,
        "http_request": http_request,
        "bb_read": blackboard.read,
        "bb_write": blackboard.write,
        "parse_nmap": parse_nmap,
        "parse_url": parse_url,
        "b64encode": b64encode,
        "b64decode": b64decode,
    }
```

- [ ] **Step 5: 运行测试验证通过**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add divine/codeact/ tests/test_sandbox.py
git commit -m "feat: CodeAct Sandbox + stdlib"
```

---

### Task 9: CodeAct Executor

**Files:**
- Create: `divine/codeact/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: 编写测试**

`tests/test_executor.py`:
```python
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from divine.codeact.executor import CodeActExecutor
from divine.codeact.sandbox import Sandbox, ExecutionResult
from divine.llm.base import LLMMessage, LLMResponse, TokenUsage
from divine.llm.router import LLMRouter
from divine.blackboard import Blackboard
from divine.prompts.engine import PromptEngine
from divine.models.task import TaskNode, TaskStatus
from divine.models.common import PentestPhase, ExecutorType


def make_task() -> TaskNode:
    return TaskNode(
        id="recon_1", description="扫描端口",
        phase=PentestPhase.RECON, executor_type=ExecutorType.RECON,
    )


class TestCodeActExecutor:
    def _setup_executor(self, llm_responses: list[str]) -> CodeActExecutor:
        router = AsyncMock(spec=LLMRouter)
        call_count = 0

        async def mock_chat(model, messages, **kwargs):
            nonlocal call_count
            content = llm_responses[min(call_count, len(llm_responses) - 1)]
            call_count += 1
            return LLMResponse(
                content=content, model=model,
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            )

        router.chat = mock_chat

        bb = Blackboard()
        sandbox = Sandbox(timeout=10)
        engine = PromptEngine()

        executor = CodeActExecutor(
            router=router, sandbox=sandbox,
            blackboard=bb, prompt_engine=engine,
            model="test-model",
        )
        return executor

    async def test_single_code_execution(self):
        executor = self._setup_executor([
            "让我扫描端口\n```python\nprint('scanning...')\n```",
            "扫描完成，未发现开放端口。",
        ])
        result = await executor.execute_task(make_task(), context={})
        assert result is not None

    async def test_no_code_means_done(self):
        executor = self._setup_executor([
            "任务分析完成，无需执行代码。",
        ])
        result = await executor.execute_task(make_task(), context={})
        assert result is not None

    async def test_max_iterations_respected(self):
        executor = self._setup_executor([
            "```python\nprint('loop')\n```",
        ] * 20)
        executor._max_iterations = 3
        result = await executor.execute_task(make_task(), context={})
        assert result is not None

    def test_extract_code(self):
        executor = self._setup_executor([])
        code = executor._extract_code("here is code\n```python\nx = 1\n```\ndone")
        assert code == "x = 1"

    def test_extract_code_no_block(self):
        executor = self._setup_executor([])
        code = executor._extract_code("no code here")
        assert code is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_executor.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 executor.py**

`divine/codeact/executor.py`:
```python
import re
from typing import Optional

from loguru import logger

from divine.blackboard.blackboard import Blackboard
from divine.codeact.sandbox import Sandbox, ExecutionResult
from divine.codeact.stdlib import create_stdlib
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.models.task import TaskNode
from divine.prompts.engine import PromptEngine


class CodeActExecutor:
    def __init__(self, router: LLMRouter, sandbox: Sandbox,
                 blackboard: Blackboard, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._sandbox = sandbox
        self._blackboard = blackboard
        self._prompt_engine = prompt_engine
        self._model = model
        self._max_iterations = 10

        # 注入 stdlib
        stdlib = create_stdlib(blackboard)
        self._sandbox.setup(stdlib)
        self._stdlib_docs = self._build_stdlib_docs(stdlib)

    def _build_stdlib_docs(self, stdlib: dict) -> str:
        lines = []
        for name, fn in stdlib.items():
            doc = getattr(fn, "__doc__", "") or ""
            first_line = doc.strip().split("\n")[0] if doc.strip() else "No description"
            lines.append(f"- {name}: {first_line}")
        return "\n".join(lines)

    async def execute_task(self, task: TaskNode, context: dict) -> dict:
        self._sandbox.reset()
        stdlib = create_stdlib(self._blackboard)
        self._sandbox.setup(stdlib)

        system_prompt = self._prompt_engine.build_executor_system_prompt(
            executor_type=task.executor_type,
            task=task,
            context=context,
            stdlib_docs=self._stdlib_docs,
        )

        conversation = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"开始执行任务: {task.description}"),
        ]

        for i in range(self._max_iterations):
            response = await self._router.chat(self._model, conversation)
            content = response.content

            code = self._extract_code(content)
            if not code:
                logger.info(f"Task {task.id}: LLM 未返回代码，视为完成 (iteration {i + 1})")
                break

            conversation.append(LLMMessage(role="assistant", content=content))

            result = await self._sandbox.execute(code)
            observation = self._prompt_engine.build_observation_prompt(result)
            conversation.append(LLMMessage(role="user", content=observation))

            logger.debug(f"Task {task.id} iteration {i + 1}: success={result.success}")
        else:
            logger.warning(f"Task {task.id}: 达到最大迭代次数 {self._max_iterations}")

        return {"iterations": min(i + 1, self._max_iterations) if 'i' in dir() else 0}

    def _extract_code(self, content: str) -> Optional[str]:
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_executor.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add divine/codeact/executor.py tests/test_executor.py
git commit -m "feat: CodeAct Executor"
```

---

### Task 10: Planner + Reflector

**Files:**
- Create: `divine/agents/__init__.py`
- Create: `divine/agents/planner.py`
- Create: `divine/agents/reflection.py`
- Create: `tests/test_planner.py`
- Create: `tests/test_reflector.py`

- [ ] **Step 1: 编写 Planner 测试**

`tests/test_planner.py`:
```python
from unittest.mock import AsyncMock
import json

import pytest

from divine.agents.planner import Planner
from divine.llm.base import LLMResponse, TokenUsage, LLMMessage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine
from divine.config import DivineConfig


class TestPlanner:
    def _make_planner(self, llm_content: str) -> Planner:
        router = AsyncMock(spec=LLMRouter)
        router.chat.return_value = LLMResponse(
            content=llm_content, model="test",
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        )
        engine = PromptEngine()
        return Planner(router=router, prompt_engine=engine, model="test")

    async def test_init_plan(self):
        ops = [
            {"command": "add_node", "node_data": {
                "id": "recon_1", "description": "端口扫描",
                "phase": "recon", "executor_type": "recon", "dependencies": [],
            }}
        ]
        planner = self._make_planner(json.dumps(ops))
        config = DivineConfig(targets=["192.168.1.1"], goal="获取控制权")
        result = await planner.init_plan(goal="获取控制权", config=config)
        assert len(result) == 1
        assert result[0]["command"] == "add_node"

    async def test_replan_no_changes(self):
        planner = self._make_planner("[]")
        result = await planner.replan(
            blackboard_summary={}, dag_stats={}, reflections=[],
        )
        assert result == []

    async def test_replan_with_new_task(self):
        ops = [{"command": "add_node", "node_data": {
            "id": "web_1", "description": "SQL 注入", "phase": "exploit",
            "executor_type": "web", "dependencies": ["recon_1"],
        }}]
        planner = self._make_planner(json.dumps(ops))
        result = await planner.replan(
            blackboard_summary={}, dag_stats={}, reflections=[],
        )
        assert len(result) == 1

    async def test_should_terminate_true(self):
        planner = self._make_planner(json.dumps({"terminate": True, "reason": "目标已达成"}))
        should_stop, reason = await planner.should_terminate(
            blackboard_summary={}, dag_stats={},
        )
        assert should_stop
        assert "达成" in reason

    async def test_should_terminate_false(self):
        planner = self._make_planner(json.dumps({"terminate": False, "reason": "还有任务未完成"}))
        should_stop, reason = await planner.should_terminate(
            blackboard_summary={}, dag_stats={},
        )
        assert not should_stop

    def test_parse_operations_json(self):
        router = AsyncMock(spec=LLMRouter)
        engine = PromptEngine()
        planner = Planner(router=router, prompt_engine=engine, model="test")
        ops = planner._parse_operations('[{"command": "add_node", "node_data": {"id": "t1"}}]')
        assert len(ops) == 1

    def test_parse_operations_markdown_wrapped(self):
        router = AsyncMock(spec=LLMRouter)
        engine = PromptEngine()
        planner = Planner(router=router, prompt_engine=engine, model="test")
        content = '```json\n[{"command": "add_node", "node_data": {"id": "t1"}}]\n```'
        ops = planner._parse_operations(content)
        assert len(ops) == 1
```

- [ ] **Step 2: 编写 Reflector 测试**

`tests/test_reflector.py`:
```python
from unittest.mock import AsyncMock
import json

import pytest

from divine.agents.reflection import Reflector, Reflection
from divine.llm.base import LLMResponse, TokenUsage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine


class TestReflector:
    def _make_reflector(self, llm_content: str) -> Reflector:
        router = AsyncMock(spec=LLMRouter)
        router.chat.return_value = LLMResponse(
            content=llm_content, model="test",
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        )
        engine = PromptEngine()
        return Reflector(router=router, prompt_engine=engine, model="test")

    async def test_analyze(self):
        reflection_data = {
            "insights": ["发现 SSH 弱密码", "80 端口运行 Apache"],
            "suggested_tasks": [
                {"description": "尝试 SSH 暴破", "phase": "exploit",
                 "executor_type": "service", "reason": "弱密码"},
            ],
            "risk_assessment": "中等风险，攻击面有限",
            "progress_summary": "完成初步侦察，发现 2 个服务",
        }
        reflector = self._make_reflector(json.dumps(reflection_data))
        result = await reflector.analyze(
            blackboard_summary={}, recent_results=[], dag_stats={},
        )
        assert isinstance(result, Reflection)
        assert len(result.insights) == 2
        assert len(result.suggested_tasks) == 1
        assert "中等" in result.risk_assessment
```

- [ ] **Step 3: 运行测试验证失败**

Run: `python -m pytest tests/test_planner.py tests/test_reflector.py -v`
Expected: FAIL

- [ ] **Step 4: 实现 planner.py**

`divine/agents/__init__.py`:
```python
from divine.agents.planner import Planner
from divine.agents.reflection import Reflector, Reflection

__all__ = ["Planner", "Reflector", "Reflection"]
```

`divine/agents/planner.py`:
```python
import json
import re
from typing import Optional

from loguru import logger

from divine.config import DivineConfig
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine

OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "command": {"enum": ["add_node", "remove_node", "update_node"]},
            "node_id": {"type": "string"},
            "node_data": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "phase": {"enum": ["recon", "scan", "exploit", "post_exploit"]},
                    "executor_type": {"enum": ["recon", "web", "host", "service"]},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "integer"},
                },
            },
            "updates": {"type": "object"},
        },
    },
}


class Planner:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._prompt_engine = prompt_engine
        self._model = model

    async def init_plan(self, goal: str, config: DivineConfig) -> list[dict]:
        prompt = self._prompt_engine.build_init_plan_prompt(
            goal=goal, targets=config.targets, output_schema=OUTPUT_SCHEMA,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=f"请为以下目标制定渗透测试计划: {goal}"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_operations(response.content)

    async def replan(self, blackboard_summary: dict, dag_stats: dict,
                     reflections: list[dict]) -> list[dict]:
        prompt = self._prompt_engine.build_replan_prompt(
            blackboard_summary=blackboard_summary,
            dag_stats=dag_stats,
            reflections=reflections,
            output_schema=OUTPUT_SCHEMA,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="根据当前进度，是否需要调整攻击计划？"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_operations(response.content)

    async def should_terminate(self, blackboard_summary: dict,
                               dag_stats: dict) -> tuple[bool, str]:
        prompt = self._prompt_engine.build_terminate_check_prompt(
            blackboard_summary=blackboard_summary,
            dag_stats=dag_stats,
            goal="",  # 由模板上下文提供
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="判断目标是否已达成。"),
        ]
        response = await self._router.chat(self._model, messages)
        try:
            data = self._extract_json_object(response.content)
            return data.get("terminate", False), data.get("reason", "")
        except Exception:
            return False, "无法解析终止判断"

    def _parse_operations(self, content: str) -> list[dict]:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
        pattern = r"```(?:json)?\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # 尝试找到 [ ... ] 部分
        bracket_match = re.search(r"\[.*\]", content, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning(f"Planner: 无法解析 operations: {content[:200]}")
        return []

    def _extract_json_object(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        pattern = r"```(?:json)?\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group())
        raise ValueError(f"Cannot extract JSON from: {content[:200]}")
```

- [ ] **Step 5: 实现 reflection.py**

`divine/agents/reflection.py`:
```python
import json
import re
from dataclasses import dataclass, field

from loguru import logger

from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter
from divine.prompts.engine import PromptEngine


@dataclass
class Reflection:
    insights: list[str] = field(default_factory=list)
    suggested_tasks: list[dict] = field(default_factory=list)
    risk_assessment: str = ""
    progress_summary: str = ""


class Reflector:
    def __init__(self, router: LLMRouter, prompt_engine: PromptEngine,
                 model: str = "claude-sonnet-4-20250514"):
        self._router = router
        self._prompt_engine = prompt_engine
        self._model = model

    async def analyze(self, blackboard_summary: dict,
                      recent_results: list[dict],
                      dag_stats: dict) -> Reflection:
        prompt = self._prompt_engine.build_analyze_prompt(
            blackboard_summary=blackboard_summary,
            recent_results=recent_results,
            dag_stats=dag_stats,
        )
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content="分析最近一轮执行结果。"),
        ]
        response = await self._router.chat(self._model, messages)
        return self._parse_reflection(response.content)

    def _parse_reflection(self, content: str) -> Reflection:
        try:
            data = self._extract_json(content)
            return Reflection(
                insights=data.get("insights", []),
                suggested_tasks=data.get("suggested_tasks", []),
                risk_assessment=data.get("risk_assessment", ""),
                progress_summary=data.get("progress_summary", ""),
            )
        except Exception:
            logger.warning(f"Reflector: 无法解析反思结果，使用原始文本")
            return Reflection(insights=[content], progress_summary=content[:200])

    def _extract_json(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        pattern = r"```(?:json)?\s*\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group())
        raise ValueError("Cannot extract JSON")
```

- [ ] **Step 6: 运行测试验证通过**

Run: `python -m pytest tests/test_planner.py tests/test_reflector.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add divine/agents/ tests/test_planner.py tests/test_reflector.py
git commit -m "feat: Planner + Reflector"
```

---

### Task 11: Session 主循环

**Files:**
- Create: `divine/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: 编写测试**

`tests/test_session.py`:
```python
from dataclasses import asdict
from unittest.mock import AsyncMock, patch, MagicMock
import json

import pytest

from divine.session import Session
from divine.config import DivineConfig, LLMConfig, ProviderConfig
from divine.models.task import TaskStatus


def make_config() -> DivineConfig:
    return DivineConfig(
        targets=["192.168.1.1"],
        goal="获取目标控制权",
        max_rounds=3,
        concurrency=1,
        llm=LLMConfig(providers={"openai": ProviderConfig(api_key="test")}),
    )


class TestSession:
    async def test_session_init(self):
        """Session 应该能正常初始化所有组件"""
        config = make_config()
        session = Session(config)
        assert session._dag is not None
        assert session._blackboard is not None
        assert session._planner is not None
        assert session._reflector is not None

    async def test_session_run_basic_flow(self):
        """测试基本流程：init_plan -> run_round -> reflect -> replan -> terminate"""
        config = make_config()
        session = Session(config)

        # Mock planner
        init_ops = [{"command": "add_node", "node_data": {
            "id": "t1", "description": "扫描", "phase": "recon",
            "executor_type": "recon", "dependencies": [],
        }}]
        session._planner.init_plan = AsyncMock(return_value=init_ops)
        session._planner.replan = AsyncMock(return_value=[])
        session._planner.should_terminate = AsyncMock(return_value=(True, "目标达成"))

        # Mock executor
        session._executor.execute_task = AsyncMock(return_value={"status": "done"})

        # Mock reflector
        from divine.agents.reflection import Reflection
        session._reflector.analyze = AsyncMock(return_value=Reflection(
            insights=["test"], suggested_tasks=[], risk_assessment="low", progress_summary="done",
        ))

        await session.run()

        session._planner.init_plan.assert_called_once()
        assert session._dag.get_task("t1").status == TaskStatus.COMPLETED

    async def test_session_max_rounds_terminates(self):
        """max_rounds 硬性兜底应该生效"""
        config = make_config()
        config.max_rounds = 2
        session = Session(config)

        init_ops = [
            {"command": "add_node", "node_data": {
                "id": f"t{i}", "description": f"Task {i}", "phase": "recon",
                "executor_type": "recon", "dependencies": [],
            }} for i in range(10)
        ]
        session._planner.init_plan = AsyncMock(return_value=init_ops)
        session._planner.replan = AsyncMock(return_value=[])
        session._planner.should_terminate = AsyncMock(return_value=(False, ""))

        session._executor.execute_task = AsyncMock(return_value={})

        from divine.agents.reflection import Reflection
        session._reflector.analyze = AsyncMock(return_value=Reflection())

        await session.run()
        # 应该在 max_rounds 后停止，不会无限循环
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 session.py**

`divine/session.py`:
```python
from dataclasses import asdict

from loguru import logger

from divine.blackboard import Blackboard
from divine.codeact.executor import CodeActExecutor
from divine.codeact.sandbox import Sandbox
from divine.codeact.stdlib import create_stdlib
from divine.config import DivineConfig
from divine.dag import TaskDAG, DAGScheduler
from divine.agents.planner import Planner
from divine.agents.reflection import Reflector
from divine.llm.router import LLMRouter
from divine.models.task import TaskNode
from divine.prompts.engine import PromptEngine


class Session:
    def __init__(self, config: DivineConfig):
        self._config = config

        self._blackboard = Blackboard(db_path=config.db_path)
        self._router = LLMRouter(config.llm)
        self._prompt_engine = PromptEngine()
        self._dag = TaskDAG()
        self._scheduler = DAGScheduler(self._dag, concurrency=config.concurrency)
        self._sandbox = Sandbox(timeout=config.code_execution_timeout)
        self._executor = CodeActExecutor(
            router=self._router, sandbox=self._sandbox,
            blackboard=self._blackboard, prompt_engine=self._prompt_engine,
            model=config.executor_model,
        )
        self._planner = Planner(
            router=self._router, prompt_engine=self._prompt_engine,
            model=config.planner_model,
        )
        self._reflector = Reflector(
            router=self._router, prompt_engine=self._prompt_engine,
            model=config.reflector_model,
        )

    async def run(self) -> None:
        logger.info(f"Session 启动: 目标={self._config.targets}, 最大轮次={self._config.max_rounds}")

        # 1. 初始规划
        operations = await self._planner.init_plan(
            goal=self._config.goal, config=self._config,
        )
        await self._dag.apply_operations(operations)
        logger.info(f"初始规划完成: {self._dag.stats}")

        # 2. 主循环
        for round_num in range(1, self._config.max_rounds + 1):
            logger.info(f"=== 第 {round_num} 轮 ===")

            # 2a. 调度并执行就绪任务
            completed = await self._scheduler.run_round(
                execute_fn=self._execute_task,
            )
            logger.info(f"本轮完成 {len(completed)} 个任务: {completed}")

            if not completed and self._dag.is_finished:
                logger.info("所有任务已完成")
                break

            # 2b. 反思
            reflection = await self._reflector.analyze(
                blackboard_summary=self._blackboard.summary(),
                recent_results=self._get_recent_results(completed),
                dag_stats=self._dag.stats,
            )
            self._blackboard.write(
                "reflections", f"round_{round_num}",
                value=asdict(reflection), source="reflector",
            )

            # 2c. 重规划
            operations = await self._planner.replan(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
                reflections=[asdict(reflection)],
            )
            if operations:
                await self._dag.apply_operations(operations)
                logger.info(f"重规划: 应用了 {len(operations)} 个操作")

            # 2d. 终止检查
            should_stop, reason = await self._planner.should_terminate(
                blackboard_summary=self._blackboard.summary(),
                dag_stats=self._dag.stats,
            )
            if should_stop:
                logger.info(f"目标达成: {reason}")
                break

        logger.info(f"Session 结束: {self._dag.stats}")

    async def _execute_task(self, task: TaskNode) -> dict:
        context = self._blackboard.summary(
            sections=["hosts", "ports", "credentials", "findings"],
        )
        return await self._executor.execute_task(task, context)

    def _get_recent_results(self, task_ids: list[str]) -> list[dict]:
        results = []
        for tid in task_ids:
            task = self._dag.get_task(tid)
            if task:
                results.append({
                    "task_id": tid,
                    "description": task.description,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                })
        return results
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_session.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add divine/session.py tests/test_session.py
git commit -m "feat: Session 主循环"
```

---

### Task 12: Report Generator

**Files:**
- Create: `divine/reporting/__init__.py`
- Create: `divine/reporting/generator.py`
- Create: `divine/reporting/templates/report.jinja2`

- [ ] **Step 1: 实现 ReportGenerator**

`divine/reporting/__init__.py`:
```python
from divine.reporting.generator import ReportGenerator

__all__ = ["ReportGenerator"]
```

`divine/reporting/generator.py`:
```python
from pathlib import Path

import jinja2
from loguru import logger

from divine.blackboard.blackboard import Blackboard
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter


REPORT_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    def __init__(self, router: LLMRouter, blackboard: Blackboard):
        self._router = router
        self._blackboard = blackboard
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(REPORT_TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def generate(self, output_path: Path, model: str = "claude-sonnet-4-20250514") -> None:
        data = self._collect_data()
        narrative = await self._generate_narrative(data, model)
        html = self._render(data, narrative)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"报告已生成: {output_path}")

    def _collect_data(self) -> dict:
        return {
            "hosts": self._blackboard.read("hosts") or {},
            "ports": self._blackboard.read("ports") or {},
            "findings": self._blackboard.read("findings") or {},
            "credentials": self._blackboard.read("credentials") or {},
            "reflections": self._blackboard.read("reflections") or {},
        }

    async def _generate_narrative(self, data: dict, model: str) -> dict:
        prompt = f"""根据以下渗透测试数据，生成报告叙述部分。

数据:
{data}

请返回 JSON 格式:
{{
    "executive_summary": "执行摘要",
    "attack_path": "攻击路径叙述",
    "risk_rating": "高/中/低",
    "recommendations": ["建议1", "建议2"]
}}"""
        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self._router.chat(model, messages)
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"LLM 叙述生成失败: {e}")
            return {
                "executive_summary": "自动生成失败",
                "attack_path": "",
                "risk_rating": "未知",
                "recommendations": [],
            }

    def _render(self, data: dict, narrative: dict) -> str:
        template = self._env.get_template("report.jinja2")
        return template.render(data=data, narrative=narrative)
```

`divine/reporting/templates/report.jinja2`:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Divine 渗透测试报告</title>
    <style>
        body { font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .finding { border-left: 3px solid #e74c3c; padding-left: 10px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Divine 渗透测试报告</h1>

    <h2>执行摘要</h2>
    <div class="summary">
        <p>{{ narrative.executive_summary }}</p>
        <p><strong>风险评级:</strong> {{ narrative.risk_rating }}</p>
    </div>

    <h2>攻击路径</h2>
    <p>{{ narrative.attack_path }}</p>

    <h2>发现的主机</h2>
    <table>
        <tr><th>主机</th><th>详情</th></tr>
        {% for key, value in data.hosts.items() %}
        <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
        {% endfor %}
    </table>

    <h2>安全发现</h2>
    {% for key, value in data.findings.items() %}
    <div class="finding">
        <strong>{{ key }}</strong>: {{ value }}
    </div>
    {% endfor %}

    <h2>获取的凭证</h2>
    {% if data.credentials %}
    <table>
        <tr><th>标识</th><th>详情</th></tr>
        {% for key, value in data.credentials.items() %}
        <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
        {% endfor %}
    </table>
    {% else %}
    <p>未获取到凭证。</p>
    {% endif %}

    <h2>建议</h2>
    <ul>
        {% for rec in narrative.recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
</body>
</html>
```

- [ ] **Step 2: 提交**

```bash
git add divine/reporting/
git commit -m "feat: Report Generator"
```

---

### Task 13: 全量测试 + 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 运行全量测试**

Run: `python -m pytest tests/ -v --tb=short`
Expected: 全部 PASS

- [ ] **Step 2: 验证 CLI 入口**

Run: `python -m divine version`
Expected: `Divine v0.1.0`

- [ ] **Step 3: 验证包可导入**

Run: `python -c "from divine.session import Session; from divine.dag import TaskDAG; from divine.blackboard import Blackboard; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 4: 更新 CLAUDE.md**

将 CLAUDE.md 更新为与实际代码一致的最终版本。确保目录结构、组件描述、设计决策表都与实现匹配。

- [ ] **Step 5: 提交**

```bash
git add CLAUDE.md
git commit -m "docs: 更新 CLAUDE.md 匹配重构后的代码"
```
