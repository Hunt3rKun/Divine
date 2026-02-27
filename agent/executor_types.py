from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


# ============================================================
# 一、枚举类型
# ============================================================

class ExecutorType(Enum):
    """执行器类型"""
    RECON = "recon"
    WEB = "web"
    HOST = "host"


class ExecutorState(Enum):
    """执行器状态机状态"""
    IDLE = "idle"  # 空闲，等待任务
    CLAIMING = "claiming"  # 正在认领任务
    PLANNING = "planning"  # 规划微任务
    EXECUTING = "executing"  # 执行中
    WAITING_COLLAB = "waiting_collab"  # 等待协作响应
    REPORTING = "reporting"  # 生成报告
    BLOCKED = "blocked"  # 阻塞
    FAILED = "failed"  # 失败


class ExecutorActionType(Enum):
    """执行器动作类型"""
    EXECUTE_CODE = "EXECUTE_CODE"
    QUERY_BLACKBOARD = "QUERY_BLACKBOARD"
    COLLABORATION_REQUEST = "COLLABORATION_REQUEST"
    REPORT_DISCOVERY = "REPORT_DISCOVERY"
    TASK_COMPLETE = "TASK_COMPLETE"
    TASK_FAILED = "TASK_FAILED"


class CollaborationType(Enum):
    """协作类型"""
    ASSIST = "ASSIST"  # 请求协助
    PREPARE = "PREPARE"  # 请求准备
    NOTIFY = "NOTIFY"  # 通知


class CollaborationStatus(Enum):
    """协作状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class CollaborationPriority(Enum):
    """协作优先级"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class DiscoveryType(Enum):
    """发现类型"""
    ASSET = "asset"
    CREDENTIAL = "credential"
    VULN = "vuln"
    KNOWLEDGE = "knowledge"


class DiscoveryImportance(Enum):
    """发现重要性"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class MicroTaskProgress(Enum):
    """微任务进度"""
    IN_PROGRESS = "进行中"
    COMPLETED = "已完成"
    FAILED = "失败"


class FailureType(Enum):
    """失败类型"""
    BLOCKED = "blocked"
    ERROR = "error"
    IMPOSSIBLE = "impossible"
    TIMEOUT = "timeout"


# ============================================================
# 二、基础数据结构
# ============================================================

@dataclass
class TaskInfo:
    """任务信息"""
    id: str
    goal: str
    description: str
    completion_criteria: str
    dependencies: List[str] = field(default_factory=list)
    priority: str = "NORMAL"
    assigned_executor: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        return cls(
            id=data.get("id", ""),
            goal=data.get("goal", ""),
            description=data.get("description", ""),
            completion_criteria=data.get("completion_criteria", ""),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", "NORMAL"),
            assigned_executor=data.get("assigned_executor"),
            status=TaskStatus(data.get("status", "pending")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "description": self.description,
            "completion_criteria": self.completion_criteria,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "assigned_executor": self.assigned_executor,
            "status": self.status.value,
        }


# ============================================================
# 三、执行器响应结构
# ============================================================

@dataclass
class ThinkingBlock:
    """思考块"""
    current_understanding: str = ""
    strategy: str = ""
    reasoning: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingBlock":
        return cls(
            current_understanding=data.get("current_understanding", ""),
            strategy=data.get("strategy", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class MicroTaskStatus:
    """微任务状态"""
    current_micro_task: str = ""
    progress: str = "in_progress"
    next_micro_task: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MicroTaskStatus":
        return cls(
            current_micro_task=data.get("current_micro_task", ""),
            progress=data.get("progress", "in_progress"),
            next_micro_task=data.get("next_micro_task"),
        )


@dataclass
class ExecutorResponse:
    """执行器响应"""
    thinking: ThinkingBlock
    action: ExecutorActionType
    action_input: Dict[str, Any]
    micro_task_status: MicroTaskStatus
    raw_response: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_response: str = None) -> "ExecutorResponse":
        return cls(
            thinking=ThinkingBlock.from_dict(data.get("thinking", {})),
            action=ExecutorActionType(data.get("action", "EXECUTE_CODE")),
            action_input=data.get("action_input", {}),
            micro_task_status=MicroTaskStatus.from_dict(data.get("micro_task_status", {})),
            raw_response=raw_response,
        )


# ============================================================
# 四、代码执行相关
# ============================================================

@dataclass
class CodeExecutionInput:
    """代码执行输入"""
    code: str
    purpose: str = ""
    expected_outcome: str = ""
    timeout: int = 60

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeExecutionInput":
        return cls(
            code=data.get("code", ""),
            purpose=data.get("purpose", ""),
            expected_outcome=data.get("expected_outcome", ""),
            timeout=data.get("timeout", 60),
        )


@dataclass
class CodeExecutionResult:
    """代码执行结果"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    error: Optional[str] = None


# ============================================================
# 五、协作相关
# ============================================================

@dataclass
class CollaborationRequest:
    """协作请求"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: CollaborationType = CollaborationType.ASSIST
    requester: str = ""
    target_agent: str = ""
    operation: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    timeout: int = 60
    priority: CollaborationPriority = CollaborationPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    status: CollaborationStatus = CollaborationStatus.PENDING

    @classmethod
    def from_dict(cls, data: Dict[str, Any], requester: str = "") -> "CollaborationRequest":
        req = data.get("request", data)
        return cls(
            type=CollaborationType(req.get("type", "ASSIST")),
            requester=requester,
            target_agent=req.get("target_agent", ""),
            operation=req.get("operation", ""),
            params=req.get("params", {}),
            reason=req.get("reason", ""),
            timeout=req.get("timeout", 60),
            priority=CollaborationPriority(req.get("priority", "NORMAL")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "requester": self.requester,
            "target_agent": self.target_agent,
            "operation": self.operation,
            "params": self.params,
            "reason": self.reason,
            "timeout": self.timeout,
            "priority": self.priority.value,
            "status": self.status.value,
        }


@dataclass
class CollaborationResponse:
    """协作响应"""
    request_id: str
    responder: str
    status: CollaborationStatus
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "responder": self.responder,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


# ============================================================
# 六、发现相关
# ============================================================

@dataclass
class Discovery:
    """发现信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "knowledge"
    category: str = ""
    importance: str = "medium"
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: int = 50
    evidence: List[str] = field(default_factory=list)
    context: Optional[str] = None
    discovered_by: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], discovered_by: str = "") -> "Discovery":
        disc = data.get("discovery", data)
        return cls(
            type=disc.get("type", "knowledge"),
            category=disc.get("category", ""),
            importance=disc.get("importance", "medium"),
            content=disc.get("content", {}),
            confidence=disc.get("confidence", 50),
            evidence=disc.get("evidence", []),
            context=disc.get("context"),
            discovered_by=discovered_by,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "importance": self.importance,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "context": self.context,
            "discovered_by": self.discovered_by,
            "discovered_at": self.discovered_at.isoformat(),
        }


# ============================================================
# 七、执行报告相关
# ============================================================

@dataclass
class ExecutionStep:
    """执行步骤"""
    step: int
    description: str
    action: str
    result: str
    status: str = "success"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        return cls(
            step=data.get("step", 0),
            description=data.get("description", ""),
            action=data.get("action", ""),
            result=data.get("result", ""),
            status=data.get("status", "success"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "action": self.action,
            "result": self.result,
            "status": self.status,
        }


@dataclass
class ExecutionSummary:
    """执行摘要"""
    total_micro_tasks: int = 0
    successful: int = 0
    failed: int = 0
    steps: List[ExecutionStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionSummary":
        return cls(
            total_micro_tasks=data.get("total_micro_tasks", 0),
            successful=data.get("successful", 0),
            failed=data.get("failed", 0),
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_micro_tasks": self.total_micro_tasks,
            "successful": self.successful,
            "failed": self.failed,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class Recommendations:
    """建议"""
    next_actions: List[str] = field(default_factory=list)
    related_tasks: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recommendations":
        return cls(
            next_actions=data.get("next_actions", []),
            related_tasks=data.get("related_tasks", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_actions": self.next_actions,
            "related_tasks": self.related_tasks,
        }


@dataclass
class CompletionReport:
    """完成报告"""
    task_id: str
    final_status: str = "success"
    completion_criteria_met: bool = True
    execution_summary: ExecutionSummary = field(default_factory=ExecutionSummary)
    discoveries: List[Discovery] = field(default_factory=list)
    recommendations: Recommendations = field(default_factory=Recommendations)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionReport":
        report = data.get("completion_report", data)
        return cls(
            task_id=report.get("task_id", ""),
            final_status=report.get("final_status", "success"),
            completion_criteria_met=report.get("completion_criteria_met", True),
            execution_summary=ExecutionSummary.from_dict(report.get("execution_summary", {})),
            discoveries=[Discovery.from_dict(d) for d in report.get("discoveries", [])],
            recommendations=Recommendations.from_dict(report.get("recommendations", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_status": self.final_status,
            "completion_criteria_met": self.completion_criteria_met,
            "execution_summary": self.execution_summary.to_dict(),
            "discoveries": [d.to_dict() for d in self.discoveries],
            "recommendations": self.recommendations.to_dict(),
        }


@dataclass
class FailureDiagnosis:
    """失败诊断"""
    root_cause: str = ""
    blocking_factors: List[str] = field(default_factory=list)
    attempted_solutions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureDiagnosis":
        return cls(
            root_cause=data.get("root_cause", ""),
            blocking_factors=data.get("blocking_factors", []),
            attempted_solutions=data.get("attempted_solutions", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "blocking_factors": self.blocking_factors,
            "attempted_solutions": self.attempted_solutions,
        }


@dataclass
class FailureSuggestions:
    """失败建议"""
    retry_viable: bool = False
    alternative_approaches: List[str] = field(default_factory=list)
    required_conditions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureSuggestions":
        return cls(
            retry_viable=data.get("retry_viable", False),
            alternative_approaches=data.get("alternative_approaches", []),
            required_conditions=data.get("required_conditions", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retry_viable": self.retry_viable,
            "alternative_approaches": self.alternative_approaches,
            "required_conditions": self.required_conditions,
        }


@dataclass
class FailureReport:
    """失败报告"""
    task_id: str
    failure_type: str = "error"
    failure_reason: str = ""
    execution_summary: ExecutionSummary = field(default_factory=ExecutionSummary)
    partial_discoveries: List[Discovery] = field(default_factory=list)
    diagnosis: FailureDiagnosis = field(default_factory=FailureDiagnosis)
    suggestions: FailureSuggestions = field(default_factory=FailureSuggestions)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureReport":
        report = data.get("failure_report", data)
        return cls(
            task_id=report.get("task_id", ""),
            failure_type=report.get("failure_type", "error"),
            failure_reason=report.get("failure_reason", ""),
            execution_summary=ExecutionSummary.from_dict(report.get("execution_summary", {})),
            partial_discoveries=[Discovery.from_dict(d) for d in report.get("partial_discoveries", [])],
            diagnosis=FailureDiagnosis.from_dict(report.get("diagnosis", {})),
            suggestions=FailureSuggestions.from_dict(report.get("suggestions", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "execution_summary": self.execution_summary.to_dict(),
            "partial_discoveries": [d.to_dict() for d in self.partial_discoveries],
            "diagnosis": self.diagnosis.to_dict(),
            "suggestions": self.suggestions.to_dict(),
        }


# ============================================================
# 八、黑板查询相关
# ============================================================

class BlackboardCategory(Enum):
    """黑板分类"""
    ASSETS = "assets"
    CREDENTIALS = "credentials"
    VULNS = "vulns"
    SESSIONS = "sessions"
    PATHS = "paths"
    KNOWLEDGE = "knowledge"


@dataclass
class BlackboardQuery:
    """黑板查询"""
    category: BlackboardCategory
    filters: Dict[str, Any] = field(default_factory=dict)
    min_confidence: int = 0
    limit: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackboardQuery":
        query = data.get("query", data)
        return cls(
            category=BlackboardCategory(query.get("category", "knowledge")),
            filters=query.get("filters", {}),
            min_confidence=query.get("min_confidence", 0),
            limit=query.get("limit", 50),
        )


# ============================================================
# 九、执行上下文相关
# ============================================================

@dataclass
class DependencyResult:
    """依赖任务结果"""
    task_id: str
    summary: str
    key_findings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyResult":
        return cls(
            task_id=data.get("task_id", ""),
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
        )


@dataclass
class IntelSummary:
    """情报摘要"""
    category: str
    summary: str
    confidence: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntelSummary":
        return cls(
            category=data.get("category", ""),
            summary=data.get("summary", ""),
            confidence=data.get("confidence", 50),
        )


@dataclass
class SessionInfo:
    """会话信息"""
    id: str
    type: str
    target: str
    status: str = "active"
    user: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInfo":
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            target=data.get("target", ""),
            status=data.get("status", "active"),
            user=data.get("user"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "target": self.target,
            "status": self.status,
            "user": self.user,
        }


@dataclass
class ExecutionContext:
    """执行上下文"""
    dependency_results: List[DependencyResult] = field(default_factory=list)
    related_intel: List[IntelSummary] = field(default_factory=list)
    available_sessions: List[SessionInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        return cls(
            dependency_results=[DependencyResult.from_dict(d) for d in data.get("dependency_results", [])],
            related_intel=[IntelSummary.from_dict(i) for i in data.get("related_intel", [])],
            available_sessions=[SessionInfo.from_dict(s) for s in data.get("available_sessions", [])],
        )


# ============================================================
# 十、黑板摘要相关
# ============================================================

@dataclass
class AssetSummary:
    """资产摘要"""
    host: str
    port: int
    service: str
    version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "service": self.service,
            "version": self.version,
        }


@dataclass
class CredentialSummary:
    """凭证摘要"""
    username: str
    target: str
    status: str = "unverified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "target": self.target,
            "status": self.status,
        }


@dataclass
class VulnSummary:
    """漏洞摘要"""
    name: str
    location: str
    severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "severity": self.severity,
        }


@dataclass
class BlackboardSummary:
    """黑板摘要"""
    assets: List[AssetSummary] = field(default_factory=list)
    credentials: List[CredentialSummary] = field(default_factory=list)
    vulns: List[VulnSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": [a.to_dict() for a in self.assets],
            "credentials": [c.to_dict() for c in self.credentials],
            "vulns": [v.to_dict() for v in self.vulns],
        }


@dataclass
class CollaborationRecord:
    """协作记录"""
    type: str
    target: str
    operation: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "operation": self.operation,
            "status": self.status,
        }


# ============================================================
# 十一、执行器配置
# ============================================================

@dataclass
class ExecutorConfig:
    """执行器配置"""
    name: str
    executor_type: ExecutorType
    max_iterations: int = 20
    code_execution_timeout: int = 60
    collaboration_timeout: int = 120
    ctf_mode: bool = False

    # LLM配置
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096




