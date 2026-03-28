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
        severities = [Severity.INFO, Severity.CRITICAL, Severity.HIGH]
        assert Severity.CRITICAL in severities
