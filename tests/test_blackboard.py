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
        bb._ensure_events()
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
