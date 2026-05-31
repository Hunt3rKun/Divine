import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Any, Callable, Optional

from divine.blackboard.models import BlackboardEntry, SECTIONS


class Blackboard:
    def __init__(self, db_path: str = ":memory:"):
        self._memory: dict[str, dict[str, BlackboardEntry]] = {s: {} for s in SECTIONS}
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._events: dict[str, asyncio.Event] = {}
        self._init_db()
        self._load_from_db()
        self._init_events()

    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blackboard (
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                source TEXT,
                timestamp TEXT,
                version INTEGER,
                PRIMARY KEY (section, key)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                source TEXT,
                timestamp TEXT,
                version INTEGER
            )
        """)
        self._conn.commit()

    def _load_from_db(self):
        cursor = self._conn.cursor()
        cursor.execute("SELECT section, key, value, source, timestamp, version FROM blackboard")
        for row in cursor.fetchall():
            section, key, value_str, source, timestamp_str, version = row
            if section not in SECTIONS:
                continue
            try:
                value = json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                value = value_str
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = datetime.now()
            entry = BlackboardEntry(
                section=section,
                key=key,
                value=value,
                source=source or "",
                timestamp=timestamp,
                version=version or 1,
            )
            self._memory[section][key] = entry

    def _init_events(self):
        try:
            asyncio.get_running_loop()
            self._events = {s: asyncio.Event() for s in SECTIONS}
        except RuntimeError:
            self._events = {}

    def _ensure_events(self):
        for section in SECTIONS:
            if section not in self._events:
                try:
                    self._events[section] = asyncio.Event()
                except RuntimeError:
                    pass

    def read(self, section: str, key: Optional[str] = None) -> Any:
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section!r}. Must be one of {SECTIONS}")
        if key is None:
            return {k: e.value for k, e in self._memory[section].items()}
        entry = self._memory[section].get(key)
        return entry.value if entry is not None else None

    def write(self, section: str, key: str, value: Any, source: str = ""):
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section!r}. Must be one of {SECTIONS}")
        existing = self._memory[section].get(key)
        version = (existing.version + 1) if existing is not None else 1
        entry = BlackboardEntry(
            section=section,
            key=key,
            value=value,
            source=source,
            timestamp=datetime.now(),
            version=version,
        )
        self._memory[section][key] = entry
        self._persist(entry)
        self._ensure_events()
        if section in self._events:
            self._events[section].set()

    def query(self, section: str, filter_fn: Optional[Callable[[BlackboardEntry], bool]] = None) -> list[BlackboardEntry]:
        if section not in SECTIONS:
            raise ValueError(f"Invalid section: {section!r}. Must be one of {SECTIONS}")
        entries = list(self._memory[section].values())
        if filter_fn is not None:
            entries = [e for e in entries if filter_fn(e)]
        return entries

    async def wait_for(self, section: str):
        self._ensure_events()
        if section not in self._events:
            raise ValueError(f"Invalid section: {section!r}. Must be one of {SECTIONS}")
        await self._events[section].wait()

    def clear_event(self, section: str):
        self._ensure_events()
        if section in self._events:
            self._events[section].clear()

    def summary(self, sections: Optional[list[str]] = None) -> dict:
        target_sections = sections if sections is not None else SECTIONS
        result = {}
        for section in target_sections:
            if section not in SECTIONS:
                continue
            entries = self._memory[section]
            result[section] = {
                "count": len(entries),
                "keys": list(entries.keys()),
                "entries": [
                    {
                        "key": e.key,
                        "value": e.value,
                        "source": e.source,
                        "version": e.version,
                        "timestamp": e.timestamp.isoformat(),
                    }
                    for e in entries.values()
                ],
            }
        return result

    def audit_log(self, limit: int = 100) -> list[dict]:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT section, key, value, source, timestamp, version FROM audit_log ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            section, key, value_str, source, timestamp_str, version = row
            try:
                value = json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                value = value_str
            result.append({
                "section": section,
                "key": key,
                "value": value,
                "source": source or "",
                "timestamp": timestamp_str,
                "version": version,
            })
        return result

    def _persist(self, entry: BlackboardEntry):
        value_str = json.dumps(entry.value)
        timestamp_str = entry.timestamp.isoformat()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO blackboard (section, key, value, source, timestamp, version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entry.section, entry.key, value_str, entry.source, timestamp_str, entry.version),
        )
        cursor.execute(
            """
            INSERT INTO audit_log (section, key, value, source, timestamp, version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entry.section, entry.key, value_str, entry.source, timestamp_str, entry.version),
        )
        self._conn.commit()
