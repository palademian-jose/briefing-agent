"""SQLite persistence: saved notes (briefings) and run records."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from briefing_agent.tools.errors import requires_human

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    markdown TEXT NOT NULL,
    sources TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    briefing_md TEXT,
    sources TEXT NOT NULL,
    tool_calls INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # -- notes --------------------------------------------------------------------------
    def save_note(self, topic: str, markdown: str, sources: list[dict[str, str]]) -> dict[str, Any]:
        if self.read_only:
            raise requires_human(
                "store_readonly",
                "notes store is read-only",
                hint="An operator must restore write access to the notes database.",
            )
        created = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO notes (topic, markdown, sources, created_at) VALUES (?, ?, ?, ?)",
                (topic, markdown, json.dumps(sources, ensure_ascii=False), created),
            )
        return {"ok": True, "note_id": cur.lastrowid, "created_at": created}

    def list_notes(self, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, topic, sources, created_at, length(markdown) AS chars "
                "FROM notes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        notes = [
            {
                "id": r["id"],
                "topic": r["topic"],
                "sources": json.loads(r["sources"]),
                "created_at": r["created_at"],
                "chars": r["chars"],
            }
            for r in rows
        ]
        return {"ok": True, "notes": notes}

    def get_note(self, note_id: int) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if r is None:
            return None
        return {
            "id": r["id"],
            "topic": r["topic"],
            "markdown": r["markdown"],
            "sources": json.loads(r["sources"]),
            "created_at": r["created_at"],
        }

    # -- runs ---------------------------------------------------------------------------
    def save_run(self, result: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, trace_id, topic, status, briefing_md, "
                "sources, "
                "tool_calls, cost_usd, latency_ms, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result["run_id"],
                    result["trace_id"],
                    result["topic"],
                    result["status"],
                    result.get("briefing_md"),
                    json.dumps(result.get("sources", []), ensure_ascii=False),
                    int(result.get("tool_calls", 0)),
                    float(result.get("cost_usd", 0.0)),
                    float(result.get("latency_ms", 0.0)),
                    result.get("created_at") or _now(),
                    json.dumps(result, ensure_ascii=False, default=str),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(r["payload"]) if r else None

    def spend_since(self, since_iso: str) -> float:
        with self._lock:
            r = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM runs WHERE created_at >= ?",
                (since_iso,),
            ).fetchone()
        return float(r["total"])
