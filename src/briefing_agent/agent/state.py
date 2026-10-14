"""Explicit run state. Every step of the loop reads and writes this object and checkpoints
it to JSON, so a run can be inspected, replayed in tests, or resumed after a crash."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from briefing_agent.llm.base import Message

Phase = Literal["init", "discover", "compose", "verify", "repair", "save", "done"]
Status = Literal["running", "ok", "needs_human", "budget_exceeded", "failed"]


class Budget(BaseModel):
    max_iterations: int = 12
    max_tool_calls: int = 16
    max_cost_usd: float = 0.25

    def exceeded(self, state: RunState) -> str | None:
        if state.iteration >= self.max_iterations:
            return f"max_iterations={self.max_iterations} reached"
        if state.tool_calls >= self.max_tool_calls:
            return f"max_tool_calls={self.max_tool_calls} reached"
        if state.cost_usd >= self.max_cost_usd:
            return f"cost ${state.cost_usd:.4f} reached cap ${self.max_cost_usd:.4f}"
        return None


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    audience: str = "engineer"
    target_sources: int = 3
    phase: Phase = "init"
    status: Status = "running"
    reason: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="milliseconds")
    )

    # discover stage (the agentic loop)
    messages: list[Message] = Field(default_factory=list)
    iteration: int = 0
    plans: list[str] = Field(default_factory=list)
    fetched: dict[str, str] = Field(default_factory=dict)  # url -> text
    titles: dict[str, str] = Field(default_factory=dict)  # url -> title

    # accounting
    tool_calls: int = 0
    llm_calls: int = 0
    retries: int = 0
    cost_usd: float = 0.0
    faults_injected: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)

    # fixed stages
    briefing_md: str | None = None
    verification: dict[str, Any] | None = None
    verification_first_pass: bool | None = None
    repair_attempted: bool = False
    note_id: int | None = None

    def sources(self) -> list[dict[str, str]]:
        return [
            {"title": self.titles.get(u, u), "url": u, "text": t} for u, t in self.fetched.items()
        ]

    def observed(self) -> dict[str, Any]:
        """Compact view handed to the think step."""
        return {
            "target": self.target_sources,
            "fetched": list(self.fetched),
            "candidates": _candidates(self.messages),
            "failed": _failed(self.messages),
            "iteration": self.iteration,
            "tool_calls": self.tool_calls,
            "cost_usd": round(self.cost_usd, 5),
        }

    def log_step(self, kind: str, **info: Any) -> None:
        self.steps.append({"phase": self.phase, "iteration": self.iteration, "kind": kind, **info})

    def finish(self, status: Status, reason: str | None = None) -> None:
        self.status = status
        self.reason = reason
        self.phase = "done"


def _pairs(messages: list[Message]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    calls: dict[str, dict[str, Any]] = {}
    out = []
    for m in messages:
        for b in m.tool_uses():
            calls[b["id"]] = b
        if m.role == "tool" and m.tool_call_id in calls:
            try:
                out.append((calls[m.tool_call_id], json.loads(m.text())))
            except json.JSONDecodeError:
                out.append((calls[m.tool_call_id], {}))
    return out


def _candidates(messages: list[Message]) -> list[str]:
    urls: list[str] = []
    for call, res in _pairs(messages):
        if call["name"] == "search_wikipedia" and res.get("ok"):
            for r in res.get("results", []):
                if r["url"] not in urls:
                    urls.append(r["url"])
    return urls


def _failed(messages: list[Message]) -> list[str]:
    return [
        call["input"].get("url", "")
        for call, res in _pairs(messages)
        if call["name"] == "fetch_page" and not res.get("ok")
    ]


# -- checkpointing ---------------------------------------------------------------------
def checkpoint_path(checkpoint_dir: str | Path, run_id: str) -> Path:
    return Path(checkpoint_dir) / f"{run_id}.json"


def save_checkpoint(state: RunState, checkpoint_dir: str | Path) -> Path:
    path = checkpoint_path(checkpoint_dir, state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX and Windows
    return path


def load_checkpoint(checkpoint_dir: str | Path, run_id: str) -> RunState:
    path = checkpoint_path(checkpoint_dir, run_id)
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))
