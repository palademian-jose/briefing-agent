"""Minimal JSONL span writer implementing the shared trace schema.

Will be swapped for ``llmops.tracing`` at integration time; the on-disk format is identical.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SPAN_KINDS = ("llm", "retrieval", "rerank", "tool", "cache", "verify", "other")


def _ts(t: float) -> str:
    base = datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{int(t * 1000) % 1000:03d}Z"


class Tracer:
    """Appends one JSON line per span. Thread-safe, cheap, no dependencies."""

    def __init__(
        self, path: str | Path, trace_id: str | None = None, feature: str = "agent.run"
    ) -> None:
        self.path = Path(path)
        self.trace_id = trace_id or str(uuid.uuid4())
        self.feature = feature
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stack: list[str] = []

    def _write(self, span: dict[str, Any]) -> None:
        with self._lock:
            self.spans.append(span)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(span, ensure_ascii=False) + "\n")

    def _record(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        kind: str,
        start: float,
        end: float,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cache_hit: bool,
        status: str,
        error_kind: str | None,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        assert kind in SPAN_KINDS, kind
        span = {
            "trace_id": self.trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": name,
            "kind": kind,
            "start_ts": _ts(start),
            "end_ts": _ts(end),
            "latency_ms": round((end - start) * 1000, 3),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit,
            "feature": self.feature,
            "user_id": "anon",
            "session_id": None,
            "status": status,
            "error_kind": error_kind,
            "meta": meta,
        }
        self._write(span)
        return span

    def emit(
        self,
        name: str,
        kind: str,
        *,
        start: float,
        end: float,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        status: str = "ok",
        error_kind: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit a leaf span (an LLM call, a tool execution, a verification)."""
        return self._record(
            span_id=str(uuid.uuid4()),
            parent_span_id=self._stack[-1] if self._stack else None,
            name=name,
            kind=kind,
            start=start,
            end=end,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
            status=status,
            error_kind=error_kind,
            meta=meta or {},
        )

    @contextmanager
    def span(self, name: str, kind: str = "other", **meta: Any) -> Iterator[dict[str, Any]]:
        """Open a parent span; leaf spans emitted inside inherit its span_id."""
        span_id = str(uuid.uuid4())
        parent = self._stack[-1] if self._stack else None
        start = time.time()
        record: dict[str, Any] = {
            "span_id": span_id,
            "status": "ok",
            "error_kind": None,
            "meta": meta,
        }
        self._stack.append(span_id)
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error_kind"] = type(exc).__name__
            raise
        finally:
            self._stack.pop()
            self._record(
                span_id=span_id,
                parent_span_id=parent,
                name=name,
                kind=kind,
                start=start,
                end=time.time(),
                model=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                cache_hit=False,
                status=record["status"],
                error_kind=record["error_kind"],
                meta=record["meta"],
            )

    def summary(self) -> dict[str, Any]:
        """Per-run cost summary (returned by the API and the CLI)."""
        llm = [s for s in self.spans if s["kind"] == "llm"]
        tools = [s for s in self.spans if s["kind"] in ("tool", "verify")]
        by_model: dict[str, float] = {}
        for s in llm:
            key = s["model"] or "unknown"
            by_model[key] = round(by_model.get(key, 0.0) + s["cost_usd"], 8)
        return {
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "input_tokens": sum(s["input_tokens"] for s in llm),
            "output_tokens": sum(s["output_tokens"] for s in llm),
            "cost_usd": round(sum(s["cost_usd"] for s in llm), 8),
            "by_model": by_model,
        }
