"""Tracing is `llmops.tracing` from llm-evalops; this module re-exports it and adds the per-run
helpers the agent needs (an in-memory copy of the run's spans, and the run cost summary).

Every span is one JSON line in ``TRACE_PATH`` in the shared schema (PLAN.md contract 2). LLM
calls are traced by ``wrap_client``, tool calls and stages by ``Tracer.span``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmops.tracing import (
    SPAN_KINDS,
    JsonlSink,
    MemorySink,
    MultiSink,
    NullSink,
    Span,
    Tracer,
    null_tracer,
    response_cost,
    wrap_client,
)

FEATURE = "agent.run"

__all__ = [
    "FEATURE",
    "SPAN_KINDS",
    "JsonlSink",
    "MemorySink",
    "NullSink",
    "Span",
    "Tracer",
    "make_tracer",
    "null_tracer",
    "response_cost",
    "run_spans",
    "summarize",
    "wrap_client",
]


def make_tracer(sink: str | Path | JsonlSink, *, feature: str = FEATURE) -> Tracer:
    """A tracer that appends to the shared JSONL (a path, or an already-open ``JsonlSink``) and
    also keeps this run's spans in memory for ``run_spans`` / ``summarize``."""
    jsonl = sink if isinstance(sink, JsonlSink) else JsonlSink(sink)
    return Tracer(MultiSink(jsonl, MemorySink()), feature=feature)


def run_spans(tracer: Tracer) -> list[Span]:
    """Spans this tracer has recorded so far, in completion order."""
    sinks = tracer.sink.sinks if isinstance(tracer.sink, MultiSink) else [tracer.sink]
    for sink in sinks:
        if isinstance(sink, MemorySink):
            return sink.spans
    return []


def summarize(spans: list[Span]) -> dict[str, Any]:
    """Per-run cost summary (returned by the API and the CLI)."""
    llm = [s for s in spans if s.kind == "llm"]
    tools = [s for s in spans if s.kind in ("tool", "verify")]
    by_model: dict[str, float] = {}
    for s in llm:
        key = s.model or "unknown"
        by_model[key] = round(by_model.get(key, 0.0) + s.cost_usd, 8)
    return {
        "llm_calls": len(llm),
        "tool_calls": len(tools),
        "input_tokens": sum(s.input_tokens for s in llm),
        "output_tokens": sum(s.output_tokens for s in llm),
        "cost_usd": round(sum(s.cost_usd for s in llm), 8),
        "by_model": by_model,
    }
