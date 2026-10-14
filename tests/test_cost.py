"""Cost via llmops pricing, traced spans (wrap_client) and the run cost summary."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from llmops.pricing import estimate_cost

from briefing_agent.llm.base import LLMResponse, Message, Usage
from briefing_agent.tracing import (
    SPAN_KINDS,
    make_tracer,
    response_cost,
    run_spans,
    summarize,
    wrap_client,
)


def test_cost_from_table() -> None:
    assert estimate_cost("anthropic/claude-sonnet-4.5", 1_000_000, 100_000) == 3.0 + 1.5
    assert estimate_cost("openai/gpt-4.1-mini", 10_000, 1_000) == 0.004 + 0.0016


def test_vendor_prefix_and_variant_fallbacks() -> None:
    sonnet = estimate_cost("anthropic/claude-sonnet-4.5", 1000, 1000)
    assert estimate_cost("claude-sonnet-4.5", 1000, 1000) == sonnet
    assert estimate_cost("claude-sonnet-4-5", 1000, 1000) == sonnet
    assert estimate_cost("claude-sonnet-4-5-20250929", 1000, 1000) == sonnet
    assert estimate_cost("someprovider/gpt-4o-mini", 1000, 1000) == estimate_cost(
        "openai/gpt-4o-mini", 1000, 1000
    )


def test_unknown_model_costs_zero_not_crash() -> None:
    assert estimate_cost("mystery/model-9", 5, 5) is None
    assert estimate_cost(None, 5, 5) is None
    resp = LLMResponse(
        text="x",
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mystery/model-9",
        latency_ms=1.0,
        stop_reason="end_turn",
    )
    assert response_cost(resp) == 0.0


class _Stub:
    provider = "stub"

    def __init__(self, reported: float | None) -> None:
        self.reported = reported

    def complete(self, messages, *, tools=None, model=None, temperature=0.0, max_tokens=1024):
        return LLMResponse(
            text="hi",
            usage=Usage(input_tokens=1000, output_tokens=100),
            model=model or "openai/gpt-4o",
            latency_ms=1.0,
            stop_reason="end_turn",
            cost_usd=self.reported,
        )


def _call(client, tracer, name: str, **kw):
    traced = wrap_client(client, tracer, name=name, feature="agent.run", meta={"run_id": "r1"})
    return traced.complete([Message(role="user", content="x")], **kw)


def test_traced_call_emits_span_with_table_cost(tmp_path: Path) -> None:
    tracer = make_tracer(tmp_path / "spans.jsonl")
    _call(_Stub(None), tracer, "agent.compose")
    span = run_spans(tracer)[-1]
    assert span.kind == "llm" and span.model == "openai/gpt-4o" and span.name == "agent.compose"
    assert span.cost_usd == (1000 * 2.5 + 100 * 10.0) / 1e6
    assert span.meta["cost_reported_usd"] is None and span.meta["cost_source"] == "computed"
    assert span.meta["run_id"] == "r1" and span.meta["gen_ai.system"] == "stub"
    line = json.loads((tmp_path / "spans.jsonl").read_text().splitlines()[-1])
    assert set(line) == {
        "trace_id",
        "span_id",
        "parent_span_id",
        "name",
        "kind",
        "start_ts",
        "end_ts",
        "latency_ms",
        "model",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "cache_hit",
        "feature",
        "user_id",
        "session_id",
        "status",
        "error_kind",
        "meta",
    }
    assert line["start_ts"].endswith("Z") and line["kind"] in SPAN_KINDS
    assert line["feature"] == "agent.run"


def test_provider_reported_cost_is_preferred(tmp_path: Path) -> None:
    tracer = make_tracer(tmp_path / "spans.jsonl")
    resp = _call(_Stub(0.0042), tracer, "agent.compose")
    span = run_spans(tracer)[-1]
    assert span.cost_usd == 0.0042 == response_cost(resp)
    assert span.meta["cost_computed_usd"] == 0.0035 and span.meta["cost_reported_usd"] == 0.0042
    assert span.meta["cost_source"] == "reported"


def test_failed_call_records_an_error_span(tmp_path: Path) -> None:
    class Boom:
        provider = "stub"

        def complete(self, *a, **k):
            raise TimeoutError("slow")

    tracer = make_tracer(tmp_path / "spans.jsonl")
    with contextlib.suppress(TimeoutError):
        _call(Boom(), tracer, "agent.act", model="openai/gpt-4o")
    span = run_spans(tracer)[-1]
    assert span.status == "error" and span.error_kind == "TimeoutError"
    assert span.kind == "llm" and span.model == "openai/gpt-4o"


def test_summary_aggregates_by_model(tmp_path: Path) -> None:
    tracer = make_tracer(tmp_path / "spans.jsonl")
    with tracer.span("agent.run", "other"):
        _call(_Stub(None), tracer, "a", model="openai/gpt-4o")
        _call(_Stub(None), tracer, "b", model="openai/gpt-4o-mini")
        with tracer.span("tool.fetch_page", "tool"):
            pass
    spans = run_spans(tracer)
    s = summarize(spans)
    assert s["llm_calls"] == 2 and s["tool_calls"] == 1
    assert s["input_tokens"] == 2000 and s["output_tokens"] == 200
    assert set(s["by_model"]) == {"openai/gpt-4o", "openai/gpt-4o-mini"}
    assert s["cost_usd"] == round(sum(s["by_model"].values()), 8)
    parent = spans[-1]
    assert parent.name == "agent.run" and all(
        sp.parent_span_id == parent.span_id for sp in spans[:-1]
    )
