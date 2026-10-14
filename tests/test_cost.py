"""Cost math, pricing lookup fallbacks, traced spans and the run cost summary."""

from __future__ import annotations

import json
from pathlib import Path

from briefing_agent.cost import cost_usd, price_for
from briefing_agent.llm.base import LLMResponse, Message, Usage
from briefing_agent.llm.traced import traced_complete
from briefing_agent.tracing import SPAN_KINDS, Tracer

PRICING = Path(__file__).resolve().parents[1] / "pricing.json"


def test_cost_from_table() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert cost_usd("anthropic/claude-sonnet-4.5", usage, PRICING) == 3.0 + 1.5
    assert (
        cost_usd("openai/gpt-4.1-mini", Usage(input_tokens=10_000, output_tokens=1_000), PRICING)
        == 0.004 + 0.0016
    )


def test_vendor_prefix_and_variant_fallbacks() -> None:
    sonnet = price_for("anthropic/claude-sonnet-4.5", PRICING)
    assert price_for("claude-sonnet-4.5", PRICING) == sonnet
    assert price_for("claude-sonnet-4-5", PRICING) == sonnet
    assert price_for("claude-sonnet-4-5-20250929", PRICING) == sonnet
    assert price_for("someprovider/gpt-4o-mini", PRICING) == price_for(
        "openai/gpt-4o-mini", PRICING
    )


def test_unknown_model_costs_zero_not_crash() -> None:
    assert price_for("mystery/model-9", PRICING) == {"input_per_m": 0.0, "output_per_m": 0.0}
    assert cost_usd(None, Usage(input_tokens=5, output_tokens=5), PRICING) == 0.0


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
            reported_cost_usd=self.reported,
        )


def test_traced_call_emits_span_with_table_cost(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "spans.jsonl", feature="agent.run")
    traced_complete(
        _Stub(None),
        tracer,
        "agent.compose",
        [Message(role="user", content="x")],
        pricing_path=str(PRICING),
    )
    span = tracer.spans[-1]
    assert span["kind"] == "llm" and span["model"] == "openai/gpt-4o"
    assert span["cost_usd"] == (1000 * 2.5 + 100 * 10.0) / 1e6
    assert span["meta"]["reported_cost_usd"] is None
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


def test_provider_reported_cost_is_preferred(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "spans.jsonl")
    traced_complete(
        _Stub(0.0042),
        tracer,
        "agent.compose",
        [Message(role="user", content="x")],
        pricing_path=str(PRICING),
    )
    span = tracer.spans[-1]
    assert span["cost_usd"] == 0.0042
    assert span["meta"]["table_cost_usd"] == 0.0035 and span["meta"]["reported_cost_usd"] == 0.0042


def test_summary_aggregates_by_model(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "spans.jsonl")
    with tracer.span("agent.run", "other"):
        traced_complete(
            _Stub(None),
            tracer,
            "a",
            [Message(role="user", content="x")],
            model="openai/gpt-4o",
            pricing_path=str(PRICING),
        )
        traced_complete(
            _Stub(None),
            tracer,
            "b",
            [Message(role="user", content="x")],
            model="openai/gpt-4o-mini",
            pricing_path=str(PRICING),
        )
        tracer.emit("tool.fetch_page", "tool", start=0.0, end=0.1)
    s = tracer.summary()
    assert s["llm_calls"] == 2 and s["tool_calls"] == 1
    assert s["input_tokens"] == 2000 and s["output_tokens"] == 200
    assert set(s["by_model"]) == {"openai/gpt-4o", "openai/gpt-4o-mini"}
    assert s["cost_usd"] == round(sum(s["by_model"].values()), 8)
    parent = tracer.spans[-1]
    assert parent["name"] == "agent.run" and all(
        sp["parent_span_id"] == parent["span_id"] for sp in tracer.spans[:-1]
    )
