"""Retry / re-plan / escalate policy per error kind."""

from __future__ import annotations

from typing import Any

import pytest

from briefing_agent.agent.policy import NeedsHuman, RetryPolicy, ToolExecutor
from briefing_agent.llm.base import ToolSpec
from briefing_agent.settings import Settings
from briefing_agent.store import Store
from briefing_agent.tools.errors import permanent, requires_human, transient
from briefing_agent.tools.faults import FaultInjector, parse_faults
from briefing_agent.tools.network import make_http_client
from briefing_agent.tools.registry import RunContext, ToolRegistry
from briefing_agent.tracing import make_tracer

SPEC = ToolSpec(
    name="flaky",
    description="test tool",
    input_schema={
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "additionalProperties": False,
    },
)


def make_registry(settings: Settings, behaviour: list[Any]) -> ToolRegistry:
    ctx = RunContext(
        http=make_http_client("fixture", fixture_dir=settings.fixture_dir),
        store=Store(settings.db_path),
        tracer=make_tracer(settings.trace_path),
    )
    reg = ToolRegistry(ctx)
    calls = iter(behaviour)

    def impl(_ctx: RunContext, _args: dict[str, Any]) -> dict[str, Any]:
        step = next(calls)
        if isinstance(step, Exception):
            raise step
        return {"ok": True, "value": step}

    reg.register(SPEC, impl)
    return reg


def test_transient_is_retried_then_succeeds(settings: Settings) -> None:
    reg = make_registry(settings, [transient("timeout", "t1"), transient("timeout", "t2"), "done"])
    sleeps: list[float] = []
    ex = ToolExecutor(reg, RetryPolicy(budgets={"flaky": 2}, base_ms=100), sleep=sleeps.append)
    out = ex.run("flaky", {"n": 1})
    assert out.ok and out.result["value"] == "done"
    assert out.attempt == 3 and ex.retries_total == 2
    assert sleeps == [0.5, 0.5]  # retry_after_ms=500 dominates the 100/200 ms backoff


def test_backoff_grows_exponentially() -> None:
    p = RetryPolicy(base_ms=100, max_backoff_ms=5000)
    assert [p.backoff_ms(a, None) for a in (1, 2, 3, 4)] == [100, 200, 400, 800]
    assert p.backoff_ms(10, None) == 5000
    assert p.backoff_ms(1, 900) == 900


def test_transient_budget_exhausted_becomes_non_retryable(settings: Settings) -> None:
    reg = make_registry(settings, [transient("timeout", f"t{i}") for i in range(5)])
    ex = ToolExecutor(reg, RetryPolicy(budgets={"flaky": 2}), sleep=lambda _s: None)
    out = ex.run("flaky", {"n": 1})
    assert not out.ok
    assert out.attempt == 3  # 1 + 2 retries
    assert out.error.kind == "TRANSIENT" and out.error.retryable is False
    assert "budget exhausted" in out.error.hint
    assert out.result["error"]["retryable"] is False  # what the model sees


def test_permanent_is_not_retried(settings: Settings) -> None:
    reg = make_registry(settings, [permanent("not_found", "gone"), "never"])
    ex = ToolExecutor(reg, sleep=lambda _s: None)
    out = ex.run("flaky", {"n": 1})
    assert not out.ok and out.attempt == 1 and ex.retries_total == 0
    assert out.error.kind == "PERMANENT" and out.error.retryable is False


def test_requires_human_escalates(settings: Settings) -> None:
    reg = make_registry(settings, [requires_human("blocked", "403")])
    ex = ToolExecutor(reg, sleep=lambda _s: None)
    with pytest.raises(NeedsHuman) as exc:
        ex.run("flaky", {"n": 1})
    assert exc.value.tool == "flaky" and exc.value.info.kind == "REQUIRES_HUMAN"


def test_fault_spec_parsing() -> None:
    assert parse_faults("fetch_timeout:0.3, search_empty:0.2") == {
        "fetch_timeout": 0.3,
        "search_empty": 0.2,
    }
    assert parse_faults("") == {}
    with pytest.raises(ValueError):
        parse_faults("bogus:0.5")
    with pytest.raises(ValueError):
        parse_faults("fetch_timeout:1.5")


def test_fault_injector_is_seeded_and_targets_one_tool() -> None:
    a = FaultInjector.from_spec("fetch_timeout:0.5", seed=1)
    b = FaultInjector.from_spec("fetch_timeout:0.5", seed=1)
    seq_a = [bool(_raises(a, "fetch_page")) for _ in range(20)]
    seq_b = [bool(_raises(b, "fetch_page")) for _ in range(20)]
    assert seq_a == seq_b and any(seq_a) and not all(seq_a)
    c = FaultInjector.from_spec("fetch_timeout:1.0", seed=1)
    assert _raises(c, "search_wikipedia") is None  # wrong tool: never injected
    assert len(c.injected) == 0


def _raises(inj: FaultInjector, tool: str) -> Exception | None:
    try:
        inj.before_tool(tool, {})
    except Exception as exc:
        return exc
    return None
