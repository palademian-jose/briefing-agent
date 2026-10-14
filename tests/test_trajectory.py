"""Fake trajectory under no faults, transient faults and permanent faults.

Faults here are injected at the transport layer (a timeout, a 404, a 403) so the test also
covers the HTTP-to-error-kind mapping, not just the injector."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings
from briefing_agent.tools.network import USER_AGENT, FixtureTransport


class FlakyTransport(FixtureTransport):
    """Fixture transport with a scripted failure per URL substring."""

    def __init__(self, fixture_dir, script: dict[str, list]) -> None:
        super().__init__(fixture_dir)
        self.script = {k: list(v) for k, v in script.items()}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for needle, steps in self.script.items():
            if needle in str(request.url) and steps:
                step = steps.pop(0)
                if step == "timeout":
                    raise httpx.ReadTimeout("injected timeout", request=request)
                if isinstance(step, int):
                    return httpx.Response(step, text=f"<html><body>{step}</body></html>")
        return super().handle_request(request)


def _client(settings: Settings, script: dict[str, list]) -> httpx.Client:
    return httpx.Client(
        transport=FlakyTransport(settings.fixture_dir, script), headers={"User-Agent": USER_AGENT}
    )


def _tool_sequence(runner_result: dict, settings: Settings) -> list[tuple[str, bool]]:
    from briefing_agent.agent.state import load_checkpoint

    state = load_checkpoint(settings.checkpoint_dir, runner_result["run_id"])
    return [(s["tool"], s["ok"]) for s in state.steps if s["kind"] == "observe"]


def test_clean_trajectory(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    res = make_runner().run("Bloom filter", {"max_sources": 3, "audience": "backend engineer"})
    assert res["status"] == "ok", res["reason"]
    seq = _tool_sequence(res, settings)
    assert seq[0] == ("search_wikipedia", True)
    assert [t for t, _ in seq[1:]] == ["fetch_page"] * 3
    assert res["retries"] == 0 and res["repair_attempted"] is False
    assert res["verification"]["ok"] is True and res["verification_first_pass"] is True
    assert len(res["sources"]) == 3 and set(s["url"] for s in res["sources"]) == set(
        res["fetched_urls"]
    )
    assert "Bloom" in res["briefing_md"] and res["note_id"] is not None
    assert len(res["plans"]) >= 4  # one think step per iteration
    assert res["cost_usd"] > 0 and res["cost"]["llm_calls"] == res["llm_calls"]


def test_transient_fault_is_retried_and_run_succeeds(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    http = _client(settings, {"/wiki/Bloom_filter": ["timeout", "timeout"]})
    res = make_runner(http=http).run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    assert res["retries"] == 2
    seq = _tool_sequence(res, settings)
    assert ("fetch_page", True) in seq
    kinds = [e["kind"] for e in res["errors"]]
    assert kinds == []  # retried successfully: the model never saw an error
    assert any(u.endswith("/Bloom_filter") for u in res["fetched_urls"])


def test_transient_budget_exhaustion_replans(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    http = _client(settings, {"/wiki/Bloom_filter": ["timeout"] * 10})
    res = make_runner(http=http).run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    assert res["retries"] == settings.retry_budget_fetch
    assert (
        res["errors"]
        and res["errors"][0]["kind"] == "TRANSIENT"
        and res["errors"][0]["retryable"] is False
    )
    assert not any(u.endswith("/Bloom_filter") for u in res["fetched_urls"])  # skipped, not looped
    assert len(res["fetched_urls"]) == 2


def test_permanent_fault_replans_without_retry(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    http = _client(settings, {"/wiki/Bloom_filter": [404]})
    res = make_runner(http=http).run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    assert res["retries"] == 0
    assert res["errors"][0]["kind"] == "PERMANENT" and res["errors"][0]["code"] == "not_found"
    seq = _tool_sequence(res, settings)
    assert seq.count(("fetch_page", False)) == 1 and seq.count(("fetch_page", True)) == 2


def test_requires_human_escalates(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    http = _client(settings, {"/wiki/Bloom_filter": [403]})
    res = make_runner(http=http).run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "needs_human"
    assert "403" in res["reason"] and "operator" in res["reason"].lower()
    assert res["briefing_md"] is None


def test_empty_search_broadens_query(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    # search_empty with p=1.0 would never recover; use the injector with a seed that fires once.
    res = make_runner(faults="search_empty:0.5", fault_seed=3).run(
        "Bloom filter", {"max_sources": 2}
    )
    assert res["status"] == "ok", res["reason"]
    assert any(f["fault"] == "search_empty" for f in res["faults_injected"])
    seq = _tool_sequence(res, settings)
    assert [t for t, _ in seq].count("search_wikipedia") >= 2


def test_injected_faults_change_the_trajectory(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    clean = make_runner().run("Consistent hashing", {"max_sources": 3})
    faulted = make_runner(faults="fetch_timeout:0.5,fetch_not_found:0.3", fault_seed=11).run(
        "Consistent hashing", {"max_sources": 3}
    )
    assert faulted["faults_injected"]
    assert faulted["tool_calls"] > clean["tool_calls"]
    assert faulted["status"] in ("ok", "budget_exceeded")
