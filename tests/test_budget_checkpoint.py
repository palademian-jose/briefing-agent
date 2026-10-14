"""Budget abort (partial result) and checkpoint / resume."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from briefing_agent.agent.state import RunState, load_checkpoint, save_checkpoint
from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings


def test_tool_call_budget_aborts_with_partial_result(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    s = settings.model_copy(update={"max_tool_calls": 2})
    res = make_runner(s).run("Bloom filter", {"max_sources": 3})
    assert res["status"] == "budget_exceeded"
    assert "max_tool_calls" in res["reason"]
    assert res["tool_calls"] <= 2
    assert res["briefing_md"] is None
    assert 0 < len(res["fetched_urls"]) < 3  # partial progress is reported, not lost


def test_cost_cap_aborts(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    s = settings.model_copy(update={"max_cost_usd_per_run": 0.0005})
    res = make_runner(s).run("Bloom filter", {"max_sources": 3})
    assert res["status"] == "budget_exceeded"
    assert "cap" in res["reason"]
    assert res["cost_usd"] >= 0.0005


def test_iteration_budget_aborts(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    s = settings.model_copy(update={"max_iterations": 1})
    res = make_runner(s).run("Bloom filter", {"max_sources": 3})
    assert res["status"] == "budget_exceeded" and "max_iterations" in res["reason"]


def test_checkpoint_round_trip(tmp_path) -> None:
    state = RunState(topic="Raft (algorithm)", target_sources=2)
    state.fetched["https://en.wikipedia.org/wiki/Raft_(algorithm)"] = (
        "Raft is a consensus algorithm"
    )
    state.iteration = 3
    path = save_checkpoint(state, tmp_path)
    assert path.exists() and path.name == f"{state.run_id}.json"
    loaded = load_checkpoint(tmp_path, state.run_id)
    assert loaded == state


class _Crash(Exception):
    pass


def test_resume_after_crash(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    crashed: list[str] = []

    def on_step(state: RunState, step: str) -> None:
        if step == "iteration" and state.iteration == 2 and not crashed:
            crashed.append(state.run_id)
            raise _Crash()

    runner = make_runner(on_step=on_step)
    with pytest.raises(_Crash):
        runner.run("Bloom filter", {"max_sources": 3})
    run_id = crashed[0]
    ckpt = load_checkpoint(settings.checkpoint_dir, run_id)
    assert ckpt.phase == "discover" and ckpt.iteration == 2 and len(ckpt.fetched) >= 1

    res = make_runner().resume(run_id)
    assert res["run_id"] == run_id
    assert res["status"] == "ok", res["reason"]
    assert res["briefing_md"] and len(res["sources"]) == 3
    # the state carries the pre-crash counts; the resumed process only executed the rest
    assert res["tool_calls"] == 6 and res["cost"]["tool_calls"] < res["tool_calls"]
    final = load_checkpoint(settings.checkpoint_dir, run_id)
    assert final.phase == "done" and final.status == "ok"


def test_resume_of_finished_run_is_idempotent(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    runner = make_runner()
    first = runner.run("Bloom filter", {"max_sources": 2})
    again = make_runner().resume(first["run_id"])
    assert again["status"] == "ok" and again["briefing_md"] == first["briefing_md"]
