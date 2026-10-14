"""End-to-end guard rails: invalid tool calls and hallucinated facts are caught and corrected."""

from __future__ import annotations

from collections.abc import Callable

from briefing_agent.agent.state import load_checkpoint
from briefing_agent.llm.fake import FakeClient
from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings
from briefing_agent.store import Store


def test_bad_call_is_rejected_then_corrected(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    runner = make_runner(client=FakeClient(settings.llm_model, misbehave="bad_call"))
    res = runner.run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    first_error = res["errors"][0]
    assert first_error["code"] == "invalid_arguments" and first_error["kind"] == "PERMANENT"
    assert "maximum" in first_error["message"]
    state = load_checkpoint(settings.checkpoint_dir, res["run_id"])
    observed = [s for s in state.steps if s["kind"] == "observe"]
    assert observed[0]["tool"] == "search_wikipedia" and observed[0]["ok"] is False
    assert observed[1]["tool"] == "search_wikipedia" and observed[1]["ok"] is True


def test_unknown_tool_is_rejected_then_corrected(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    runner = make_runner(client=FakeClient(settings.llm_model, misbehave="unknown_tool"))
    res = runner.run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    assert res["errors"][0]["code"] == "unknown_tool"


def test_hallucination_fails_verifier_and_is_repaired(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    runner = make_runner(client=FakeClient(settings.llm_model, misbehave="hallucinate"))
    res = runner.run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok", res["reason"]
    assert res["verification_first_pass"] is False and res["repair_attempted"] is True
    assert res["verification"]["ok"] is True
    assert "Kepler-442" not in res["briefing_md"]
    state = load_checkpoint(settings.checkpoint_dir, res["run_id"])
    verify_steps = [s for s in state.steps if s["kind"] in ("verify", "verify_after_repair")]
    assert verify_steps[0]["kind"] == "verify" and {"fact_unsupported", "citation_unmapped"} <= set(
        verify_steps[0]["issues"]
    )
    assert verify_steps[1]["kind"] == "verify_after_repair" and verify_steps[1]["ok"] is True
    kinds = [sp["kind"] for sp in state.steps]
    assert (
        kinds.count("verify") == 1 and kinds.count("verify_after_repair") == 1
    )  # exactly one repair pass


def test_repair_is_bounded_to_one_pass(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    class AlwaysBad(FakeClient):
        def _repair(self, text: str) -> str:  # repair that does not fix anything
            return self._compose(text.replace("TASK: repair", "TASK: compose"))

    runner = make_runner(client=AlwaysBad(settings.llm_model, misbehave="hallucinate"))
    res = runner.run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "failed" and "verification failed" in res["reason"]
    assert res["repair_attempted"] is True and res["verification"]["ok"] is False
    assert res["briefing_md"]  # the unverified draft is still returned, clearly labelled failed
    assert Store(settings.db_path).list_notes()["notes"] == []  # and never saved


def test_llm_level_fault_is_measured(
    settings: Settings, make_runner: Callable[..., Runner]
) -> None:
    res = make_runner(faults="compose_hallucinate:1.0").run("Bloom filter", {"max_sources": 2})
    assert res["status"] == "ok" and res["repair_attempted"] is True
    assert any(f["fault"] == "compose_hallucinate" for f in res["faults_injected"])
