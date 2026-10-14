"""The research-briefing workflow: fixed stages around one bounded agentic stage.

    normalize -> [discover_sources: agentic loop] -> compose -> verify (+1 repair) -> save

Only ``discover_sources`` lets the model choose actions, because only source discovery has
an open-ended, observation-dependent branching factor (what to search, which result to
fetch, what to do after a 404). The other stages have a known input and a known output,
so they are plain function calls: cheaper, testable in isolation, and impossible to loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from briefing_agent.agent import prompts
from briefing_agent.agent.loop import AgentEnv, discover_sources
from briefing_agent.agent.policy import NeedsHuman, RetryPolicy, ToolExecutor
from briefing_agent.agent.state import Budget, RunState, load_checkpoint, save_checkpoint
from briefing_agent.briefing import parse_briefing
from briefing_agent.llm.base import LLMClient, Message
from briefing_agent.llm.factory import get_client
from briefing_agent.settings import Settings, get_settings
from briefing_agent.store import Store
from briefing_agent.tools.faults import FaultInjector
from briefing_agent.tools.network import make_http_client
from briefing_agent.tools.registry import RunContext, ToolRegistry
from briefing_agent.tracing import FEATURE, JsonlSink, make_tracer, run_spans, summarize

log = logging.getLogger(__name__)


class Runner:
    """Owns the per-run wiring (tracer, store, HTTP client, registry, executor)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: LLMClient | None = None,
        store: Store | None = None,
        http: httpx.Client | None = None,
        faults: str | None = None,
        fault_seed: int | None = None,
        sleep: Callable[[float], None] | None = None,
        on_step: Callable[[RunState, str], None] | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        s = self.settings
        self.client = client or get_client(s)
        self.store = store or Store(s.db_path)
        self.http = http or make_http_client(
            s.effective_network_mode, timeout_s=s.http_timeout_s, fixture_dir=s.fixture_dir
        )
        self.fault_spec = s.faults if faults is None else faults
        self.fault_seed = s.fault_seed if fault_seed is None else fault_seed
        self.sleep = sleep or time.sleep
        self.on_step = on_step
        self.trace_path = Path(trace_path or s.trace_path)
        # One open JSONL sink per runner; every run gets its own tracer (and in-memory span list).
        self._sink = JsonlSink(self.trace_path)

    # -- wiring ----------------------------------------------------------------------
    def _env(self, state: RunState) -> AgentEnv:
        s = self.settings
        tracer = make_tracer(self._sink)
        faults = FaultInjector.from_spec(self.fault_spec, seed=self.fault_seed + state.iteration)
        if hasattr(self.client, "faults"):
            self.client.faults = faults  # fake provider: LLM-level soft faults
        ctx = RunContext(
            http=self.http,
            store=self.store,
            tracer=tracer,
            faults=faults,
            verify_threshold=s.verify_match_threshold,
            min_facts=s.min_facts,
            max_facts=s.max_facts,
        )
        registry = ToolRegistry(ctx)
        executor = ToolExecutor(
            registry,
            RetryPolicy(
                budgets={
                    "search_wikipedia": s.retry_budget_search,
                    "fetch_page": s.retry_budget_fetch,
                },
                base_ms=s.retry_base_ms,
            ),
            sleep=self.sleep,
        )
        return AgentEnv(
            client=self.client,
            registry=registry,
            executor=executor,
            tracer=tracer,
            budget=Budget(
                max_iterations=s.max_iterations,
                max_tool_calls=s.max_tool_calls,
                max_cost_usd=s.max_cost_usd_per_run,
            ),
            model=s.llm_model,
            cheap_model=s.cheap_model,
            checkpoint_dir=s.checkpoint_dir,
            min_facts=s.min_facts,
            max_facts=s.max_facts,
            on_step=self.on_step,
        )

    # -- public API ------------------------------------------------------------------
    def run(self, topic: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        c = constraints or {}
        state = RunState(
            topic=topic.strip(),
            audience=str(c.get("audience") or "engineer"),
            target_sources=int(c.get("max_sources") or 3),
        )
        return self._execute(state)

    def resume(self, run_id: str) -> dict[str, Any]:
        state = load_checkpoint(self.settings.checkpoint_dir, run_id)
        if state.phase == "done":
            return self._result(state, self._env(state), started=time.time())
        state.status = "running"
        return self._execute(state)

    # -- the workflow ----------------------------------------------------------------
    def _execute(self, state: RunState) -> dict[str, Any]:
        env = self._env(state)
        started = time.time()
        with (
            env.tracer.trace(trace_id=state.trace_id, feature=FEATURE),
            env.tracer.span("agent.run", "other", run_id=state.run_id, topic=state.topic),
        ):
            try:
                if state.phase in ("init", "discover"):
                    discover_sources(state, env)
                if state.status == "running" and state.phase in ("discover", "compose"):
                    self._compose(state, env)
                if state.status == "running" and state.phase in ("verify", "repair"):
                    self._verify_and_repair(state, env)
                if state.status == "running" and state.phase == "save":
                    self._save(state, env)
                if state.status == "running":
                    state.finish("ok")
            except NeedsHuman as exc:
                state.errors.append({"tool": exc.tool, **exc.info.model_dump(mode="json")})
                state.finish("needs_human", f"{exc.tool}: {exc.info.message} ({exc.info.hint})")
        save_checkpoint(state, env.checkpoint_dir)
        result = self._result(state, env, started)
        self.store.save_run(result)
        return result

    def _compose(self, state: RunState, env: AgentEnv) -> None:
        state.phase = "compose"
        if env.budget.exceeded(state):
            state.finish("budget_exceeded", env.budget.exceeded(state))
            return
        with env.tracer.span("stage.compose", "other", run_id=state.run_id):
            msgs = [
                Message(role="system", content=prompts.COMPOSE_SYSTEM),
                Message(
                    role="user",
                    content=prompts.compose_user(
                        state.topic, state.audience, state.sources(), env.min_facts, env.max_facts
                    ),
                ),
            ]
            resp = env.llm(state, "agent.compose", msgs, max_tokens=1800)
            state.briefing_md = resp.text.strip() + "\n"
        state.phase = "verify"
        env.checkpoint(state, "compose")

    def _verify_and_repair(self, state: RunState, env: AgentEnv) -> None:
        assert state.briefing_md is not None
        state.phase = "verify"
        with env.tracer.span("stage.verify", "other", run_id=state.run_id):
            verdict = env.registry.execute("verify_briefing", {"briefing": state.briefing_md})
            state.tool_calls += 1
            state.verification = verdict.result
            if state.verification_first_pass is None:
                state.verification_first_pass = bool(verdict.result.get("ok"))
            state.log_step(
                "verify",
                ok=verdict.result.get("ok"),
                issues=[i["code"] for i in verdict.result.get("issues", [])],
            )
            env.checkpoint(state, "verify")

            if not verdict.result.get("ok") and not state.repair_attempted:
                state.phase = "repair"
                state.repair_attempted = True
                if env.budget.exceeded(state):
                    state.finish("budget_exceeded", env.budget.exceeded(state))
                    return
                msgs = [
                    Message(role="system", content=prompts.REPAIR_SYSTEM),
                    Message(
                        role="user",
                        content=prompts.repair_user(
                            state.topic,
                            state.briefing_md,
                            verdict.result.get("issues", []),
                            state.sources(),
                            env.min_facts,
                            env.max_facts,
                        ),
                    ),
                ]
                resp = env.llm(state, "agent.repair", msgs, max_tokens=1800)
                state.briefing_md = resp.text.strip() + "\n"
                verdict = env.registry.execute("verify_briefing", {"briefing": state.briefing_md})
                state.tool_calls += 1
                state.verification = verdict.result
                state.log_step(
                    "verify_after_repair",
                    ok=verdict.result.get("ok"),
                    issues=[i["code"] for i in verdict.result.get("issues", [])],
                )
                env.checkpoint(state, "repair")

        if not state.verification or not state.verification.get("ok"):
            state.finish("failed", "verification failed after one repair pass")
            return
        state.phase = "save"

    def _save(self, state: RunState, env: AgentEnv) -> None:
        assert state.briefing_md is not None
        with env.tracer.span("stage.save", "other", run_id=state.run_id):
            parsed = parse_briefing(state.briefing_md, state.topic)
            sources = [{"title": s.title, "url": s.url} for s in parsed.sources]
            outcome = env.executor.run(
                "save_note",
                {"topic": state.topic, "markdown": state.briefing_md, "sources": sources},
            )
            state.tool_calls += outcome.attempt
            if outcome.error is not None:
                state.errors.append({"tool": "save_note", **outcome.error.model_dump(mode="json")})
                state.finish("failed", f"save_note: {outcome.error.message}")
                return
            state.note_id = int(outcome.result["note_id"])
        env.checkpoint(state, "save")

    def _result(self, state: RunState, env: AgentEnv, started: float) -> dict[str, Any]:
        parsed = parse_briefing(state.briefing_md or "", state.topic)
        summary = summarize(run_spans(env.tracer))
        return {
            "run_id": state.run_id,
            "trace_id": state.trace_id,
            "topic": state.topic,
            "status": state.status,
            "reason": state.reason,
            "briefing_md": state.briefing_md,
            "sources": [{"n": s.n, "title": s.title, "url": s.url} for s in parsed.sources],
            "fetched_urls": list(state.fetched),
            "tool_calls": state.tool_calls,
            "llm_calls": state.llm_calls,
            "retries": state.retries,
            "iterations": state.iteration,
            "cost_usd": round(state.cost_usd, 6),
            "cost": summary,
            "latency_ms": round((time.time() - started) * 1000, 1),
            "verification": state.verification,
            "verification_first_pass": state.verification_first_pass,
            "repair_attempted": state.repair_attempted,
            "faults_injected": list(env.registry.ctx.faults.injected),
            "errors": state.errors,
            "plans": state.plans,
            "note_id": state.note_id,
            "created_at": state.created_at,
            "provider": getattr(self.client, "provider", "unknown"),
            "model": env.model,
        }


def run_briefing(
    topic: str, constraints: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    return Runner(**kwargs).run(topic, constraints)
