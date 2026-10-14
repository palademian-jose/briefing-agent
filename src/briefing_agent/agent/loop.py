"""The bounded agentic loop: think -> act -> observe -> verify, one stage only.

This is the only place where the model chooses actions. Everything around it (compose,
verify, repair, save) is a fixed pipeline in ``pipeline.py``. The loop is hand-written on
purpose: the state is explicit (``RunState``), each step is checkpointed, and the stop
conditions are enumerable: the model says DONE, the stage goal is met, or a budget trips.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from briefing_agent.agent import prompts
from briefing_agent.agent.policy import NeedsHuman, ToolExecutor
from briefing_agent.agent.state import Budget, RunState, save_checkpoint
from briefing_agent.llm.base import LLMClient, LLMResponse, Message, ToolSpec, tool_result_message
from briefing_agent.tools.registry import ToolRegistry
from briefing_agent.tools.schemas import SEARCH_STAGE_TOOLS
from briefing_agent.tracing import FEATURE, Tracer, response_cost, wrap_client


@dataclass
class AgentEnv:
    """Everything a stage needs besides the state. Built once per run by the pipeline."""

    client: LLMClient
    registry: ToolRegistry
    executor: ToolExecutor
    tracer: Tracer
    budget: Budget = field(default_factory=Budget)
    model: str = "anthropic/claude-sonnet-4.5"
    cheap_model: str = "anthropic/claude-haiku-4.5"
    checkpoint_dir: Path = Path("checkpoints")
    min_facts: int = 5
    max_facts: int = 8
    think: bool = True
    on_step: Callable[[RunState, str], None] | None = None

    def llm(
        self,
        state: RunState,
        name: str,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Every LLM call goes through here: traced, costed, counted against the budget."""
        # One llm span per call (llmops.tracing.wrap_client): tokens, latency, and cost where the
        # provider-reported cost (OpenRouter usage.cost) wins over tokens x pricing.json.
        traced = wrap_client(
            self.client,
            self.tracer,
            name=name,
            feature=FEATURE,
            meta={
                "run_id": state.run_id,
                "phase": state.phase,
                "provider": getattr(self.client, "provider", "unknown"),
            },
        )
        resp = traced.complete(
            messages, tools=tools, model=model or self.model, max_tokens=max_tokens
        )
        state.llm_calls += 1
        state.cost_usd = round(state.cost_usd + response_cost(resp), 8)
        return resp

    def checkpoint(self, state: RunState, step: str) -> None:
        save_checkpoint(state, self.checkpoint_dir)
        if self.on_step:
            self.on_step(state, step)


def _budget_trip(state: RunState, env: AgentEnv) -> bool:
    reason = env.budget.exceeded(state)
    if reason:
        state.finish("budget_exceeded", reason)
        state.log_step("budget_exceeded", reason=reason)
        return True
    return False


def discover_sources(state: RunState, env: AgentEnv) -> RunState:
    """Run the agentic source-discovery stage until DONE, goal met, or budget exhausted."""
    state.phase = "discover"
    if not state.messages:
        state.messages = [
            Message(role="system", content=prompts.SEARCH_SYSTEM),
            Message(
                role="user",
                content=prompts.search_user(state.topic, state.audience, state.target_sources),
            ),
        ]
    env.registry.ctx.pages.update(state.fetched)
    env.registry.ctx.page_titles.update(state.titles)

    with env.tracer.span("stage.discover", "other", run_id=state.run_id):
        while True:
            if _budget_trip(state, env):
                return state

            # THINK: cheap model proposes the next action from a compact state view.
            if env.think:
                think_msgs = [
                    Message(role="system", content=prompts.THINK_SYSTEM),
                    Message(role="user", content=prompts.think_user(state.topic, state.observed())),
                ]
                plan = env.llm(
                    state, "agent.think", think_msgs, model=env.cheap_model, max_tokens=120
                )
                state.plans.append(plan.text.strip())
                state.log_step("think", plan=plan.text.strip())
                state.messages.append(
                    Message(role="user", content=f"Planner note: {plan.text.strip()}")
                )

            # ACT: the main model chooses a tool call (or says DONE).
            resp = env.llm(state, "agent.act", state.messages, tools=SEARCH_STAGE_TOOLS)
            state.messages.append(resp.as_assistant_message())
            state.log_step(
                "act", text=resp.text[:200], tool_calls=[c.name for c in resp.tool_calls]
            )
            if not resp.tool_calls:
                break

            # OBSERVE: execute with the retry/escalation policy; feed structured results back.
            for call in resp.tool_calls:
                if state.tool_calls >= env.budget.max_tool_calls:
                    state.messages.append(
                        tool_result_message(
                            call.id,
                            {
                                "ok": False,
                                "error": {
                                    "kind": "PERMANENT",
                                    "code": "budget",
                                    "message": "tool-call budget exhausted",
                                    "retryable": False,
                                },
                            },
                        )
                    )
                    state.finish(
                        "budget_exceeded", f"max_tool_calls={env.budget.max_tool_calls} reached"
                    )
                    env.checkpoint(state, "budget_exceeded")
                    return state
                try:
                    outcome = env.executor.run(call.name, call.arguments)
                except NeedsHuman as exc:
                    state.tool_calls += 1
                    state.errors.append({"tool": exc.tool, **exc.info.model_dump(mode="json")})
                    state.finish("needs_human", f"{exc.tool}: {exc.info.message} ({exc.info.hint})")
                    env.checkpoint(state, "needs_human")
                    return state
                state.tool_calls += outcome.attempt
                state.retries += outcome.attempt - 1
                if outcome.error is not None:
                    state.errors.append(
                        {"tool": call.name, **outcome.error.model_dump(mode="json")}
                    )
                elif call.name == "fetch_page":
                    state.fetched[outcome.result["url"]] = outcome.result["text"]
                    state.titles[outcome.result["url"]] = outcome.result["title"]
                state.messages.append(tool_result_message(call.id, outcome.result))
                state.log_step(
                    "observe",
                    tool=call.name,
                    ok=outcome.ok,
                    attempts=outcome.attempt,
                    error=None if outcome.ok else outcome.error.code,
                )

            # VERIFY (stage-level): is the discovery goal met?
            state.iteration += 1
            state.faults_injected = list(env.registry.ctx.faults.injected)
            env.checkpoint(state, "iteration")
            if len(state.fetched) >= state.target_sources:
                state.log_step("goal_met", fetched=len(state.fetched))
                break

    if not state.fetched:
        state.finish("failed", "no sources could be fetched")
    env.checkpoint(state, "discover_done")
    return state


def transcript(state: RunState) -> str:
    """Human-readable dump of the discover-stage exchange (v0 'log the raw exchange')."""
    lines = []
    for m in state.messages:
        if m.role == "tool":
            try:
                body = json.loads(m.text())
                if isinstance(body, dict) and "text" in body:
                    body = {**body, "text": body["text"][:160] + "..."}
                lines.append(
                    f"[tool:{m.tool_call_id}] {json.dumps(body, ensure_ascii=False)[:400]}"
                )
            except json.JSONDecodeError:
                lines.append(f"[tool] {m.text()[:400]}")
        else:
            lines.append(f"[{m.role}] {m.text()[:400]}")
    return "\n".join(lines)
