"""Fault injection for evals and demos.

Spec format (env ``FAULTS`` or ``--faults``): ``name:probability,name:probability``.
Each fault targets one tool and is sampled independently per call from a seeded RNG, so a
given (seed, call sequence) is reproducible but the trajectory changes with the seed.

Known faults
------------
fetch_timeout        fetch_page       TRANSIENT       upstream timeout (retryable)
fetch_rate_limit     fetch_page       TRANSIENT       HTTP 429 with retry_after_ms
fetch_not_found      fetch_page       PERMANENT       HTTP 404, pick another source
fetch_blocked        fetch_page       REQUIRES_HUMAN  HTTP 403 bot block, operator must act
search_empty         search_wikipedia soft            returns zero hits (model must re-plan)
search_rate_limit    search_wikipedia TRANSIENT       HTTP 429
notes_readonly       save_note        REQUIRES_HUMAN  store is read-only
compose_hallucinate  (fake LLM)       soft            compose step emits an unsupported fact
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from briefing_agent.tools.errors import ToolError, permanent, requires_human, transient

FAULT_TOOL = {
    "fetch_timeout": "fetch_page",
    "fetch_rate_limit": "fetch_page",
    "fetch_not_found": "fetch_page",
    "fetch_blocked": "fetch_page",
    "search_empty": "search_wikipedia",
    "search_rate_limit": "search_wikipedia",
    "notes_readonly": "save_note",
    "compose_hallucinate": "__llm_compose__",
}


def parse_faults(spec: str | None) -> dict[str, float]:
    """``"fetch_timeout:0.3,search_empty:0.2"`` -> ``{"fetch_timeout": 0.3, ...}``."""
    out: dict[str, float] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, prob = part.partition(":")
        name = name.strip()
        if name not in FAULT_TOOL:
            raise ValueError(f"unknown fault {name!r}; known: {sorted(FAULT_TOOL)}")
        p = float(prob or "1.0")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"fault probability for {name} must be in [0, 1], got {p}")
        out[name] = p
    return out


@dataclass
class FaultInjector:
    faults: dict[str, float] = field(default_factory=dict)
    seed: int = 7
    injected: list[dict[str, Any]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @classmethod
    def from_spec(cls, spec: str | None, seed: int = 7) -> FaultInjector:
        return cls(parse_faults(spec), seed=seed)

    @property
    def enabled(self) -> bool:
        return bool(self.faults)

    def _roll(self, name: str) -> bool:
        p = self.faults.get(name, 0.0)
        return p > 0.0 and self._rng.random() < p

    def should_inject(self, name: str, context: dict[str, Any] | None = None) -> bool:
        """Soft faults (no exception): the caller changes its own behaviour."""
        if self._roll(name):
            self.injected.append({"fault": name, "tool": FAULT_TOOL[name], **(context or {})})
            return True
        return False

    def before_tool(self, tool: str, arguments: dict[str, Any]) -> None:
        """Raise a structured ToolError for hard faults aimed at ``tool``."""
        for name in self.faults:
            if FAULT_TOOL.get(name) != tool:
                continue
            if name in ("search_empty", "compose_hallucinate"):
                continue  # soft faults are consulted by the tool / LLM itself
            if not self._roll(name):
                continue
            self.injected.append({"fault": name, "tool": tool, "arguments": arguments})
            raise _fault_error(name)


def _fault_error(name: str) -> ToolError:
    if name == "fetch_timeout":
        return transient("timeout", "injected: upstream read timed out", retry_after_ms=300)
    if name == "fetch_rate_limit":
        return transient("rate_limited", "injected: HTTP 429 Too Many Requests", retry_after_ms=800)
    if name == "search_rate_limit":
        return transient("rate_limited", "injected: HTTP 429 from search API", retry_after_ms=800)
    if name == "fetch_not_found":
        return permanent(
            "not_found",
            "injected: HTTP 404 Not Found",
            hint="This page does not exist; fetch a different search result instead.",
        )
    if name == "fetch_blocked":
        return requires_human(
            "blocked",
            "injected: HTTP 403, request identified as automated traffic",
            hint="An operator must change the User-Agent or egress IP before retrying.",
        )
    if name == "notes_readonly":
        return requires_human(
            "store_readonly",
            "injected: notes store is read-only",
            hint="An operator must restore write access to the notes database.",
        )
    raise ValueError(name)
