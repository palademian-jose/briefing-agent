"""Error-handling policy: what the executor does with each error kind.

TRANSIENT       retry with exponential backoff within the per-tool budget; if the budget
                runs out the model sees the error with retryable=false and a hint.
PERMANENT       return immediately; the model sees the error and a hint to re-plan.
REQUIRES_HUMAN  raise NeedsHuman; the run ends with status needs_human and a reason.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from briefing_agent.tools.errors import ErrorKind, ToolErrorInfo
from briefing_agent.tools.registry import ToolOutcome, ToolRegistry


class NeedsHuman(Exception):
    def __init__(self, info: ToolErrorInfo, tool: str) -> None:
        super().__init__(f"{tool}: {info.message}")
        self.info = info
        self.tool = tool


@dataclass
class RetryPolicy:
    budgets: dict[str, int] = field(
        default_factory=lambda: {"search_wikipedia": 2, "fetch_page": 2}
    )
    default_budget: int = 1
    base_ms: int = 200
    max_backoff_ms: int = 5000

    def budget_for(self, tool: str) -> int:
        return self.budgets.get(tool, self.default_budget)

    def backoff_ms(self, attempt: int, retry_after_ms: int | None) -> int:
        exp = self.base_ms * (2 ** (attempt - 1))
        return min(max(exp, retry_after_ms or 0), self.max_backoff_ms)


@dataclass
class ToolExecutor:
    registry: ToolRegistry
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    sleep: Callable[[float], None] = time.sleep
    attempts_total: int = 0
    retries_total: int = 0

    def run(self, name: str, arguments: Any) -> ToolOutcome:
        attempt = 1
        while True:
            outcome = self.registry.execute(name, arguments, attempt=attempt)
            self.attempts_total += 1
            if outcome.error is None:
                outcome.attempt = attempt
                return outcome
            kind = outcome.error.kind
            if kind == ErrorKind.REQUIRES_HUMAN:
                raise NeedsHuman(outcome.error, name)
            if kind == ErrorKind.PERMANENT:
                outcome.attempt = attempt
                return outcome
            # TRANSIENT
            if attempt > self.policy.budget_for(name):
                outcome.error.retryable = False
                outcome.error.hint = (
                    f"Retry budget exhausted after {attempt} attempts; treat as permanent and "
                    "choose a different source or query."
                )
                outcome.result = outcome.error.as_result()
                outcome.attempt = attempt
                return outcome
            self.retries_total += 1
            self.sleep(self.policy.backoff_ms(attempt, outcome.error.retry_after_ms) / 1000.0)
            attempt += 1
