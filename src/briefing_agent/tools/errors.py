"""Structured tool errors. Every failure a tool can raise carries a *kind* that the
agent's policy maps to an action: retry (TRANSIENT), re-plan (PERMANENT), or escalate
(REQUIRES_HUMAN). The model sees the same structure, so it can react too."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorKind(StrEnum):
    TRANSIENT = "TRANSIENT"  # timeouts, 429/5xx, flaky network: retry with backoff
    PERMANENT = "PERMANENT"  # 404, invalid arguments, empty result: do not retry, re-plan
    REQUIRES_HUMAN = "REQUIRES_HUMAN"  # blocked, read-only store, policy violation: escalate


class ToolErrorInfo(BaseModel):
    kind: ErrorKind
    code: str = Field(description="machine-readable code, e.g. timeout|not_found|invalid_arguments")
    message: str
    retryable: bool
    retry_after_ms: int | None = None
    hint: str = Field(default="", description="what the model should do next")
    details: dict[str, Any] = Field(default_factory=dict)

    def as_result(self) -> dict[str, Any]:
        """The JSON object handed back to the model."""
        return {"ok": False, "error": self.model_dump(mode="json")}


class ToolError(Exception):
    """Raised by tool implementations and the fault injector."""

    def __init__(
        self,
        kind: ErrorKind,
        code: str,
        message: str,
        *,
        retry_after_ms: int | None = None,
        hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.info = ToolErrorInfo(
            kind=kind,
            code=code,
            message=message,
            retryable=kind == ErrorKind.TRANSIENT,
            retry_after_ms=retry_after_ms,
            hint=hint,
            details=details or {},
        )

    @property
    def kind(self) -> ErrorKind:
        return self.info.kind

    @property
    def code(self) -> str:
        return self.info.code


def transient(code: str, message: str, *, retry_after_ms: int = 500, hint: str = "") -> ToolError:
    return ToolError(
        ErrorKind.TRANSIENT,
        code,
        message,
        retry_after_ms=retry_after_ms,
        hint=hint or "Transient failure; the executor retries automatically within budget.",
    )


def permanent(code: str, message: str, *, hint: str = "", details: dict | None = None) -> ToolError:
    return ToolError(
        ErrorKind.PERMANENT,
        code,
        message,
        hint=hint or "Do not repeat this call with the same arguments; choose another action.",
        details=details,
    )


def requires_human(code: str, message: str, *, hint: str = "") -> ToolError:
    return ToolError(
        ErrorKind.REQUIRES_HUMAN,
        code,
        message,
        hint=hint or "Stop and report to an operator; the run cannot complete on its own.",
    )
