"""Tool registry: the only path from a model's tool call to real side effects.

Pipeline for one call: schema validation -> fault injection -> execution -> span.
Validation failures never reach the implementation; they come back to the model as a
PERMANENT ``invalid_arguments`` error with the exact JSON-Schema violation as the hint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from jsonschema import Draft202012Validator

from briefing_agent.llm.base import ToolSpec
from briefing_agent.tools import schemas
from briefing_agent.tools.errors import ErrorKind, ToolError, ToolErrorInfo, permanent
from briefing_agent.tools.faults import FaultInjector
from briefing_agent.tools.verify import verify_briefing
from briefing_agent.tools.wikipedia import fetch_page, search_wikipedia
from briefing_agent.tracing import Tracer

if TYPE_CHECKING:
    from briefing_agent.store import Store


@dataclass
class ToolOutcome:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    error: ToolErrorInfo | None
    latency_ms: float
    attempt: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RunContext:
    """Everything a tool may touch during one run. Pages fetched here are what the verifier
    trusts, so the cache lives on the context, not in a global."""

    http: httpx.Client
    store: Store
    tracer: Tracer
    faults: FaultInjector = field(default_factory=FaultInjector)
    pages: dict[str, str] = field(default_factory=dict)  # url -> extracted text
    page_titles: dict[str, str] = field(default_factory=dict)
    verify_threshold: float = 0.6
    min_facts: int = 5
    max_facts: int = 8


ToolImpl = Callable[[RunContext, dict[str, Any]], dict[str, Any]]


def _apply_defaults(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    out = dict(args)
    for key, prop in spec.input_schema.get("properties", {}).items():
        if key not in out and "default" in prop:
            out[key] = prop["default"]
    return out


# -- implementations ---------------------------------------------------------------------
def _search(ctx: RunContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.faults.should_inject("search_empty", {"query": args["query"]}):
        return {"ok": True, "query": args["query"], "total_hits": 0, "results": []}
    return search_wikipedia(ctx.http, args["query"], limit=args["limit"], lang=args["lang"])


def _fetch(ctx: RunContext, args: dict[str, Any]) -> dict[str, Any]:
    res = fetch_page(ctx.http, args["url"], max_chars=args["max_chars"])
    ctx.pages[res["url"]] = res["text"]
    ctx.page_titles[res["url"]] = res["title"]
    return res


def _save_note(ctx: RunContext, args: dict[str, Any]) -> dict[str, Any]:
    return ctx.store.save_note(args["topic"], args["markdown"], args["sources"])


def _list_notes(ctx: RunContext, args: dict[str, Any]) -> dict[str, Any]:
    return ctx.store.list_notes(limit=args["limit"])


def _verify(ctx: RunContext, args: dict[str, Any]) -> dict[str, Any]:
    v = verify_briefing(
        args["briefing"],
        ctx.pages,
        threshold=args.get("threshold", ctx.verify_threshold),
        min_facts=ctx.min_facts,
        max_facts=ctx.max_facts,
    )
    return v.as_result()


class ToolRegistry:
    def __init__(self, ctx: RunContext, specs: list[ToolSpec] | None = None) -> None:
        self.ctx = ctx
        self.specs: dict[str, ToolSpec] = {}
        self.impls: dict[str, ToolImpl] = {}
        self.validators: dict[str, Draft202012Validator] = {}
        self.calls: int = 0
        for spec in specs or schemas.ALL_TOOLS:
            impl = _IMPLS.get(spec.name)
            if impl is None:
                raise KeyError(f"no implementation registered for tool {spec.name}")
            self.register(spec, impl)

    def register(self, spec: ToolSpec, impl: ToolImpl) -> None:
        if spec.input_schema.get("additionalProperties", True) is not False:
            raise ValueError(f"tool {spec.name}: additionalProperties must be false")
        Draft202012Validator.check_schema(spec.input_schema)
        self.specs[spec.name] = spec
        self.impls[spec.name] = impl
        self.validators[spec.name] = Draft202012Validator(spec.input_schema)

    def specs_for(self, names: list[str]) -> list[ToolSpec]:
        return [self.specs[n] for n in names]

    # -- validation --------------------------------------------------------------------
    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        """Return normalised arguments (defaults applied) or raise a PERMANENT ToolError."""
        if name not in self.specs:
            raise permanent(
                "unknown_tool",
                f"unknown tool {name!r}",
                hint=f"Available tools: {sorted(self.specs)}",
            )
        if not isinstance(arguments, dict):
            raise permanent("invalid_arguments", "arguments must be a JSON object")
        errors = sorted(self.validators[name].iter_errors(arguments), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            where = "/".join(str(p) for p in first.absolute_path) or "<root>"
            raise permanent(
                "invalid_arguments",
                f"{name}: {first.message} (at {where})",
                hint="Fix the arguments to match the tool schema exactly; do not add extra keys.",
                details={"violations": [e.message for e in errors[:5]]},
            )
        return _apply_defaults(self.specs[name], arguments)

    # -- execution ---------------------------------------------------------------------
    def execute(self, name: str, arguments: Any, *, attempt: int = 1) -> ToolOutcome:
        """One attempt. Retry policy lives in the agent executor, not here."""
        start = time.time()
        normalised = arguments if isinstance(arguments, dict) else {}
        error: ToolErrorInfo | None = None
        result: dict[str, Any]
        try:
            normalised = self.validate(name, arguments)
            self.calls += 1
            self.ctx.faults.before_tool(name, normalised)
            result = self.impls[name](self.ctx, normalised)
        except ToolError as exc:
            error = exc.info
            result = exc.info.as_result()
        end = time.time()
        meta: dict[str, Any] = {"arguments": _redact(normalised), "attempt": attempt}
        if error is not None:
            meta["error"] = error.model_dump(mode="json")
        else:
            meta["result_summary"] = _summarise(name, result)
        self.ctx.tracer.emit(
            f"tool.{name}",
            "verify" if name == "verify_briefing" else "tool",
            start=start,
            end=end,
            status="ok" if error is None else "error",
            error_kind=None if error is None else str(error.kind),
            meta=meta,
        )
        return ToolOutcome(
            name=name,
            arguments=normalised,
            result=result,
            error=error,
            latency_ms=round((end - start) * 1000, 3),
            attempt=attempt,
        )


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        out[k] = (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
    return out


def _summarise(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if name == "search_wikipedia":
        return {"hits": len(result.get("results", []))}
    if name == "fetch_page":
        return {"title": result.get("title"), "chars": result.get("chars")}
    if name == "verify_briefing":
        return {"ok": result.get("ok"), "issues": [i["code"] for i in result.get("issues", [])]}
    if name == "save_note":
        return {"note_id": result.get("note_id")}
    if name == "list_notes":
        return {"count": len(result.get("notes", []))}
    return {}


_IMPLS: dict[str, ToolImpl] = {
    "search_wikipedia": _search,
    "fetch_page": _fetch,
    "save_note": _save_note,
    "list_notes": _list_notes,
    "verify_briefing": _verify,
}

__all__ = ["ErrorKind", "RunContext", "ToolOutcome", "ToolRegistry"]
