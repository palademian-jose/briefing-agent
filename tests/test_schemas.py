"""Schema validation rejects bad calls before execution and returns a structured error."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from briefing_agent.settings import Settings
from briefing_agent.store import Store
from briefing_agent.tools.errors import ErrorKind, ToolError
from briefing_agent.tools.network import make_http_client
from briefing_agent.tools.registry import RunContext, ToolRegistry
from briefing_agent.tools.schemas import ALL_TOOLS
from briefing_agent.tracing import Tracer


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    ctx = RunContext(
        http=make_http_client("fixture", fixture_dir=settings.fixture_dir),
        store=Store(settings.db_path),
        tracer=Tracer(settings.trace_path),
    )
    return ToolRegistry(ctx)


def test_every_schema_is_strict() -> None:
    for spec in ALL_TOOLS:
        Draft202012Validator.check_schema(spec.input_schema)
        assert spec.input_schema["additionalProperties"] is False, spec.name
        assert spec.input_schema["type"] == "object"


@pytest.mark.parametrize(
    ("tool", "args", "fragment"),
    [
        ("search_wikipedia", {"query": "x"}, "too short"),
        ("search_wikipedia", {"query": "bloom filter", "limit": 50}, "maximum"),
        ("search_wikipedia", {"query": "bloom filter", "lang": "fr"}, "is not one of"),
        ("search_wikipedia", {"query": "bloom filter", "pages": 3}, "Additional properties"),
        ("search_wikipedia", {}, "required"),
        ("fetch_page", {"url": "http://localhost:8080/admin"}, "does not match"),
        ("fetch_page", {"url": "https://example.com/wiki/Bloom_filter"}, "does not match"),
        ("fetch_page", {"url": "file:///etc/passwd"}, "does not match"),
        (
            "fetch_page",
            {"url": "https://en.wikipedia.org/wiki/Bloom_filter", "max_chars": 10},
            "minimum",
        ),
        ("save_note", {"topic": "t", "markdown": "short", "sources": []}, "too short"),
        ("verify_briefing", {"briefing": ""}, "non-empty"),
    ],
)
def test_invalid_calls_are_rejected(
    registry: ToolRegistry, tool: str, args: dict, fragment: str
) -> None:
    with pytest.raises(ToolError) as exc:
        registry.validate(tool, args)
    assert exc.value.kind == ErrorKind.PERMANENT
    assert exc.value.code == "invalid_arguments"
    assert fragment.lower() in exc.value.info.message.lower()


def test_unknown_tool_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError) as exc:
        registry.validate("search_web", {"query": "bloom"})
    assert exc.value.code == "unknown_tool"
    assert "search_wikipedia" in exc.value.info.hint


def test_defaults_are_applied(registry: ToolRegistry) -> None:
    args = registry.validate("search_wikipedia", {"query": "bloom filter"})
    assert args == {"query": "bloom filter", "limit": 5, "lang": "en"}


def test_execute_returns_structured_error_without_running(registry: ToolRegistry) -> None:
    outcome = registry.execute("fetch_page", {"url": "javascript:alert(1)"})
    assert not outcome.ok
    assert outcome.result["ok"] is False
    err = outcome.result["error"]
    assert err["kind"] == "PERMANENT" and err["retryable"] is False
    assert err["code"] == "invalid_arguments"
    assert registry.calls == 0  # nothing executed
    assert registry.ctx.tracer.spans[-1]["status"] == "error"
