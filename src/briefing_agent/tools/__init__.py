"""Tools: strict schemas, structured errors, fault injection, registry, verifier."""

from briefing_agent.tools.errors import ErrorKind, ToolError, ToolErrorInfo
from briefing_agent.tools.faults import FaultInjector, parse_faults
from briefing_agent.tools.registry import RunContext, ToolOutcome, ToolRegistry
from briefing_agent.tools.schemas import ALL_TOOLS, SEARCH_STAGE_TOOLS

__all__ = [
    "ALL_TOOLS",
    "SEARCH_STAGE_TOOLS",
    "ErrorKind",
    "FaultInjector",
    "RunContext",
    "ToolError",
    "ToolErrorInfo",
    "ToolOutcome",
    "ToolRegistry",
    "parse_faults",
]
