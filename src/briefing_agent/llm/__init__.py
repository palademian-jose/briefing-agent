"""LLM provider abstraction (shared contract across the portfolio repos)."""

from briefing_agent.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    tool_result_message,
)
from briefing_agent.llm.factory import get_client

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "get_client",
    "tool_result_message",
]
