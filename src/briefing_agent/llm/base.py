"""Provider-neutral message, tool and response types.

Conversation encoding (shared across backends):

- an assistant turn that calls tools has ``content`` as a list of blocks:
  ``{"type": "text", "text": ...}`` and ``{"type": "tool_use", "id", "name", "input"}``;
- a tool result is a ``role="tool"`` message whose ``content`` is the JSON-encoded result
  and whose ``tool_call_id`` points at the call it answers.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None

    def text(self) -> str:
        """Plain-text view of the content (tool_use blocks are rendered as JSON)."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                parts.append(json.dumps({"tool": block.get("name"), "input": block.get("input")}))
        return "\n".join(parts)

    def tool_uses(self) -> list[dict[str, Any]]:
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if b.get("type") == "tool_use"]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema; additionalProperties must be false


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage
    model: str
    latency_ms: float
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | ...
    cost_usd: float | None = None  # provider-billed cost when the API reports it (OpenRouter)

    def as_assistant_message(self) -> Message:
        if not self.tool_calls:
            return Message(role="assistant", content=self.text)
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for call in self.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return Message(role="assistant", content=blocks)


class LLMClient(Protocol):
    provider: str

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


def tool_result_message(tool_call_id: str, result: Any) -> Message:
    """Encode a tool execution result as a conversation message."""
    return Message(
        role="tool",
        content=json.dumps(result, ensure_ascii=False, default=str),
        tool_call_id=tool_call_id,
    )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used by the fake provider (roughly 4 chars per token)."""
    return max(1, len(text) // 4)
