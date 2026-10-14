"""Anthropic Messages API backend with native tool use."""

from __future__ import annotations

import json
import time
from typing import Any

from briefing_agent.llm.base import LLMResponse, Message, ToolCall, ToolSpec, Usage


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split system text out and merge consecutive tool results into one user turn."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.text())
            continue
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text()}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if m.role == "assistant" and m.tool_uses():
            out.append({"role": "assistant", "content": list(m.content)})
            continue
        if m.role == "user" and out and out[-1]["role"] == "user":
            # The API rejects consecutive same-role turns; fold into the previous one.
            prev = out[-1]["content"]
            blocks = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
            blocks.append({"type": "text", "text": m.text()})
            out[-1]["content"] = blocks
            continue
        out.append({"role": m.role, "content": m.text()})
    return "\n\n".join(system_parts), out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": {k: v for k, v in t.input_schema.items() if k != "$schema"},
        }
        for t in tools
    ]


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, *, api_key: str, default_model: str, timeout_s: float = 60.0) -> None:
        from anthropic import Anthropic

        self.default_model = default_model
        self._client = Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        system, msgs = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        start = time.time()
        resp = self._client.messages.create(**kwargs)
        latency = (time.time() - start) * 1000
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = (
                    block.input if isinstance(block.input, dict) else json.loads(str(block.input))
                )
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens
            ),
            model=resp.model,
            latency_ms=round(latency, 3),
            stop_reason=resp.stop_reason or ("tool_use" if calls else "end_turn"),
        )
