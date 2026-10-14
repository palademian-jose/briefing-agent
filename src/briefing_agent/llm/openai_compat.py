"""OpenAI-compatible chat-completions backend. Used for both OpenRouter (default real
backend, ``base_url=https://openrouter.ai/api/v1``) and the OpenAI API directly."""

from __future__ import annotations

import json
import time
from typing import Any

from briefing_agent.llm.base import LLMResponse, Message, ToolCall, ToolSpec, Usage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text()})
        elif m.role == "assistant" and m.tool_uses():
            text = "".join(
                b.get("text", "")
                for b in m.content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            out.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                        }
                        for b in m.tool_uses()
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.text()})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {k: v for k, v in t.input_schema.items() if k != "$schema"},
            },
        }
        for t in tools
    ]


class OpenAICompatClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        default_model: str,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.provider = provider
        self.default_model = default_model
        self.is_openrouter = provider == "openrouter"
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or None,
            timeout=timeout_s,
            max_retries=2,
        )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": to_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if self.is_openrouter:
            # Ask OpenRouter to report the billed cost alongside token usage.
            kwargs["extra_body"] = {"usage": {"include": True}}
        start = time.time()
        resp = self._client.chat.completions.create(**kwargs)
        latency = (time.time() - start) * 1000
        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = resp.usage
        reported_cost = None
        if usage is not None:
            reported_cost = getattr(usage, "cost", None)
            if reported_cost is None and getattr(usage, "model_extra", None):
                reported_cost = usage.model_extra.get("cost")
        out = LLMResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=resp.model or kwargs["model"],
            latency_ms=round(latency, 3),
            stop_reason="tool_use" if calls else (choice.finish_reason or "end_turn"),
        )
        if reported_cost is not None:
            out.cost_usd = float(reported_cost)
        return out
