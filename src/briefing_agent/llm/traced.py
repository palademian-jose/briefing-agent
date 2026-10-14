"""One wrapper for every LLM call: emits an ``llm`` span with token usage and cost.

Cost preference: provider-reported cost (OpenRouter ``usage.cost``) when present, otherwise
tokens x ``pricing.json``. Both are kept in span meta so the two can be reconciled later.
"""

from __future__ import annotations

import time
from typing import Any

from briefing_agent.cost import cost_usd
from briefing_agent.llm.base import LLMClient, LLMResponse, Message, ToolSpec
from briefing_agent.tracing import Tracer


def traced_complete(
    client: LLMClient,
    tracer: Tracer,
    name: str,
    messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    pricing_path: str = "pricing.json",
    meta: dict[str, Any] | None = None,
) -> LLMResponse:
    start = time.time()
    status, error_kind = "ok", None
    resp: LLMResponse | None = None
    try:
        resp = client.complete(
            messages, tools=tools, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return resp
    except Exception as exc:
        status, error_kind = "error", type(exc).__name__
        raise
    finally:
        end = time.time()
        if resp is not None:
            table_cost = cost_usd(resp.model, resp.usage, pricing_path)
            cost = resp.reported_cost_usd if resp.reported_cost_usd is not None else table_cost
            tracer.emit(
                name,
                "llm",
                start=start,
                end=end,
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cost_usd=round(cost, 8),
                status=status,
                meta={
                    "provider": getattr(client, "provider", "unknown"),
                    "stop_reason": resp.stop_reason,
                    "tool_calls": [c.name for c in resp.tool_calls],
                    "table_cost_usd": table_cost,
                    "reported_cost_usd": resp.reported_cost_usd,
                    **(meta or {}),
                },
            )
        else:
            tracer.emit(
                name,
                "llm",
                start=start,
                end=end,
                model=model,
                status=status,
                error_kind=error_kind,
                meta=meta or {},
            )
