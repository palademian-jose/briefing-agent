"""Token-to-USD accounting from pricing.json (USD per million tokens).

Lookup order for a model id: exact key -> key without the vendor prefix
(``anthropic/claude-sonnet-4.5`` -> ``claude-sonnet-4.5``) -> dotted/dashed variant
(``claude-sonnet-4.5`` ~ ``claude-sonnet-4-5``) -> prefix match for dated aliases.
Unknown models cost 0 and log a warning, so a typo cannot silently inflate a bill.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from briefing_agent.llm.base import Usage

log = logging.getLogger(__name__)
ZERO = {"input_per_m": 0.0, "output_per_m": 0.0}


@lru_cache(maxsize=4)
def load_pricing(path: str = "pricing.json") -> dict[str, dict[str, float]]:
    p = Path(path)
    if not p.exists():
        # Fall back to the copy shipped next to the package (Docker / installed wheel).
        p = Path(__file__).resolve().parents[2] / "pricing.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _variants(model: str) -> list[str]:
    bare = model.split("/", 1)[1] if "/" in model else model
    out = [model, bare, bare.replace(".", "-"), bare.replace("-", ".")]
    seen: list[str] = []
    for v in out:
        if v not in seen:
            seen.append(v)
    return seen


def price_for(model: str | None, pricing_path: str | Path = "pricing.json") -> dict[str, float]:
    if not model:
        return ZERO
    table = load_pricing(str(pricing_path))
    for variant in _variants(model):
        if variant in table:
            return table[variant]
    for variant in _variants(model):
        for key, value in table.items():
            if variant.startswith(key):
                return value
    log.warning("No pricing entry for model %r; cost recorded as 0", model)
    return ZERO


def cost_usd(model: str | None, usage: Usage, pricing_path: str | Path = "pricing.json") -> float:
    p = price_for(model, pricing_path)
    cost = (usage.input_tokens * p["input_per_m"] + usage.output_tokens * p["output_per_m"]) / 1e6
    return round(cost, 8)
