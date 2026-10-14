"""Binary, criterion-specific judge (shared contract 4).

``judge(criterion, input, output, reference) -> {"pass": bool, "reason": str}``

With a real provider the verdict comes from ``JUDGE_MODEL`` (a different family than the
generator). With the fake provider the verdict comes from a deterministic lexical rubric so
the eval pipeline runs offline; the calibration script measures how far that rubric is from
human labels (Cohen's kappa) and probes order and length bias.
"""

from __future__ import annotations

import json
import re
from typing import Any

from briefing_agent.agent import prompts
from briefing_agent.briefing import parse_briefing
from briefing_agent.llm.base import LLMClient, Message
from briefing_agent.llm.factory import get_client
from briefing_agent.settings import Settings, get_settings
from briefing_agent.tools.verify import lexical_support

CRITERION_USEFUL_FAITHFUL = (
    "The briefing is useful for the stated audience (on topic, a real summary, specific key "
    "facts) and faithful to its sources (every key fact is supported by the text of the source "
    "it cites; nothing is invented)."
)


def lexical_judge(
    criterion: str,
    input_: dict[str, Any],
    output: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Deterministic fallback rubric. Expects ``output["briefing_md"]`` and
    ``output["sources_text"]`` (url -> text) and uses ``input_["topic"]``."""
    md = output.get("briefing_md") or ""
    sources_text: dict[str, str] = output.get("sources_text") or {}
    topic = str(input_.get("topic", ""))
    b = parse_briefing(md, topic)
    reasons: list[str] = []

    if len(b.summary.split()) < 15:
        reasons.append("summary missing or too short")
    if len(b.facts) < 3:
        reasons.append(f"only {len(b.facts)} key facts")
    topic_terms = [t for t in re.findall(r"[a-z0-9]+", topic.lower()) if len(t) >= 3]
    # Judge the body (summary + facts); the title and templated questions always echo the topic.
    body = (b.summary + " " + " ".join(f.text for f in b.facts)).lower()
    if topic_terms and not any(t in body for t in topic_terms):
        reasons.append("summary and facts never mention the topic")
    by_n = {s.n: s.url for s in b.sources}
    unsupported = 0
    for fact in b.facts:
        if not fact.citations:
            unsupported += 1
            continue
        ok = False
        for n in fact.citations:
            page = sources_text.get(by_n.get(n, ""), "")
            if page and lexical_support(fact.text, page) >= threshold:
                ok = True
        if not ok:
            unsupported += 1
    if unsupported:
        reasons.append(f"{unsupported} fact(s) not supported by their cited source")
    must = (reference or {}).get("must_mention") or []
    missing = [m for m in must if m.lower() not in body]
    if missing:
        reasons.append(f"missing expected terms: {missing}")
    passed = not reasons
    return {
        "pass": passed,
        "reason": "meets the criterion under the lexical rubric" if passed else "; ".join(reasons),
    }


_CLIENT_CACHE: dict[str, LLMClient] = {}


def _default_client(s: Settings) -> LLMClient:
    """One client per provider for the process (avoids re-logging the fallback warning)."""
    key = s.effective_provider
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = get_client(s)
    return _CLIENT_CACHE[key]


def judge(
    criterion: str,
    input_: dict[str, Any],
    output: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    c = client or _default_client(s)
    if getattr(c, "provider", "fake") == "fake":
        verdict = lexical_judge(
            criterion, input_, output, reference, threshold=s.verify_match_threshold
        )
        verdict["judge"] = "lexical-fallback"
        return verdict
    shown_output = {k: v for k, v in output.items() if k != "sources_text"}
    shown_output["sources_text"] = {
        url: text[:3000] for url, text in (output.get("sources_text") or {}).items()
    }
    msgs = [
        Message(role="system", content=prompts.JUDGE_SYSTEM),
        Message(
            role="user", content=prompts.judge_user(criterion, input_, shown_output, reference)
        ),
    ]
    resp = c.complete(msgs, model=s.judge_model, max_tokens=200)
    m = re.search(r"\{.*\}", resp.text, re.DOTALL)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        parsed = {}
    if "pass" not in parsed:
        return {
            "pass": False,
            "reason": f"unparseable judge output: {resp.text[:120]}",
            "judge": s.judge_model,
        }
    return {
        "pass": bool(parsed["pass"]),
        "reason": str(parsed.get("reason", "")),
        "judge": s.judge_model,
    }
