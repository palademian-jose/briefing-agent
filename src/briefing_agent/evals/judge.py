"""Binary, criterion-specific judge (shared contract 4).

``judge(criterion, input, output, reference) -> {"pass": bool, "reason": str}``

With a real provider the verdict comes from ``evalkit.judge.Judge`` (llm-evalops) on
``JUDGE_MODEL``, a different family than the generator: the criterion text goes through
evalkit's default prompt template, the reply is parsed strictly as ``{"pass", "reason"}`` and a
malformed reply gets one repair turn. With the fake provider the verdict comes from a
deterministic lexical rubric so the eval pipeline runs offline (evalkit's fake judge is a
different heuristic, so switching would move the committed fake-provider baseline); the
calibration script measures how far that rubric is from human labels (Cohen's kappa) and probes
order and length bias.
"""

from __future__ import annotations

import re
from typing import Any

from briefing_agent.briefing import parse_briefing
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


def _llmops_client(s: Settings) -> Any:
    """An llmops client for the judge role, built from this repo's settings."""
    from llmops.llm import get_client
    from llmops.settings import Settings as LlmopsSettings

    return get_client(
        LlmopsSettings(
            llm_provider=s.effective_provider,
            llm_model=s.llm_model,
            judge_model=s.judge_model,
            cheap_model=s.cheap_model,
            openrouter_api_key=s.openrouter_api_key or None,
            openrouter_site_url=s.openrouter_site_url or None,
            openrouter_app_name=s.openrouter_app_name or None,
            anthropic_api_key=s.anthropic_api_key or None,
            openai_api_key=s.openai_api_key or None,
        ),
        role="judge",
    )


def judge(
    criterion: str,
    input_: dict[str, Any],
    output: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    judge_client: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Contract 4.

    ``client`` is the run's generator client: its ``provider`` decides between the lexical
    rubric (fake) and the evalkit judge (anything else); it is never sent judge prompts, because
    its message types are this repo's own. ``judge_client`` is an llmops-style client
    (``complete(messages, model=...)``) to send them to instead of the one built from the
    settings, and forces the evalkit judge.
    """
    s = settings or get_settings()
    provider = getattr(client, "provider", None) or s.effective_provider
    if judge_client is None and provider == "fake":
        verdict = lexical_judge(
            criterion, input_, output, reference, threshold=s.verify_match_threshold
        )
        verdict["judge"] = "lexical-fallback"
        return verdict
    from evalkit.judge import Judge, check_family

    check_family(s.llm_model, s.judge_model, strict=False)
    shown_output = {k: v for k, v in output.items() if k != "sources_text"}
    shown_output["sources_text"] = {
        url: text[:3000] for url, text in (output.get("sources_text") or {}).items()
    }
    evalkit_judge = Judge(criterion, judge_client or _llmops_client(s), model=s.judge_model)
    try:
        verdict = evalkit_judge.judge(input_, shown_output, reference)
    except ValueError as exc:  # JudgeParseError after the repair turn
        return {"pass": False, "reason": f"unparseable judge output: {exc}", "judge": s.judge_model}
    return {**verdict, "judge": s.judge_model}
