"""Stage prompts. Each user turn starts with a ``TASK:`` line so that stages are explicit
for real models and parseable by the fake policy. Keep these in one place: prompt changes
are gated by the eval in CI."""

from __future__ import annotations

import json
from typing import Any

SEARCH_SYSTEM = """You are the source-discovery stage of a research-briefing agent.
Your only job is to find and fetch a small set of relevant Wikipedia articles for the topic.
Rules:
- Call search_wikipedia first. Prefer keyword queries. If a search returns no results or an
  error that is not retryable, simplify the query instead of repeating it.
- Fetch the most relevant results one at a time with fetch_page until you have the target
  number of sources. Skip pages that fail permanently; never fetch the same URL twice.
- Tool errors are JSON objects with ok=false and an error.kind of TRANSIENT, PERMANENT or
  REQUIRES_HUMAN. The executor already retried TRANSIENT errors; when you see one, treat it as
  permanent and choose a different action.
- When you have enough sources (or no more candidates), reply with the single word DONE and
  no tool call. Do not write the briefing in this stage."""

THINK_SYSTEM = """You are the planning step of a research agent. Given the observed state,
write at most two short lines: what has been achieved and the single next action
(search, fetch <url>, or finish). No prose, no markdown."""

COMPOSE_SYSTEM = """You write research briefings in Markdown using only the sources provided.
Output exactly these sections: '# Briefing: <topic>', '## Summary' (one paragraph),
'## Key facts' (a numbered list; every item ends with one or more citations like [1]),
'## Open questions' (bullet list), '## Sources' (lines of the form '[n] Title — URL' copied
verbatim from the provided sources). Every fact must be directly supported by the cited
source text; do not add outside knowledge, numbers or names that are not in the sources."""

REPAIR_SYSTEM = """You fix a research briefing so that it passes a deterministic verifier.
Keep the same section structure. Remove or rewrite unsupported facts using only the provided
source text, drop citations that do not map to a source, drop sources that were not fetched,
remove duplicate sources, and keep the fact count within the stated bounds. Return the full
corrected briefing only."""

JUDGE_SYSTEM = """You are a strict binary grader. Answer with a JSON object
{"pass": true|false, "reason": "<one sentence>"} and nothing else."""


def search_user(topic: str, audience: str, target_sources: int) -> str:
    return (
        "TASK: discover_sources\n"
        f"Topic: {topic}\n"
        f"Audience: {audience}\n"
        f"Target sources: {target_sources}\n"
        "Find and fetch the sources now."
    )


def think_user(topic: str, observed: dict[str, Any]) -> str:
    return (
        f"TASK: think\nTopic: {topic}\nObserved state: {json.dumps(observed, ensure_ascii=False)}"
    )


def sources_block(sources: list[dict[str, str]]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"### [{i}] {s['title']} — {s['url']}\n{s['text']}")
    return "\n\n".join(parts)


def compose_user(
    topic: str, audience: str, sources: list[dict[str, str]], min_facts: int, max_facts: int
) -> str:
    return (
        "TASK: compose\n"
        f"Topic: {topic}\n"
        f"Audience: {audience}\n"
        f"Facts: between {min_facts} and {max_facts}\n"
        "Sources:\n\n"
        f"{sources_block(sources)}"
    )


def repair_user(
    topic: str,
    briefing: str,
    issues: list[dict[str, Any]],
    sources: list[dict[str, str]],
    min_facts: int,
    max_facts: int,
) -> str:
    return (
        "TASK: repair\n"
        f"Topic: {topic}\n"
        f"Facts: between {min_facts} and {max_facts}\n"
        f"Verifier issues: {json.dumps(issues, ensure_ascii=False)}\n"
        "Sources:\n\n"
        f"{sources_block(sources)}\n\n"
        "Briefing to fix:\n\n"
        f"{briefing}"
    )


def judge_user(
    criterion: str, input_: dict[str, Any], output: dict[str, Any], reference: dict | None
) -> str:
    return (
        "TASK: judge\n"
        f"Criterion: {criterion}\n"
        f"Input: {json.dumps(input_, ensure_ascii=False)}\n"
        f"Output: {json.dumps(output, ensure_ascii=False)}\n"
        f"Reference: {json.dumps(reference, ensure_ascii=False) if reference else 'none'}"
    )
