"""Deterministic fake provider.

It is a *policy over the observed conversation*, not a transcript: it reads the tool results
in the messages and decides the next call, so injected faults change the trajectory
(re-search after an empty result, skip a 404'd page, retry handled by the executor, repair
after a verifier failure). Stages are recognised by the ``TASK:`` line of the user turn.

Knobs (``misbehave``) exist only to exercise guard rails in tests and demos:
``bad_call`` (first search violates the schema), ``unknown_tool`` (calls a tool that does not
exist), ``hallucinate`` (compose emits an unsupported fact and a dangling citation).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from briefing_agent.briefing import Briefing, Fact, Source, parse_briefing
from briefing_agent.llm.base import LLMResponse, Message, ToolCall, ToolSpec, Usage, estimate_tokens
from briefing_agent.tools.faults import FaultInjector
from briefing_agent.tools.verify import content_tokens

_TASK = re.compile(r"^TASK:\s*(\w+)", re.MULTILINE)
_FIELD = re.compile(r"^(Topic|Audience|Target sources|Facts|Criterion):\s*(.+)$", re.MULTILINE)
_SOURCE_HDR = re.compile(r"^### \[(\d+)\] (.+?) — (https?://\S+)\s*$", re.MULTILINE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass
class Observed:
    topic: str = ""
    target: int = 3
    queries: list[str] = field(default_factory=list)
    candidates: list[dict[str, str]] = field(default_factory=list)  # ordered unique
    fetched_ok: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    last_call: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    searches_ok: int = 0


def _fields(text: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _FIELD.findall(text)}


def observe(messages: list[Message]) -> Observed:
    obs = Observed()
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for m in messages:
        if (
            m.role == "user"
            and isinstance(m.content, str)
            and "TASK: discover_sources" in m.content
        ):
            f = _fields(m.content)
            obs.topic = f.get("Topic", obs.topic)
            obs.target = int(f.get("Target sources", obs.target) or 3)
        for block in m.tool_uses():
            calls[block["id"]] = {"name": block["name"], "args": block["input"], "result": None}
            order.append(block["id"])
        if m.role == "tool" and m.tool_call_id in calls:
            try:
                calls[m.tool_call_id]["result"] = json.loads(m.text())
            except json.JSONDecodeError:
                calls[m.tool_call_id]["result"] = {"ok": False, "error": {"kind": "PERMANENT"}}
    seen: set[str] = set()
    for cid in order:
        c = calls[cid]
        res = c["result"] or {}
        obs.last_call, obs.last_result = c, res
        if c["name"] == "search_wikipedia":
            obs.queries.append(str(c["args"].get("query", "")))
            if res.get("ok") and res.get("results"):
                obs.searches_ok += 1
                for r in res["results"]:
                    if r["url"] not in seen:
                        seen.add(r["url"])
                        obs.candidates.append({"title": r["title"], "url": r["url"]})
        elif c["name"] == "fetch_page":
            url = str(c["args"].get("url", ""))
            if res.get("ok"):
                if url not in obs.fetched_ok:
                    obs.fetched_ok.append(url)
            elif res:
                obs.failed.append(url)
    return obs


def broaden(topic: str, attempt: int) -> str:
    """Progressively simpler queries: strip qualifiers, then keep the head words."""
    base = re.sub(r"\(.*?\)", "", topic).strip()
    base = re.split(r"\s+(?:in|for|of|and|vs\.?|versus)\s+", base, maxsplit=1)[0].strip() or topic
    words = base.split()
    if attempt <= 0:
        return topic
    if attempt == 1:
        return base if base.lower() != topic.lower() else " ".join(words[:3])
    if attempt == 2:
        return " ".join(words[:2]) if len(words) >= 2 else words[0]
    return words[0]


def _sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        for s in _SENT_SPLIT.split(para.strip()):
            s = s.strip()
            if 60 <= len(s) <= 220 and len(content_tokens(s)) >= 6 and "may refer to" not in s:
                out.append(s)
    return out


def _parse_sources(text: str) -> list[dict[str, str]]:
    matches = list(_SOURCE_HDR.finditer(text))
    sources = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end]
        if "\n\nBriefing to fix:" in body:
            body = body.split("\n\nBriefing to fix:", 1)[0]
        sources.append(
            {"n": int(m.group(1)), "title": m.group(2), "url": m.group(3), "text": body.strip()}
        )
    return sources


class FakeClient:
    provider = "fake"

    def __init__(
        self,
        default_model: str = "anthropic/claude-sonnet-4.5",
        *,
        misbehave: str | None = None,
        faults: FaultInjector | None = None,
    ) -> None:
        self.default_model = default_model
        self.misbehave = misbehave
        self.faults = faults or FaultInjector()
        self.calls = 0
        self._misbehaved = False

    # -- public contract ---------------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls += 1
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        task = ""
        if last_user is not None:
            mt = _TASK.search(last_user.text())
            task = mt.group(1) if mt else ""
        first_user = next((m for m in messages if m.role == "user"), None)
        if task == "" and first_user is not None:
            mt = _TASK.search(first_user.text())
            task = mt.group(1) if mt else ""

        text, calls = "", []
        if task == "discover_sources":
            text, calls = self._discover(messages, tools or [])
        elif task == "think":
            text = self._think(last_user.text() if last_user else "")
        elif task == "compose":
            text = self._compose(last_user.text() if last_user else "")
        elif task == "repair":
            text = self._repair(last_user.text() if last_user else "")
        elif task == "judge":
            text = self._judge(last_user.text() if last_user else "")
        else:
            text = "OK"

        in_tokens = sum(estimate_tokens(m.text()) for m in messages) + sum(
            estimate_tokens(json.dumps(t.input_schema)) for t in (tools or [])
        )
        out_tokens = estimate_tokens(text) + sum(
            estimate_tokens(json.dumps(c.arguments)) for c in calls
        )
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=in_tokens, output_tokens=out_tokens),
            model=model or self.default_model,
            latency_ms=0.0,
            stop_reason="tool_use" if calls else "end_turn",
        )

    # -- stage policies ----------------------------------------------------------------
    def _call(self, name: str, **args: Any) -> ToolCall:
        return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)

    def _discover(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> tuple[str, list[ToolCall]]:
        obs = observe(messages)
        tool_names = {t.name for t in tools}

        # Demonstrate the validation layer once, then behave.
        if self.misbehave in ("bad_call", "unknown_tool") and not self._misbehaved:
            self._misbehaved = True
            if self.misbehave == "bad_call":
                return "Searching.", [self._call("search_wikipedia", query=obs.topic, limit=50)]
            return "Searching.", [self._call("search_web", query=obs.topic)]

        last_err = (obs.last_result or {}).get("error") if obs.last_result else None
        # An invalid call came back: re-issue it corrected.
        if last_err and last_err.get("code") in ("invalid_arguments", "unknown_tool"):
            return "Correcting the call.", [
                self._call("search_wikipedia", query=obs.topic, limit=5)
            ]

        untried = [
            c
            for c in obs.candidates
            if c["url"] not in obs.fetched_ok and c["url"] not in obs.failed
        ]

        if len(obs.fetched_ok) >= obs.target:
            return f"DONE: {len(obs.fetched_ok)} sources fetched.", []

        if untried and "fetch_page" in tool_names:
            nxt = untried[0]
            return f"Fetching {nxt['title']}.", [
                self._call("fetch_page", url=nxt["url"], max_chars=8000)
            ]

        # No candidate left: search (again) with a progressively broader query.
        if (
            "search_wikipedia" in tool_names
            and len(obs.queries) < 4
            and len(obs.fetched_ok) < max(2, obs.target)
        ):
            query = broaden(obs.topic, len(obs.queries))
            if query.lower() in (q.lower() for q in obs.queries):
                query = broaden(obs.topic, len(obs.queries) + 1)
            if query.lower() not in (q.lower() for q in obs.queries):
                why = "no results yet" if not obs.candidates else "need more sources"
                return f"Re-planning ({why}): searching '{query}'.", [
                    self._call("search_wikipedia", query=query, limit=5)
                ]

        if obs.fetched_ok:
            return f"DONE: {len(obs.fetched_ok)} sources fetched (no more candidates).", []
        return "DONE: no sources could be found.", []

    def _think(self, text: str) -> str:
        f = _fields(text)
        m = re.search(r"Observed state: (\{.*\})", text, re.DOTALL)
        state = json.loads(m.group(1)) if m else {}
        fetched = state.get("fetched", [])
        candidates = state.get("candidates", [])
        target = state.get("target", 3)
        done = f"Have {len(fetched)}/{target} sources."
        if len(fetched) >= target:
            return f"{done} Next: finish."
        untried = [c for c in candidates if c not in fetched and c not in state.get("failed", [])]
        if untried:
            return f"{done} Next: fetch {untried[0]}"
        return f"{done} Next: search Wikipedia for '{f.get('Topic', '')}'."

    def _compose(self, text: str) -> str:
        f = _fields(text)
        topic = f.get("Topic", "")
        lo, hi = 5, 8
        mf = re.match(r"between (\d+) and (\d+)", f.get("Facts", ""))
        if mf:
            lo, hi = int(mf.group(1)), int(mf.group(2))
        sources = _parse_sources(text)
        briefing = self._build(topic, sources, lo, hi, used=set())
        if self.misbehave == "hallucinate" or self.faults.should_inject(
            "compose_hallucinate", {"topic": topic}
        ):
            fake_fact = Fact(
                text=(
                    f"{topic} was first deployed on the Kepler-442 colony in 1897 by the "
                    "Zeta Consortium, cutting latency by 400 percent"
                ),
                citations=[1],
            )
            if len(briefing.facts) >= 3:
                briefing.facts[2] = fake_fact
            else:
                briefing.facts.append(fake_fact)
            if briefing.facts:
                briefing.facts[-1].citations = (
                    [max(s.n for s in briefing.sources) + 5] if briefing.sources else [9]
                )
        return briefing.render()

    def _repair(self, text: str) -> str:
        f = _fields(text)
        topic = f.get("Topic", "")
        lo, hi = 5, 8
        mf = re.match(r"between (\d+) and (\d+)", f.get("Facts", ""))
        if mf:
            lo, hi = int(mf.group(1)), int(mf.group(2))
        sources = _parse_sources(text)
        issues_m = re.search(r"Verifier issues: (\[.*?\])\nSources:", text, re.DOTALL)
        issues = json.loads(issues_m.group(1)) if issues_m else []
        old_md = text.split("Briefing to fix:\n\n", 1)[-1]
        old = parse_briefing(old_md, topic)
        fetched_urls = {s["url"] for s in sources}
        bad_facts = {i["fact_index"] for i in issues if i.get("fact_index") is not None}
        valid_n = {s.n for s in old.sources if s.url in fetched_urls}
        kept = [
            fct
            for i, fct in enumerate(old.facts)
            if i not in bad_facts and fct.citations and all(c in valid_n for c in fct.citations)
        ]
        used = {fct.text for fct in kept}
        rebuilt = self._build(topic, sources, lo, hi, used=used, seed_facts=kept)
        rebuilt.summary = old.summary or rebuilt.summary
        return rebuilt.render()

    def _build(
        self,
        topic: str,
        sources: list[dict[str, Any]],
        lo: int,
        hi: int,
        *,
        used: set[str],
        seed_facts: list[Fact] | None = None,
    ) -> Briefing:
        want = min(hi, max(lo, 6))
        facts: list[Fact] = list(seed_facts or [])
        pools = [(s["n"], _sentences(s["text"])) for s in sources]
        idx = 0
        while len(facts) < want and any(pool for _, pool in pools):
            n, pool = pools[idx % len(pools)]
            idx += 1
            while pool:
                s = pool.pop(0)
                if s not in used:
                    used.add(s)
                    facts.append(Fact(text=s.rstrip("."), citations=[n]))
                    break
        first_text = sources[0]["text"] if sources else ""
        summary_sents = _SENT_SPLIT.split(first_text.split("\n")[0])[:2] if first_text else []
        summary = " ".join(summary_sents) or f"No source text was available for {topic}."
        questions = [
            f"How does {topic} compare with the main alternatives in practice?",
            f"What are the known limitations or failure modes of {topic}?",
            "Which primary sources beyond Wikipedia corroborate these facts?",
        ]
        return Briefing(
            topic=topic,
            summary=summary,
            facts=facts[:hi],
            open_questions=questions,
            sources=[Source(n=s["n"], title=s["title"], url=s["url"]) for s in sources],
        )

    def _judge(self, text: str) -> str:
        # The judge module already applies a lexical fallback for the fake provider; this
        # path exists so the client contract is complete.
        return json.dumps(
            {"pass": True, "reason": "fake provider: lexical fallback handled by judge module"}
        )
