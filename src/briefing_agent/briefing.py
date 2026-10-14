"""The briefing document: a small structured model plus its Markdown rendering and parser.

The agent composes Markdown (that is what models are good at); the verifier parses it back
into this structure. Keeping render/parse in one module keeps the two in lock-step.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_CITE = re.compile(r"\[(\d+)\]")
_FACT_LINE = re.compile(r"^\s*(?:\d+\.|-|\*)\s+(.*\S)\s*$")
_SOURCE_LINE = re.compile(r"^\s*\[(\d+)\]\s+(.*?)\s+(?:—|-|–)\s+(https?://\S+)\s*$")
_HEADING = re.compile(r"^##\s+(.+?)\s*$")


class Fact(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)


class Source(BaseModel):
    n: int
    title: str
    url: str


class Briefing(BaseModel):
    topic: str
    summary: str
    facts: list[Fact]
    open_questions: list[str]
    sources: list[Source]

    def render(self) -> str:
        lines = [f"# Briefing: {self.topic}", "", "## Summary", self.summary.strip(), ""]
        lines.append("## Key facts")
        for i, fact in enumerate(self.facts, 1):
            cites = "".join(f"[{c}]" for c in fact.citations)
            lines.append(f"{i}. {fact.text.strip()} {cites}".rstrip())
        lines += ["", "## Open questions"]
        lines += [f"- {q.strip()}" for q in self.open_questions]
        lines += ["", "## Sources"]
        lines += [f"[{s.n}] {s.title} — {s.url}" for s in self.sources]
        return "\n".join(lines) + "\n"

    def source_urls(self) -> list[str]:
        return [s.url for s in self.sources]


def _sections(markdown: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = "_preamble"
    for raw in markdown.splitlines():
        m = _HEADING.match(raw)
        if m:
            current = m.group(1).strip().lower()
            out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(raw)
    return out


def parse_briefing(markdown: str, topic: str = "") -> Briefing:
    """Parse the Markdown briefing. Tolerant of blank lines and bullet styles; strict about
    the section names, which are part of the output contract."""
    sec = _sections(markdown)
    if not topic:
        for line in sec.get("_preamble", []):
            if line.startswith("# "):
                topic = re.sub(r"^#\s*(Briefing:)?\s*", "", line).strip()
                break
    summary = " ".join(line.strip() for line in sec.get("summary", []) if line.strip())
    facts: list[Fact] = []
    for line in sec.get("key facts", []):
        m = _FACT_LINE.match(line)
        if not m:
            continue
        body = m.group(1)
        cites = [int(c) for c in _CITE.findall(body)]
        text = _CITE.sub("", body).strip()
        facts.append(Fact(text=text, citations=cites))
    questions = [
        m.group(1) for line in sec.get("open questions", []) if (m := _FACT_LINE.match(line))
    ]
    sources: list[Source] = []
    for line in sec.get("sources", []):
        m = _SOURCE_LINE.match(line)
        if m:
            sources.append(Source(n=int(m.group(1)), title=m.group(2).strip(), url=m.group(3)))
    return Briefing(
        topic=topic, summary=summary, facts=facts, open_questions=questions, sources=sources
    )
