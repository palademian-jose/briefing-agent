"""verify_briefing: the deterministic verifier.

It is the one component in the system that cannot hallucinate. It checks structure and
grounding against the pages this run actually fetched, so a confident-sounding briefing with
an invented source, a dangling [7], or a fact its cited page does not support fails here, not
in front of the reader.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from briefing_agent.briefing import Briefing, parse_briefing

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the and or of to in on at for by with from as is are was were be been being this
    that these those it its into than then which who whom whose where when while also such
    can may might will would should could not no nor but if so because about over under
    between through during before after above below up down out off very more most some any
    each both few other same only own too just has have had having does do did doing there
    their them they he she his her we our you your i me my one two three per via use used
    using""".split()
)


def content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 3 and t not in STOPWORDS]


def lexical_support(fact: str, page_text: str) -> float:
    """Fraction of the fact's content tokens that occur in the page. Loose by design: the
    model paraphrases; a 0.6 threshold catches invented numbers, names and claims while
    tolerating rewording."""
    toks = content_tokens(fact)
    if not toks:
        return 0.0
    page = set(_TOKEN.findall(page_text.lower()))
    hits = sum(1 for t in toks if t in page)
    return hits / len(toks)


class Issue(BaseModel):
    code: str
    message: str
    fact_index: int | None = None
    citation: int | None = None
    score: float | None = None


class Verification(BaseModel):
    ok: bool
    issues: list[Issue] = Field(default_factory=list)
    fact_count: int = 0
    source_count: int = 0
    min_support: float | None = None

    def as_result(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def verify_briefing(
    markdown: str,
    fetched_pages: dict[str, str],
    *,
    threshold: float = 0.6,
    min_facts: int = 5,
    max_facts: int = 8,
) -> Verification:
    """``fetched_pages`` maps URL -> extracted text for every successful fetch this run."""
    b: Briefing = parse_briefing(markdown)
    issues: list[Issue] = []

    if len(b.summary.split()) < 15:
        issues.append(Issue(code="no_summary", message="Summary section missing or too short"))

    if len(b.facts) < min_facts:
        issues.append(
            Issue(
                code="too_few_facts", message=f"{len(b.facts)} key facts, need at least {min_facts}"
            )
        )
    if len(b.facts) > max_facts:
        issues.append(
            Issue(code="too_many_facts", message=f"{len(b.facts)} key facts, at most {max_facts}")
        )

    by_n = {s.n: s for s in b.sources}
    seen_urls: set[str] = set()
    for s in b.sources:
        if s.url in seen_urls:
            issues.append(
                Issue(
                    code="duplicate_source", message=f"Duplicate source URL {s.url}", citation=s.n
                )
            )
        seen_urls.add(s.url)
        if s.url not in fetched_pages:
            issues.append(
                Issue(
                    code="source_not_fetched",
                    message=f"Source [{s.n}] {s.url} was not fetched during this run",
                    citation=s.n,
                )
            )
    if len({s.n for s in b.sources}) != len(b.sources):
        issues.append(Issue(code="duplicate_source", message="Duplicate source numbers"))

    min_support: float | None = None
    for i, fact in enumerate(b.facts):
        if not fact.citations:
            issues.append(
                Issue(code="fact_no_citation", message="Key fact has no [n] citation", fact_index=i)
            )
            continue
        for n in fact.citations:
            src = by_n.get(n)
            if src is None:
                issues.append(
                    Issue(
                        code="citation_unmapped",
                        message=f"[{n}] has no entry in Sources",
                        fact_index=i,
                        citation=n,
                    )
                )
                continue
            page = fetched_pages.get(src.url)
            if page is None:
                continue  # already reported as source_not_fetched
            score = lexical_support(fact.text, page)
            min_support = score if min_support is None else min(min_support, score)
            if score < threshold:
                issues.append(
                    Issue(
                        code="fact_unsupported",
                        message=(
                            f"Fact {i + 1} has lexical support {score:.2f} < {threshold} in [{n}]"
                        ),
                        fact_index=i,
                        citation=n,
                        score=round(score, 3),
                    )
                )

    return Verification(
        ok=not issues,
        issues=issues,
        fact_count=len(b.facts),
        source_count=len(b.sources),
        min_support=None if min_support is None else round(min_support, 3),
    )
