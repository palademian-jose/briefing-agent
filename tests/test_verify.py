"""Verifier: positive and negative cases, and the briefing render/parse round trip."""

from __future__ import annotations

from briefing_agent.briefing import Briefing, Fact, Source, parse_briefing
from briefing_agent.tools.verify import lexical_support, verify_briefing

PAGE_A = (
    "A Bloom filter is a space-efficient probabilistic data structure that is used to test "
    "whether an element is a member of a set. False positive matches are possible, but false "
    "negatives are not. Elements can be added to the set, but not removed."
)
PAGE_B = (
    "Consistent hashing is a special kind of hashing technique such that when a hash table is "
    "resized, only a fraction of the keys need to be remapped. It is used in distributed caches."
)
URL_A = "https://en.wikipedia.org/wiki/Bloom_filter"
URL_B = "https://en.wikipedia.org/wiki/Consistent_hashing"
PAGES = {URL_A: PAGE_A, URL_B: PAGE_B}


def good_briefing() -> Briefing:
    return Briefing(
        topic="Bloom filter",
        summary=(
            "A Bloom filter is a space-efficient probabilistic data structure used to test set "
            "membership, trading a small false-positive rate for memory."
        ),
        facts=[
            Fact(
                text="A Bloom filter is a space-efficient probabilistic data structure",
                citations=[1],
            ),
            Fact(text="It is used to test whether an element is a member of a set", citations=[1]),
            Fact(
                text="False positive matches are possible, but false negatives are not",
                citations=[1],
            ),
            Fact(text="Elements can be added to the set, but not removed", citations=[1]),
            Fact(
                text="Consistent hashing remaps only a fraction of the keys when a hash table is resized",
                citations=[2],
            ),
        ],
        open_questions=["How does the false-positive rate scale?"],
        sources=[
            Source(n=1, title="Bloom filter", url=URL_A),
            Source(n=2, title="Consistent hashing", url=URL_B),
        ],
    )


def test_render_parse_round_trip() -> None:
    b = good_briefing()
    parsed = parse_briefing(b.render())
    assert parsed.topic == "Bloom filter"
    assert [f.text for f in parsed.facts] == [f.text for f in b.facts]
    assert [f.citations for f in parsed.facts] == [f.citations for f in b.facts]
    assert [(s.n, s.url) for s in parsed.sources] == [(1, URL_A), (2, URL_B)]
    assert parsed.open_questions == b.open_questions


def test_good_briefing_passes() -> None:
    v = verify_briefing(good_briefing().render(), PAGES, threshold=0.6, min_facts=5, max_facts=8)
    assert v.ok, v.issues
    assert v.fact_count == 5 and v.source_count == 2 and v.min_support >= 0.6


def test_dangling_citation_fails() -> None:
    b = good_briefing()
    b.facts[0].citations = [7]
    v = verify_briefing(b.render(), PAGES)
    assert not v.ok and {i.code for i in v.issues} == {"citation_unmapped"}
    assert v.issues[0].fact_index == 0 and v.issues[0].citation == 7


def test_unfetched_source_fails() -> None:
    b = good_briefing()
    b.sources[1].url = "https://en.wikipedia.org/wiki/Not_fetched"
    v = verify_briefing(b.render(), PAGES)
    assert "source_not_fetched" in {i.code for i in v.issues}


def test_unsupported_fact_fails() -> None:
    b = good_briefing()
    b.facts[2] = Fact(
        text="Bloom filters were invented on Mars by the Zeta Consortium in 1897", citations=[1]
    )
    v = verify_briefing(b.render(), PAGES)
    codes = {i.code for i in v.issues}
    assert codes == {"fact_unsupported"}
    issue = v.issues[0]
    assert issue.fact_index == 2 and issue.score is not None and issue.score < 0.6


def test_wrong_citation_for_true_fact_fails() -> None:
    b = good_briefing()
    b.facts[4].citations = [1]  # consistent hashing fact cited to the Bloom filter page
    v = verify_briefing(b.render(), PAGES)
    assert {i.code for i in v.issues} == {"fact_unsupported"}


def test_duplicate_source_fails() -> None:
    b = good_briefing()
    b.sources.append(Source(n=3, title="Bloom filter again", url=URL_A))
    v = verify_briefing(b.render(), PAGES)
    assert "duplicate_source" in {i.code for i in v.issues}


def test_missing_summary_and_too_few_facts_fail() -> None:
    b = good_briefing()
    b.summary = "Short."
    b.facts = b.facts[:2]
    v = verify_briefing(b.render(), PAGES, min_facts=5)
    assert {"no_summary", "too_few_facts"} <= {i.code for i in v.issues}


def test_fact_without_citation_fails() -> None:
    b = good_briefing()
    b.facts[1].citations = []
    v = verify_briefing(b.render(), PAGES)
    assert "fact_no_citation" in {i.code for i in v.issues}


def test_threshold_is_configurable() -> None:
    fact = "Bloom filters test membership using probabilistic hashing bits"  # partial overlap
    score = lexical_support(fact, PAGE_A)
    assert 0.3 < score < 0.9
    b = good_briefing()
    b.facts[0] = Fact(text=fact, citations=[1])
    assert verify_briefing(b.render(), PAGES, threshold=score - 0.05).ok
    assert not verify_briefing(b.render(), PAGES, threshold=score + 0.05).ok
