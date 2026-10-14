"""The judge: lexical rubric on the fake provider, evalkit.judge.Judge on a real one."""

from __future__ import annotations

from typing import Any

from llmops.llm import LLMResponse

from briefing_agent.evals.judge import CRITERION_USEFUL_FAITHFUL, judge
from briefing_agent.settings import Settings

BRIEFING = """# Briefing: Bloom filter

## Summary
A Bloom filter is a space-efficient probabilistic data structure used to test whether an element
is a member of a set, allowing false positives but never false negatives.

## Key facts
1. A Bloom filter never gives false negatives. [1]
2. It can return false positives. [1]
3. It uses several hash functions over a bit array. [1]

## Sources
[1] Bloom filter — https://en.wikipedia.org/wiki/Bloom_filter
"""
SOURCE = (
    "A Bloom filter is a space-efficient probabilistic data structure. A Bloom filter never gives "
    "false negatives. It can return false positives. It uses several hash functions over a bit "
    "array."
)
OUT = {
    "briefing_md": BRIEFING,
    "sources_text": {"https://en.wikipedia.org/wiki/Bloom_filter": SOURCE},
}
INP = {"topic": "Bloom filter"}


class _Scripted:
    """llmops-style client returning queued texts and recording the prompts."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    def complete(self, messages, *, tools=None, model=None, temperature=0.0, max_tokens=1024):
        self.prompts.append("\n".join(m.text() for m in messages))
        self.models.append(model)
        return LLMResponse(text=self.texts.pop(0), model=model or "stub")


def test_fake_provider_uses_the_lexical_rubric(settings: Settings) -> None:
    verdict = judge(CRITERION_USEFUL_FAITHFUL, INP, OUT, None, settings=settings)
    assert verdict["pass"] is True and verdict["judge"] == "lexical-fallback"
    bad = {**OUT, "sources_text": {"https://en.wikipedia.org/wiki/Bloom_filter": "unrelated text"}}
    failed = judge(CRITERION_USEFUL_FAITHFUL, INP, bad, None, settings=settings)
    assert failed["pass"] is False and "not supported" in failed["reason"]


def test_run_client_only_selects_the_judge_and_is_never_sent_judge_prompts(
    settings: Settings,
) -> None:
    from briefing_agent.llm.fake import FakeClient

    run_client = FakeClient()  # the eval passes the generator client of the run
    verdict = judge(CRITERION_USEFUL_FAITHFUL, INP, OUT, None, client=run_client, settings=settings)
    assert verdict["judge"] == "lexical-fallback" and verdict["pass"] is True


def test_real_provider_goes_through_evalkit_judge(settings: Settings) -> None:
    real = settings.model_copy(update={"llm_provider": "openrouter", "openrouter_api_key": "k"})
    client = _Scripted('{"pass": true, "reason": "every fact is in the source"}')
    verdict = judge(
        CRITERION_USEFUL_FAITHFUL,
        INP,
        OUT,
        {"must_mention": ["false"]},
        judge_client=client,
        settings=real,
    )
    assert verdict == {
        "pass": True,
        "reason": "every fact is in the source",
        "judge": "openai/gpt-4.1-mini",
    }
    assert client.models == ["openai/gpt-4.1-mini"]
    prompt = client.prompts[0]
    assert f"<criterion>{CRITERION_USEFUL_FAITHFUL}</criterion>" in prompt
    assert "Bloom filter" in prompt, "the briefing and its sources reach the judge"
    assert "<reference>\nfalse\n</reference>" in prompt, "the reference is rendered"


def test_real_provider_malformed_reply_gets_one_repair_turn(settings: Settings) -> None:
    client = _Scripted("looks fine to me", '{"pass": false, "reason": "fact 2 invented"}')
    verdict = judge(
        CRITERION_USEFUL_FAITHFUL, INP, OUT, None, judge_client=client, settings=settings
    )
    assert verdict["pass"] is False and verdict["reason"] == "fact 2 invented"
    assert len(client.prompts) == 2


def test_real_provider_still_malformed_after_repair_fails_closed(settings: Settings) -> None:
    client = _Scripted("nope", "still nope")
    verdict: dict[str, Any] = judge(
        CRITERION_USEFUL_FAITHFUL, INP, OUT, None, judge_client=client, settings=settings
    )
    assert verdict["pass"] is False and "unparseable judge output" in verdict["reason"]


def test_source_text_is_truncated_for_the_judge(settings: Settings) -> None:
    client = _Scripted('{"pass": true, "reason": "ok"}')
    long_out = {**OUT, "sources_text": {"https://x.test/a": "word " * 2000}}
    judge(CRITERION_USEFUL_FAITHFUL, INP, long_out, None, judge_client=client, settings=settings)
    assert "word " * 600 in client.prompts[0] and "word " * 700 not in client.prompts[0]
