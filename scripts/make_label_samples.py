"""Generate candidate samples for evals/human_labels.jsonl.

Produces briefings in four conditions from the fake pipeline over recorded fixtures:

- clean: the normal output                                        -> expected label: pass
- hallucinated: one invented fact and one dangling citation       -> fail
- paraphrased: a true fact rewritten with synonyms (lexically weak) -> pass
- off_topic: body taken from a different topic's run               -> fail

The script writes the samples with a provisional label; the shipped file was then checked by
hand and the ``note`` field records the reasoning. Re-run only if you intend to relabel.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from briefing_agent.briefing import parse_briefing  # noqa: E402
from briefing_agent.llm.fake import FakeClient  # noqa: E402
from briefing_agent.pipeline import Runner  # noqa: E402
from briefing_agent.settings import Settings  # noqa: E402
from briefing_agent.store import Store  # noqa: E402

TOPICS = [
    "Bloom filter",
    "Consistent hashing",
    "PageRank",
    "Merkle tree",
    "CAP theorem",
    "Beam search",
    "Gradient descent",
    "Public-key cryptography",
    "Kalman filter",
    "Word2vec",
]
SYNONYMS = {
    "uses": "relies on",
    "used": "employed",
    "algorithm": "procedure",
    "method": "technique",
    "data structure": "structure for holding data",
    "number": "count",
    "set": "collection",
    "element": "item",
    "function": "mapping",
    "value": "quantity",
    "values": "quantities",
    "system": "setup",
    "systems": "setups",
    "allows": "lets",
    "called": "known as",
    "compute": "work out",
    "computed": "worked out",
    "stored": "kept",
    "large": "big",
    "small": "little",
    "designed": "built",
    "process": "workflow",
    "key": "identifier",
    "keys": "identifiers",
    "network": "web",
    "model": "representation",
    "error": "mistake",
    "estimate": "guess",
}


def paraphrase(text: str) -> str:
    out = text
    for a, b in SYNONYMS.items():
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.IGNORECASE)
    return out


def main() -> int:
    settings = Settings(
        db_path=ROOT / "data" / "labels.db",
        trace_path=ROOT / "traces" / "labels.jsonl",
        checkpoint_dir=ROOT / "checkpoints" / "labels",
    )
    store = Store(settings.db_path)
    clean = Runner(settings, store=store, sleep=lambda _: None)
    halluc = Runner(
        settings,
        store=store,
        sleep=lambda _: None,
        client=FakeClient(settings.llm_model, misbehave="hallucinate"),
    )
    samples = []
    runs = {t: clean.run(t, {"max_sources": 3}) for t in TOPICS}

    def pages(res: dict) -> dict[str, str]:
        from briefing_agent.agent.state import load_checkpoint

        return load_checkpoint(settings.checkpoint_dir, res["run_id"]).fetched

    i = 0
    for t in TOPICS[:8]:
        i += 1
        r = runs[t]
        samples.append(
            {
                "id": f"h-{i:03d}",
                "condition": "clean",
                "input": {"topic": t, "audience": "engineer"},
                "output": {"briefing_md": r["briefing_md"], "sources_text": pages(r)},
                "reference": {},
                "label": True,
                "note": "clean fake-pipeline output; every fact is a sentence from its cited page",
            }
        )
    for t in TOPICS[:6]:
        i += 1
        # Hallucinated variant: bypass the repair pass by composing directly.
        state_pages = pages(runs[t])
        sources = [
            {"title": u.rsplit("/", 1)[-1].replace("_", " "), "url": u, "text": txt}
            for u, txt in state_pages.items()
        ]
        from briefing_agent.agent import prompts
        from briefing_agent.llm.base import Message

        md = halluc.client.complete(
            [
                Message(role="system", content=prompts.COMPOSE_SYSTEM),
                Message(role="user", content=prompts.compose_user(t, "engineer", sources, 5, 8)),
            ]
        ).text
        samples.append(
            {
                "id": f"h-{i:03d}",
                "condition": "hallucinated",
                "input": {"topic": t, "audience": "engineer"},
                "output": {"briefing_md": md, "sources_text": state_pages},
                "reference": {},
                "label": False,
                "note": "fact 3 is invented (Kepler-442 colony) and the last fact cites a source that does not exist",
            }
        )
    for t in TOPICS[:3]:
        i += 1
        r = runs[t]
        b = parse_briefing(r["briefing_md"], t)
        for f in b.facts[:3]:
            f.text = paraphrase(f.text)
        samples.append(
            {
                "id": f"h-{i:03d}",
                "condition": "paraphrased",
                "input": {"topic": t, "audience": "engineer"},
                "output": {"briefing_md": b.render(), "sources_text": pages(r)},
                "reference": {},
                "label": True,
                "note": "first three facts rewritten with synonyms; meaning preserved, lexical overlap reduced",
            }
        )
    pairs = [(TOPICS[0], TOPICS[8]), (TOPICS[1], TOPICS[9]), (TOPICS[2], TOPICS[5])]
    for t, other in pairs:
        i += 1
        r_other = runs[other]
        b = parse_briefing(r_other["briefing_md"], other)
        b.topic = t
        samples.append(
            {
                "id": f"h-{i:03d}",
                "condition": "off_topic",
                "input": {"topic": t, "audience": "engineer"},
                "output": {"briefing_md": b.render(), "sources_text": pages(r_other)},
                "reference": {},
                "label": False,
                "note": f"faithful to its sources but the sources are about {other}, not {t}",
            }
        )
    out = ROOT / "evals" / "human_labels.jsonl"
    out.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(samples)} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
