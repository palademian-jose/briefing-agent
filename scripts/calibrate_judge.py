"""Judge calibration against human labels.

    uv run python scripts/calibrate_judge.py [--labels evals/human_labels.jsonl] [--out reports]

Reports Cohen's kappa and raw agreement between judge verdicts and human labels, plus two
bias probes that a pointwise binary judge can still suffer from:

- position swap: reverse the order of the key facts (content unchanged) and re-judge;
  a verdict that flips means the judge is reading order, not content;
- length bias: pad the summary with neutral filler and re-judge; also compare pass rates
  for the shorter and longer halves of the human-passing samples.

Writes reports/judge-calibration.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from briefing_agent.briefing import parse_briefing  # noqa: E402
from briefing_agent.evals.judge import CRITERION_USEFUL_FAITHFUL, judge  # noqa: E402

FILLER = (
    " This section restates the context for completeness. The topic is covered in the sources"
    " listed below, which were consulted while preparing this briefing. Further background is"
    " available in the cited material."
)


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def swap_positions(md: str, topic: str) -> str:
    b = parse_briefing(md, topic)
    b.facts = list(reversed(b.facts))
    return b.render()


def pad_summary(md: str, topic: str) -> str:
    b = parse_briefing(md, topic)
    b.summary = b.summary + FILLER
    return b.render()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "evals" / "human_labels.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    human: list[bool] = []
    verdicts: list[bool] = []
    swap_flips = 0
    pad_flips = 0
    lengths: list[tuple[int, bool, bool]] = []
    per_item: list[dict[str, Any]] = []
    judge_name = "?"
    for row in rows:
        topic = row["input"]["topic"]
        out = row["output"]
        v = judge(CRITERION_USEFUL_FAITHFUL, row["input"], out, row.get("reference"))
        judge_name = v.get("judge", judge_name)
        v_swap = judge(
            CRITERION_USEFUL_FAITHFUL,
            row["input"],
            {**out, "briefing_md": swap_positions(out["briefing_md"], topic)},
            row.get("reference"),
        )
        v_pad = judge(
            CRITERION_USEFUL_FAITHFUL,
            row["input"],
            {**out, "briefing_md": pad_summary(out["briefing_md"], topic)},
            row.get("reference"),
        )
        human.append(bool(row["label"]))
        verdicts.append(bool(v["pass"]))
        swap_flips += int(v_swap["pass"] != v["pass"])
        pad_flips += int(v_pad["pass"] != v["pass"])
        lengths.append((len(out["briefing_md"]), bool(row["label"]), bool(v["pass"])))
        per_item.append(
            {
                "id": row["id"],
                "label": bool(row["label"]),
                "verdict": bool(v["pass"]),
                "agree": bool(row["label"]) == bool(v["pass"]),
                "reason": v["reason"],
                "note": row.get("note", ""),
            }
        )

    n = len(rows)
    tp = sum(h and v for h, v in zip(human, verdicts, strict=True))
    tn = sum((not h) and (not v) for h, v in zip(human, verdicts, strict=True))
    fp = sum((not h) and v for h, v in zip(human, verdicts, strict=True))
    fn = sum(h and (not v) for h, v in zip(human, verdicts, strict=True))
    passing = [item for item in lengths if item[1]]
    median = statistics.median(item[0] for item in passing) if passing else 0
    short = [item[2] for item in passing if item[0] <= median]
    long_ = [item[2] for item in passing if item[0] > median]
    report = {
        "judge": judge_name,
        "n": n,
        "agreement": round((tp + tn) / n, 4) if n else 0.0,
        "kappa": round(cohens_kappa(human, verdicts), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "position_swap_flip_rate": round(swap_flips / n, 4) if n else 0.0,
        "length_padding_flip_rate": round(pad_flips / n, 4) if n else 0.0,
        "pass_rate_short_half": round(sum(short) / len(short), 4) if short else None,
        "pass_rate_long_half": round(sum(long_) / len(long_), 4) if long_ else None,
        "items": per_item,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge-calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = [
        f"judge: `{judge_name}`; n={n}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| agreement | {report['agreement']:.2f} |",
        f"| Cohen's kappa | {report['kappa']:.2f} |",
        f"| confusion (tp/tn/fp/fn) | {tp}/{tn}/{fp}/{fn} |",
        f"| position-swap flip rate | {report['position_swap_flip_rate']:.2f} |",
        f"| length-padding flip rate | {report['length_padding_flip_rate']:.2f} |",
        f"| pass rate, shorter half of human-pass | {report['pass_rate_short_half']} |",
        f"| pass rate, longer half of human-pass | {report['pass_rate_long_half']} |",
        "",
        "Disagreements:",
    ]
    md += [
        f"- {i['id']}: label={i['label']} verdict={i['verdict']} ({i['reason']}) {i['note']}"
        for i in per_item
        if not i["agree"]
    ] or ["- none"]
    text = "\n".join(md)
    (out_dir / "judge-calibration.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
