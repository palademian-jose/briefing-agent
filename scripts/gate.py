"""CI gate: compare a candidate eval report with a baseline against evals.yaml thresholds.

    uv run python scripts/gate.py --baseline reports/baseline.json --candidate reports/latest.json \
        --config evals.yaml

Exit 1 on any regression. Prints a Markdown table. This is a vendored stand-in for
``evalkit gate`` from llm-evalops; the CLI shape and config format are identical so the swap
is a one-line change in CI.

Config keys per metric: ``min`` (absolute floor), ``max`` (absolute ceiling), ``max_drop``
(candidate may not be lower than baseline by more than this), ``max_rise`` (for cost-like
metrics), ``optional: true`` (skip when the metric is absent instead of failing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def evaluate(
    baseline: dict[str, Any] | None, candidate: dict[str, Any], config: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ok = True
    base_metrics = (baseline or {}).get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    for name, rule in config.get("metrics", {}).items():
        cand = cand_metrics.get(name)
        base = base_metrics.get(name)
        if cand is None:
            status = "skip" if rule.get("optional") else "FAIL (missing)"
            ok = ok and bool(rule.get("optional"))
            rows.append(
                {
                    "metric": name,
                    "baseline": base,
                    "candidate": None,
                    "rule": rule,
                    "status": status,
                }
            )
            continue
        reasons = []
        if "min" in rule and cand < rule["min"]:
            reasons.append(f"below min {rule['min']}")
        if "max" in rule and cand > rule["max"]:
            reasons.append(f"above max {rule['max']}")
        if base is not None and "max_drop" in rule and cand < base - rule["max_drop"]:
            reasons.append(f"dropped more than {rule['max_drop']} vs baseline")
        if base is not None and "max_rise" in rule and cand > base + rule["max_rise"]:
            reasons.append(f"rose more than {rule['max_rise']} vs baseline")
        status = "ok" if not reasons else "FAIL: " + "; ".join(reasons)
        ok = ok and not reasons
        rows.append(
            {"metric": name, "baseline": base, "candidate": cand, "rule": rule, "status": status}
        )
    return ok, rows


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| metric | baseline | candidate | rule | status |", "|---|---|---|---|---|"]
    for r in rows:
        rule = ", ".join(f"{k}={v}" for k, v in r["rule"].items())
        b = "-" if r["baseline"] is None else f"{r['baseline']}"
        c = "-" if r["candidate"] is None else f"{r['candidate']}"
        lines.append(f"| {r['metric']} | {b} | {c} | {rule} | {r['status']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", default="evals.yaml")
    args = ap.parse_args()
    baseline = (
        json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if Path(args.baseline).exists()
        else None
    )
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ok, rows = evaluate(baseline, candidate, config)
    print(
        f"gate: baseline={args.baseline if baseline else 'none'} candidate={args.candidate} "
        f"(provider={candidate.get('provider')}, commit={candidate.get('commit')})\n"
    )
    print(table(rows))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
