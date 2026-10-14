"""Trajectory eval over the golden set.

    uv run python scripts/eval.py --k 3 [--subset N] [--faults SPEC] [--suffix faults] [--out reports]

Each task runs k times. A run passes when the agent finishes with status ok, the briefing
meets the task's expected properties, and the judge passes. Reported:

- task_success: mean pass rate over all k*N runs
- pass_k: share of tasks whose k runs ALL passed (pass^k, never best-of-k)
- recovery_rate: among runs with >= 1 injected fault, the share that still passed
- mean_tool_calls, cost_usd_mean, latency_p95_ms, verifier_pass_rate (first pass), repair_rate

Writes reports/eval-<run_id>.json (shared contract 3), reports/latest[-suffix].json and
reports/latest[-suffix].md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from briefing_agent.briefing import parse_briefing  # noqa: E402
from briefing_agent.evals.judge import CRITERION_USEFUL_FAITHFUL, judge  # noqa: E402
from briefing_agent.pipeline import Runner  # noqa: E402
from briefing_agent.settings import Settings  # noqa: E402
from briefing_agent.store import Store  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_golden(path: Path, subset: int | None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return cases[:subset] if subset else cases


def check_expected(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Deterministic task-level checks from the golden case."""
    failures: list[str] = []
    md = result.get("briefing_md") or ""
    b = parse_briefing(md, result["topic"])
    if len(b.facts) < expected.get("min_facts", 1):
        failures.append(f"facts {len(b.facts)} < {expected['min_facts']}")
    if len(b.sources) < expected.get("min_sources", 1):
        failures.append(f"sources {len(b.sources)} < {expected['min_sources']}")
    dom = expected.get("required_source_domain")
    if dom and not any(dom in s.url for s in b.sources):
        failures.append(f"no source from {dom}")
    body = (b.summary + " " + " ".join(f.text for f in b.facts)).lower()
    for term in expected.get("must_mention", []):
        if term.lower() not in body:
            failures.append(f"missing term {term!r} in summary/facts")
    low = md.lower()
    for term in expected.get("must_not_contain", []):
        if term.lower() in low:
            failures.append(f"contains forbidden {term!r}")
    if any(not f.citations for f in b.facts):
        failures.append("fact without citation")
    fetched = set(result.get("fetched_urls", []))
    if any(s.url not in fetched for s in b.sources):
        failures.append("source not fetched")
    return failures


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return vals[idx]


def run_eval(
    *,
    golden: Path,
    k: int,
    subset: int | None,
    faults: str,
    seed: int,
    out_dir: Path,
    suffix: str,
    settings_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    settings = Settings(
        db_path=ROOT / "data" / "eval.db",
        trace_path=ROOT / "traces" / f"eval-{run_id}.jsonl",
        checkpoint_dir=ROOT / "checkpoints" / "eval",
        faults=faults,
        fixture_dir=ROOT / "tests" / "fixtures" / "wikipedia",
        **(settings_overrides or {}),
    )
    store = Store(settings.db_path)
    runner = Runner(settings, store=store, sleep=lambda _s: None, fault_seed=seed)
    cases = load_golden(golden, subset)
    case_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    started = time.time()
    for case in cases:
        runs: list[dict[str, Any]] = []
        for i in range(k):
            runner.fault_seed = seed + 1000 * i + int(case["id"].split("-")[-1])
            res = runner.run(case["input"]["topic"], case["input"].get("constraints"))
            failures = (
                check_expected(res, case["expected"])
                if res["status"] == "ok"
                else [f"status {res['status']}: {res.get('reason')}"]
            )
            verdict = {"pass": False, "reason": "not judged (run failed)"}
            if not failures:
                verdict = judge(
                    CRITERION_USEFUL_FAITHFUL,
                    {
                        "topic": case["input"]["topic"],
                        "audience": case["input"].get("constraints", {}).get("audience"),
                    },
                    {
                        "briefing_md": res["briefing_md"],
                        "sources_text": {u: t for u, t in _pages(runner, res).items()},
                    },
                    {"must_mention": case["expected"].get("must_mention", [])},
                    client=runner.client,
                    settings=settings,
                )
                if not verdict["pass"]:
                    failures.append(f"judge: {verdict['reason']}")
            run = {
                "run_id": res["run_id"],
                "passed": not failures,
                "failures": failures,
                "status": res["status"],
                "tool_calls": res["tool_calls"],
                "llm_calls": res["llm_calls"],
                "retries": res["retries"],
                "cost_usd": res["cost_usd"],
                "latency_ms": res["latency_ms"],
                "faults_injected": len(res["faults_injected"]),
                "verifier_first_pass": res["verification_first_pass"],
                "repair_attempted": res["repair_attempted"],
                "judge": verdict,
            }
            runs.append(run)
            all_runs.append(run)
        case_rows.append(
            {
                "id": case["id"],
                "topic": case["input"]["topic"],
                "failure_mode": case.get("failure_mode"),
                "tags": case.get("tags", []),
                "scores": {
                    "task_success": sum(r["passed"] for r in runs) / k,
                    "judge_pass": sum(r["judge"]["pass"] for r in runs) / k,
                    "verifier_first_pass": sum(bool(r["verifier_first_pass"]) for r in runs) / k,
                },
                "passed": all(r["passed"] for r in runs),
                "latency_ms": round(statistics.mean(r["latency_ms"] for r in runs), 1),
                "cost_usd": round(statistics.mean(r["cost_usd"] for r in runs), 6),
                "runs": runs,
            }
        )
        row = case_rows[-1]
        mean_tools = statistics.mean(r["tool_calls"] for r in runs)
        print(
            f"{case['id']} {case['input']['topic'][:40]:40s} "
            f"pass^{k}={'yes' if row['passed'] else 'no '} "
            f"success={row['scores']['task_success']:.2f} tools={mean_tools:.1f}"
        )

    faulted = [r for r in all_runs if r["faults_injected"] > 0]
    metrics = {
        "task_success": _mean([r["passed"] for r in all_runs]),
        "pass_k": _mean([c["passed"] for c in case_rows]),
        "k": k,
        "mean_tool_calls": round(_mean([r["tool_calls"] for r in all_runs]), 2),
        "mean_llm_calls": round(_mean([r["llm_calls"] for r in all_runs]), 2),
        "recovery_rate": _mean([r["passed"] for r in faulted]) if faulted else None,
        "faulted_runs": len(faulted),
        "faults_per_run": round(_mean([r["faults_injected"] for r in all_runs]), 2),
        "cost_usd_mean": round(_mean([r["cost_usd"] for r in all_runs]), 6),
        "latency_p95_ms": round(p95([r["latency_ms"] for r in all_runs]), 1),
        "latency_mean_ms": round(_mean([r["latency_ms"] for r in all_runs]), 1),
        "verifier_pass_rate": _mean([bool(r["verifier_first_pass"]) for r in all_runs]),
        "repair_rate": _mean([r["repair_attempted"] for r in all_runs]),
        "judge_pass_rate": _mean([r["judge"]["pass"] for r in all_runs]),
        "needs_human_rate": _mean([r["status"] == "needs_human" for r in all_runs]),
        "budget_exceeded_rate": _mean([r["status"] == "budget_exceeded" for r in all_runs]),
        "runs": len(all_runs),
        "tasks": len(case_rows),
        "wall_s": round(time.time() - started, 1),
    }
    report = {
        "run_id": run_id,
        "commit": git_sha(),
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "provider": getattr(runner.client, "provider", "unknown"),
        "model": settings.llm_model,
        "judge_model": "lexical-fallback"
        if getattr(runner.client, "provider", "fake") == "fake"
        else settings.judge_model,
        "faults": faults or "",
        "seed": seed,
        "metrics": metrics,
        "cases": case_rows,
    }
    name = f"eval-{run_id}" + (f"-{suffix}" if suffix else "")
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    latest = "latest" + (f"-{suffix}" if suffix else "")
    (out_dir / f"{latest}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / f"{latest}.md").write_text(markdown_table(report), encoding="utf-8")
    print(f"\nwrote {out_dir / (name + '.json')} and {out_dir / (latest + '.md')}")
    print(markdown_table(report))
    return report


def _pages(runner: Runner, res: dict[str, Any]) -> dict[str, str]:
    """Fetched page text for the judge, read back from the run checkpoint."""
    from briefing_agent.agent.state import load_checkpoint

    try:
        state = load_checkpoint(runner.settings.checkpoint_dir, res["run_id"])
        return state.fetched
    except FileNotFoundError:
        return {}


def _mean(values: list[Any]) -> float:
    return round(sum(float(v) for v in values) / len(values), 4) if values else 0.0


def markdown_table(report: dict[str, Any]) -> str:
    m = report["metrics"]
    label = f"provider: **{report['provider']}**"
    if report["provider"] == "fake":
        label += " (fake-provider numbers: deterministic policy over recorded fixtures, not a real model)"
    rec = (
        "n/a (no faults injected)"
        if m["recovery_rate"] is None
        else f"{m['recovery_rate']:.2f} ({m['faulted_runs']} faulted runs)"
    )
    rows = [
        ("task_success", f"{m['task_success']:.2f}"),
        (f"pass^{m['k']}", f"{m['pass_k']:.2f}"),
        ("recovery_rate", rec),
        ("mean_tool_calls", f"{m['mean_tool_calls']:.1f}"),
        ("mean_llm_calls", f"{m['mean_llm_calls']:.1f}"),
        ("$/run (mean)", f"{m['cost_usd_mean']:.4f}"),
        ("latency p95 (ms)", f"{m['latency_p95_ms']:.0f}"),
        ("verifier first-pass rate", f"{m['verifier_pass_rate']:.2f}"),
        ("repair rate", f"{m['repair_rate']:.2f}"),
        ("judge pass rate", f"{m['judge_pass_rate']:.2f}"),
        ("needs_human rate", f"{m['needs_human_rate']:.2f}"),
        ("budget_exceeded rate", f"{m['budget_exceeded_rate']:.2f}"),
    ]
    lines = [
        f"# Eval {report['run_id']}",
        "",
        f"{label}; model `{report['model']}`; judge `{report['judge_model']}`; commit `{report['commit']}`; "
        f"faults `{report['faults'] or 'none'}`; {m['tasks']} tasks x k={m['k']} = {m['runs']} runs.",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    failed = [c for c in report["cases"] if not c["passed"]]
    if failed:
        lines += [
            "",
            "## Failed tasks",
            "",
            "| id | topic | failure_mode | first failure |",
            "|---|---|---|---|",
        ]
        for c in failed:
            first = next((r["failures"][0] for r in c["runs"] if r["failures"]), "")
            lines.append(f"| {c['id']} | {c['topic']} | {c['failure_mode']} | {first[:80]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "evals" / "golden" / "tasks.jsonl"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--faults", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    run_eval(
        golden=Path(args.golden),
        k=args.k,
        subset=args.subset,
        faults=args.faults,
        seed=args.seed,
        out_dir=Path(args.out),
        suffix=args.suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
