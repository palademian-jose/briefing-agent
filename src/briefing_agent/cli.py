"""CLI: ``uv run briefing "topic" [--max-sources N] [--audience X] [--faults SPEC] [--resume]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from briefing_agent.agent.state import RunState
from briefing_agent.pipeline import Runner
from briefing_agent.settings import get_settings


class _StepPrinter:
    """Prints every step appended to the state since the previous callback."""

    def __init__(self) -> None:
        self.seen = 0

    def __call__(self, state: RunState, step: str) -> None:
        for last in state.steps[self.seen :]:
            kind = last.get("kind")
            if kind == "think":
                sys.stderr.write(f"  think  | {last.get('plan')}\n")
            elif kind == "act":
                calls = ",".join(last.get("tool_calls") or []) or "DONE"
                sys.stderr.write(f"  act    | {calls}: {last.get('text', '')[:80]}\n")
            elif kind == "observe":
                status = "ok" if last.get("ok") else f"error:{last.get('error')}"
                sys.stderr.write(
                    f"  observe| {last.get('tool')} -> {status} (attempts={last.get('attempts')})\n"
                )
            elif kind in ("verify", "verify_after_repair"):
                sys.stderr.write(f"  verify | ok={last.get('ok')} issues={last.get('issues')}\n")
            elif kind in ("budget_exceeded", "goal_met"):
                sys.stderr.write(f"  {kind} | {last}\n")
        self.seen = len(state.steps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="briefing", description="Research-briefing agent")
    ap.add_argument("topic", nargs="?", help="topic to research")
    ap.add_argument("--max-sources", type=int, default=3)
    ap.add_argument("--audience", default="engineer")
    ap.add_argument("--faults", default=None, help='e.g. "fetch_timeout:0.3,search_empty:0.2"')
    ap.add_argument("--seed", type=int, default=None, help="fault-injection seed")
    ap.add_argument("--resume", default=None, metavar="RUN_ID", help="resume a checkpointed run")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--quiet", action="store_true", help="do not print the step log")
    args = ap.parse_args(argv)
    if not args.topic and not args.resume:
        ap.error("topic is required unless --resume is given")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # em dashes on Windows consoles
    settings = get_settings()
    runner = Runner(
        settings,
        faults=args.faults,
        fault_seed=args.seed,
        on_step=None if args.quiet else _StepPrinter(),
    )
    sys.stderr.write(
        f"provider={getattr(runner.client, 'provider', '?')} "
        f"network={settings.effective_network_mode}"
        f" faults={runner.fault_spec or 'none'}\n"
    )
    if args.resume:
        result = runner.resume(args.resume)
    else:
        result = runner.run(
            args.topic, {"max_sources": args.max_sources, "audience": args.audience}
        )

    if args.json:
        print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        print(result["briefing_md"] or "(no briefing produced)")
        c = result["cost"]
        print(
            f"\n---\nstatus={result['status']}"
            + (f" reason={result['reason']}" if result.get("reason") else "")
            + f" run_id={result['run_id']} tool_calls={result['tool_calls']} "
            f"llm_calls={c['llm_calls']} cost=${result['cost_usd']:.4f} "
            f"latency={result['latency_ms']:.0f}ms faults={len(result['faults_injected'])}"
        )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
