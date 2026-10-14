# Eval 20261014T200200-ff3652

provider: **fake** (fake-provider numbers: deterministic policy over recorded fixtures, not a real model); model `anthropic/claude-sonnet-4.5`; judge `lexical-fallback`; commit `253546e`; faults `none`; 30 tasks x k=3 = 90 runs.

| metric | value |
|---|---|
| task_success | 0.93 |
| pass^3 | 0.93 |
| recovery_rate | n/a (no faults injected) |
| mean_tool_calls | 6.3 |
| mean_llm_calls | 9.6 |
| $/run (mean) | 0.0513 |
| latency p95 (ms) | 124 |
| verifier first-pass rate | 1.00 |
| repair rate | 0.00 |
| judge pass rate | 0.93 |
| needs_human rate | 0.00 |
| budget_exceeded rate | 0.00 |

## Failed tasks

| id | topic | failure_mode | first failure |
|---|---|---|---|
| g-0022 | A* search algorithm | ambiguous_topic | missing term 'heuristic' in summary/facts |
| g-0030 | Reciprocal rank fusion in search | off_topic_source | missing term 'rank fusion' in summary/facts |
