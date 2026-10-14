# Eval 20261014T200209-b21dd8

provider: **fake** (fake-provider numbers: deterministic policy over recorded fixtures, not a real model); model `anthropic/claude-sonnet-4.5`; judge `lexical-fallback`; commit `253546e`; faults `fetch_timeout:0.3,search_empty:0.2,fetch_not_found:0.15`; 30 tasks x k=1 = 30 runs.

| metric | value |
|---|---|
| task_success | 0.87 |
| pass^1 | 0.87 |
| recovery_rate | 0.83 (24 faulted runs) |
| mean_tool_calls | 8.8 |
| mean_llm_calls | 12.0 |
| $/run (mean) | 0.0620 |
| latency p95 (ms) | 125 |
| verifier first-pass rate | 1.00 |
| repair rate | 0.00 |
| judge pass rate | 0.87 |
| needs_human rate | 0.00 |
| budget_exceeded rate | 0.00 |

## Failed tasks

| id | topic | failure_mode | first failure |
|---|---|---|---|
| g-0011 | Lamport timestamp | tool_loop | sources 1 < 2 |
| g-0022 | A* search algorithm | ambiguous_topic | missing term 'heuristic' in summary/facts |
| g-0024 | Gradient descent | hallucination | sources 1 < 2 |
| g-0030 | Reciprocal rank fusion in search | off_topic_source | missing term 'rank fusion' in summary/facts |
