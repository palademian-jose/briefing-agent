.PHONY: setup test lint fmt demo eval eval-faults calibrate gate serve clean

setup:
	uv python pin 3.12
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	uv run briefing "Consistent hashing" --faults "fetch_timeout:0.5,search_empty:0.3"

eval:
	uv run python scripts/eval.py --k 3 --out reports

eval-faults:
	uv run python scripts/eval.py --k 1 --faults "fetch_timeout:0.3,search_empty:0.2,fetch_not_found:0.15" --out reports --suffix faults

calibrate:
	uv run python scripts/calibrate_judge.py

gate:
	uv run evalkit gate --baseline reports/baseline.json --candidate reports/latest.json --config evals.yaml --md reports/gate.md

serve:
	uv run uvicorn briefing_agent.api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache checkpoints/*.json traces/*.jsonl data/*.db
