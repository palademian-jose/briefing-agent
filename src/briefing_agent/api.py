"""FastAPI surface: POST /run, GET /runs/{id}, GET /notes, GET /health.

Protection: a per-IP token bucket (RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST) and a daily USD
cap (DAILY_USD_CAP) computed from the runs table plus the per-run cap inside the agent.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from briefing_agent import __version__
from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings, get_settings
from briefing_agent.store import Store


class TokenBucket:
    def __init__(self, rate_per_minute: float, burst: int) -> None:
        self.rate = rate_per_minute / 60.0
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True


class Constraints(BaseModel):
    max_sources: int = Field(default=3, ge=1, le=6)
    audience: str = Field(default="engineer", min_length=1, max_length=80)


class RunRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200, pattern=r"^[^\n\r]+$")
    constraints: Constraints = Field(default_factory=Constraints)


class RunResponse(BaseModel):
    run_id: str
    status: str
    reason: str | None = None
    briefing_md: str | None
    sources: list[dict[str, Any]]
    trace_id: str
    tool_calls: int
    cost_usd: float
    cost: dict[str, Any]
    latency_ms: float
    verification: dict[str, Any] | None = None


def create_app(settings: Settings | None = None, runner: Runner | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="briefing-agent", version=__version__)
    store = runner.store if runner else Store(s.db_path)
    rn = runner or Runner(s, store=store)
    bucket = TokenBucket(s.rate_limit_per_minute, s.rate_limit_burst)
    app.state.settings = s
    app.state.store = store
    app.state.runner = rn
    app.state.bucket = bucket

    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def rate_limited(request: Request) -> None:
        if not bucket.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded; retry later")

    def day_start() -> str:
        return datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": getattr(rn.client, "provider", "unknown"),
            "network": s.effective_network_mode,
            "spend_today_usd": round(store.spend_since(day_start()), 6),
            "daily_usd_cap": s.daily_usd_cap,
        }

    @app.post("/run", response_model=RunResponse, dependencies=[Depends(rate_limited)])
    def run(req: RunRequest) -> RunResponse:
        spent = store.spend_since(day_start())
        if spent + s.max_cost_usd_per_run > s.daily_usd_cap:
            raise HTTPException(
                status_code=429,
                detail=f"daily USD cap reached (spent ${spent:.4f} of ${s.daily_usd_cap:.2f})",
            )
        result = rn.run(req.topic, req.constraints.model_dump())
        return RunResponse(**{k: result[k] for k in RunResponse.model_fields if k in result})

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        return row

    @app.get("/notes")
    def notes(limit: int = 20) -> dict[str, Any]:
        return store.list_notes(limit=max(1, min(limit, 100)))

    @app.get("/notes/{note_id}")
    def note(note_id: int) -> dict[str, Any]:
        row = store.get_note(note_id)
        if row is None:
            raise HTTPException(status_code=404, detail="note not found")
        return row

    return app


app = create_app()
