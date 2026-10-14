"""API endpoints via TestClient, plus rate limit and daily cap."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from briefing_agent.api import TokenBucket, create_app
from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings


@pytest.fixture
def client(settings: Settings, make_runner: Callable[..., Runner]) -> TestClient:
    s = settings.model_copy(update={"rate_limit_per_minute": 60, "rate_limit_burst": 3})
    return TestClient(create_app(s, runner=make_runner(s)))


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["provider"] == "fake" and body["network"] == "fixture"
    assert body["spend_today_usd"] == 0.0


def test_run_then_fetch_run_and_notes(client: TestClient) -> None:
    r = client.post("/run", json={"topic": "Merkle tree", "constraints": {"max_sources": 2}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert set(body) >= {
        "run_id",
        "briefing_md",
        "sources",
        "trace_id",
        "tool_calls",
        "cost_usd",
        "latency_ms",
    }
    assert body["briefing_md"].startswith("# Briefing: Merkle tree")
    assert len(body["sources"]) == 2 and body["cost_usd"] > 0

    r2 = client.get(f"/runs/{body['run_id']}")
    assert r2.status_code == 200 and r2.json()["trace_id"] == body["trace_id"]
    assert client.get("/runs/nope").status_code == 404

    notes = client.get("/notes").json()["notes"]
    assert notes and notes[0]["topic"] == "Merkle tree"
    note = client.get(f"/notes/{notes[0]['id']}").json()
    assert note["markdown"] == body["briefing_md"]

    health = client.get("/health").json()
    assert health["spend_today_usd"] == pytest.approx(body["cost_usd"], abs=1e-6)


def test_request_validation(client: TestClient) -> None:
    assert client.post("/run", json={"topic": "x"}).status_code == 422
    assert (
        client.post(
            "/run", json={"topic": "ok topic", "constraints": {"max_sources": 99}}
        ).status_code
        == 422
    )
    assert client.post("/run", json={"topic": "line\nbreak"}).status_code == 422


def test_rate_limit_per_ip(client: TestClient) -> None:
    codes = [
        client.post(
            "/run", json={"topic": "Bloom filter", "constraints": {"max_sources": 1}}
        ).status_code
        for _ in range(4)
    ]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_daily_usd_cap(settings: Settings, make_runner: Callable[..., Runner]) -> None:
    s = settings.model_copy(update={"daily_usd_cap": 0.001, "max_cost_usd_per_run": 0.25})
    c = TestClient(create_app(s, runner=make_runner(s)))
    r = c.post("/run", json={"topic": "Bloom filter"})
    assert r.status_code == 429 and "daily USD cap" in r.json()["detail"]


def test_token_bucket_refills() -> None:
    b = TokenBucket(rate_per_minute=60, burst=2)
    assert b.allow("ip", now=0.0) and b.allow("ip", now=0.0) and not b.allow("ip", now=0.0)
    assert b.allow("ip", now=1.1)  # one token per second
    assert b.allow("other", now=0.0)  # independent buckets
