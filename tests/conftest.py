"""Shared fixtures: an isolated Settings per test (fake provider, fixture network, temp
storage, no retry sleeps) and a Runner factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from briefing_agent.llm.base import LLMClient
from briefing_agent.pipeline import Runner
from briefing_agent.settings import Settings
from briefing_agent.store import Store

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia"


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FAULTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("NETWORK_MODE", "fixture")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="fake",
        network_mode="fixture",
        db_path=tmp_path / "test.db",
        trace_path=tmp_path / "spans.jsonl",
        checkpoint_dir=tmp_path / "checkpoints",
        fixture_dir=FIXTURES,
        pricing_path=ROOT / "pricing.json",
        retry_base_ms=0,
    )


@pytest.fixture
def make_runner(settings: Settings) -> Callable[..., Runner]:
    def _make(
        s: Settings | None = None,
        *,
        client: LLMClient | None = None,
        faults: str | None = None,
        fault_seed: int | None = None,
        http: Any = None,
        on_step: Any = None,
    ) -> Runner:
        cfg = s or settings
        return Runner(
            cfg,
            client=client,
            store=Store(cfg.db_path),
            http=http,
            faults=faults,
            fault_seed=fault_seed,
            sleep=lambda _s: None,
            on_step=on_step,
        )

    return _make
