"""Single source of configuration. Every variable is listed in .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "anthropic", "openai"]
NetworkMode = Literal["auto", "fixture", "live"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "fake"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5"
    cheap_model: str = "claude-haiku-4-5"
    judge_model: str = "gpt-4o-mini"

    network_mode: NetworkMode = "auto"
    http_timeout_s: float = 10.0

    faults: str = ""
    fault_seed: int = 7

    max_iterations: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=16, ge=1)
    max_cost_usd_per_run: float = Field(default=0.25, ge=0.0)
    retry_budget_search: int = Field(default=2, ge=0)
    retry_budget_fetch: int = Field(default=2, ge=0)
    retry_base_ms: int = Field(default=200, ge=0)
    verify_match_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_facts: int = Field(default=5, ge=1)
    max_facts: int = Field(default=8, ge=1)

    db_path: Path = Path("data/briefing.db")
    trace_path: Path = Path("traces/spans.jsonl")
    checkpoint_dir: Path = Path("checkpoints")
    pricing_path: Path = Path("pricing.json")
    fixture_dir: Path = Path("tests/fixtures/wikipedia")

    rate_limit_per_minute: int = Field(default=10, ge=1)
    rate_limit_burst: int = Field(default=5, ge=1)
    daily_usd_cap: float = Field(default=2.0, ge=0.0)

    @property
    def effective_provider(self) -> Provider:
        """Fall back to the fake provider when the configured real provider has no key."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "fake"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "fake"
        return self.llm_provider

    @property
    def effective_network_mode(self) -> Literal["fixture", "live"]:
        if self.network_mode == "auto":
            return "fixture" if self.effective_provider == "fake" else "live"
        return self.network_mode


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache (tests mutate the environment between cases)."""
    get_settings.cache_clear()
    return get_settings()
