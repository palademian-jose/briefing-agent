"""The provider/model/tracing settings are shared across docs-rag, briefing-agent and llm-evalops:
same variable names in .env.example, same defaults as llmops' Settings."""

from __future__ import annotations

from pathlib import Path

from llmops.settings import Settings as LlmopsSettings

from briefing_agent.settings import Settings

SHARED = [
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "LLM_MODEL",
    "JUDGE_MODEL",
    "CHEAP_MODEL",
    "TRACE_PATH",
]
ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
ROOT = Path(__file__).resolve().parents[1]


def _env_names() -> set[str]:
    names = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            names.add(line.split("=", 1)[0].strip())
    return names


def test_env_example_lists_the_shared_variables() -> None:
    assert set(SHARED) <= _env_names()


def test_every_env_example_variable_is_a_settings_field() -> None:
    fields = {name.upper() for name in Settings.model_fields}
    # PRICING_PATH is read by llmops directly, not through this repo's Settings.
    unknown = _env_names() - fields - {"PRICING_PATH"}
    assert not unknown, f".env.example lists variables Settings does not read: {sorted(unknown)}"


def test_shared_settings_match_llmops_defaults() -> None:
    mine, theirs = Settings.model_fields, LlmopsSettings.model_fields
    for var in SHARED:
        name = var.lower()
        assert name in mine and name in theirs, name
    for name in ("llm_model", "judge_model", "cheap_model"):
        assert mine[name].default == theirs[name].default, name
    assert Path(mine["trace_path"].default).as_posix() == theirs["trace_path"].default


def test_no_local_copy_of_the_price_table() -> None:
    assert not (ROOT / "pricing.json").exists(), "pricing.json lives in llm-evalops (llmops)"
