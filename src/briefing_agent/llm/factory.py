"""``get_client()``: pick the backend from settings.

Order: explicit ``LLM_PROVIDER`` (if its key exists) -> openrouter when ``OPENROUTER_API_KEY``
is set -> anthropic / openai when their key is set -> fake, with a one-line warning.
"""

from __future__ import annotations

import logging

from briefing_agent.llm.base import LLMClient
from briefing_agent.settings import Settings, get_settings

log = logging.getLogger(__name__)


def get_client(settings: Settings | None = None, **fake_kwargs: object) -> LLMClient:
    s = settings or get_settings()
    provider = s.effective_provider
    if provider == "openrouter":
        from briefing_agent.llm.openai_compat import OPENROUTER_BASE_URL, OpenAICompatClient

        headers: dict[str, str] = {}
        if s.openrouter_site_url:
            headers["HTTP-Referer"] = s.openrouter_site_url
        if s.openrouter_app_name:
            headers["X-Title"] = s.openrouter_app_name
        return OpenAICompatClient(
            provider="openrouter",
            api_key=s.openrouter_api_key,
            default_model=s.llm_model,
            base_url=OPENROUTER_BASE_URL,
            extra_headers=headers,
        )
    if provider == "openai":
        from briefing_agent.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            provider="openai", api_key=s.openai_api_key, default_model=s.llm_model
        )
    if provider == "anthropic":
        from briefing_agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(api_key=s.anthropic_api_key, default_model=s.llm_model)

    from briefing_agent.llm.fake import FakeClient

    if s.llm_provider not in ("fake", ""):
        log.warning(
            "LLM_PROVIDER=%s has no API key; falling back to the fake provider", s.llm_provider
        )
    else:
        log.warning("No LLM API key found; using the deterministic fake provider")
    return FakeClient(default_model=s.llm_model, **fake_kwargs)  # type: ignore[arg-type]
