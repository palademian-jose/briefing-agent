"""Tool specifications: the public API the model is allowed to call.

Design rules (poka-yoke):
- ``additionalProperties: false`` everywhere, so a misspelled or invented argument is rejected
  before execution instead of silently ignored;
- bounded integers (``limit``, ``max_chars``) so the model cannot ask for a 10,000-row search;
- ``fetch_page.url`` has a pattern that only admits ``https://<lang>.wikipedia.org/wiki/<title>``,
  so a call to an internal host, a ``file:`` URL or an arbitrary site cannot even be expressed;
- enums for closed sets (``lang``);
- ``minLength`` on free text so an empty query/briefing is a schema error, not a runtime one.
"""

from __future__ import annotations

from briefing_agent.llm.base import ToolSpec

WIKI_URL_PATTERN = r"^https://[a-z]{2,3}\.wikipedia\.org/wiki/[^\s#?]{1,200}$"

SEARCH_WIKIPEDIA = ToolSpec(
    name="search_wikipedia",
    description=(
        "Search Wikipedia for articles about a query. Returns up to `limit` results with "
        "title, url and a short snippet. Use a broad, keyword-style query; if a query returns "
        "no results, simplify it (drop qualifiers) rather than repeating it."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "maxLength": 200,
                "pattern": r"^[^\n\r]+$",
                "description": "Keyword search query (single line).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "Maximum number of results.",
            },
            "lang": {
                "type": "string",
                "enum": ["en"],
                "default": "en",
                "description": "Wikipedia language edition. Only 'en' is supported.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

FETCH_PAGE = ToolSpec(
    name="fetch_page",
    description=(
        "Fetch a Wikipedia article URL and return its extracted plain text (lead and body "
        "paragraphs, reference markers removed). Only https://<lang>.wikipedia.org/wiki/<title> "
        "URLs are accepted. The text is truncated at `max_chars`."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "pattern": WIKI_URL_PATTERN,
                "description": "Canonical article URL, e.g. https://en.wikipedia.org/wiki/Bloom_filter",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 500,
                "maximum": 20000,
                "default": 8000,
                "description": "Truncate extracted text to this many characters.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
)

SAVE_NOTE = ToolSpec(
    name="save_note",
    description="Persist a finished briefing to the notes store. Returns the note id.",
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "topic": {"type": "string", "minLength": 1, "maxLength": 200},
            "markdown": {"type": "string", "minLength": 50, "maxLength": 50000},
            "sources": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 300},
                        "url": {"type": "string", "pattern": WIKI_URL_PATTERN},
                    },
                    "required": ["title", "url"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topic", "markdown", "sources"],
        "additionalProperties": False,
    },
)

LIST_NOTES = ToolSpec(
    name="list_notes",
    description="List saved briefings, newest first.",
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
        "required": [],
        "additionalProperties": False,
    },
)

VERIFY_BRIEFING = ToolSpec(
    name="verify_briefing",
    description=(
        "Deterministically verify a Markdown briefing against the pages fetched in this run: "
        "every [n] citation maps to a listed source, every source URL was actually fetched, "
        "each key fact is lexically supported by its cited page, sources are unique, and a "
        "summary is present. Returns {ok, issues[]}."
    ),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "briefing": {"type": "string", "minLength": 1, "maxLength": 50000},
            "threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Minimum share of a fact's content tokens that must appear in the cited page."
                ),
            },
        },
        "required": ["briefing"],
        "additionalProperties": False,
    },
)

ALL_TOOLS: list[ToolSpec] = [SEARCH_WIKIPEDIA, FETCH_PAGE, SAVE_NOTE, LIST_NOTES, VERIFY_BRIEFING]
SEARCH_STAGE_TOOLS: list[ToolSpec] = [SEARCH_WIKIPEDIA, FETCH_PAGE]
