"""HTTP access for tools. Two transports share one httpx.Client interface:

- ``live``: real requests to Wikipedia with a polite User-Agent;
- ``fixture``: replays recorded responses from ``tests/fixtures/wikipedia`` so tests, CI and
  the fake-provider demo never touch the network. Unrecorded search queries fall back to a
  tiny local ranking over the recorded page titles; unrecorded page URLs return 404.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote

import httpx

USER_AGENT = (
    "briefing-agent/0.1 (portfolio research agent; https://github.com/palademian-jose) httpx"
)

_WORD = re.compile(r"[a-z0-9]+")


def normalize_query(q: str) -> str:
    return " ".join(_WORD.findall(q.lower()))


def title_to_url(title: str, lang: str = "en") -> str:
    return f"https://{lang}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,'-.:!*")


def url_to_title(url: str) -> str:
    return unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ")


class FixtureTransport(httpx.BaseTransport):
    """Serve recorded Wikipedia responses. See ``scripts/record_fixtures.py``."""

    def __init__(self, fixture_dir: str | Path) -> None:
        self.dir = Path(fixture_dir)
        index_path = self.dir / "index.json"
        if index_path.exists():
            self.index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            self.index = {"searches": {}, "pages": {}}
        self.requests: list[str] = []

    # -- helpers -------------------------------------------------------------------------
    def _page_entries(self) -> list[dict[str, Any]]:
        return [{"url": url, **meta} for url, meta in self.index["pages"].items()]

    def _local_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Rank recorded pages by token overlap with the query (offline fallback)."""
        q_tokens = set(normalize_query(query).split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._page_entries():
            t_tokens = set(normalize_query(entry["title"]).split())
            if not q_tokens or not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens)
            if overlap == 0:
                continue
            score = overlap / len(q_tokens) + 0.1 * overlap / len(t_tokens)
            scored.append((score, entry))
        scored.sort(key=lambda s: (-s[0], s[1]["title"]))
        return [
            {
                "ns": 0,
                "title": e["title"],
                "pageid": e.get("pageid", 0),
                "size": 0,
                "wordcount": 0,
                "snippet": e.get("snippet", ""),
                "timestamp": "2026-01-01T00:00:00Z",
            }
            for _, e in scored[:limit]
        ]

    def _json(self, status: int, payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(status, json=payload, headers={"content-type": "application/json"})

    def _html(self, status: int, body: str) -> httpx.Response:
        return httpx.Response(
            status,
            content=body.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    # -- transport -----------------------------------------------------------------------
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.requests.append(str(url))
        if url.path == "/w/api.php":
            params = parse_qs(url.query.decode())
            if params.get("list", [""])[0] != "search":
                return self._json(
                    400, {"error": {"code": "unsupported", "info": "fixture transport"}}
                )
            query = params.get("srsearch", [""])[0]
            limit = int(params.get("srlimit", ["5"])[0])
            key = normalize_query(query)
            rel = self.index["searches"].get(key)
            if rel:
                payload = json.loads((self.dir / rel).read_text(encoding="utf-8"))
                payload["query"]["search"] = payload["query"]["search"][:limit]
                return self._json(200, payload)
            hits = self._local_search(query, limit)
            return self._json(
                200,
                {
                    "batchcomplete": "",
                    "query": {"searchinfo": {"totalhits": len(hits)}, "search": hits},
                    "_fixture": "local-fallback",
                },
            )
        if url.path.startswith("/wiki/"):
            full = f"https://{url.host}{url.path}"
            meta = self.index["pages"].get(full)
            if meta is None:
                # tolerate percent-encoding differences
                wanted = url_to_title(full)
                for candidate, m in self.index["pages"].items():
                    if url_to_title(candidate) == wanted:
                        meta = m
                        break
            if meta is None:
                return self._html(
                    404, "<html><head><title>Not found</title></head><body>404</body></html>"
                )
            body = (self.dir / meta["file"]).read_text(encoding="utf-8")
            return self._html(200, body)
        return self._html(404, "<html><body>fixture transport: unknown path</body></html>")


def make_http_client(
    mode: str, *, timeout_s: float = 10.0, fixture_dir: str | Path = "tests/fixtures/wikipedia"
) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    if mode == "fixture":
        return httpx.Client(
            transport=FixtureTransport(fixture_dir), headers=headers, timeout=timeout_s
        )
    if mode == "live":
        return httpx.Client(headers=headers, timeout=timeout_s, follow_redirects=True)
    raise ValueError(f"unknown network mode {mode!r}")
