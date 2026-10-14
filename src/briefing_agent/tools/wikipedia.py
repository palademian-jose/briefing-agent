"""search_wikipedia and fetch_page: keyless tools over the public MediaWiki API."""

from __future__ import annotations

import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from briefing_agent.tools.errors import ToolError, permanent, requires_human, transient
from briefing_agent.tools.network import title_to_url

_REF_MARK = re.compile(r"\[(?:\d+|[a-z]|note \d+|citation needed)\]", re.IGNORECASE)
_WS = re.compile(r"[ \t ]+")
_TAGS = re.compile(r"<[^>]+>")
_PUNCT_SPACE = re.compile(r"\s+([,.;:)\]])")


def _map_http_error(exc: Exception, url: str) -> ToolError:
    if isinstance(exc, httpx.TimeoutException):
        return transient("timeout", f"timed out fetching {url}", retry_after_ms=500)
    if isinstance(exc, httpx.ConnectError | httpx.ReadError | httpx.RemoteProtocolError):
        return transient("network", f"network error fetching {url}: {exc}", retry_after_ms=500)
    return permanent("http_error", f"unexpected error fetching {url}: {exc}")


def _raise_for_status(resp: httpx.Response, url: str) -> None:
    code = resp.status_code
    if code == 200:
        return
    if code == 429:
        retry_after = resp.headers.get("retry-after")
        ms = int(float(retry_after) * 1000) if retry_after and retry_after.isdigit() else 1000
        raise transient("rate_limited", f"HTTP 429 for {url}", retry_after_ms=ms)
    if code >= 500:
        raise transient("upstream_error", f"HTTP {code} for {url}", retry_after_ms=500)
    if code == 404:
        raise permanent(
            "not_found",
            f"HTTP 404 for {url}",
            hint="This article does not exist; fetch a different search result instead.",
        )
    if code == 403:
        raise requires_human(
            "blocked",
            f"HTTP 403 for {url}: request identified as automated traffic",
            hint="An operator must adjust the User-Agent or egress IP before retrying.",
        )
    raise permanent("http_error", f"HTTP {code} for {url}")


def search_wikipedia(
    client: httpx.Client, query: str, limit: int = 5, lang: str = "en"
) -> dict[str, Any]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    api = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        resp = client.get(api, params=params)
    except httpx.HTTPError as exc:
        raise _map_http_error(exc, api) from exc
    _raise_for_status(resp, api)
    payload = resp.json()
    hits = payload.get("query", {}).get("search", [])
    results = [
        {
            "title": h["title"],
            "url": title_to_url(h["title"], lang),
            "snippet": _WS.sub(" ", _TAGS.sub("", h.get("snippet", ""))).strip(),
            "pageid": h.get("pageid"),
        }
        for h in hits
    ]
    return {
        "ok": True,
        "query": query,
        "total_hits": payload.get("query", {}).get("searchinfo", {}).get("totalhits", len(results)),
        "results": results,
    }


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text). Lightweight extraction tuned for Wikipedia but generic enough:
    body paragraphs with reference markers, tables, navboxes and scripts removed."""
    soup = BeautifulSoup(html, "html.parser")
    # Replace rendered math with its TeX source so sentences keep their symbols.
    for m in soup.select(".mwe-math-element"):
        ann = m.find("annotation")
        tex = ann.get_text(strip=True) if ann else ""
        tex = re.sub(r"^\{\\displaystyle\s*(.*)\}$", r"\1", tex).strip()
        m.replace_with(f" {tex} " if tex else " ")
    for tag in soup.select(
        "script, style, table, sup.reference, sup, span.mw-editsection, .navbox, .hatnote, "
        ".infobox, math, .mw-empty-elt, .IPA, .noprint"
    ):
        tag.decompose()
    heading = soup.select_one("h1#firstHeading") or soup.find("h1")
    title = (
        heading.get_text(" ", strip=True)
        if heading
        else (soup.title.get_text(strip=True) if soup.title else "")
    )
    title = re.sub(r"\s+-\s+Wikipedia$", "", title)
    root = soup.select_one("div.mw-parser-output") or soup.body or soup
    candidates = root.find_all("p") or soup.find_all("p")
    paragraphs: list[str] = []
    for p in candidates:
        txt = _WS.sub(" ", p.get_text("", strip=False))
        txt = _REF_MARK.sub("", txt)
        txt = _PUNCT_SPACE.sub(r"\1", txt).strip()
        if len(txt) >= 40:
            paragraphs.append(txt)
    return title, "\n\n".join(paragraphs)


def fetch_page(client: httpx.Client, url: str, max_chars: int = 8000) -> dict[str, Any]:
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        raise _map_http_error(exc, url) from exc
    _raise_for_status(resp, url)
    title, text = html_to_text(resp.text)
    if not text:
        raise permanent(
            "empty_page",
            f"no article text could be extracted from {url}",
            hint="Try a different search result.",
        )
    truncated = len(text) > max_chars
    return {
        "ok": True,
        "url": url,
        "title": title,
        "text": text[:max_chars],
        "chars": min(len(text), max_chars),
        "truncated": truncated,
    }
