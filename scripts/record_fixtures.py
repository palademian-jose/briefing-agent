"""Record real Wikipedia responses once so tests, CI and the fake-provider demo run offline.

For every topic in the golden set: one search (top 5) and the HTML of the top N result
pages. Pages are trimmed to the article heading plus the first body paragraphs (the lead and
early sections) to keep the repository small; the structure the extractor relies on
(``div.mw-parser-output`` with ``<p>`` children, ``sup.reference`` markers) is preserved.

Usage: uv run python scripts/record_fixtures.py [--pages-per-topic 4] [--max-paragraphs 40]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from briefing_agent.tools.network import make_http_client, normalize_query  # noqa: E402
from briefing_agent.tools.wikipedia import search_wikipedia  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def trim_html(html: str, max_paragraphs: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.select_one("h1#firstHeading")
    title = (
        heading.get_text(" ", strip=True)
        if heading
        else (soup.title.get_text(strip=True) if soup.title else "")
    )
    body = soup.select_one("div.mw-parser-output")
    paragraphs = []
    if body is not None:
        for p in body.find_all("p"):
            if p.find_parent("table") is not None:
                continue
            paragraphs.append(str(p))
            if len(paragraphs) >= max_paragraphs:
                break
    inner = "\n".join(paragraphs)
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{title} - Wikipedia</title></head>\n<body>\n"
        f'<h1 id="firstHeading">{title}</h1>\n'
        f'<div id="mw-content-text"><div class="mw-parser-output">\n{inner}\n</div></div>\n'
        "</body></html>\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "evals" / "golden" / "tasks.jsonl"))
    ap.add_argument("--pages-per-topic", type=int, default=4)
    ap.add_argument("--max-paragraphs", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    (FIXTURES / "search").mkdir(parents=True, exist_ok=True)
    (FIXTURES / "pages").mkdir(parents=True, exist_ok=True)
    index_path = FIXTURES / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"searches": {}, "pages": {}}
    )

    topics = [
        json.loads(line)["input"]["topic"]
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    client = make_http_client("live", timeout_s=20)
    for topic in topics:
        key = normalize_query(topic)
        if key in index["searches"]:
            print(f"skip (recorded): {topic}")
            continue
        api = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": "5",
            "format": "json",
            "utf8": "1",
        }
        raw = client.get(api, params=params)
        raw.raise_for_status()
        search_file = f"search/{slug(topic)}.json"
        (FIXTURES / search_file).write_text(
            json.dumps(raw.json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        index["searches"][key] = search_file
        results = search_wikipedia(client, topic, limit=5)["results"]
        print(f"{topic}: {len(results)} hits")
        for r in results[: args.pages_per_topic]:
            url = r["url"]
            if url in index["pages"]:
                continue
            time.sleep(args.sleep)
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"  {url}: HTTP {resp.status_code}, skipped")
                continue
            page_file = f"pages/{slug(r['title'])}.html"
            (FIXTURES / page_file).write_text(
                trim_html(resp.text, args.max_paragraphs), encoding="utf-8"
            )
            index["pages"][url] = {
                "file": page_file,
                "title": r["title"],
                "pageid": r.get("pageid"),
                "snippet": r.get("snippet", "")[:200],
            }
            print(f"  recorded {r['title']}")
        time.sleep(args.sleep)
        index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(f"done: {len(index['searches'])} searches, {len(index['pages'])} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
