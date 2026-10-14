"""Optional Slack slash-command adapter (stub; Slack is not required to use this project).

How to wire ``/briefing <topic>``:

1. Create a Slack app, add a slash command ``/briefing`` whose Request URL points at this
   service, e.g. ``https://<host>/slack/command``. Copy the app's *signing secret* into
   ``SLACK_SIGNING_SECRET``.
2. Run this module next to the API (``uv run python scripts/slack_command.py`` serves on
   port 8001) or mount ``router`` into ``briefing_agent.api.create_app``.
3. Slack requires a response within 3 seconds, so the handler acknowledges immediately and
   posts the finished briefing to the ``response_url`` from a background thread.
4. Verify every request with the signing secret (HMAC-SHA256 over ``v0:<ts>:<body>``) and
   reject stale timestamps; without that anyone could trigger paid runs.

The agent's own guard rails still apply: per-run cost cap, per-IP rate limit on the HTTP
layer, and the daily USD cap from settings.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from urllib.parse import parse_qs

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request

from briefing_agent.pipeline import Runner

router = APIRouter()
_runner: Runner | None = None


def _get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner


def verify_signature(body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    if abs(time.time() - float(timestamp or 0)) > 300:
        return False
    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    expected = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def _deliver(response_url: str, topic: str) -> None:
    result = _get_runner().run(topic, {"max_sources": 3})
    text = result["briefing_md"] or f"Run ended with status {result['status']}: {result['reason']}"
    footer = (
        f"\n_status={result['status']} cost=${result['cost_usd']:.4f} run_id={result['run_id']}_"
    )
    httpx.post(
        response_url, json={"response_type": "in_channel", "text": text + footer}, timeout=10
    )


@router.post("/slack/command")
async def slack_command(request: Request) -> dict[str, str]:
    body = await request.body()
    secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret or not verify_signature(
        body,
        request.headers.get("x-slack-request-timestamp", ""),
        request.headers.get("x-slack-signature", ""),
        secret,
    ):
        raise HTTPException(status_code=401, detail="invalid Slack signature")
    form = parse_qs(body.decode())
    topic = (form.get("text") or [""])[0].strip()
    response_url = (form.get("response_url") or [""])[0]
    if not topic:
        return {"response_type": "ephemeral", "text": "Usage: /briefing <topic>"}
    threading.Thread(target=_deliver, args=(response_url, topic), daemon=True).start()
    return {
        "response_type": "ephemeral",
        "text": f"Researching *{topic}*; the briefing will follow.",
    }


app = FastAPI(title="briefing-agent slack adapter")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
