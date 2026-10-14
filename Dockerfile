# Multi-stage build with uv. Runs the API with the fake provider unless an API key is passed.
# `llm-evalops` is a path dependency (../llm-evalops) outside this build context; pass it as a
# named context:
#   docker build --build-context llm-evalops=../llm-evalops -t briefing-agent .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# `../llm-evalops` relative to /app is /llm-evalops; --no-editable installs it into the venv.
COPY --from=llm-evalops pyproject.toml README.md /llm-evalops/
COPY --from=llm-evalops src /llm-evalops/src
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable
COPY src ./src
COPY pricing.json evals.yaml ./
COPY tests/fixtures ./tests/fixtures
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    DB_PATH=/app/data/briefing.db \
    TRACE_PATH=/app/traces/spans.jsonl \
    CHECKPOINT_DIR=/app/checkpoints \
    FIXTURE_DIR=/app/tests/fixtures/wikipedia
RUN mkdir -p /app/data /app/traces /app/checkpoints
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "briefing_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
