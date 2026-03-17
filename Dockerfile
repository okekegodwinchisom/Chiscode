# ╔══════════════════════════════════════════════════════════════╗
# ║  ChisCode — Hugging Face Spaces Dockerfile                  ║
# ║  Port: 7860  |  User: UID 1000  |  Build: multi-stage       ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Repo layout (all paths relative to repo root = Docker build context):
#
#   your-repo/
#   ├── Dockerfile
#   ├── pyproject.toml
#   ├── backend/app/        ← FastAPI source
#   └── frontend/           ← templates + static

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── KEY FIX: build the venv at /app/.venv, not /build/.venv ──────
# uv embeds the venv path into every script shebang at creation time.
# If you build at /build/.venv and copy to /app/.venv, all shebang
# lines still point to /build/.venv/bin/python — which doesn't exist
# in the runtime stage → "no such file or directory" on exec.
# Building at /app/.venv means the paths survive the COPY unchanged.
WORKDIR /app

COPY pyproject.toml .

RUN uv venv /app/.venv && \
    uv sync --no-group dev --no-cache

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ChisCode Team"
LABEL description="ChisCode AI Agent Builder — FastAPI on Hugging Face Spaces"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# HF Spaces requires UID/GID exactly 1000
RUN groupadd --gid 1000 chiscode \
    && useradd  --uid 1000 \
                --gid 1000 \
                --home-dir /home/chiscode \
                --shell /bin/bash \
                --create-home \
                chiscode

WORKDIR /app

# Copy venv — paths are already /app/.venv/... so shebangs are valid
COPY --from=builder /app/.venv /app/.venv

# Activate venv for all subsequent RUN commands and the final CMD
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Copy application source
COPY --chown=1000:1000 backend/app ./app
COPY --chown=1000:1000 frontend    ./frontend

# Writable runtime dirs
RUN mkdir -p /app/logs /app/temp \
    && chown -R 1000:1000 /app/logs /app/temp

# Python flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    APP_ENV=production \
    PORT=8000

# Verify uvicorn is reachable before dropping to non-root —
# fails the build immediately if the venv is broken, not at runtime
RUN /app/.venv/bin/uvicorn --version

USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ... (everything above stays the same)

CMD ["/app/.venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "h11", \
     "--no-access-log", \
     "--log-level", "warning"]
