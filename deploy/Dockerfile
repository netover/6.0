# ============================================================
# Production Dockerfile — multi-stage, non-root, minimal image
# Build:  docker build -t resync:prod -f deploy/Dockerfile .
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tools (only in builder stage — not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix directory (keeps final image clean)
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH"

# Runtime system deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user — SECURITY: never run as root in production
RUN groupadd --gid 1001 resync \
    && useradd --uid 1001 --gid resync --shell /bin/bash --create-home resync

WORKDIR /app

# Copy application source
COPY --chown=resync:resync . .

# Switch to non-root user
USER resync

EXPOSE 8000

# Health check so orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Gunicorn + UvicornWorker
CMD ["gunicorn", "-c", "deploy/gunicorn_conf.py", "resync.main:app"]
