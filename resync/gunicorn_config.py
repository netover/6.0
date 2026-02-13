"""
Gunicorn Configuration for RESYNC v6.2.0

Production command:
    gunicorn -c gunicorn_config.py resync.main:app

Or with explicit overrides:
    gunicorn -c gunicorn_config.py --workers 4 --bind 0.0.0.0:8000 resync.main:app

Environment variables (all optional — sensible defaults below):
    GUNICORN_WORKERS        Number of worker processes (default: CPU cores * 2 + 1, max 8)
    GUNICORN_BIND           Bind address (default: 127.0.0.1:8000)
    GUNICORN_TIMEOUT        Worker timeout in seconds (default: 120)
    GUNICORN_MAX_REQUESTS   Requests before worker restart (default: 2000)
    GUNICORN_LOG_LEVEL      Log level (default: info)
    GUNICORN_ACCESS_LOG     Access log path (default: - i.e. stdout)
    GUNICORN_ERROR_LOG      Error log path (default: - i.e. stderr)
"""

import multiprocessing
import os

# ── Worker Configuration ─────────────────────────────────────────────────────

# Uvicorn workers for async FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Workers: default = min(CPU * 2 + 1, 8)
# More workers = more RAM. Each worker loads the full app (~500 MB base + Docling models).
# For a 16 GB server: 4 workers is safe. For 8 GB: 2 workers.
_default_workers = min(multiprocessing.cpu_count() * 2 + 1, 8)
workers = int(os.getenv("GUNICORN_WORKERS", _default_workers))

# ── Network ──────────────────────────────────────────────────────────────────

# Bind to localhost by default — use nginx/caddy as reverse proxy
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

# Keep-alive for connection reuse behind reverse proxy
keepalive = 10

# ── Timeouts ─────────────────────────────────────────────────────────────────

# Worker timeout — Docling PDF conversion can take 60-120s for large documents
# Set higher than default 30s to avoid killing workers during document ingestion
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Graceful shutdown timeout
graceful_timeout = 30

# ── Worker Lifecycle ─────────────────────────────────────────────────────────

# Restart workers after N requests to prevent memory leaks
# Docling loads ~3-4 GB of ML models — periodic restart keeps memory clean
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "2000"))
max_requests_jitter = 200  # Random jitter to avoid all workers restarting at once

# Preload app in master process — shares memory across workers via fork COW
# Saves ~200 MB per worker for shared Python bytecode
preload_app = True

# ── Logging ──────────────────────────────────────────────────────────────────

loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")

# Use structured format for access logs (parseable by log aggregators)
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ── Process Naming ───────────────────────────────────────────────────────────

proc_name = "resync"

# ── Server Hooks ─────────────────────────────────────────────────────────────


def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting RESYNC with %d workers", server.app.cfg.workers)
    server.log.info("Bind: %s", server.app.cfg.bind)
    server.log.info("Worker class: %s", server.app.cfg.worker_class)
    server.log.info("Timeout: %ds", server.app.cfg.timeout)
    server.log.info("Max requests: %d (jitter: %d)",
                     server.app.cfg.max_requests, server.app.cfg.max_requests_jitter)


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def worker_exit(server, worker):
    """Called when a worker exits."""
    server.log.info("Worker exited (pid: %s)", worker.pid)


def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("RESYNC shutting down")
