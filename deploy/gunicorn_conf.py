"""Gunicorn configuration for Resync (enterprise-grade defaults).

Run::

    gunicorn -c deploy/gunicorn_conf.py resync.main:app

Environment variables (override in .env or shell):
    WEB_CONCURRENCY      — number of worker processes (default: 2*CPU + 1)
    BIND                 — address:port (default: 0.0.0.0:8000)
    TIMEOUT              — hard kill timeout in seconds (default: 60)
    GRACEFUL_TIMEOUT     — SIGTERM grace period in seconds (default: 30)
    KEEPALIVE            — keep-alive seconds (default: 5)
    MAX_REQUESTS         — restart worker after N requests (leak prevention)
    MAX_REQUESTS_JITTER  — jitter for max_requests to avoid thundering herd
    LOG_LEVEL            — gunicorn log level (default: info)
    FORWARDED_ALLOW_IPS  — comma-separated proxy IPs allowed to set X-Forwarded-*
                           SECURITY: Default is "127.0.0.1" (localhost only).
                           Set to your load-balancer IP(s) in production.
                           Use "*" ONLY in trusted internal networks.
"""

from __future__ import annotations

import multiprocessing
import os

# ── Worker settings ──────────────────────────────────────────
worker_class = "uvicorn_worker.UvicornWorker"

_default_workers = (multiprocessing.cpu_count() * 2) + 1
workers = int(os.getenv("WEB_CONCURRENCY", str(_default_workers)))

# ── Bind ─────────────────────────────────────────────────────
bind = os.getenv("BIND", "0.0.0.0:8000")

# ── Timeouts ─────────────────────────────────────────────────
timeout = int(os.getenv("TIMEOUT", "60"))           # hard kill
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))  # SIGTERM grace
keepalive = int(os.getenv("KEEPALIVE", "5"))

# ── Memory leak prevention ───────────────────────────────────
max_requests = int(os.getenv("MAX_REQUESTS", "2000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "200"))

# ── Logging ──────────────────────────────────────────────────
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"  # stdout
errorlog = "-"   # stderr

# ── Security: trusted proxy headers ─────────────────────────
# SECURITY FIX: Default to "127.0.0.1" (localhost only) instead of "*".
# The wildcard "*" allows ANY client to spoof X-Forwarded-For, which breaks
# IP-based rate limiting and audit logging in production.
#
# Production config:
#   - Behind nginx on same host:  FORWARDED_ALLOW_IPS=127.0.0.1
#   - Behind external LB:         FORWARDED_ALLOW_IPS=10.0.0.5,10.0.0.6
#   - Kubernetes pod CIDR:        FORWARDED_ALLOW_IPS=10.244.0.0/16
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")

# ── Prometheus multiprocess support ─────────────────────────
# IMPORTANT: Set PROMETHEUS_MULTIPROC_DIR to a writable directory.
# Without it, each worker reports independent metrics and Prometheus
# aggregation (sum/rate across workers) will be incorrect.
# Example: PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
import os as _os
_prometheus_multiproc_dir = _os.getenv("PROMETHEUS_MULTIPROC_DIR", "")
if _prometheus_multiproc_dir:
    # Ensure directory exists
    import pathlib as _pathlib
    _pathlib.Path(_prometheus_multiproc_dir).mkdir(parents=True, exist_ok=True)


def child_exit(server: object, worker: object) -> None:  # pragma: no cover
    """Gunicorn hook for prometheus-client multiprocess mode."""
    try:
        from prometheus_client import multiprocess  # type: ignore
        multiprocess.mark_process_dead(getattr(worker, "pid", 0))
    except Exception:
        return
