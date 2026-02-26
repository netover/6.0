"""
Prometheus metrics exposition endpoint.

This module exposes a /metrics endpoint compatible with Prometheus scraping,
using the official prometheus-client library.

Supports multiprocess mode (Gunicorn) when PROMETHEUS_MULTIPROC_DIR is set.
See: https://prometheus.github.io/client_python/multiprocess/
"""

from __future__ import annotations

import os
from typing import Final

from fastapi import APIRouter, Request, Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY, generate_latest, multiprocess
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):  # pragma: no cover
    # If prometheus-client is missing, expose an empty payload.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = None  # type: ignore[assignment]
    REGISTRY = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    multiprocess = None  # type: ignore[assignment]

router = APIRouter(tags=["Monitoring - Prometheus"])

_METRICS_TOKEN_ENV: Final[str] = "METRICS_TOKEN"

def _is_authorized(request: Request) -> bool:
    """
    Optional bearer token protection for /metrics.

    If METRICS_TOKEN is not set, /metrics is open (useful in private networks).
    If set, requires: Authorization: Bearer <token>
    """
    token = os.getenv(_METRICS_TOKEN_ENV, "").strip()
    if not token:
        return True

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    return auth[7:].strip() == token

def _build_registry() -> object:
    """
    Build the correct registry for single-process or multiprocess mode.

    Best practice per client_python docs: create a new CollectorRegistry per-request
    when using MultiProcessCollector, to avoid duplicate metric registration.
    """
    if CollectorRegistry is None or REGISTRY is None:
        return None

    mp_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if mp_dir and multiprocess is not None:
        registry = CollectorRegistry(support_collectors_without_names=True)
        multiprocess.MultiProcessCollector(registry)
        return registry

    return REGISTRY

@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    if not _is_authorized(request):
        return Response(status_code=401, content="unauthorized\n", media_type="text/plain")

    registry = _build_registry()
    if registry is None or generate_latest is None:
        return Response(status_code=200, content="# prometheus-client not installed\n", media_type="text/plain")

    data = generate_latest(registry)  # bytes
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
