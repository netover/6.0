"""Resync API middleware package.

This module intentionally keeps imports lightweight.

Avoid importing heavy modules at package import time because FastAPI/uvicorn
(and gunicorn --preload) may import modules in contexts where an event loop
is not yet available.

Downstream code should import specific middleware directly, e.g.:

    from resync.api.middleware.cors_middleware import LoggingCORSMiddleware
"""

from __future__ import annotations

from .correlation_id import CorrelationIdMiddleware
from .csrf_protection import CSRFMiddleware

# Optional middleware (may require optional deps / configuration).
# Import lazily and degrade gracefully when unavailable.
try:  # pragma: no cover
    from .redis_validation import RedisHealthMiddleware, RedisValidationMiddleware
except ImportError:  # pragma: no cover
    RedisHealthMiddleware = None  # type: ignore[assignment]
    RedisValidationMiddleware = None  # type: ignore[assignment]

__all__ = [
    "CorrelationIdMiddleware",
    "CSRFMiddleware",
    "RedisHealthMiddleware",
    "RedisValidationMiddleware",
]
