"""Utility helpers for global context and environment tags.

This module intentionally avoids importing from ``resync.core`` to prevent
import cycles during app initialization.
"""

from __future__ import annotations

import os
import platform
import threading
import uuid

_global_correlation_id: str | None = None
_global_environment_tags: dict[str, str] | None = None
_global_lock = threading.Lock()


def get_global_correlation_id() -> str:
    """Return a stable process-wide correlation identifier."""
    global _global_correlation_id
    if _global_correlation_id is None:
        with _global_lock:
            if _global_correlation_id is None:
                _global_correlation_id = str(uuid.uuid4())
    return _global_correlation_id


def get_environment_tags() -> dict[str, str]:
    """Return cached environment tags used by logs/metrics."""
    global _global_environment_tags
    if _global_environment_tags is None:
        with _global_lock:
            if _global_environment_tags is None:
                _global_environment_tags = {
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "version": os.getenv("APP_VERSION", "6.0.0"),
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "node": platform.node(),
                    "global_id": get_global_correlation_id(),
                }

    # Return a copy so callers cannot mutate shared state.
    return _global_environment_tags.copy()


__all__ = [
    "get_environment_tags",
    "get_global_correlation_id",
]
