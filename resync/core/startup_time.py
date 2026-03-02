"""Application startup time tracking.

Extracted from the deprecated ``lifespan.py`` module.  Used by the admin
dashboard to show uptime.
"""

from __future__ import annotations

from datetime import datetime, timezone

_startup_time: datetime | None = None

def get_startup_time() -> datetime | None:
    """Return the application startup time (UTC), or *None* before boot."""
    return _startup_time

def set_startup_time() -> None:
    """Record the current UTC time as the application startup time."""
    global _startup_time
    _startup_time = datetime.now(timezone.utc)
