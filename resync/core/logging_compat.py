from __future__ import annotations

from typing import Any

def log_event(logger: Any, level: str, event: str, **fields: Any) -> None:
    """Log an event with consistent structured fields for structlog or stdlib logging."""
    lvl = level.lower().strip()
    method = getattr(logger, lvl, None)
    if method is None:
        log_method = getattr(logger, "log", None)
        if log_method is None:
            return
        log_method(level.upper(), event, extra=fields)
        return

    try:
        method(event, **fields)
    except TypeError:
        method(event, extra=fields)
