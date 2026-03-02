from __future__ import annotations

import logging
from typing import Any

import structlog

def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging with consistent structured output."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
