from __future__ import annotations

import logging
from typing import Any

import structlog

from resync.core.logging_utils import SecretRedactor


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging with consistent structured output."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl)

    root_logger = logging.getLogger()
    if not any(isinstance(log_filter, SecretRedactor) for log_filter in root_logger.filters):
        root_logger.addFilter(SecretRedactor())

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
