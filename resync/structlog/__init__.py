from __future__ import annotations

"""Minimal structlog stub.

The real project uses the external `structlog` package. This repository ships a
tiny compatibility layer so local tools/tests can import the codebase even when
structlog isn't installed.

In production you SHOULD install the real structlog package.
"""

import logging
from typing import Any, Optional

from .types import EventDict, WrappedLogger  # noqa: F401
from . import contextvars, processors, stdlib, dev  # noqa: F401

BoundLogger = Any  # compatibility alias


class PrintLoggerFactory:  # pragma: no cover
    def __call__(self, *args: Any, **kwargs: Any) -> logging.Logger:
        return logging.getLogger(kwargs.get("logger_name", __name__))


class _ShimLogger:
    def __init__(self, name: str):
        self._l = logging.getLogger(name)

    def _log(self, method: Any, event: str, *args: Any, **kwargs: Any) -> None:
        """Log an event in a way that won't crash stdlib logging.

        The stdlib logging machinery reserves a handful of attribute names on
        LogRecord (e.g. ``exc_info``). Passing these through ``extra=`` raises
        ``KeyError: Attempt to overwrite ...``.

        This stub emulates the most important part of structlog behaviour:
        - allow structured kwargs
        - support ``exc_info`` / ``stack_info`` the standard logging way
        - avoid collisions by nesting structured fields under ``event_data``
        """

        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", None)

        # Nest all custom fields under a single key to avoid collisions with
        # LogRecord attributes (message/asctime/etc.).
        extra = {"event_data": kwargs} if kwargs else None
        if extra is None:
            method(event, *args, exc_info=exc_info, stack_info=stack_info)
        else:
            method(event, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)

    def bind(self, **_kwargs: Any) -> "_ShimLogger":
        return self

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(self._l.debug, event, *args, **kwargs)

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(self._l.info, event, *args, **kwargs)

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(self._l.warning, event, *args, **kwargs)

    def error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(self._l.error, event, *args, **kwargs)

    def critical(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(self._l.critical, event, *args, **kwargs)


def get_logger(name: Optional[str] = None) -> _ShimLogger:
    return _ShimLogger(name or __name__)


def configure(*args: Any, **kwargs: Any) -> None:
    # no-op for stub
    return None


def make_filtering_bound_logger(_level: int):
    # return a logger factory compatible with structlog expectations
    def _factory(name: Optional[str] = None):
        return get_logger(name)
    return _factory
