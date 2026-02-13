"""
Exception handling utilities.

Provides utilities for graceful exception handling without silently
swallowing errors (a security anti-pattern).

SECURITY (v5.4.1):
- All exception handlers log at minimum DEBUG level
- Critical paths use appropriate log levels
- No more bare `except Exception: pass`
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_call(
    func: Callable[..., T],
    *args,
    default: T = None,
    log_level: int = logging.DEBUG,
    context: str = "",
    **kwargs,
) -> T:
    """
    Call a function with exception handling.

    Unlike bare `except Exception: pass`, this logs the error.

    Args:
        func: Function to call
        *args: Positional arguments
        default: Default value on error
        log_level: Logging level for errors
        context: Context string for error message
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        ctx = f" [{context}]" if context else ""
        logger.log(
            log_level,
            f"safe_call_failed{ctx}",
            func=getattr(func, "__name__", str(func)),
            error=str(e),
            error_type=type(e).__name__,
        )
        return default


async def safe_call_async(
    func: Callable[..., T],
    *args,
    default: T = None,
    log_level: int = logging.DEBUG,
    context: str = "",
    **kwargs,
) -> T:
    """
    Call an async function with exception handling.

    Args:
        func: Async function to call
        *args: Positional arguments
        default: Default value on error
        log_level: Logging level for errors
        context: Context string for error message
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        ctx = f" [{context}]" if context else ""
        logger.log(
            log_level,
            f"safe_call_async_failed{ctx}",
            func=getattr(func, "__name__", str(func)),
            error=str(e),
            error_type=type(e).__name__,
        )
        return default


def graceful_degradation(
    default: Any = None,
    log_level: int = logging.DEBUG,
    reraise: bool = False,
):
    """
    Decorator for graceful degradation on errors.

    Use instead of bare try/except blocks.

    Args:
        default: Default value to return on error
        log_level: Logging level for errors
        reraise: If True, re-raise after logging

    Example:
        @graceful_degradation(default={}, log_level=logging.WARNING)
        def get_statistics() -> dict:
            # May fail but shouldn't crash the app
            return expensive_operation()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    "graceful_degradation",
                    func=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


def graceful_degradation_async(
    default: Any = None,
    log_level: int = logging.DEBUG,
    reraise: bool = False,
):
    """
    Async decorator for graceful degradation on errors.

    Example:
        @graceful_degradation_async(default=[], log_level=logging.WARNING)
        async def fetch_metrics() -> list:
            return await slow_api_call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    "graceful_degradation_async",
                    func=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


class SuppressedExceptionTracker:
    """
    Track suppressed exceptions for debugging.

    Instead of bare `except Exception: pass`, use this to track issues.

    Example:
        tracker = SuppressedExceptionTracker("cache_operations")

        try:
            cache.get(key)
        except Exception as e:
            tracker.record(e, context="get_key")

        # Later, check for issues
        if tracker.count > 10:
            logger.warning("high_suppressed_errors", count=tracker.count)
    """

    def __init__(self, name: str, max_samples: int = 10):
        self.name = name
        self.count = 0
        self.samples: list[dict] = []
        self.max_samples = max_samples

    def record(self, error: Exception, context: str = "") -> None:
        """Record a suppressed exception."""
        self.count += 1

        if len(self.samples) < self.max_samples:
            self.samples.append(
                {
                    "error": str(error),
                    "type": type(error).__name__,
                    "context": context,
                }
            )

        logger.debug(
            "exception_suppressed",
            tracker=self.name,
            count=self.count,
            error=str(error),
            context=context,
        )

    def get_stats(self) -> dict:
        """Get suppression statistics."""
        return {
            "name": self.name,
            "total_suppressed": self.count,
            "samples": self.samples,
        }


__all__ = [
    "safe_call",
    "safe_call_async",
    "graceful_degradation",
    "graceful_degradation_async",
    "SuppressedExceptionTracker",
]
