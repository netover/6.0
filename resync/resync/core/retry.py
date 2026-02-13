"""
Retry helpers for robust error handling across the application.

This module provides standardized retry configurations for different types of operations,
ensuring consistent behavior and proper logging of retry attempts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)

logger = logging.getLogger(__name__)

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])
ExceptionTypes = type[Exception] | list[type[Exception]]


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log information about the retry attempt."""
    if retry_state.outcome is not None:
        exc = retry_state.outcome.exception()
        if exc:
            logger.warning(
                "Retry attempt %d/%d after %0.2fs for %s: %s",
                retry_state.attempt_number,
                (
                    retry_state.retry_object.stop.max_attempt_number
                    if hasattr(retry_state.retry_object.stop, "max_attempt_number")
                    else "∞"
                ),
                retry_state.seconds_since_start,
                getattr(retry_state.fn, "__qualname__", str(retry_state.fn)),
                exc,
            )
        else:
            logger.warning(
                "Retry attempt %d/%d after %0.2fs for %s due to unexpected result",
                retry_state.attempt_number,
                (
                    retry_state.retry_object.stop.max_attempt_number
                    if hasattr(retry_state.retry_object.stop, "max_attempt_number")
                    else "∞"
                ),
                retry_state.seconds_since_start,
                getattr(retry_state.fn, "__qualname__", str(retry_state.fn)),
            )


def http_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: ExceptionTypes | None = None,
) -> Callable[[F], F]:
    """
    Decorator for HTTP requests with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        exceptions: Exception types to retry on (defaults to network-related exceptions)

    Returns:
        Decorated function with retry logic
    """
    if exceptions is None:
        exceptions = [
            httpx.RequestError,
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            ConnectionError,
        ]

    # Convert exceptions to a tuple for tenacity
    exception_tuple: tuple[type[Exception], ...]
    if isinstance(exceptions, list):
        exception_tuple = tuple(exceptions)
    elif isinstance(exceptions, type):
        exception_tuple = (exceptions,)
    else:
        exception_tuple = (
            exceptions
            if isinstance(exceptions, tuple)
            else (
                httpx.RequestError,
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                ConnectionError,
            )
        )

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_tuple),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def database_retry(
    max_attempts: int = 5,
    min_wait: float = 0.1,
    max_wait: float = 2.0,
    exceptions: ExceptionTypes | None = None,
) -> Callable[[F], F]:
    """
    Decorator for database operations with exponential backoff.
    Optimized for quick retries on transient database errors.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        exceptions: Exception types to retry on (customize based on your DB driver)

    Returns:
        Decorated function with retry logic
    """
    # Default to common database exceptions - adjust based on your DB driver
    if exceptions is None:
        exceptions = [ConnectionError, TimeoutError]

    # Convert exceptions to a tuple for tenacity
    exception_tuple: tuple[type[Exception], ...]
    if isinstance(exceptions, list):
        exception_tuple = tuple(exceptions)
    elif isinstance(exceptions, type):
        exception_tuple = (exceptions,)
    else:
        exception_tuple = (
            exceptions if isinstance(exceptions, tuple) else (ConnectionError, TimeoutError)
        )

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_tuple),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def external_service_retry(
    max_attempts: int = 3,
    max_delay: float = 30.0,
    wait_time: float = 2.0,
    exceptions: ExceptionTypes | None = None,
) -> Callable[[F], F]:
    """
    Decorator for external service calls with fixed wait and maximum total delay.
    Useful for third-party APIs with rate limits or unstable connections.

    Args:
        max_attempts: Maximum number of retry attempts
        max_delay: Maximum total delay before giving up, in seconds
        wait_time: Fixed wait time between retries in seconds
        exceptions: Exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    if exceptions is None:
        exceptions = [ConnectionError, TimeoutError, httpx.RequestError]

    # Convert exceptions to a tuple for tenacity
    exception_tuple: tuple[type[Exception], ...]
    if isinstance(exceptions, list):
        exception_tuple = tuple(exceptions)
    elif isinstance(exceptions, type):
        exception_tuple = (exceptions,)
    else:
        exception_tuple = (
            exceptions
            if isinstance(exceptions, tuple)
            else (ConnectionError, TimeoutError, httpx.RequestError)
        )

    return retry(
        stop=(stop_after_attempt(max_attempts) | stop_after_delay(max_delay)),
        wait=wait_fixed(wait_time),
        retry=retry_if_exception_type(exception_tuple),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def retry_on_result(
    result_checker: Callable[[Any], bool],
    max_attempts: int = 3,
    wait_time: float = 1.0,
) -> Callable[[F], F]:
    """
    Retry based on return value evaluation.
    Useful when a function succeeds but returns an undesirable result.

    Args:
        result_checker: Function that returns True if retry is needed
        max_attempts: Maximum number of retry attempts
        wait_time: Wait time between retries in seconds

    Returns:
        Decorated function with retry logic
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_time),
        retry=retry_if_result(result_checker),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
