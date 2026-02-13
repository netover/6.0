"""
Common error handling utilities to eliminate code duplication.

This module contains standard error handling patterns that can be reused
across multiple modules in the application.
"""

import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from ..exceptions import ResyncException

logger = logging.getLogger(__name__)

# Create a type variable for preserving function signatures
F = TypeVar("F", bound=Callable[..., Any])


def handle_parsing_errors(
    error_message: str = "Error occurred during parsing",
) -> Callable[[F], F]:
    """
    Decorator to handle parsing errors consistently.

    Args:
        error_message: Base error message to use when raising ParsingError
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.debug("%s: %s", error_message, e)
                from ..exceptions import ParsingError

                raise ParsingError(f"{error_message}: {e}") from e

        return cast(F, wrapper)

    return decorator


def handle_llm_errors(
    error_message: str = "Error occurred during LLM call",
) -> Callable[[F], F]:
    """
    Decorator to handle LLM-related errors consistently.

    Args:
        error_message: Base error message to use when raising LLMError
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("%s: %s", error_message, e, exc_info=True)
                from ..exceptions import LLMError

                raise LLMError(f"{error_message}: {e}") from e

        return cast(F, wrapper)

    return decorator


def handle_api_errors(
    exception_class: type[ResyncException],
    error_message: str = "Error occurred in API call",
) -> Callable[[F], F]:
    """
    Decorator to handle API-related errors consistently.

    Args:
        exception_class: The ResyncException subclass to raise
        error_message: Base error message to use
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("%s: %s", error_message, e, exc_info=True)
                raise exception_class(f"{error_message}: {e}") from e

        return cast(F, wrapper)

    return decorator


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """
    Decorator to retry a function if specific exceptions are raised.

    Supports both synchronous and asynchronous functions with proper
    event loop management.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch
        logger: Logger to use for retry messages

    Returns:
        Decorated function that retries on specified exceptions
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger_instance = logger or logging.getLogger(func.__module__)

                # Extract retry-specific arguments from kwargs
                current_max_retries = kwargs.pop("max_retries", max_retries)
                current_delay = kwargs.pop("initial_backoff", delay)

                cleaned_kwargs = kwargs

                for attempt in range(current_max_retries + 1):
                    try:
                        return await func(*args, **cleaned_kwargs)
                    except Exception as e:
                        # Resolve lazy exceptions if callable
                        allowed_exceptions = exceptions() if callable(exceptions) else exceptions
                        if not isinstance(e, allowed_exceptions):
                            raise

                        if attempt < current_max_retries:
                            logger_instance.info(
                                f"Attempt {attempt + 1} failed: {e}. "
                                "Retrying in {current_delay:.2f} seconds..."
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger_instance.error(
                                f"Failed after {max_retries} retries: {e}",
                                exc_info=True,
                            )
                            raise
                return None  # This should never be reached

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time

            logger_instance = logger or logging.getLogger(func.__module__)
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Resolve lazy exceptions if callable
                    allowed_exceptions = exceptions() if callable(exceptions) else exceptions
                    if not isinstance(e, allowed_exceptions):
                        raise

                    if attempt < max_retries:
                        logger_instance.info(
                            f"Attempt {attempt + 1} failed: {e}. "
                            "Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)  # Use sync sleep for sync functions
                        current_delay *= backoff
                    else:
                        logger_instance.error(
                            f"Failed after {max_retries} retries: {e}",
                            exc_info=True,
                        )
                        raise
            return None  # This should never be reached

        return cast(F, sync_wrapper)

    return decorator


def log_and_handle_exception(
    exception_class: type[ResyncException], message: str, log_level: int = logging.ERROR
) -> Callable[[F], F]:
    """
    Context manager or decorator to handle exceptions consistently.

    Args:
        exception_class: The ResyncException subclass to raise
        message: Error message to use
        log_level: Logging level to use for logging the exception
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"{message}: {e}", exc_info=True)
                raise exception_class(f"{message}: {e}") from e

        return cast(F, wrapper)

    return decorator
