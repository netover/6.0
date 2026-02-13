"""
Health Check Retry Utility

This module provides retry mechanisms with exponential backoff for health checks,
ensuring reliable health monitoring even in the face of transient failures.
"""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class HealthCheckRetry:
    """
    Provides retry functionality with exponential backoff for health checks.

    This utility helps ensure health checks are reliable by automatically retrying
    failed checks with increasing delays between attempts.
    """

    @staticmethod
    async def with_retry(
        func: Callable[[], Any],
        component_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """
        Execute a health check function with retry and exponential backoff.

        Args:
            func: Async function to execute
            component_name: Name of the component being checked (for logging)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            Result of the function call

        Raises:
            Exception: Final exception after all retries are exhausted
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                last_exception = e

                if attempt == max_retries - 1:
                    logger.error(
                        "health_check_failed_after_retries",
                        component_name=component_name,
                        max_retries=max_retries,
                        final_error=str(e),
                    )
                    raise

                wait_time = base_delay * (2**attempt)  # 1s, 2s, 4s
                logger.warning(
                    "health_check_failed_retrying",
                    component_name=component_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        return None

    @staticmethod
    def create_retry_wrapper(
        component_name: str, max_retries: int = 3, base_delay: float = 1.0
    ) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
        """
        Create a retry wrapper function for a specific component.

        Args:
            component_name: Name of the component being checked
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            Decorator function that can be applied to health check methods
        """

        def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
            async def wrapper(*args, **kwargs) -> Any:
                return await HealthCheckRetry.with_retry(
                    lambda: func(*args, **kwargs),
                    component_name,
                    max_retries,
                    base_delay,
                )

            return wrapper

        return decorator
