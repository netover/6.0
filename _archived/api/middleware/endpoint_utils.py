"""
Utility functions and decorators for endpoint cross-cutting concerns.
This module separates logging, error handling, and other cross-cutting concerns
from business logic in endpoints.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps

from fastapi import Request

from resync.core.logger import log_with_correlation
from resync.core.metrics import runtime_metrics
from resync.core.utils.error_utils import create_error_response_from_exception

logger = logging.getLogger(__name__)


def with_monitoring(operation_name: str):
    """
    Decorator to add monitoring and logging to endpoint functions.
    Separates monitoring concerns from business logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request if available to get correlation ID
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            correlation_id = getattr(request, "correlation_id", None) if request else None
            if not correlation_id:
                correlation_id = runtime_metrics.create_correlation_id(
                    {"component": "api_endpoint", "operation": operation_name}
                )

            start_time = time.time()

            try:
                # Log the start of the operation
                log_with_correlation(logging.INFO, f"Starting {operation_name}", correlation_id)

                # Execute the actual function
                result = await func(*args, **kwargs)

                # Record successful metrics
                runtime_metrics.tws_status_requests_success.increment(1)
                runtime_metrics.api_response_time.observe(time.time() - start_time)

                # Log successful completion
                log_with_correlation(logging.INFO, f"Completed {operation_name}", correlation_id)

                return result

            except Exception as e:
                # Record error metrics
                runtime_metrics.tws_status_requests_failed.increment(1)

                # Log error with correlation ID
                log_with_correlation(
                    logging.ERROR,
                    f"Error in {operation_name}: {str(e)}",
                    correlation_id,
                    exc_info=True,
                )

                # Re-raise the exception to be handled by FastAPI
                raise

            finally:
                # Close correlation ID to prevent memory leaks
                runtime_metrics.close_correlation_id(correlation_id)

        return wrapper

    return decorator


def handle_endpoint_errors(operation: str):
    """
    Decorator to handle errors consistently across endpoints.
    Separates error handling from business logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Create standardized error response
                error_response = create_error_response_from_exception(
                    e,
                    request=kwargs.get("request")
                    or next((arg for arg in args if isinstance(arg, Request)), None),
                )

                logger.error(f"Error in {operation}: {str(e)}", exc_info=True)

                # Return appropriate error based on exception type
                from fastapi import HTTPException

                if isinstance(e, HTTPException):
                    raise e
                # In the context of FastAPI endpoints, raise HTTPException for proper error response
                raise HTTPException(
                    status_code=500,
                    detail=error_response.message or str(e),
                ) from None

        return wrapper

    return decorator


def with_security_validation():
    """
    Decorator to add security validation to endpoints.
    Separates security concerns from business logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Add security validation here if needed
            # For now, just pass through to the function
            return await func(*args, **kwargs)

        return wrapper

    return decorator
