"""
Centralized error handling utilities for API operations.

This module provides standardized error handling and response mapping
for consistent API error responses across all endpoints.
"""

from fastapi import HTTPException, status

from resync.core.constants import ErrorMessages
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Error status code mapping for consistent API responses
ERROR_STATUS_MAP: dict[str, tuple[int, str]] = {
    "timeout": (status.HTTP_504_GATEWAY_TIMEOUT, ErrorMessages.TIMEOUT.value),
    "connection": (status.HTTP_504_GATEWAY_TIMEOUT, ErrorMessages.CONNECTION.value),
    "auth": (status.HTTP_401_UNAUTHORIZED, ErrorMessages.AUTH_REQUIRED.value),
    "unauthorized": (status.HTTP_401_UNAUTHORIZED, ErrorMessages.UNAUTHORIZED.value),
    "forbidden": (status.HTTP_403_FORBIDDEN, ErrorMessages.FORBIDDEN.value),
    "not found": (status.HTTP_404_NOT_FOUND, ErrorMessages.NOT_FOUND.value),
    "404": (status.HTTP_404_NOT_FOUND, ErrorMessages.NOT_FOUND.value),
    "validation": (
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorMessages.VALIDATION_ERROR.value,
    ),
    "invalid": (
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorMessages.VALIDATION_ERROR.value,
    ),
    "conflict": (status.HTTP_409_CONFLICT, ErrorMessages.CONFLICT.value),
    "duplicate": (status.HTTP_409_CONFLICT, ErrorMessages.CONFLICT.value),
    "unavailable": (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorMessages.SERVICE_UNAVAILABLE.value,
    ),
    "rate limit": (
        status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorMessages.RATE_LIMIT_EXCEEDED.value,
    ),
    "quota": (status.HTTP_429_TOO_MANY_REQUESTS, ErrorMessages.QUOTA_EXCEEDED.value),
}


def handle_api_error(
    exception: Exception,
    operation: str,
    default_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> HTTPException:
    """
    Centralizes error handling for API operations with consistent mapping.

    Maps exception details to appropriate HTTP status codes and messages,
    providing standardized error responses across all API endpoints.

    Args:
        exception: The exception that occurred during the operation
        operation: Description of the operation that failed (for logging)
        default_status: Default HTTP status code if no specific match found

    Returns:
        HTTPException with appropriate status code and detail message

    Raises:
        HTTPException: Always raises with mapped status code and message
    """
    # If it's already an HTTPException, return as-is
    if isinstance(exception, HTTPException):
        return exception

    # Log the error with structured context
    logger.error(
        "api_operation_failed",
        operation=operation,
        error=str(exception),
        error_type=type(exception).__name__,
        exc_info=True,
    )

    # Convert error message to lowercase for pattern matching
    error_lower = str(exception).lower()

    # Find matching error pattern and return appropriate HTTP exception
    for keyword, (status_code, message_template) in ERROR_STATUS_MAP.items():
        if keyword in error_lower:
            # Use safe detail without leaking internal exception info
            safe_detail = message_template.format(
                operation=operation,
                detail="See server logs for details",
            )
            return HTTPException(status_code=status_code, detail=safe_detail)

    # Default error response for unmatched exceptions — no internal details
    safe_detail = ErrorMessages.INTERNAL_ERROR.value.format(
        operation=operation,
        detail="An unexpected error occurred. See server logs for details",
    )
    return HTTPException(status_code=default_status, detail=safe_detail)


def create_error_response(
    status_code: int,
    message: str,
    operation: str = "operation",
    details: dict | None = None,
) -> HTTPException:
    """
    Creates a standardized HTTPException for API responses.

    Args:
        status_code: HTTP status code
        message: Error message
        operation: Operation description for context
        details: Additional error details (optional)

    Returns:
        HTTPException with standardized format
    """
    error_detail = {
        "message": message,
        "operation": operation,
    }

    if details:
        error_detail["details"] = details

    return HTTPException(status_code=status_code, detail=error_detail)
