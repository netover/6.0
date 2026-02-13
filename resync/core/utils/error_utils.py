"""Error response utilities for standardized error handling."""

import logging
import re
import traceback
import uuid
from functools import lru_cache
from re import Pattern
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from resync.models.error_models import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    BaseErrorResponse,
    BusinessLogicErrorResponse,
    ErrorCategory,
    ErrorSeverity,
    ExternalServiceErrorResponse,
    RateLimitErrorResponse,
    SystemErrorResponse,
    ValidationErrorResponse,
)
from resync.settings import settings

logger = logging.getLogger(__name__)


class ErrorResponseBuilder:
    """Builder class for creating standardized error responses."""

    def __init__(self) -> None:
        self._correlation_id: str | None = None
        self._path: str | None = None
        self._method: str | None = None
        self._include_stack_trace: bool = False

    def with_correlation_id(self, correlation_id: str) -> "ErrorResponseBuilder":
        """Set the correlation ID for error tracking."""
        self._correlation_id = correlation_id
        return self

    def with_request_context(self, request: Request) -> "ErrorResponseBuilder":
        """Set request context information."""
        self._path = request.url.path
        self._method = request.method
        return self

    def with_stack_trace(self, include: bool = True) -> "ErrorResponseBuilder":
        """Set whether to include stack traces in error responses."""
        self._include_stack_trace = include
        return self

    def build_validation_error(
        self, validation_errors: list[dict[str, Any]], message: str | None = None
    ) -> ValidationErrorResponse:
        """Build validation error response."""
        response = ValidationErrorResponse.from_pydantic_errors(
            validation_errors,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
        )
        if message:
            response.message = message
        return response

    def build_authentication_error(
        self, error_type: str, **kwargs: Any
    ) -> AuthenticationErrorResponse:
        """Build authentication error response."""
        if error_type == "unauthorized":
            return AuthenticationErrorResponse.unauthorized(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "invalid_credentials":
            return AuthenticationErrorResponse.invalid_credentials(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "token_expired":
            return AuthenticationErrorResponse.token_expired(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        return AuthenticationErrorResponse(
            error_code="AUTHENTICATION_ERROR",
            message="Authentication failed",
            category=ErrorCategory.AUTHENTICATION,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )

    def build_authorization_error(
        self, error_type: str, **kwargs: Any
    ) -> AuthorizationErrorResponse:
        """Build authorization error response."""
        if error_type == "forbidden":
            return AuthorizationErrorResponse.forbidden(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        return AuthorizationErrorResponse.insufficient_permissions(
            resource=kwargs.pop("resource", "resource"),
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )

    def build_business_logic_error(
        self, error_type: str, **kwargs: Any
    ) -> BusinessLogicErrorResponse:
        """Build business logic error response."""
        if error_type == "resource_not_found":
            return BusinessLogicErrorResponse.resource_not_found(
                resource=kwargs.pop("resource", "Resource"),
                identifier=kwargs.pop("identifier", None),
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "resource_already_exists":
            return BusinessLogicErrorResponse.resource_already_exists(
                resource=kwargs.pop("resource", "Resource"),
                identifier=kwargs.pop("identifier", None),
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "invalid_operation":
            return BusinessLogicErrorResponse.invalid_operation(
                operation=kwargs.pop("operation", "operation"),
                reason=kwargs.pop("reason", None),
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        return BusinessLogicErrorResponse(
            error_code="BUSINESS_LOGIC_ERROR",
            message="Business logic error occurred",
            category=ErrorCategory.BUSINESS_LOGIC,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )

    def build_system_error(
        self, error_type: str, exception: Exception | None = None, **kwargs: Any
    ) -> SystemErrorResponse:
        """Build system error response."""
        if self._include_stack_trace and exception:
            kwargs["details"] = kwargs.get("details", {})
            kwargs["details"]["stack_trace"] = traceback.format_exc()

        if error_type == "internal_server_error":
            return SystemErrorResponse.internal_server_error(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "service_unavailable":
            return SystemErrorResponse.service_unavailable(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        if error_type == "database_error":
            return SystemErrorResponse.database_error(
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        return SystemErrorResponse(
            error_code="SYSTEM_ERROR",
            message="System error occurred",
            category=ErrorCategory.SYSTEM,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )

    def build_external_service_error(
        self, service: str, error_type: str = "service_error", **kwargs: Any
    ) -> ExternalServiceErrorResponse:
        """Build external service error response."""
        if error_type == "timeout":
            return ExternalServiceErrorResponse.timeout(
                service=service,
                correlation_id=self._correlation_id,
                path=self._path,
                method=self._method,
                **kwargs,
            )
        return ExternalServiceErrorResponse.service_error(
            service=service,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )

    def build_rate_limit_error(
        self, limit: int, window: str | None = None, **kwargs
    ) -> RateLimitErrorResponse:
        """Build rate limit error response."""
        return RateLimitErrorResponse.rate_limit_exceeded(
            limit=limit,
            window=window,
            correlation_id=self._correlation_id,
            path=self._path,
            method=self._method,
            **kwargs,
        )


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for error tracking."""
    return str(uuid.uuid4())


def extract_validation_errors(
    validation_error: RequestValidationError | ValidationError,
) -> list[dict[str, Any]]:
    """Extract validation error details from FastAPI or Pydantic validation errors."""
    errors = []

    if isinstance(validation_error, (RequestValidationError, ValidationError)):
        for error in validation_error.errors():
            errors.append(
                {
                    "loc": error["loc"],
                    "msg": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

    return errors


def should_include_stack_trace() -> bool:
    """Determine if stack traces should be included in error responses."""
    # In development environment, include stack traces for debugging
    if hasattr(settings, "APP_ENV") and settings.APP_ENV == "development":
        return True

    # Check specific error detail level setting
    if hasattr(settings, "ERROR_DETAIL_LEVEL"):
        return settings.ERROR_DETAIL_LEVEL == "detailed"

    # Default to not including stack traces in production
    return False


def log_error_response(
    error_response: BaseErrorResponse, original_exception: Exception | None = None
) -> None:
    """Log error response with appropriate level based on severity."""
    log_data = {
        "error_code": error_response.error_code,
        "message": error_response.message,
        "category": error_response.category,
        "severity": error_response.severity,
        "correlation_id": error_response.correlation_id,
        "path": error_response.path,
        "method": error_response.method,
    }

    if original_exception:
        log_data["exception_type"] = type(original_exception).__name__

    # Log based on severity
    if error_response.severity == ErrorSeverity.CRITICAL:
        logger.critical(
            f"Critical error occurred: {log_data}",
            exc_info=original_exception if should_include_stack_trace() else None,
        )
    elif error_response.severity == ErrorSeverity.HIGH:
        logger.error(
            f"High severity error occurred: {log_data}",
            exc_info=original_exception if should_include_stack_trace() else None,
        )
    elif error_response.severity == ErrorSeverity.MEDIUM:
        logger.warning("Medium severity error occurred: %s", log_data)
    else:
        logger.info("Low severity error occurred: %s", log_data)


def get_error_status_code(error_category: ErrorCategory) -> int:
    """Get HTTP status code based on error category."""
    status_code_map = {
        ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,
        ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,
        ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,
        ErrorCategory.BUSINESS_LOGIC: status.HTTP_404_NOT_FOUND,  # Changed from 400 to 404 for resource not found
        ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.EXTERNAL_SERVICE: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
    }
    return status_code_map.get(error_category, status.HTTP_500_INTERNAL_SERVER_ERROR)


class ErrorSanitizer:
    """
    Advanced error message sanitizer with pre-compiled regex patterns
    for secure and efficient sensitive data removal.
    """

    SENSITIVE_PATTERNS = [
        re.compile(r"(?i)password\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)token\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)(?:api_)?key\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)credentials?\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)connection_string\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)secret\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)auth\s*[:=]\s*[^\s,;]+"),
        re.compile(r"(?i)bearer\s+[^\s,;]+"),
        re.compile(r"(?i)authorization\s*:\s*[^\s,;]+"),
    ]

    @classmethod
    def sanitize(cls, message: str) -> str:
        """
        Sanitize error messages to prevent sensitive information disclosure.

        Args:
            message: The error message to sanitize

        Returns:
            Sanitized message with sensitive data redacted
        """
        if not message:
            return "An error occurred"

        sanitized = str(message)  # Ensure string type
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        return sanitized


@lru_cache(maxsize=128)
def get_sensitive_patterns() -> frozenset[Pattern]:
    """
    Get cached sensitive patterns for efficient reuse.

    Returns memoized set of compiled regex patterns for sensitive data detection.
    """
    return frozenset(
        [
            re.compile(pattern)
            for pattern in [
                r"(?i)password\s*[:=]\s*[^\s,;]+",
                r"(?i)token\s*[:=]\s*[^\s,;]+",
                r"(?i)(?:api_)?key\s*[:=]\s*[^\s,;]+",
                r"(?i)credentials?\s*[:=]\s*[^\s,;]+",
                r"(?i)connection_string\s*[:=]\s*[^\s,;]+",
                r"(?i)secret\s*[:=]\s*[^\s,;]+",
                r"(?i)auth\s*[:=]\s*[^\s,;]+",
                r"(?i)bearer\s+[^\s,;]+",
                r"(?i)authorization\s*:\s*[^\s,;]+",
            ]
        ]
    )


def sanitize_error_message(message: str) -> str:
    """
    Sanitize error messages to prevent sensitive information disclosure.

    This function is maintained for backward compatibility.
    Use ErrorSanitizer.sanitize() for new code.
    """
    return ErrorSanitizer.sanitize(message)


def create_json_response_from_error(error_response: BaseErrorResponse) -> JSONResponse:
    """Create FastAPI JSONResponse from standardized error response."""
    status_code = get_error_status_code(error_response.category)
    # Use model_dump_json to properly serialize with JSON encoders
    content = error_response.model_dump(exclude_none=True, mode="json")

    return JSONResponse(status_code=status_code, content=content)


def register_exception_handlers(app):
    """Register standardized exception handlers for the FastAPI application."""
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from resync.core.exceptions import ResyncException as BaseResyncException
    from resync.core.exceptions import (
        ResyncException as EnhancedResyncException,
    )

    # Register handler for validation errors
    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(request: Request, exc: RequestValidationError):
        correlation_id = generate_correlation_id()
        error_response = (
            ErrorResponseBuilder()
            .with_correlation_id(correlation_id)
            .with_request_context(request)
            .build_validation_error(extract_validation_errors(exc))
        )

        log_error_response(error_response, exc)

        return create_json_response_from_error(error_response)

    # Register handler for standard HTTP exceptions
    @app.exception_handler(StarletteHTTPException)
    def http_exception_handler(request: Request, exc: StarletteHTTPException):
        correlation_id = generate_correlation_id()
        builder = (
            ErrorResponseBuilder().with_correlation_id(correlation_id).with_request_context(request)
        )

        # Map HTTP status codes to appropriate error types
        if exc.status_code == 404:
            error_response = builder.build_business_logic_error(
                "resource_not_found", resource="Resource"
            )
        elif exc.status_code == 401:
            error_response = builder.build_authentication_error("unauthorized")
        elif exc.status_code == 403:
            error_response = builder.build_authorization_error("forbidden")
        elif exc.status_code == 429:
            error_response = builder.build_rate_limit_error(
                limit=getattr(request.state, "rate_limit", {}).get("limit", 0)
            )
        else:
            error_response = builder.build_system_error("internal_server_error")

        error_response.error_code = f"HTTP_{exc.status_code}"
        error_response.message = exc.detail if exc.detail else f"HTTP Error {exc.status_code}"

        log_error_response(error_response)

        return create_json_response_from_error(error_response)

    # Register handler for enhanced Resync custom exceptions first
    @app.exception_handler(EnhancedResyncException)
    def enhanced_resync_exception_handler(request: Request, exc: EnhancedResyncException):
        correlation_id = generate_correlation_id()
        error_response = create_error_response_from_exception(exc, request, correlation_id)

        log_error_response(error_response, exc)

        return create_json_response_from_error(error_response)

    # Register handler for base Resync custom exceptions
    @app.exception_handler(BaseResyncException)
    def base_resync_exception_handler(request: Request, exc: BaseResyncException):
        correlation_id = generate_correlation_id()
        error_response = create_error_response_from_exception(exc, request, correlation_id)

        log_error_response(error_response, exc)

        return create_json_response_from_error(error_response)

    return app


def create_error_response_from_exception(
    exception: Exception,
    request: Request | None = None,
    correlation_id: str | None = None,
) -> BaseErrorResponse:
    """Create standardized error response from any exception with security considerations."""
    # Lazy import to avoid circular dependency
    from .error_factories import ErrorFactory

    return ErrorFactory.create_error_response(exception, request, correlation_id)
