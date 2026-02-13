"""
Error Response Factories using Factory Pattern.

This module implements the Factory pattern for creating standardized error responses
based on different exception types, making the code more modular, testable, and maintainable.
"""

from fastapi import Request

from resync.core.exceptions import (
    DatabaseError,
    LLMError,
    NotFoundError,
    TWSConnectionError,
)
from resync.core.exceptions import (
    DatabaseError as EnhancedDatabaseError,
)
from resync.core.exceptions import (
    LLMError as EnhancedLLMError,
)
from resync.core.exceptions import (
    NotFoundError as EnhancedNotFoundError,
)
from resync.core.exceptions import (
    ResyncException as BaseResyncException,
)
from resync.core.exceptions import (
    ResyncException as EnhancedResyncException,
)
from resync.core.exceptions import (
    TWSConnectionError as EnhancedTWSConnectionError,
)
from resync.models.error_models import (
    BaseErrorResponse,
)


# Lazy imports to avoid circular dependency
def _get_error_response_builder():
    """Lazy import to avoid circular dependency."""
    from resync.core.utils.error_utils import ErrorResponseBuilder

    return ErrorResponseBuilder


def _get_generate_correlation_id():
    """Lazy import to avoid circular dependency."""
    from resync.core.utils.error_utils import generate_correlation_id

    return generate_correlation_id


def _get_should_include_stack_trace():
    """Lazy import to avoid circular dependency."""
    from resync.core.utils.error_utils import should_include_stack_trace

    return should_include_stack_trace


class ErrorFactory:
    """Base factory class for creating error responses."""

    @staticmethod
    def create_error_response(
        exception: Exception,
        request: Request | None = None,
        correlation_id: str | None = None,
    ) -> BaseErrorResponse:
        """Factory method to create appropriate error response based on exception type."""
        error_response_builder_cls = _get_error_response_builder()
        builder = error_response_builder_cls()

        # Set correlation ID
        if correlation_id:
            builder.with_correlation_id(correlation_id)
        else:
            generate_correlation_id = _get_generate_correlation_id()
            builder.with_correlation_id(generate_correlation_id())

        # Set request context
        if request:
            builder.with_request_context(request)

        # Enhanced security: only include stack traces in non-production environments
        is_production = (
            getattr(request.state, "app_env", "development") == "production"
            if request
            else "production"
        )
        should_include_stack_trace_func = _get_should_include_stack_trace()
        include_stack_trace = should_include_stack_trace_func() and not is_production

        # In production, sanitize error messages to prevent information disclosure
        if is_production:
            builder.with_stack_trace(include_stack_trace and is_production is False)
        else:
            builder.with_stack_trace(include_stack_trace)

        # Handle different exception types
        if isinstance(exception, EnhancedResyncException):
            return EnhancedResyncExceptionFactory.create_response(builder, exception, is_production)
        if isinstance(exception, (TWSConnectionError, EnhancedTWSConnectionError)):
            return TWSConnectionExceptionFactory.create_response(builder, exception, is_production)
        if isinstance(exception, (LLMError, EnhancedLLMError)):
            return LLMExceptionFactory.create_response(builder, exception, is_production)
        if isinstance(exception, (DatabaseError, EnhancedDatabaseError)):
            return DatabaseExceptionFactory.create_response(builder, exception, is_production)
        if isinstance(exception, (NotFoundError, EnhancedNotFoundError)):
            return NotFoundExceptionHandler.create_response(builder, exception, is_production)
        if isinstance(exception, BaseResyncException):
            return BaseResyncExceptionFactory.create_response(builder, exception, is_production)
        return UnknownExceptionFactory.create_response(builder, exception, is_production)


class EnhancedResyncExceptionFactory:
    """Factory for handling enhanced Resync exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: EnhancedResyncException,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for enhanced Resync exceptions."""
        # Use the enhanced exception's information to create a more detailed response
        message = exception.message
        user_friendly_message = exception.user_friendly_message
        details = exception.details.copy()  # Copy to avoid modifying original

        # Sanitize details in production to prevent sensitive data leakage
        if is_production:
            # Remove potentially sensitive information
            details.pop("original_exception", None)
            if "details" in details and isinstance(details["details"], dict):
                # Further sanitize nested details
                details["details"] = {
                    k: v
                    for k, v in details["details"].items()
                    if k not in ["password", "token", "credentials", "key"]
                }

        # Map the enhanced exception to the appropriate response type based on category
        if exception.error_category == "VALIDATION":
            return builder.build_validation_error(
                [
                    {
                        "loc": ["validation"],
                        "msg": sanitize_error_message(message),
                        "type": "value_error",
                    }
                ],
                sanitize_error_message(user_friendly_message),
            )
        if exception.error_category == "BUSINESS_LOGIC":
            return _handle_business_logic_exception(
                builder, exception, message, user_friendly_message, details, is_production
            )
        if exception.error_category == "EXTERNAL_SERVICE":
            service_name = (
                details.get("service", "External Service") if details else "External Service"
            )
            return builder.build_external_service_error(
                service_name,
                user_friendly_message=sanitize_error_message(user_friendly_message),
                details=details,
            )
        if exception.error_category == "SYSTEM":
            return builder.build_system_error(
                "internal_server_error",
                user_friendly_message=sanitize_error_message(user_friendly_message),
                details=details,
                exception=exception if not is_production else None,
            )
        # For other enhanced exceptions
        return builder.build_system_error(
            "internal_server_error",
            exception=exception if not is_production else None,
            user_friendly_message=sanitize_error_message(user_friendly_message),
            details=details,
        )


class TWSConnectionExceptionFactory:
    """Factory for handling TWS connection exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: Exception,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for TWS connection exceptions."""
        return builder.build_external_service_error("TWS", "service_error")


class LLMExceptionFactory:
    """Factory for handling LLM exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: Exception,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for LLM exceptions."""
        return builder.build_external_service_error("LLM", "service_error")


class DatabaseExceptionFactory:
    """Factory for handling database exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: Exception,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for database exceptions."""
        return builder.build_system_error("database_error")


class NotFoundExceptionHandler:
    """Factory for handling not found exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: Exception,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for not found exceptions."""
        return builder.build_business_logic_error("resource_not_found", resource="Resource")


class BaseResyncExceptionFactory:
    """Factory for handling base Resync exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: BaseResyncException,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for base Resync exceptions."""
        return builder.build_system_error(
            "internal_server_error", exception=exception if not is_production else None
        )


class UnknownExceptionFactory:
    """Factory for handling unknown exceptions."""

    @staticmethod
    def create_response(
        builder,  # ErrorResponseBuilder
        exception: Exception,
        is_production: bool,
    ) -> BaseErrorResponse:
        """Create response for unknown exceptions."""
        return builder.build_system_error(
            "internal_server_error", exception=exception if not is_production else None
        )


def _handle_business_logic_exception(
    builder,  # ErrorResponseBuilder
    exception: Exception,
    message: str,
    user_friendly_message: str,
    details: dict,
    is_production: bool,
) -> BaseErrorResponse:
    """Handle business logic exceptions."""
    from resync.core.exceptions import NotFoundError as EnhancedNotFoundError

    if isinstance(exception, EnhancedNotFoundError):
        return builder.build_business_logic_error("resource_not_found", resource="Resource")
    return builder.build_business_logic_error(
        "invalid_operation",
        user_friendly_message=sanitize_error_message(user_friendly_message),
        details=details,
    )


def sanitize_error_message(message: str) -> str:
    """
    Sanitize error messages to prevent sensitive information disclosure.

    This function is maintained for backward compatibility.
    Use ErrorSanitizer.sanitize() for new code.
    """
    from resync.core.utils.error_utils import ErrorSanitizer

    return ErrorSanitizer.sanitize(message)
