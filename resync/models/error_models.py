"""Standardized error response models for the FastAPI application."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ErrorCategory(str, Enum):
    """Error category enumeration for consistent error classification."""

    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"
    SYSTEM = "SYSTEM"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    RATE_LIMIT = "RATE_LIMIT"


class ErrorSeverity(str, Enum):
    """Error severity levels for prioritization and handling."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BaseErrorResponse(BaseModel):
    """Base error response model with common fields."""

    error_code: str = Field(..., description="Unique error code for identification")
    message: str = Field(..., description="Technical error message")
    correlation_id: str = Field(..., description="Correlation ID for tracing")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of error occurrence"
    )
    severity: ErrorSeverity = Field(ErrorSeverity.MEDIUM, description="Error severity level")
    category: ErrorCategory = Field(..., description="Error category")
    path: str | None = Field(None, description="Request path that caused the error")
    method: str | None = Field(None, description="HTTP method of the request that caused the error")
    user_friendly_message: str | None = Field(None, description="User-friendly error message")
    troubleshooting_hints: list[str] | None = Field(None, description="Troubleshooting suggestions")
    stack_trace: str | None = Field(
        None, description="Stack trace for debugging (production disabled)"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(None, description="Value that caused the validation error")
    location: str | None = Field(
        None, description="Location of the field (body, query, path, header)"
    )


class ValidationErrorResponse(BaseErrorResponse):
    """Error response model for validation errors with detailed field information."""

    category: ErrorCategory = ErrorCategory.VALIDATION
    details: list[ValidationErrorDetail] = Field(
        ..., description="List of validation error details"
    )

    @classmethod
    def from_pydantic_errors(
        cls,
        errors: list[dict[str, Any]],
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "ValidationErrorResponse":
        """Create validation error response from Pydantic validation errors."""
        details = []
        for error in errors:
            field = ".".join(str(loc) for loc in error.get("loc", []))
            details.append(
                ValidationErrorDetail(
                    field=field,
                    message=error.get("msg", "Validation failed"),
                    value=error.get("input"),
                    location="body",  # Default location for most validation errors
                )
            )

        # Build contextual message with request information
        context_message = "Validation failed"
        if path and method:
            context_message = f"Validation failed for {method} {path}"

        return cls(
            error_code="VALIDATION_ERROR",
            message=context_message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.VALIDATION,
            details=details,
            path=path,
            method=method,
            severity=ErrorSeverity.LOW,
            user_friendly_message="Please check your input and try again.",
            troubleshooting_hints=["Check field requirements", "Verify data types"],
            stack_trace=None,
        )


class AuthenticationErrorResponse(BaseErrorResponse):
    """Error response model for authentication errors."""

    category: ErrorCategory = ErrorCategory.AUTHENTICATION

    @classmethod
    def unauthorized(
        cls,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "AuthenticationErrorResponse":
        """Create unauthorized error response."""
        # Build contextual message with request information
        message = "Authentication required"
        if path and method:
            message = f"Authentication required for {method} {path}"

        return cls(
            error_code="UNAUTHORIZED",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.AUTHENTICATION,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="Please authenticate to access this resource.",
            troubleshooting_hints=[
                "Check your credentials",
                "Ensure your session hasn't expired",
            ],
            stack_trace=None,
        )

    @classmethod
    def invalid_credentials(
        cls,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "AuthenticationErrorResponse":
        """Create invalid credentials error response."""
        # Build contextual message with request information
        message = "Invalid authentication credentials"
        if path and method:
            message = f"Invalid authentication credentials for {method} {path}"

        return cls(
            error_code="INVALID_CREDENTIALS",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.AUTHENTICATION,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="The provided credentials are invalid. Please check and try again.",
            troubleshooting_hints=[
                "Verify your username and password",
                "Check for typos",
            ],
            stack_trace=None,
        )


class AuthorizationErrorResponse(BaseErrorResponse):
    """Error response model for authorization errors."""

    category: ErrorCategory = ErrorCategory.AUTHORIZATION
    required_permissions: list[str] | None = Field(None, description="Required permissions")
    user_permissions: list[str] | None = Field(None, description="User's current permissions")

    @classmethod
    def forbidden(
        cls,
        resource: str,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "AuthorizationErrorResponse":
        """Create forbidden error response."""
        # Build contextual message with request information
        message = f"Access denied to {resource}"
        if path and method:
            message = f"Access denied to {resource} for {method} {path}"

        return cls(
            error_code="FORBIDDEN",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.AUTHORIZATION,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="You don't have permission to access this resource.",
            troubleshooting_hints=[
                "Contact your administrator for access",
                "Check if you have the required permissions",
            ],
            required_permissions=["access"],
            user_permissions=["basic"],
            stack_trace=None,
        )

    @classmethod
    def insufficient_permissions(
        cls,
        required: list[str] | None = None,
        user: list[str] | None = None,
        resource: str | None = None,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "AuthorizationErrorResponse":
        """Create insufficient permissions error response."""
        # Handle case where resource is provided instead of required/user permissions
        if resource is not None and required is None and user is None:
            required = [f"access_{resource}"]
            user = ["basic"]

        # Ensure we have valid required and user permissions
        required = required or ["unknown"]
        user = user or ["none"]

        # Build contextual message with request information
        message = f"Insufficient permissions. Required: {', '.join(required)}"
        if path and method:
            message = (
                f"Insufficient permissions for {method} {path}. Required: {', '.join(required)}"
            )

        return cls(
            error_code="INSUFFICIENT_PERMISSIONS",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.AUTHORIZATION,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="You don't have the required permissions to perform this action.",
            troubleshooting_hints=["Contact your administrator to request additional permissions"],
            required_permissions=required,
            user_permissions=user,
            stack_trace=None,
        )


class BusinessLogicErrorResponse(BaseErrorResponse):
    """Error response model for business logic errors."""

    category: ErrorCategory = ErrorCategory.BUSINESS_LOGIC
    business_rule: str | None = Field(None, description="Business rule that was violated")

    @classmethod
    def resource_not_found(
        cls,
        resource_type: str | None = None,
        resource_id: str | None = None,
        resource: str | None = None,
        identifier: str | None = None,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "BusinessLogicErrorResponse":
        """Create resource not found error response."""
        # Handle case where resource and identifier are provided instead of resource_type and resource_id
        if resource is not None:
            resource_type = resource_type or resource
        if identifier is not None:
            resource_id = resource_id or identifier

        # Build contextual message with request information
        message = f"{resource_type} with ID '{resource_id}' not found"
        if path and method:
            message = f"{resource_type} with ID '{resource_id}' not found for {method} {path}"

        return cls(
            error_code="RESOURCE_NOT_FOUND",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            severity=ErrorSeverity.MEDIUM,
            path=path,
            method=method,
            user_friendly_message=f"The requested {resource_type} could not be found.",
            troubleshooting_hints=[
                "Check the resource ID",
                "Verify the resource exists",
            ],
            business_rule="Resource existence check",
            stack_trace=None,
        )

    @classmethod
    def business_rule_violation(
        cls,
        rule: str,
        details: str,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "BusinessLogicErrorResponse":
        """Create business rule violation error response."""
        return cls(
            error_code="BUSINESS_RULE_VIOLATION",
            message=f"Business rule violation: {rule} - {details}",
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.BUSINESS_LOGIC,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="A business rule was violated. Please check your request.",
            troubleshooting_hints=[
                "Review the business rules",
                "Modify your request accordingly",
            ],
            business_rule=rule,
            stack_trace=None,
        )


class SystemErrorResponse(BaseErrorResponse):
    """Error response model for system/internal errors."""

    category: ErrorCategory = ErrorCategory.SYSTEM
    error_details: dict[str, Any] | None = Field(None, description="Additional error details")

    @classmethod
    def internal_server_error(
        cls,
        correlation_id: str | None = None,
        error_details: dict[str, Any] | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "SystemErrorResponse":
        """Create internal server error response."""
        # Build contextual message with request information
        message = "An unexpected error occurred"
        if path and method:
            message = f"An unexpected error occurred for {method} {path}"

        return cls(
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            severity=ErrorSeverity.HIGH,
            path=path,
            method=method,
            user_friendly_message="Something went wrong on our end. Please try again later.",
            troubleshooting_hints=[
                "Try again in a few minutes",
                "Contact support if the problem persists",
            ],
            error_details=error_details,
            stack_trace=None,
        )

    @classmethod
    def service_unavailable(
        cls,
        service: str,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "SystemErrorResponse":
        """Create service unavailable error response."""
        # Build contextual message with request information
        message = f"Service '{service}' is temporarily unavailable"
        if path and method:
            message = f"Service '{service}' is temporarily unavailable for {method} {path}"

        return cls(
            error_code="SERVICE_UNAVAILABLE",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            severity=ErrorSeverity.HIGH,
            path=path,
            method=method,
            user_friendly_message=f"The {service} service is currently unavailable. Please try again later.",
            troubleshooting_hints=[
                "Try again in a few minutes",
                "Check service status",
            ],
            error_details={"service": service},
            stack_trace=None,
        )


class ExternalServiceErrorResponse(BaseErrorResponse):
    """Error response model for external service errors."""

    category: ErrorCategory = ErrorCategory.EXTERNAL_SERVICE
    service_name: str | None = Field(None, description="Name of the external service")
    http_status: int | None = Field(None, description="HTTP status code from external service")

    @classmethod
    def external_service_error(
        cls,
        service_name: str,
        http_status: int,
        error_message: str,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "ExternalServiceErrorResponse":
        """Create external service error response."""
        return cls(
            error_code="EXTERNAL_SERVICE_ERROR",
            message=f"External service error from {service_name}: {error_message}",
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.EXTERNAL_SERVICE,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message=f"An error occurred while communicating with {service_name}. Please try again later.",
            troubleshooting_hints=[
                "Try again in a few minutes",
                "Check if {service_name} is operational",
            ],
            service_name=service_name,
            http_status=http_status,
            stack_trace=None,
        )


class RateLimitErrorResponse(BaseErrorResponse):
    """Error response model for rate limiting errors."""

    category: ErrorCategory = ErrorCategory.RATE_LIMIT
    limit: int | None = Field(None, description="Rate limit threshold")
    reset_time: datetime | None = Field(None, description="When the rate limit resets")

    @classmethod
    def rate_limit_exceeded(
        cls,
        limit: int,
        reset_time: datetime | None = None,
        window: str | None = None,
        correlation_id: str | None = None,
        path: str | None = None,
        method: str | None = None,
    ) -> "RateLimitErrorResponse":
        """Create rate limit exceeded error response."""
        # Handle case where window is provided instead of reset_time
        if window is not None and reset_time is None:
            # For simplicity, we'll create a future datetime based on the window
            from datetime import datetime, timezone, timedelta

            # This is a simplified implementation - in a real scenario, you'd calculate the actual reset time
            reset_time = datetime.now(timezone.utc) + timedelta(seconds=60)  # Default to 1 minute

        # Build contextual message with request information
        message = f"Rate limit exceeded. Limit: {limit} requests"
        if path and method:
            message = f"Rate limit exceeded for {method} {path}. Limit: {limit} requests"

        return cls(
            error_code="RATE_LIMIT_EXCEEDED",
            message=message,
            correlation_id=correlation_id or str(uuid4()),
            category=ErrorCategory.RATE_LIMIT,
            path=path,
            method=method,
            severity=ErrorSeverity.MEDIUM,
            user_friendly_message="You've made too many requests recently. Please wait before trying again.",
            troubleshooting_hints=[
                (
                    f"Wait until {reset_time.isoformat()} to retry"
                    if reset_time
                    else "Try again later"
                ),
                "Consider implementing request batching",
            ],
            limit=limit,
            reset_time=reset_time,
            stack_trace=None,
        )


# Export all error response models
__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "BaseErrorResponse",
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "AuthorizationErrorResponse",
    "BusinessLogicErrorResponse",
    "SystemErrorResponse",
    "ExternalServiceErrorResponse",
    "RateLimitErrorResponse",
]
