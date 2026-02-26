"""Modelos de resposta de erro padronizados (RFC 7807 - Problem Details)."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

def _utc_now_iso8601() -> str:
    """Generate UTC timestamp in ISO 8601 format with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# NOVO MODELO ADICIONADO
class HealthCheckResponse(BaseModel):
    """Resposta de health check padronizada.

    Attributes:
        status: Status da saúde (UP/DOWN)
        timestamp: Timestamp do health check
        version: Versão da aplicação
        environment: Ambiente de execução
    """

    status: str = Field(..., json_schema_extra={"example": "UP"})
    timestamp: str = Field(default_factory=_utc_now_iso8601)
    version: str | None = Field(None, json_schema_extra={"example": "1.0.0"})
    environment: str | None = Field(None, json_schema_extra={"example": "production"})

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "UP",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "environment": "production",
            }
        }
    )

# RFC 7807 Problem Details models
class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response model."""

    type: str = Field(..., description="URI reference identifying the problem type")
    title: str = Field(..., description="Human-readable summary of the problem")
    detail: str | None = Field(None, description="Human-readable explanation")
    instance: str | None = Field(
        None, description="URI reference identifying specific occurrence"
    )
    status: int = Field(..., description="HTTP status code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "https://api.example.com/errors/validation-error",
                "title": "Validation Error",
                "detail": "The request contains invalid data",
                "instance": "/api/v1/users/123",
                "status": 400,
            }
        }
    )

class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    code: str | None = Field(None, description="Error code")
    value: Any | None = Field(None, description="Invalid value provided")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field": "email",
                "message": "Invalid email format",
                "code": "invalid_email",
                "value": "invalid-email",
            }
        }
    )

class ValidationProblemDetail(ProblemDetail):
    """Problem Details with validation errors."""

    errors: list[ValidationErrorDetail] = Field(
        default_factory=list, description="List of validation errors"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "https://api.example.com/errors/validation-error",
                "title": "Validation Failed",
                "detail": "Multiple validation errors occurred",
                "status": 400,
                "errors": [
                    {
                        "field": "email",
                        "message": "Invalid email format",
                        "code": "invalid_email",
                    }
                ],
            }
        }
    )

class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = Field(True, description="Success indicator")
    message: str = Field(..., description="Success message")
    data: Any | None = Field(None, description="Response data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"id": 123, "name": "example"},
            }
        }
    )

class PaginatedResponse(BaseModel):
    """Paginated response with metadata."""

    items: list[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}],
                "total": 25,
                "page": 1,
                "page_size": 10,
                "total_pages": 3,
            }
        }
    )

# Factory functions
def create_problem_detail(
    type_uri: str,
    title: str,
    status: int,
    detail: str | None = None,
    instance: str | None = None,
) -> ProblemDetail:
    """Create a ProblemDetail instance."""
    return ProblemDetail(
        type=type_uri, title=title, status=status, detail=detail, instance=instance
    )

def create_validation_problem_detail(
    title: str,
    detail: str,
    errors: list[ValidationErrorDetail],
    status: int = 400,
    instance: str | None = None,
) -> ValidationProblemDetail:
    """Create a ValidationProblemDetail instance."""
    return ValidationProblemDetail(
        type="https://api.example.com/errors/validation-error",
        title=title,
        detail=detail,
        status=status,
        instance=instance,
        errors=errors,
    )

def create_success_response(message: str, data: Any | None = None) -> SuccessResponse:
    """Create a SuccessResponse instance."""
    return SuccessResponse(success=True, message=message, data=data)

def create_paginated_response(
    items: list[Any],
    total: int,
    page: int,
    page_size: int,
    base_path: str | None = None,
    query_params: dict[str, Any] | None = None,
) -> PaginatedResponse:
    """Create a PaginatedResponse instance.

    Args:
        items: List of items for the current page
        total: Total number of items
        page: Current page number
        page_size: Number of items per page
        base_path: Optional base path for pagination links (unused, kept for compatibility)
        query_params: Optional query parameters for pagination links (unused, kept for compatibility)
    """
    total_pages = (total + page_size - 1) // page_size  # Ceiling division
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )

# Response helpers
def error_response(
    status_code: int, message: str, details: Any | None = None
) -> dict[str, Any]:
    """Create a standardized error response."""
    return {
        "success": False,
        "error": {"code": status_code, "message": message, "details": details},
        "timestamp": _utc_now_iso8601(),
    }

def success_response(message: str, data: Any | None = None) -> dict[str, Any]:
    """Create a standardized success response."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": _utc_now_iso8601(),
    }

def paginated_response(
    items: list[Any], total: int, page: int, page_size: int
) -> dict[str, Any]:
    """Create a standardized paginated response."""
    total_pages = (total + page_size - 1) // page_size
    return {
        "success": True,
        "data": {
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            },
        },
        "timestamp": _utc_now_iso8601(),
    }
