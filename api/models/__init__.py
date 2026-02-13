"""API Models Package.

This package contains all Pydantic models used in the Resync API,
organized by functionality for better maintainability.
"""

# Base models and common classes
# Agent management
from .agents import (
    AgentConfig,
    AgentType,
)

# Authentication and user management
from .auth import (
    LoginRequest,
    PasswordChangeRequest,
    Token,
    TokenData,
    UserRegistrationRequest,
)
from .base import (
    BaseModelWithTime,
    PaginationRequest,
    PaginationResponse,
)

# Health and system monitoring
from .health import (
    SystemMetric,
)

# RAG (Retrieval-Augmented Generation)
from .rag import (
    RAGFileCreate,
    RAGFileDetail,
    RAGFileMetaData,
)

# Response models (standardized API responses)
from .responses import (
    HealthCheckResponse,
    PaginatedResponse,
    ProblemDetail,
    SuccessResponse,
    ValidationErrorDetail,
    ValidationProblemDetail,
    create_paginated_response,
    create_problem_detail,
    create_success_response,
    create_validation_problem_detail,
    error_response,
    paginated_response,
    success_response,
)

# Database models (re-exports for backward compatibility)
from resync.core.database.models import AdminUser, User, UserRole

__all__ = [
    # Base models
    "BaseModelWithTime",
    "PaginationRequest",
    "PaginationResponse",
    # Authentication models
    "LoginRequest",
    "UserRegistrationRequest",
    "PasswordChangeRequest",
    "Token",
    "TokenData",
    # RAG models
    "RAGFileMetaData",
    "RAGFileCreate",
    "RAGFileDetail",
    # Agent models
    "AgentConfig",
    "AgentType",
    # Health models
    "SystemMetric",
    # Response models
    "HealthCheckResponse",
    "PaginatedResponse",
    "ProblemDetail",
    "SuccessResponse",
    "ValidationErrorDetail",
    "ValidationProblemDetail",
    "create_paginated_response",
    "create_problem_detail",
    "create_success_response",
    "create_validation_problem_detail",
    "error_response",
    "paginated_response",
    "success_response",
    # Database models (re-exports)
    "AdminUser",
    "User",
    "UserRole",
]
