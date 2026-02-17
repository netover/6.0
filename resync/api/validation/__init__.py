"""Enhanced input validation using pydantic models with strict validation rules."""

from .agents import (
    AgentBulkActionRequest,
    AgentConfig,
    AgentCreateRequest,
    AgentUpdateRequest,
)
from .auth import (
    APIKeyRequest,
    LoginRequest,
    LogoutRequest,
    MFARequest,
    PasswordChangeRequest,
    TokenRefreshRequest,
    TokenRequest,
    UserRegistrationRequest,
)
from .chat import (
    ChatExportRequest,
    ChatHistoryRequest,
    ChatMessage,
    ChatSession,
    MessageReaction,
    WebSocketMessage,
)
from .common import (
    BaseValidatedModel,
    NumericConstraints,
    StringConstraints,
    ValidationErrorResponse,
    ValidationPatterns,
    ValidationSeverity,
    sanitize_input,
    validate_enum_value,
    validate_numeric_range,
    validate_pattern,
    validate_string_length,
)
from .config import (
    AgentValidationConfig,
    ChatValidationConfig,
    RateLimitConfig,
    SanitizationLevel,
    SecurityValidationConfig,
    ValidationConfigModel,
    ValidationMode,
    ValidationSettings,
    get_validation_settings,
    set_validation_settings,
)
from .files import (
    FileChunkUploadRequest,
    FileInfo,
    FileProcessingRequest,
    FileType,
    FileUpdateRequest,
    FileUploadRequest,
    ProcessingStatus,
    RAGUploadRequest,
)
from .middleware import (
    ValidationConfig,
    ValidationMiddleware,
    create_validation_middleware,
    validate_json_body,
    validate_query_params,
)
from .monitoring import (
    AlertQueryParams,
    AlertRequest,
    AlertSeverity,
    AlertStatus,
    CustomMetricRequest,
    HealthCheckRequest,
    HealthStatus,
    LogQueryParams,
    MetricType,
    PerformanceTestRequest,
    SystemMetricRequest,
)
from .query_params import (
    AgentQueryParams,
    AuditQueryParams,
    CombinedQueryParams,
    DateRangeParams,
    FileQueryParams,
    FilterOperator,
    FilterParams,
    PaginationParams,
    SearchParams,
    SortOrder,
    SortParams,
    SystemQueryParams,
)

__all__ = [
    # Common validation utilities
    "BaseValidatedModel",
    "ValidationPatterns",
    "StringConstraints",
    "NumericConstraints",
    "ValidationSeverity",
    "ValidationErrorResponse",
    "sanitize_input",
    "validate_string_length",
    "validate_numeric_range",
    "validate_pattern",
    "validate_enum_value",
    # Agent validation models
    "AgentConfig",
    "AgentCreateRequest",
    "AgentUpdateRequest",
    "AgentQueryParams",
    "AgentBulkActionRequest",
    # Authentication validation models
    "LoginRequest",
    "TokenRequest",
    "PasswordChangeRequest",
    "UserRegistrationRequest",
    "APIKeyRequest",
    "MFARequest",
    "TokenRefreshRequest",
    "LogoutRequest",
    # Chat validation models
    "ChatMessage",
    "WebSocketMessage",
    "ChatSession",
    "ChatHistoryRequest",
    "MessageReaction",
    "ChatExportRequest",
    # Query parameter validation models
    "PaginationParams",
    "SearchParams",
    "FilterParams",
    "SortParams",
    "DateRangeParams",
    "AgentQueryParams",
    "SystemQueryParams",
    "AuditQueryParams",
    "FileQueryParams",
    "CombinedQueryParams",
    "SortOrder",
    "FilterOperator",
    # File upload validation models
    "FileUploadRequest",
    "FileChunkUploadRequest",
    "FileUpdateRequest",
    "FileProcessingRequest",
    "RAGUploadRequest",
    "FileInfo",
    "FileType",
    "ProcessingStatus",
    # Monitoring validation models
    "SystemMetricRequest",
    "CustomMetricRequest",
    "AlertRequest",
    "AlertQueryParams",
    "HealthCheckRequest",
    "LogQueryParams",
    "PerformanceTestRequest",
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "HealthStatus",
    # Middleware
    "ValidationMiddleware",
    "ValidationConfig",
    "create_validation_middleware",
    "validate_json_body",
    "validate_query_params",
    # Configuration
    "ValidationMode",
    "SanitizationLevel",
    "ValidationConfigModel",
    "AgentValidationConfig",
    "ChatValidationConfig",
    "SecurityValidationConfig",
    "RateLimitConfig",
    "ValidationSettings",
    "get_validation_settings",
    "set_validation_settings",
]

# Version information
__version__ = "1.0.0"
__author__ = "Resync Team"
__description__ = "Enhanced input validation using pydantic models with strict validation rules"
