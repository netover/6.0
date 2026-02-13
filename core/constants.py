"""
Core constants and configuration defaults for the Resync application.

This module centralizes all magic numbers, thresholds, and default values
to improve maintainability and eliminate code duplication.
"""

from enum import Enum

# ============================================================================
# IA AUDITOR THRESHOLDS
# ============================================================================

AUDIT_DELETION_CONFIDENCE_THRESHOLD = 0.85
AUDIT_FLAGGING_CONFIDENCE_THRESHOLD = 0.6
AUDIT_HIGH_RATING_THRESHOLD = 3
RECENT_MEMORIES_FETCH_LIMIT = 100


# ============================================================================
# HTTP CLIENT DEFAULTS
# ============================================================================

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 5.0
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

LLM_DEFAULT_MAX_TOKENS = 200
LLM_DEFAULT_TEMPERATURE = 0.1
LLM_DEFAULT_MAX_RETRIES = 3
LLM_DEFAULT_BACKOFF_BASE_DELAY = 1.0
LLM_DEFAULT_BACKOFF_MAX_DELAY = 30.0


# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

CACHE_DEFAULT_TTL_SECONDS = 3600
RESPONSE_CACHE_DEFAULT_TTL_SECONDS = 300


# ============================================================================
# AUDIT AND LOCKING
# ============================================================================

AUDIT_LOCK_DEFAULT_TIMEOUT = 30
AUDIT_LOCK_CLEANUP_MAX_AGE = 60


# ============================================================================
# ERROR MESSAGES
# ============================================================================


class ErrorMessages(Enum):
    """Standardized error messages for consistent user communication."""

    TIMEOUT = "Request timeout during {operation}"
    CONNECTION = "Connection error during {operation}"
    AUTH_REQUIRED = "Authentication required for this operation"
    UNAUTHORIZED = "Unauthorized access to {operation}"
    FORBIDDEN = "Access forbidden for this operation"
    NOT_FOUND = "Resource not found during {operation}"
    VALIDATION_ERROR = "Validation error during {operation}: {detail}"
    CONFLICT = "Conflict during {operation}: {detail}"
    INTERNAL_ERROR = "An error occurred during {operation}: {detail}"
    SERVICE_UNAVAILABLE = "Service unavailable during {operation}"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded for {operation}"
    QUOTA_EXCEEDED = "Quota exceeded for {operation}"


# ============================================================================
# LOG LEVEL MAPPINGS
# ============================================================================

LOG_LEVEL_MAP = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


# ============================================================================
# API RESPONSE CONSTANTS
# ============================================================================

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 30


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

MIN_MESSAGE_LENGTH = 1
MAX_MESSAGE_LENGTH = 1000
MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 2000


# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

SLOW_QUERY_THRESHOLD_MS = 1000
SLOW_EXTERNAL_CALL_THRESHOLD_MS = 5000
CONNECTION_POOL_WARNING_THRESHOLD = 80  # percentage


# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

MAX_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_DURATION_MINUTES = 15
SESSION_TIMEOUT_HOURS = 24
PASSWORD_MIN_LENGTH = 8
JWT_DEFAULT_EXPIRATION_HOURS = 1


# ============================================================================
# WEBSOCKET CONSTANTS
# ============================================================================

WEBSOCKET_PING_INTERVAL_SECONDS = 30
WEBSOCKET_PING_TIMEOUT_SECONDS = 10
WEBSOCKET_MAX_CONNECTIONS_PER_CLIENT = 5


# ============================================================================
# METRICS AND MONITORING
# ============================================================================

HEALTH_CHECK_INTERVAL_SECONDS = 30
METRICS_RETENTION_DAYS = 30
ALERT_ESCALATION_TIMEOUT_MINUTES = 5


# ============================================================================
# TWS INTEGRATION CONSTANTS
# ============================================================================

TWS_DEFAULT_PORT = 31111
TWS_CONNECTION_RETRY_ATTEMPTS = 3
TWS_CONNECTION_RETRY_DELAY_SECONDS = 2.0
TWS_COMMAND_TIMEOUT_SECONDS = 60
TWS_HEALTH_CHECK_TIMEOUT_SECONDS = 10


# ============================================================================
# KNOWLEDGE GRAPH CONSTANTS
# ============================================================================

KNOWLEDGE_GRAPH_DEFAULT_LIMIT = 100
KNOWLEDGE_GRAPH_MAX_LIMIT = 1000
KNOWLEDGE_GRAPH_SIMILARITY_THRESHOLD = 0.8
KNOWLEDGE_GRAPH_EMBEDDING_DIMENSIONS = 384  # Common for sentence transformers


# ============================================================================
# FILE UPLOAD CONSTANTS
# ============================================================================

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".json", ".yaml", ".yml"}
UPLOAD_CHUNK_SIZE = 8192  # 8KB chunks


# ============================================================================
# RATE LIMITING CONSTANTS
# ============================================================================

DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
BURST_RATE_LIMIT_REQUESTS = 20
BURST_RATE_LIMIT_WINDOW_SECONDS = 10
