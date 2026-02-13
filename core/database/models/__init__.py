"""
Database Models Package.

Contains all SQLAlchemy models for PostgreSQL storage.
"""

# Auth Models (v5.4.7 consolidation)
from .auth import (
    AuditLog,
    User,
    UserRole,
)

# AdminUser is an alias for User (for backward compatibility with admin repositories)
AdminUser = User
from .stores import (
    ActiveLearningCandidate,
    # Audit Models
    AuditEntry,
    AuditQueueItem,
    # Base
    Base,
    ContentType,
    ContextContent,
    # Context Models
    Conversation,
    EventSeverity,
    # Learning Models
    Feedback,
    # Enums
    JobStatusEnum,
    LearningThreshold,
    MetricAggregation,
    # Metrics Models
    MetricDataPoint,
    SessionHistory,
    TWSEvent,
    TWSJobStatus,
    TWSPattern,
    TWSProblemSolution,
    # TWS Models
    TWSSnapshot,
    TWSWorkstationStatus,
    # Analytics Models
    UserProfile,
    # Helper
    get_all_models,
)

__all__ = [
    # Base
    "Base",
    # Enums
    "JobStatusEnum",
    "EventSeverity",
    "ContentType",
    "UserRole",
    # TWS Models
    "TWSSnapshot",
    "TWSJobStatus",
    "TWSWorkstationStatus",
    "TWSEvent",
    "TWSPattern",
    "TWSProblemSolution",
    # Context Models
    "Conversation",
    "ContextContent",
    # Audit Models
    "AuditEntry",
    "AuditQueueItem",
    "AuditLog",
    # Auth Models
    "User",
    "AdminUser",
    # Analytics Models
    "UserProfile",
    "SessionHistory",
    # Learning Models
    "Feedback",
    "LearningThreshold",
    "ActiveLearningCandidate",
    # Metrics Models
    "MetricDataPoint",
    "MetricAggregation",
    # Helper
    "get_all_models",
]
