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
from .connector import Connector
from .orchestration import (
    OrchestrationCallback,
    OrchestrationConfig,
    OrchestrationExecution,
    OrchestrationStepRun,
)
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
    get_all_models as _get_all_models_stores,
)

def get_all_models():
    """Return all model classes including new ones."""
    models = _get_all_models_stores()
    models.extend([
        Connector,
        OrchestrationConfig,
        OrchestrationExecution,
        OrchestrationStepRun,
        OrchestrationCallback,
    ])
    return models

AdminUser = User

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
    # Connector Models
    "Connector",
    # Orchestration Models
    "OrchestrationConfig",
    "OrchestrationExecution",
    "OrchestrationStepRun",
    "OrchestrationCallback",
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
