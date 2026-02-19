"""
Database Package.

Provides PostgreSQL database connectivity, models, and repositories.
All data stores are now consolidated in PostgreSQL.

Usage:
    from resync.core.database import get_session, TWSStore, ContextStore

    async with get_session() as session:
        store = TWSStore()
        await store.update_job_status(job)
"""

from .config import (
    DatabaseConfig,
    DatabaseDriver,
    get_database_config,
)
from .engine import (
    get_db_session,
    get_engine,
    get_session,
    get_session_factory,
)
from .models import (
    ActiveLearningCandidate,
    AuditEntry,
    AuditQueueItem,
    Base,
    ContentType,
    ContextContent,
    Conversation,
    EventSeverity,
    Feedback,
    # Enums
    JobStatusEnum,
    LearningThreshold,
    MetricAggregation,
    MetricDataPoint,
    SessionHistory,
    TWSEvent,
    TWSJobStatus,
    TWSPattern,
    TWSProblemSolution,
    # All models
    TWSSnapshot,
    TWSWorkstationStatus,
    UserProfile,
    get_all_models,
)
from .repositories import (
    # Audit
    AuditEntryRepository,
    AuditQueueRepository,
    # Base
    BaseRepository,
    # Context
    ContextStore,
    # Learning
    FeedbackStore,
    # TWS
    JobStatus,
    # Metrics
    MetricsStore,
    PatternMatch,
    TimestampedRepository,
    TWSStore,
    # Analytics
    UserBehaviorStore,
)
from .schema import (
    check_database_connection,
    create_schemas,
    create_tables,
    initialize_database,
)

# Alias for backward compatibility
get_db = get_db_session
get_async_session = get_session  # Alias for workflows

__all__ = [
    # Config
    "DatabaseConfig",
    "DatabaseDriver",
    "get_database_config",
    # Engine
    "get_engine",
    "get_session",
    "get_db_session",
    "get_db",  # Alias for get_db_session
    "get_async_session",  # Alias for get_session (workflows)
    "get_session_factory",
    # Schema
    "initialize_database",
    "create_schemas",
    "create_tables",
    "check_database_connection",
    # Base
    "Base",
    "BaseRepository",
    "TimestampedRepository",
    # Enums
    "JobStatusEnum",
    "EventSeverity",
    "ContentType",
    # Stores (main interfaces)
    "TWSStore",
    "ContextStore",
    "UserBehaviorStore",
    "FeedbackStore",
    "MetricsStore",
    # Data classes
    "JobStatus",
    "PatternMatch",
    # Audit repositories
    "AuditEntryRepository",
    "AuditQueueRepository",
    # All models
    "TWSSnapshot",
    "TWSJobStatus",
    "TWSWorkstationStatus",
    "TWSEvent",
    "TWSPattern",
    "TWSProblemSolution",
    "Conversation",
    "ContextContent",
    "AuditEntry",
    "AuditQueueItem",
    "UserProfile",
    "SessionHistory",
    "Feedback",
    "LearningThreshold",
    "ActiveLearningCandidate",
    "MetricDataPoint",
    "MetricAggregation",
    "get_all_models",
]
