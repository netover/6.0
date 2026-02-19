"""
Database Repositories Package.

Provides repository pattern implementations for all data stores.
All repositories use PostgreSQL via SQLAlchemy async.
"""

from .base import BaseRepository, TimestampedRepository
from .stores import (
    ActiveLearningRepository,
    # Audit
    AuditEntryRepository,
    AuditQueueRepository,
    ContextContentRepository,
    ContextStore,
    # Context
    ConversationRepository,
    # Learning
    FeedbackRepository,
    FeedbackStore,
    LearningThresholdRepository,
    MetricAggregationRepository,
    # Metrics
    MetricDataPointRepository,
    MetricsStore,
    SessionHistoryRepository,
    UserBehaviorStore,
    # Analytics
    UserProfileRepository,
)
from .tws_repository import (
    # Data classes
    JobStatus,
    PatternMatch,
    TWSEventRepository,
    TWSJobStatusRepository,
    TWSPatternRepository,
    TWSProblemSolutionRepository,
    # Repositories
    TWSSnapshotRepository,
    # Unified Store
    TWSStore,
)

# Auth (v5.4.7 consolidation)
from .user_repository import UserRepository, UserService

__all__ = [
    # Base
    "BaseRepository",
    "TimestampedRepository",
    # Auth
    "UserRepository",
    "UserService",  # Backward compatibility alias
    # TWS
    "JobStatus",
    "PatternMatch",
    "TWSSnapshotRepository",
    "TWSJobStatusRepository",
    "TWSEventRepository",
    "TWSPatternRepository",
    "TWSProblemSolutionRepository",
    "TWSStore",
    # Context
    "ConversationRepository",
    "ContextContentRepository",
    "ContextStore",
    # Audit
    "AuditEntryRepository",
    "AuditQueueRepository",
    # Analytics
    "UserProfileRepository",
    "SessionHistoryRepository",
    "UserBehaviorStore",
    # Learning
    "FeedbackRepository",
    "LearningThresholdRepository",
    "ActiveLearningRepository",
    "FeedbackStore",
    # Metrics
    "MetricDataPointRepository",
    "MetricAggregationRepository",
    "MetricsStore",
]
