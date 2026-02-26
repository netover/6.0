"""
SQLAlchemy Models for All Data Stores.

Consolidated models replacing SQLite stores with PostgreSQL.
All models use async-compatible SQLAlchemy 2.0 style.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from resync.core.database.engine import Base

# =============================================================================
# ENUMS
# =============================================================================

class JobStatusEnum(str, enum.Enum):
    """TWS Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    HELD = "held"
    UNKNOWN = "unknown"

class EventSeverity(str, enum.Enum):
    """Event severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ContentType(str, enum.Enum):
    """Content types for context store."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    SOLUTION = "solution"
    OBSERVATION = "observation"

# =============================================================================
# TWS STATUS STORE MODELS (from tws_status_store.py)
# =============================================================================

class TWSSnapshot(Base):
    """Snapshot of TWS status at a point in time."""

    __tablename__ = "tws_snapshots"
    __table_args__ = (
        Index("idx_tws_snapshots_timestamp", "timestamp"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    snapshot_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    job_count: Mapped[int] = mapped_column(Integer, default=0)
    workstation_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    # lazy="raise" força erro se acessar sem eager loading (previne N+1 queries)
    job_statuses: Mapped[list["TWSJobStatus"]] = relationship(
        back_populates="snapshot",
        lazy="raise",  # Força uso de selectinload() para prevenir N+1
    )

class TWSJobStatus(Base):
    """Individual job status record."""

    __tablename__ = "tws_job_status"
    __table_args__ = (
        Index("idx_tws_job_status_job_name", "job_name"),
        Index("idx_tws_job_status_status", "status"),
        Index("idx_tws_job_status_timestamp", "timestamp"),
        Index("idx_tws_job_status_workstation", "workstation"),
        UniqueConstraint("job_name", "run_number", "timestamp", name="uq_job_run_time"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("tws.tws_snapshots.id")
    )
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_stream: Mapped[str | None] = mapped_column(String(255))
    workstation: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    run_number: Mapped[int] = mapped_column(Integer, default=1)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    return_code: Mapped[int | None] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    # Relationships
    # lazy="raise" força erro se acessar sem eager loading (previne N+1 queries)
    snapshot: Mapped["TWSSnapshot" | None] = relationship(
        back_populates="job_statuses",
        lazy="raise",  # Força uso de joinedload() para prevenir N+1
    )

class TWSWorkstationStatus(Base):
    """Workstation status record."""

    __tablename__ = "tws_workstation_status"
    __table_args__ = (
        Index("idx_tws_ws_name", "workstation_name"),
        Index("idx_tws_ws_timestamp", "timestamp"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workstation_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    cpu_usage: Mapped[float | None] = mapped_column(Float)
    memory_usage: Mapped[float | None] = mapped_column(Float)
    active_jobs: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

class TWSEvent(Base):
    """TWS events and alerts."""

    __tablename__ = "tws_events"
    __table_args__ = (
        Index("idx_tws_events_type", "event_type"),
        Index("idx_tws_events_severity", "severity"),
        Index("idx_tws_events_timestamp", "timestamp"),
        Index("idx_tws_events_job", "job_name"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="info")
    job_name: Mapped[str | None] = mapped_column(String(255))
    workstation: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSONB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(255))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

class TWSPattern(Base):
    """Detected patterns in TWS data."""

    __tablename__ = "tws_patterns"
    __table_args__ = (
        Index("idx_tws_patterns_type", "pattern_type"),
        Index("idx_tws_patterns_job", "job_name"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False)
    job_name: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    pattern_data: Mapped[dict | None] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class TWSProblemSolution(Base):
    """Known problems and their solutions."""

    __tablename__ = "tws_problem_solutions"
    __table_args__ = (
        Index("idx_tws_ps_problem_type", "problem_type"),
        Index("idx_tws_ps_job", "job_name"),
        {"schema": "tws"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    problem_type: Mapped[str] = mapped_column(String(100), nullable=False)
    job_name: Mapped[str | None] = mapped_column(String(255))
    problem_description: Mapped[str] = mapped_column(Text, nullable=False)
    solution: Mapped[str] = mapped_column(Text, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

# =============================================================================
# CONTEXT STORE MODELS (from context_store.py)
# =============================================================================

class Conversation(Base):
    """Conversation history for context."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_session", "session_id"),
        Index("idx_conversations_timestamp", "timestamp"),
        Index("idx_conversations_user", "user_id"),
        {"schema": "context"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    embedding_id: Mapped[str | None] = mapped_column(
        String(255)
    )  # Reference to vector store
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True)

class ContextContent(Base):
    """General content for context retrieval."""

    __tablename__ = "context_content"
    __table_args__ = (
        Index("idx_context_content_type", "content_type"),
        Index("idx_context_content_source", "source"),
        {"schema": "context"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    embedding_id: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

# =============================================================================
# AUDIT MODELS (from audit_db.py, audit_queue.py)
# =============================================================================

class AuditEntry(Base):
    """Audit log entries."""

    __tablename__ = "audit_entries"
    __table_args__ = (
        Index("idx_audit_entries_action", "action"),
        Index("idx_audit_entries_user", "user_id"),
        Index("idx_audit_entries_timestamp", "timestamp"),
        Index("idx_audit_entries_entity", "entity_type", "entity_id"),
        {"schema": "audit"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(255))
    entity_type: Mapped[str | None] = mapped_column(String(100))
    entity_id: Mapped[str | None] = mapped_column(String(255))
    old_value: Mapped[dict | None] = mapped_column(JSONB)
    new_value: Mapped[dict | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None] = mapped_column(String(50))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

class AuditQueueItem(Base):
    """Pending audit items queue."""

    __tablename__ = "audit_queue"
    __table_args__ = (
        Index("idx_audit_queue_status", "status"),
        Index("idx_audit_queue_priority", "priority"),
        Index("idx_audit_queue_created", "created_at"),
        {"schema": "audit"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, processing, completed, failed
    priority: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

# =============================================================================
# USER BEHAVIOR MODELS (from user_behavior.py)
# =============================================================================

class UserProfile(Base):
    """User profile and preferences."""

    __tablename__ = "user_profiles"
    __table_args__ = (
        Index("idx_user_profiles_user", "user_id"),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    preferences: Mapped[dict | None] = mapped_column(JSONB)
    behavior_patterns: Mapped[dict | None] = mapped_column(JSONB)
    skill_level: Mapped[str | None] = mapped_column(String(50))
    total_sessions: Mapped[int] = mapped_column(Integer, default=0)
    total_queries: Mapped[int] = mapped_column(Integer, default=0)
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

class SessionHistory(Base):
    """User session history."""

    __tablename__ = "session_history"
    __table_args__ = (
        Index("idx_session_history_user", "user_id"),
        Index("idx_session_history_session", "session_id"),
        Index("idx_session_history_start", "started_at"),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("analytics.user_profiles.user_id")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    query_count: Mapped[int] = mapped_column(Integer, default=0)
    actions: Mapped[dict | None] = mapped_column(JSONB)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

# =============================================================================
# FEEDBACK & LEARNING MODELS (from feedback_store.py, active_learning.py)
# =============================================================================

class Feedback(Base):
    """User feedback on responses."""

    __tablename__ = "feedback"
    __table_args__ = (
        Index("idx_feedback_session", "session_id"),
        Index("idx_feedback_rating", "rating"),
        Index("idx_feedback_timestamp", "created_at"),
        Index("idx_feedback_curation_status", "curation_status"),
        {"schema": "learning"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    query_id: Mapped[str | None] = mapped_column(String(255))
    query_text: Mapped[str | None] = mapped_column(Text)
    response_text: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int | None] = mapped_column(Integer)  # 1-5
    feedback_type: Mapped[str] = mapped_column(
        String(50), default="general"
    )  # general, accuracy, helpfulness
    feedback_text: Mapped[str | None] = mapped_column(Text)
    is_positive: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    # v5.2.3.20: Golden Record fields for knowledge incorporation
    user_correction: Mapped[str | None] = mapped_column(Text)  # Expert's correct answer
    curation_status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, approved, rejected, incorporated
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    incorporated_doc_id: Mapped[str | None] = mapped_column(
        String(255)
    )  # Vector store doc ID

class LearningThreshold(Base):
    """Dynamic learning thresholds."""

    __tablename__ = "learning_thresholds"
    __table_args__ = (
        Index("idx_learning_thresholds_name", "threshold_name"),
        {"schema": "learning"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    threshold_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    current_value: Mapped[float] = mapped_column(Float, nullable=False)
    min_value: Mapped[float] = mapped_column(Float, default=0.0)
    max_value: Mapped[float] = mapped_column(Float, default=1.0)
    adjustment_rate: Mapped[float] = mapped_column(Float, default=0.01)
    last_adjusted: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    adjustment_history: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

class ActiveLearningCandidate(Base):
    """Candidates for active learning."""

    __tablename__ = "active_learning_candidates"
    __table_args__ = (
        Index("idx_al_candidates_status", "status"),
        Index("idx_al_candidates_uncertainty", "uncertainty_score"),
        {"schema": "learning"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str | None] = mapped_column(Text)
    uncertainty_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, reviewed, incorporated
    selected_label: Mapped[str | None] = mapped_column(String(255))
    reviewer_id: Mapped[str | None] = mapped_column(String(255))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

# =============================================================================
# METRICS MODELS (from lightweight_store.py)
# =============================================================================

class MetricDataPoint(Base):
    """Time-series metric data points."""

    __tablename__ = "metric_data_points"
    __table_args__ = (
        Index("idx_metrics_name", "metric_name"),
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_name_time", "metric_name", "timestamp"),
        {"schema": "metrics"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(50))
    tags: Mapped[dict | None] = mapped_column(JSONB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

class MetricAggregation(Base):
    """Pre-aggregated metrics for faster queries."""

    __tablename__ = "metric_aggregations"
    __table_args__ = (
        Index("idx_metric_agg_name", "metric_name"),
        Index("idx_metric_agg_period", "period_start", "period_end"),
        UniqueConstraint(
            "metric_name", "aggregation_type", "period_start", name="uq_metric_agg"
        ),
        {"schema": "metrics"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    aggregation_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # hourly, daily, weekly
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    min_value: Mapped[float] = mapped_column(Float)
    max_value: Mapped[float] = mapped_column(Float)
    avg_value: Mapped[float] = mapped_column(Float)
    sum_value: Mapped[float] = mapped_column(Float)
    count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

# =============================================================================
# HELPER FUNCTION TO GET ALL MODELS
# =============================================================================

def get_all_models():
    """Return all model classes for migration/schema creation."""
    return [
        # TWS
        TWSSnapshot,
        TWSJobStatus,
        TWSWorkstationStatus,
        TWSEvent,
        TWSPattern,
        TWSProblemSolution,
        # Context
        Conversation,
        ContextContent,
        # Audit
        AuditEntry,
        AuditQueueItem,
        # Analytics
        UserProfile,
        SessionHistory,
        # Learning
        Feedback,
        LearningThreshold,
        ActiveLearningCandidate,
        # Metrics
        MetricDataPoint,
        MetricAggregation,
    ]
