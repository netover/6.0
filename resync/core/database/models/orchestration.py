"""
Orchestration Models - SQLAlchemy Async
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy import UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from resync.core.database.engine import Base

class OrchestrationStrategy(str, Enum):
    """Enumeration of orchestration strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    FALLBACK = "fallback"

class ExecutionStatus(str, Enum):
    """Status states for orchestration execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATION = "compensation"

class StepStatus(str, Enum):
    """Status states for individual step runs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"

class OrchestrationConfig(Base):
    """
    Model for orchestration configuration.

    Stores the definition of an orchestration workflow including
    all steps, agents, dependencies, and execution settings.
    """

    __tablename__ = "orchestration_configs"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Strategy
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)

    # Steps definition (JSON)
    steps: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Metadata
    meta_data: Mapped[dict] = mapped_column(JSON, default=lambda: {})

    # Access control
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_global: Mapped[bool] = mapped_column(Boolean, default=False)

    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    executions: Mapped[list["OrchestrationExecution"]] = relationship(
        "OrchestrationExecution", back_populates="config", cascade="all, delete-orphan"
    )

    # Table args
    __table_args__ = (
        Index("idx_configs_owner", "owner_id"),
        Index("idx_configs_tenant", "tenant_id"),
        Index("idx_configs_strategy", "strategy"),
        Index("idx_configs_active", "is_active", postgresql_where=is_active.is_(True)),
    )

    def __repr__(self) -> str:
        return f"<OrchestrationConfig(id={self.id}, name={self.name}, strategy={self.strategy})>"

class OrchestrationExecution(Base):
    """
    Model for orchestration execution instance.

    Represents a single execution of an orchestration configuration,
    tracking status, input, output, and metrics.
    """

    __tablename__ = "orchestration_executions"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Trace ID for correlation
    trace_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )

    # Configuration reference
    config_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    config_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    # Input/Output
    input_data: Mapped[dict] = mapped_column(JSON, default=lambda: {})
    output_data: Mapped[dict] = mapped_column(JSON, default=lambda: {})

    # Context
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paused_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Metadata
    meta_data: Mapped[dict] = mapped_column(JSON, default=lambda: {})

    # Metrics
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(
        Numeric(precision=10, scale=6, asdecimal=False), nullable=True
    )

    # Control
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    callback_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    config: Mapped["OrchestrationConfig" | None] = relationship(
        "OrchestrationConfig", back_populates="executions"
    )
    step_runs: Mapped[list["OrchestrationStepRun"]] = relationship(
        "OrchestrationStepRun",
        back_populates="execution",
        cascade="all, delete-orphan",
        order_by="OrchestrationStepRun.step_index",
    )
    callbacks: Mapped[list["OrchestrationCallback"]] = relationship(
        "OrchestrationCallback",
        back_populates="execution",
        cascade="all, delete-orphan",
    )

    # Table args
    __table_args__ = (
        Index("idx_executions_config", "config_id"),
        Index("idx_executions_status", "status"),
        Index("idx_executions_user", "user_id"),
        Index("idx_executions_session", "session_id"),
        Index("idx_executions_created", "created_at", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<OrchestrationExecution(id={self.id}, trace_id={self.trace_id}, status={self.status})>"

class OrchestrationStepRun(Base):
    """
    Model for individual step execution within an orchestration.

    Tracks the execution of each step including status, output,
    metrics, and errors.
    """

    __tablename__ = "orchestration_step_runs"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Execution reference
    execution_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_executions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Step identification
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_id: Mapped[str] = mapped_column(String(100), nullable=False)
    step_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    # Output
    output: Mapped[dict] = mapped_column(JSON, default=lambda: {})
    output_truncated: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_trace: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dependencies
    dependencies_json: Mapped[list] = mapped_column(JSON, default=lambda: [])

    # Metrics
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[float | None] = mapped_column(nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Context
    agent_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    execution: Mapped["OrchestrationExecution"] = relationship(
        "OrchestrationExecution", back_populates="step_runs"
    )

    # Table args
    __table_args__ = (
        UniqueConstraint("execution_id", "step_index", name="uq_step_execution_index"),
        UniqueConstraint("execution_id", "step_id", name="uq_step_execution_id"),
        Index("idx_steps_status", "status"),
        Index(
            "idx_steps_latency", "latency_ms", postgresql_where=status == "completed"
        ),  # noqa: E712
    )

    def __repr__(self) -> str:
        return f"<OrchestrationStepRun(id={self.id}, step_id={self.step_id}, status={self.status})>"

class OrchestrationCallback(Base):
    """
    Model for orchestration callbacks/webhooks.

    Stores callback configuration and delivery status for
    asynchronous notifications.
    """

    __tablename__ = "orchestration_callbacks"

    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Execution reference
    execution_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("orchestration_executions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Callback configuration
    callback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    callback_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    callback_method: Mapped[str] = mapped_column(String(10), default="POST")
    callback_headers: Mapped[dict] = mapped_column(JSON, default=lambda: {})
    callback_body_template: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="pending")

    # Retry
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    last_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Result
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    execution: Mapped["OrchestrationExecution"] = relationship(
        "OrchestrationExecution", back_populates="callbacks"
    )

    def __repr__(self) -> str:
        return f"<OrchestrationCallback(id={self.id}, type={self.callback_type}, status={self.status})>"
