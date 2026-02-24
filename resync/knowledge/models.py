"""
Knowledge Graph Models v5.9.3

Simplified models after removing persistent graph storage.
Graph is now built on-demand from TWS API using NetworkX.

Remaining models:
- Enums: NodeType, RelationType (used for typing)
- ExtractedTriplet: LLM-extracted entities pending review

Removed in v5.9.3:
- GraphNode: Graph now built on-demand from TWS API
- GraphEdge: Graph now built on-demand from TWS API
- GraphSnapshot: No longer needed

Version: 6.0.0 - Fixed datetime lambda bug, added validations
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from resync.core.database.engine import Base


# =============================================================================
# ENUMS (kept for type hints and ontology)
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the TWS knowledge graph."""

    JOB = "job"
    JOB_STREAM = "job_stream"
    WORKSTATION = "workstation"
    RESOURCE = "resource"
    SCHEDULE = "schedule"
    POLICY = "policy"
    APPLICATION = "application"
    ENVIRONMENT = "environment"
    EVENT = "event"
    ALERT = "alert"


class RelationType(str, Enum):
    """Types of relationships (edges) in the TWS knowledge graph."""

    # Job relationships
    DEPENDS_ON = "depends_on"  # Job → Job
    TRIGGERS = "triggers"  # Job → Job (downstream)
    RUNS_ON = "runs_on"  # Job → Workstation
    BELONGS_TO = "belongs_to"  # Job → JobStream
    USES = "uses"  # Job → Resource
    FOLLOWS = "follows"  # Job → Schedule
    GOVERNED_BY = "governed_by"  # Job → Policy

    # Hierarchy relationships
    PART_OF = "part_of"  # JobStream → Application
    HOSTED_ON = "hosted_on"  # Application → Environment
    CONTAINS = "contains"  # Parent → Child (generic)

    # Event relationships
    OCCURRED_ON = "occurred_on"  # Event → Workstation
    AFFECTED = "affected"  # Event → Job
    NEXT = "next"  # Event → Event (temporal chain)
    CAUSED_BY = "caused_by"  # Event → Event (causal)

    # Resource relationships
    SHARED_BY = "shared_by"  # Resource → Job (multiple)
    EXCLUSIVE_TO = "exclusive_to"  # Resource → Job (single)


class TripletStatus(str, Enum):
    """Status values for extracted triplets."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# =============================================================================
# REMAINING MODELS
# =============================================================================


class ExtractedTriplet(Base):
    """
    Stores triplets extracted by LLM for review before adding to main graph.

    Allows human-in-the-loop validation of LLM extractions.
    Used by ontology-driven extraction (v5.9.2).

    Constraints:
    - confidence must be between 0.0 and 1.0
    - reviewed_by and reviewed_at must both be NULL or both be set
    - subject, predicate, object sanitized before storage
    """

    __tablename__ = "kg_extracted_triplets"

    # Table-level constraints
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_confidence_range",
        ),
        CheckConstraint(
            "(reviewed_by IS NULL AND reviewed_at IS NULL) OR "
            "(reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)",
            name="ck_reviewed_consistency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Triplet data - sanitized via @validates decorators
    subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    predicate: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    object: Mapped[str] = mapped_column(String(255), nullable=False)

    # Source text - max 10MB
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_document: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Extraction metadata
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
    )

    # Review status - uses Enum for type safety
    status: Mapped[TripletStatus] = mapped_column(
        String(20),
        nullable=False,
        default=TripletStatus.PENDING.value,
        index=True,
    )
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Timestamps - use server_default for database-side timestamp
    # FIXED: Was using lambda which evaluated once at class definition
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),  # Database-side default using SQL function
        index=True,
    )

    def to_dict(self, truncate_text: int = 200) -> dict[str, Any]:
        """
        Serialize to dictionary for API responses.

        Args:
            truncate_text: Maximum length for source_text (0 = no truncation)

        Returns:
            Dictionary representation safe for JSON serialization
        """
        source_text = self.source_text
        if truncate_text > 0 and len(source_text) > truncate_text:
            source_text = source_text[:truncate_text] + "..."

        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),  # Limit precision
            "status": self.status,
            "source_text": source_text,
            "source_document": self.source_document,
            "model_used": self.model_used,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# REMOVED IN v5.9.3
# =============================================================================
#
# The following models were removed as graph is now built on-demand from TWS API:
# - GraphNode: Persistent node storage
# - GraphEdge: Persistent edge storage
# - GraphSnapshot: Graph statistics snapshots
#
# Rationale:
# - TWS API is the single source of truth for job dependencies
# - Building graph on-demand ensures data is always fresh
# - NetworkX in-memory graph is sufficient for ~100K nodes
# - Eliminates sync complexity and potential data staleness
