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
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

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


# =============================================================================
# REMAINING MODELS
# =============================================================================


class ExtractedTriplet(Base):
    """
    Stores triplets extracted by LLM for review before adding to main graph.

    Allows human-in-the-loop validation of LLM extractions.
    Used by ontology-driven extraction (v5.9.2).
    """

    __tablename__ = "kg_extracted_triplets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Triplet data
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    predicate: Mapped[str] = mapped_column(String(100), nullable=False)
    object: Mapped[str] = mapped_column(String(255), nullable=False)

    # Source text
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_document: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Extraction metadata
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    # Review status
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",  # pending, approved, rejected
        index=True,
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "status": self.status,
            "source_text": self.source_text[:200] + "..."
            if len(self.source_text) > 200
            else self.source_text,
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
