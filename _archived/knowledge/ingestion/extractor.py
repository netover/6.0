"""
Knowledge Graph Triplet Extractor.

Uses LLM to extract structured relationships from unstructured text.
Follows the schema-based extraction approach from the article.

Key Features:
- Constrained relation types (prevents hallucination)
- Entity normalization (prevents duplicates)
- Confidence scoring
- Human-in-the-loop review queue
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from resync.core.database.engine import get_db_session as get_async_session
from resync.core.structured_logger import get_logger
from resync.knowledge.models import ExtractedTriplet, NodeType

logger = get_logger(__name__)


@dataclass
class Triplet:
    """Represents a subject-predicate-object triplet."""

    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 1.0
    source_text: str = ""


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Allowed relations for TWS domain
ALLOWED_RELATIONS = {
    # Job relationships
    "DEPENDS_ON": ("job", "job"),
    "TRIGGERS": ("job", "job"),
    "RUNS_ON": ("job", "workstation"),
    "BELONGS_TO": ("job", "job_stream"),
    "USES": ("job", "resource"),
    "FOLLOWS": ("job", "schedule"),
    # Hierarchy
    "PART_OF": ("*", "*"),
    "CONTAINS": ("*", "*"),
    # Events
    "AFFECTED": ("event", "job"),
    "OCCURRED_ON": ("event", "workstation"),
    "CAUSED_BY": ("event", "event"),
}

# Entity normalization patterns
ENTITY_PATTERNS = {
    "job": [
        r"\bjob[:\s]+([A-Z][A-Z0-9_]+)\b",
        r"\b([A-Z][A-Z0-9_]{2,})\s+(?:job|batch|process)\b",
        r"\bjob\s+(?:called|named)\s+['\"]?([A-Z][A-Z0-9_]+)['\"]?\b",
    ],
    "workstation": [
        r"\b(?:workstation|ws|server|agent)[:\s]+([A-Z0-9_]+)\b",
        r"\b([A-Z]{2,3}\d{3,})\b",  # e.g., WS001, SRV123
    ],
    "resource": [
        r"\bresource[:\s]+([A-Z0-9_]+)\b",
        r"\b(?:file|database|queue|lock)[:\s]+([A-Z0-9_./]+)\b",
    ],
}


class TripletExtractor:
    """
    Extracts knowledge graph triplets from text using LLM.

    Uses schema-based extraction to ensure only valid relationships
    are extracted, preventing hallucination.
    """

    def __init__(self, llm_client: Any | None = None):
        """
        Initialize extractor.

        Args:
            llm_client: LLM client for extraction (uses LiteLLM if None)
        """
        self._llm = llm_client

    async def _get_llm(self):
        """Get or create LLM client."""
        if self._llm is None:
            from resync.services.llm_service import get_llm_service

            self._llm = get_llm_service()
        return self._llm

    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================

    async def extract_from_text(
        self, text: str, source_document: str | None = None, auto_approve: bool = False
    ) -> list[Triplet]:
        """
        Extract triplets from text using LLM.

        Args:
            text: Text to extract from
            source_document: Optional source document reference
            auto_approve: If True, add directly to graph; else queue for review

        Returns:
            List of extracted triplets
        """
        if not text or len(text.strip()) < 10:
            return []

        # Build extraction prompt
        prompt = self._build_extraction_prompt(text)

        try:
            llm = await self._get_llm()
            response = await llm.generate(prompt)

            # Parse response
            triplets = self._parse_extraction_response(response, text)

            # Normalize entities
            triplets = [self._normalize_triplet(t) for t in triplets]

            # Filter by allowed schema
            triplets = [t for t in triplets if self._is_valid_triplet(t)]

            # Save to review queue if not auto-approving
            if not auto_approve:
                await self._save_to_review_queue(triplets, text, source_document)

            logger.info("triplets_extracted", count=len(triplets), source=source_document)

            return triplets

        except Exception as e:
            logger.error("triplet_extraction_failed", error=str(e))
            return []

    def _build_extraction_prompt(self, text: str) -> str:
        """Build the extraction prompt with schema constraints."""
        relations_list = "\n".join([f"  - {r}" for r in ALLOWED_RELATIONS])

        return f"""You are a Knowledge Graph Engineer for a TWS/HWA workload automation system.
Extract relationships from the text below.

ALLOWED RELATION TYPES:
{relations_list}

OUTPUT FORMAT (one per line):
Subject | RELATION | Object

RULES:
1. Use ONLY the relations listed above
2. Subject and Object should be entity names (jobs, workstations, resources)
3. Job names are usually UPPERCASE with underscores (e.g., BATCH_PROCESS, INIT_JOB)
4. Workstation names often have format like WS001, SRV123
5. If no relationships found, output: NONE
6. Maximum 10 relationships per extraction

TEXT:
{text[:2000]}

RELATIONSHIPS:"""

    def _parse_extraction_response(self, response: str, source_text: str) -> list[Triplet]:
        """Parse LLM response into triplets."""
        triplets = []

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            if not line or line.upper() == "NONE":
                continue

            # Skip lines that don't look like triplets
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) >= 3:
                subject = parts[0]
                predicate = parts[1].upper().replace(" ", "_")
                obj = parts[2]

                # Infer types from context
                subj_type = self._infer_entity_type(subject)
                obj_type = self._infer_entity_type(obj)

                triplets.append(
                    Triplet(
                        subject=subject,
                        subject_type=subj_type,
                        predicate=predicate,
                        object=obj,
                        object_type=obj_type,
                        confidence=0.8,  # Default confidence for LLM extraction
                        source_text=source_text[:500],
                    )
                )

        return triplets

    def _infer_entity_type(self, entity: str) -> str:
        """Infer entity type from name patterns."""
        entity_upper = entity.upper()

        # Check patterns
        for etype, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, entity, re.IGNORECASE):
                    return etype

        # Heuristics
        if re.match(r"^[A-Z][A-Z0-9_]{2,}$", entity_upper):
            return "job"  # Most likely a job name

        if re.match(r"^[A-Z]{2,3}\d{3,}$", entity_upper):
            return "workstation"

        return "unknown"

    def _normalize_triplet(self, triplet: Triplet) -> Triplet:
        """Normalize entity names to prevent duplicates."""
        # Remove common prefixes/suffixes
        subject = self._normalize_entity_name(triplet.subject)
        obj = self._normalize_entity_name(triplet.object)

        return Triplet(
            subject=subject,
            subject_type=triplet.subject_type,
            predicate=triplet.predicate,
            object=obj,
            object_type=triplet.object_type,
            confidence=triplet.confidence,
            source_text=triplet.source_text,
        )

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize an entity name."""
        # Remove quotes
        name = name.strip("'\"")

        # Remove common prefixes
        for prefix in ["job:", "Job:", "JOB:", "ws:", "WS:", "resource:"]:
            if name.startswith(prefix):
                name = name[len(prefix) :]

        # Uppercase for consistency
        return name.strip().upper()

    def _is_valid_triplet(self, triplet: Triplet) -> bool:
        """Check if triplet matches allowed schema."""
        predicate = triplet.predicate

        if predicate not in ALLOWED_RELATIONS:
            return False

        expected_types = ALLOWED_RELATIONS[predicate]

        # Wildcard match
        if (
            expected_types[0] != "*"
            and triplet.subject_type != expected_types[0]
            and triplet.subject_type != "unknown"
        ):
            return False

        return not (expected_types[1] != "*" and triplet.object_type != expected_types[1] and triplet.object_type != "unknown")

    async def _save_to_review_queue(
        self, triplets: list[Triplet], source_text: str, source_document: str | None
    ) -> None:
        """Save extracted triplets to review queue."""
        async with get_async_session() as session:
            for t in triplets:
                record = ExtractedTriplet(
                    subject=t.subject,
                    predicate=t.predicate,
                    object=t.object,
                    source_text=source_text[:1000],
                    source_document=source_document,
                    model_used="llm_extraction",
                    confidence=t.confidence,
                    status="pending",
                )
                session.add(record)

            await session.commit()

    # =========================================================================
    # RULE-BASED EXTRACTION (for structured data)
    # =========================================================================

    def extract_from_tws_data(self, job_data: dict[str, Any]) -> list[Triplet]:
        """
        Extract triplets from structured TWS job data.

        This is more reliable than LLM extraction for structured sources.

        Args:
            job_data: Job definition from TWS API

        Returns:
            List of triplets
        """
        triplets = []

        job_name = job_data.get("name", job_data.get("job_name", ""))
        if not job_name:
            return triplets

        # Workstation relationship
        workstation = job_data.get("workstation", job_data.get("ws", ""))
        if workstation:
            triplets.append(
                Triplet(
                    subject=job_name,
                    subject_type="job",
                    predicate="RUNS_ON",
                    object=workstation,
                    object_type="workstation",
                    confidence=1.0,
                    source_text="TWS API",
                )
            )

        # Job stream relationship
        job_stream = job_data.get("job_stream", job_data.get("stream", ""))
        if job_stream:
            triplets.append(
                Triplet(
                    subject=job_name,
                    subject_type="job",
                    predicate="BELONGS_TO",
                    object=job_stream,
                    object_type="job_stream",
                    confidence=1.0,
                    source_text="TWS API",
                )
            )

        # Dependencies
        dependencies = job_data.get("dependencies", job_data.get("deps", []))
        for dep in dependencies:
            if isinstance(dep, dict):
                dep = dep.get("job_name", dep.get("name", ""))
            if dep:
                triplets.append(
                    Triplet(
                        subject=job_name,
                        subject_type="job",
                        predicate="DEPENDS_ON",
                        object=dep,
                        object_type="job",
                        confidence=1.0,
                        source_text="TWS API",
                    )
                )

        # Resources
        resources = job_data.get("resources", job_data.get("resource_requirements", []))
        if isinstance(resources, dict):
            resources = list(resources.keys())
        for res in resources:
            if isinstance(res, dict):
                res = res.get("name", "")
            if res:
                triplets.append(
                    Triplet(
                        subject=job_name,
                        subject_type="job",
                        predicate="USES",
                        object=res,
                        object_type="resource",
                        confidence=1.0,
                        source_text="TWS API",
                    )
                )

        return triplets

    def extract_from_event(self, event_data: dict[str, Any]) -> list[Triplet]:
        """
        Extract triplets from event/log data.

        Args:
            event_data: Event record

        Returns:
            List of triplets
        """
        triplets = []

        event_id = event_data.get("event_id", event_data.get("id", ""))
        if not event_id:
            return triplets

        # Affected job
        job_id = event_data.get("job_id", event_data.get("job_name", ""))
        if job_id:
            triplets.append(
                Triplet(
                    subject=str(event_id),
                    subject_type="event",
                    predicate="AFFECTED",
                    object=job_id,
                    object_type="job",
                    confidence=1.0,
                    source_text="Event log",
                )
            )

        # Workstation
        workstation = event_data.get("workstation", event_data.get("source", ""))
        if workstation:
            triplets.append(
                Triplet(
                    subject=str(event_id),
                    subject_type="event",
                    predicate="OCCURRED_ON",
                    object=workstation,
                    object_type="workstation",
                    confidence=1.0,
                    source_text="Event log",
                )
            )

        return triplets


# =============================================================================
# REVIEW QUEUE MANAGEMENT
# =============================================================================


async def get_pending_triplets(limit: int = 50) -> list[dict[str, Any]]:
    """Get triplets pending review."""
    from sqlalchemy import select

    async with get_async_session() as session:
        result = await session.execute(
            select(ExtractedTriplet)
            .where(ExtractedTriplet.status == "pending")
            .order_by(ExtractedTriplet.created_at.desc())
            .limit(limit)
        )
        triplets = result.scalars().all()
        return [t.to_dict() for t in triplets]


async def approve_triplet(triplet_id: int, reviewer: str) -> bool:
    """Approve a triplet and add to graph."""
    from sqlalchemy import select

    from resync.knowledge.retrieval.graph import get_knowledge_graph

    async with get_async_session() as session:
        result = await session.execute(
            select(ExtractedTriplet).where(ExtractedTriplet.id == triplet_id)
        )
        triplet = result.scalar_one_or_none()

        if not triplet:
            return False

        # Update status
        triplet.status = "approved"
        triplet.reviewed_by = reviewer
        triplet.reviewed_at = datetime.now(timezone.utc)

        await session.commit()

        # Add to graph
        kg = get_knowledge_graph()
        await kg.add_node(
            f"job:{triplet.subject}", NodeType.JOB, triplet.subject, source="llm_extracted"
        )
        await kg.add_node(
            f"job:{triplet.object}", NodeType.JOB, triplet.object, source="llm_extracted"
        )
        await kg.add_edge(
            f"job:{triplet.subject}",
            f"job:{triplet.object}",
            triplet.predicate,
            confidence=triplet.confidence,
            source="llm_extracted",
        )

        logger.info("triplet_approved", triplet_id=triplet_id, reviewer=reviewer)

        return True


async def reject_triplet(triplet_id: int, reviewer: str) -> bool:
    """Reject a triplet."""
    from sqlalchemy import select

    async with get_async_session() as session:
        result = await session.execute(
            select(ExtractedTriplet).where(ExtractedTriplet.id == triplet_id)
        )
        triplet = result.scalar_one_or_none()

        if not triplet:
            return False

        triplet.status = "rejected"
        triplet.reviewed_by = reviewer
        triplet.reviewed_at = datetime.now(timezone.utc)

        await session.commit()

        logger.info("triplet_rejected", triplet_id=triplet_id, reviewer=reviewer)

        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_triplet_extractor() -> TripletExtractor:
    """Get a triplet extractor instance."""
    return TripletExtractor()
