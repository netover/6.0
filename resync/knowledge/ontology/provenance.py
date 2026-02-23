"""
TWS Provenance Tracker v5.9.2

Tracks the origin and lifecycle of every entity in the knowledge graph.
Enables auditability: "Where did this information come from?"

Features:
1. Source Tracking - Which document, chunk, section
2. Extraction Tracking - Which model, when, confidence
3. Verification Tracking - Human validation status
4. Lineage Tracking - Entity relationships and derivations

Author: Resync Team
Version: 5.9.2
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class SourceType(str, Enum):
    """Type of source document."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    LOG = "log"
    API = "api"
    MANUAL_ENTRY = "manual"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Method used to extract entity."""

    LLM = "llm"
    REGEX = "regex"
    HYBRID = "hybrid"
    MANUAL = "manual"
    SYSTEM = "system"


class VerificationStatus(str, Enum):
    """Entity verification status."""

    UNVERIFIED = "unverified"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    AUTO_VERIFIED = "auto_verified"  # High confidence + validation passed


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class SourceInfo:
    """Information about the source of extracted data."""

    # Document identification
    source_type: SourceType = SourceType.UNKNOWN
    source_file: str = ""
    source_uri: str = ""

    # Content identification
    document_title: str = ""
    section_path: str = ""  # e.g., "Chapter 5 > Troubleshooting > Error Codes"
    page_number: int | None = None
    chunk_id: str = ""

    # Content hash for integrity verification
    content_hash: str = ""

    # Position in source
    start_char: int = 0
    end_char: int = 0
    start_line: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_type": self.source_type.value,
            "source_file": self.source_file,
            "source_uri": self.source_uri,
            "document_title": self.document_title,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class ExtractionInfo:
    """Information about how entity was extracted."""

    # Extraction method
    method: ExtractionMethod = ExtractionMethod.LLM

    # Model information
    model_name: str = ""
    model_version: str = ""
    prompt_template: str = ""

    # Quality metrics
    confidence_score: float = 0.0
    validation_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)

    # Timing
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_template": self.prompt_template,
            "confidence_score": self.confidence_score,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_duration_ms": self.extraction_duration_ms,
        }


@dataclass
class VerificationInfo:
    """Information about entity verification."""

    status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_by: str | None = None
    verified_at: datetime | None = None
    verification_notes: str = ""
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verification_notes": self.verification_notes,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ProvenanceRecord:
    """Complete provenance record for an entity."""

    # Entity identification
    entity_id: str
    entity_type: str

    # Source information
    source: SourceInfo

    # Extraction information
    extraction: ExtractionInfo

    # Verification information
    verification: VerificationInfo = field(default_factory=VerificationInfo)

    # Record metadata
    record_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Generate record ID if not provided."""
        if not self.record_id:
            hash_input = (
                f"{self.entity_id}:{self.source.source_file}:"
                f"{self.created_at.isoformat()}"
            )
            self.record_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "source": self.source.to_dict(),
            "extraction": self.extraction.to_dict(),
            "verification": self.verification.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def is_verified(self) -> bool:
        """Check if entity is verified."""
        return self.verification.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.AUTO_VERIFIED,
        )

    @property
    def is_trustworthy(self) -> bool:
        """Check if entity is trustworthy (verified + validation passed)."""
        return (
            self.is_verified
            and self.extraction.validation_passed
            and self.extraction.confidence_score >= 0.8
        )


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================


class ProvenanceTracker:
    """
    Tracks provenance for all extracted entities.

    Provides:
    - Create provenance records for new entities
    - Update verification status
    - Query provenance by entity or source
    - Export provenance for auditing
    """

    def __init__(self):
        """Initialize ProvenanceTracker."""
        # In-memory storage (can be replaced with database backend)
        self._records: dict[str, ProvenanceRecord] = {}  # record_id -> record
        self._entity_index: dict[str, list[str]] = {}  # entity_id -> record_ids
        self._source_index: dict[str, list[str]] = {}  # source_file -> record_ids

        logger.info("provenance_tracker_initialized")

    # =========================================================================
    # RECORD CREATION
    # =========================================================================

    def create_record(
        self,
        entity_id: str,
        entity_type: str,
        source: SourceInfo,
        extraction: ExtractionInfo,
    ) -> ProvenanceRecord:
        """
        Create a provenance record for an entity.

        Args:
            entity_id: ID of the extracted entity
            entity_type: Type of entity (Job, ErrorCode, etc.)
            source: Source document information
            extraction: Extraction method and quality info

        Returns:
            Created ProvenanceRecord
        """
        record = ProvenanceRecord(
            entity_id=entity_id,
            entity_type=entity_type,
            source=source,
            extraction=extraction,
        )

        # Store record
        self._records[record.record_id] = record

        # Index by entity
        if entity_id not in self._entity_index:
            self._entity_index[entity_id] = []
        self._entity_index[entity_id].append(record.record_id)

        # Index by source
        if source.source_file not in self._source_index:
            self._source_index[source.source_file] = []
        self._source_index[source.source_file].append(record.record_id)

        logger.debug(
            "provenance_record_created",
            record_id=record.record_id,
            entity_id=entity_id,
            entity_type=entity_type,
            source_file=source.source_file,
        )

        return record

    def create_from_chunk_metadata(
        self,
        entity_id: str,
        entity_type: str,
        chunk_metadata: dict[str, Any],
        model_name: str,
        confidence: float = 0.0,
        validation_passed: bool = False,
    ) -> ProvenanceRecord:
        """
        Create provenance record from ChunkMetadata.

        Convenience method that extracts relevant fields from ChunkMetadata.

        Args:
            entity_id: ID of the extracted entity
            entity_type: Type of entity
            chunk_metadata: ChunkMetadata as dict
            model_name: LLM model used for extraction
            confidence: Extraction confidence score
            validation_passed: Whether validation succeeded

        Returns:
            Created ProvenanceRecord
        """
        # Build SourceInfo from chunk metadata
        source = SourceInfo(
            source_type=self._detect_source_type(chunk_metadata.get("source_file", "")),
            source_file=chunk_metadata.get("source_file", ""),
            document_title=chunk_metadata.get("document_title", ""),
            section_path=chunk_metadata.get("section_path", ""),
            chunk_id=str(chunk_metadata.get("chunk_index", 0)),
            start_char=chunk_metadata.get("start_char", 0),
            end_char=chunk_metadata.get("end_char", 0),
        )

        # Build ExtractionInfo
        extraction = ExtractionInfo(
            method=ExtractionMethod.LLM,
            model_name=model_name,
            confidence_score=confidence,
            validation_passed=validation_passed,
        )

        return self.create_record(entity_id, entity_type, source, extraction)

    @staticmethod
    def _detect_source_type(filename: str) -> SourceType:
        """Detect source type from filename."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            return SourceType.PDF
        if filename_lower.endswith((".md", ".markdown")):
            return SourceType.MARKDOWN
        if filename_lower.endswith((".log", ".txt")):
            return SourceType.LOG
        return SourceType.UNKNOWN

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify_entity(
        self,
        entity_id: str,
        verified_by: str,
        notes: str = "",
    ) -> list[ProvenanceRecord]:
        """
        Mark all provenance records for an entity as verified.

        Args:
            entity_id: ID of entity to verify
            verified_by: Who verified (username or system)
            notes: Optional verification notes

        Returns:
            Updated provenance records
        """
        record_ids = self._entity_index.get(entity_id, [])
        updated = []

        for record_id in record_ids:
            record = self._records.get(record_id)
            if record:
                record.verification.status = VerificationStatus.VERIFIED
                record.verification.verified_by = verified_by
                record.verification.verified_at = datetime.now(timezone.utc)
                record.verification.verification_notes = notes
                record.updated_at = datetime.now(timezone.utc)
                updated.append(record)

        logger.info(
            "entity_verified",
            entity_id=entity_id,
            verified_by=verified_by,
            records_updated=len(updated),
        )

        return updated

    def reject_entity(
        self,
        entity_id: str,
        rejected_by: str,
        reason: str,
    ) -> list[ProvenanceRecord]:
        """
        Mark all provenance records for an entity as rejected.

        Args:
            entity_id: ID of entity to reject
            rejected_by: Who rejected
            reason: Rejection reason

        Returns:
            Updated provenance records
        """
        record_ids = self._entity_index.get(entity_id, [])
        updated = []

        for record_id in record_ids:
            record = self._records.get(record_id)
            if record:
                record.verification.status = VerificationStatus.REJECTED
                record.verification.verified_by = rejected_by
                record.verification.verified_at = datetime.now(timezone.utc)
                record.verification.rejection_reason = reason
                record.updated_at = datetime.now(timezone.utc)
                updated.append(record)

        logger.info(
            "entity_rejected",
            entity_id=entity_id,
            rejected_by=rejected_by,
            reason=reason,
            records_updated=len(updated),
        )

        return updated

    def auto_verify_if_trustworthy(
        self,
        entity_id: str,
        confidence_threshold: float = 0.9,
    ) -> bool:
        """
        Auto-verify entity if it meets trust criteria.

        Criteria:
        - Confidence score >= threshold
        - Validation passed
        - No validation errors

        Args:
            entity_id: ID of entity to check
            confidence_threshold: Minimum confidence for auto-verify

        Returns:
            True if auto-verified, False otherwise
        """
        record_ids = self._entity_index.get(entity_id, [])

        for record_id in record_ids:
            record = self._records.get(record_id)
            if not record:
                continue

            if (
                record.extraction.confidence_score >= confidence_threshold
                and record.extraction.validation_passed
                and not record.extraction.validation_errors
            ):
                record.verification.status = VerificationStatus.AUTO_VERIFIED
                record.verification.verified_at = datetime.now(timezone.utc)
                record.verification.verification_notes = (
                    "Auto-verified: confidence={record.extraction.confidence_score:.2f}"
                )
                record.updated_at = datetime.now(timezone.utc)

                logger.debug(
                    "entity_auto_verified",
                    entity_id=entity_id,
                    confidence=record.extraction.confidence_score,
                )
                return True

        return False

    # =========================================================================
    # QUERYING
    # =========================================================================

    def get_provenance(self, entity_id: str) -> list[ProvenanceRecord]:
        """
        Get all provenance records for an entity.

        Args:
            entity_id: ID of entity

        Returns:
            List of provenance records
        """
        record_ids = self._entity_index.get(entity_id, [])
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_source_entities(self, source_file: str) -> list[ProvenanceRecord]:
        """
        Get all entities extracted from a source file.

        Args:
            source_file: Source filename

        Returns:
            List of provenance records
        """
        record_ids = self._source_index.get(source_file, [])
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_unverified_entities(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[ProvenanceRecord]:
        """
        Get entities pending verification.

        Args:
            entity_type: Filter by entity type (optional)
            limit: Maximum number to return

        Returns:
            List of unverified provenance records
        """
        unverified = []

        for record in self._records.values():
            if record.verification.status == VerificationStatus.UNVERIFIED:
                if entity_type is None or record.entity_type == entity_type:
                    unverified.append(record)
                    if len(unverified) >= limit:
                        break

        return unverified

    def get_low_confidence_entities(
        self,
        threshold: float = 0.7,
        limit: int = 100,
    ) -> list[ProvenanceRecord]:
        """
        Get entities with low extraction confidence.

        Args:
            threshold: Confidence threshold
            limit: Maximum number to return

        Returns:
            List of low-confidence provenance records
        """
        low_confidence = []

        for record in self._records.values():
            if record.extraction.confidence_score < threshold:
                low_confidence.append(record)
                if len(low_confidence) >= limit:
                    break

        # Sort by confidence (lowest first)
        low_confidence.sort(key=lambda r: r.extraction.confidence_score)
        return low_confidence

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_all(self) -> list[dict]:
        """Export all provenance records as list of dicts."""
        return [record.to_dict() for record in self._records.values()]

    def export_for_entity(self, entity_id: str) -> list[dict]:
        """Export provenance records for a specific entity."""
        records = self.get_provenance(entity_id)
        return [record.to_dict() for record in records]

    def get_stats(self) -> dict[str, Any]:
        """Get provenance statistics."""
        total = len(self._records)
        if total == 0:
            return {"total": 0}

        verified = sum(1 for r in self._records.values() if r.is_verified)
        rejected = sum(
            1
            for r in self._records.values()
            if r.verification.status == VerificationStatus.REJECTED
        )
        auto_verified = sum(
            1
            for r in self._records.values()
            if r.verification.status == VerificationStatus.AUTO_VERIFIED
        )

        # Confidence distribution
        confidences = [r.extraction.confidence_score for r in self._records.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Entity type distribution
        type_counts: dict[str, int] = {}
        for record in self._records.values():
            type_counts[record.entity_type] = type_counts.get(record.entity_type, 0) + 1

        return {
            "total": total,
            "verified": verified,
            "auto_verified": auto_verified,
            "rejected": rejected,
            "unverified": total - verified - auto_verified - rejected,
            "verification_rate": (verified + auto_verified) / total,
            "avg_confidence": avg_confidence,
            "entity_types": type_counts,
            "unique_sources": len(self._source_index),
            "unique_entities": len(self._entity_index),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================


_provenance_tracker: ProvenanceTracker | None = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get or create the singleton ProvenanceTracker instance."""
    global _provenance_tracker
    if _provenance_tracker is None:
        _provenance_tracker = ProvenanceTracker()
    return _provenance_tracker


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def track_entity(
    entity_id: str,
    entity_type: str,
    chunk_metadata: dict[str, Any],
    model_name: str,
    confidence: float = 0.0,
) -> ProvenanceRecord:
    """
    Convenience function to track entity provenance.

    Args:
        entity_id: ID of extracted entity
        entity_type: Type of entity
        chunk_metadata: Source chunk metadata
        model_name: LLM model used
        confidence: Extraction confidence

    Returns:
        Created provenance record
    """
    tracker = get_provenance_tracker()
    return tracker.create_from_chunk_metadata(
        entity_id=entity_id,
        entity_type=entity_type,
        chunk_metadata=chunk_metadata,
        model_name=model_name,
        confidence=confidence,
    )


def get_entity_source(entity_id: str) -> dict[str, Any] | None:
    """
    Get source information for an entity.

    Args:
        entity_id: ID of entity

    Returns:
        Source info dict or None
    """
    tracker = get_provenance_tracker()
    records = tracker.get_provenance(entity_id)
    if records:
        return records[0].source.to_dict()
    return None
