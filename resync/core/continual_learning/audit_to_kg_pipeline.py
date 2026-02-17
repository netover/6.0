"""
Audit to Knowledge Graph Pipeline.

Converte insights de auditoria em conhecimento estruturado no Knowledge Graph.
Quando o IA Auditor identifica erros, este pipeline:

1. Extrai padrões de erro (triplets)
2. Adiciona ao KG como "INCORRECT_ASSOCIATION" ou "COMMON_ERROR"
3. Notifica RAG para penalizar documentos relacionados
4. Cria knowledge entries para evitar erros futuros
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from resync.core.continual_learning.feedback_store import get_feedback_store
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class ErrorRelationType(str, Enum):
    """Types of error relationships in the KG."""

    INCORRECT_ASSOCIATION = "INCORRECT_ASSOCIATION"
    COMMON_ERROR = "COMMON_ERROR"
    CONFUSION_WITH = "CONFUSION_WITH"
    MISLEADING_CONTEXT = "MISLEADING_CONTEXT"
    DEPRECATED_INFO = "DEPRECATED_INFO"


@dataclass
class AuditResult:
    """Result from IA Auditor analysis."""

    memory_id: str
    user_query: str
    agent_response: str
    is_incorrect: bool
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entities_mentioned: list[str] = field(default_factory=list)
    error_type: str | None = None
    suggested_correction: str | None = None


@dataclass
class ErrorTriplet:
    """Triplet representing an error pattern."""

    subject: str
    subject_type: str
    predicate: ErrorRelationType
    object: str
    object_type: str
    confidence: float
    error_reason: str
    source_query: str
    source_memory_id: str


class AuditToKGPipeline:
    """
    Pipeline that converts audit findings into Knowledge Graph entries.

    Workflow:
    1. Receive audit result (incorrect response identified)
    2. Extract entities and relationships from the error
    3. Create triplets representing the error pattern
    4. Add to Knowledge Graph as negative associations
    5. Notify feedback store to penalize related docs
    """

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        feedback_store: Any | None = None,
        auto_penalize_rag: bool = True,
        min_confidence_for_kg: float = 0.7,
    ):
        """
        Initialize the pipeline.

        Args:
            knowledge_graph: KG instance (lazy loaded if None)
            feedback_store: Feedback store (uses default if None)
            auto_penalize_rag: Whether to auto-penalize RAG docs
            min_confidence_for_kg: Minimum confidence to add to KG
        """
        self._kg = knowledge_graph
        self._feedback_store = feedback_store
        self.auto_penalize_rag = auto_penalize_rag
        self.min_confidence_for_kg = min_confidence_for_kg

        # Entity extraction patterns (TWS domain)
        self._entity_patterns = {
            "job": [
                r"\bjob[:\s]+([A-Z][A-Z0-9_]+)\b",
                r"\b([A-Z][A-Z0-9_]{2,})\s+(?:job|batch|process)\b",
                r"(?:job|processo)\s+(?:chamado|named)\s+['\"]?([A-Z][A-Z0-9_]+)['\"]?",
            ],
            "workstation": [
                r"\b(?:workstation|ws|server|agent|agente)[:\s]+([A-Z0-9_]+)\b",
                r"\b([A-Z]{2,3}\d{3,})\b",
            ],
            "error_code": [
                r"\b(AWSB[A-Z0-9]+)\b",
                r"\b(ERR[_-]?\d+)\b",
                r"\berror\s+(?:code\s+)?(\d{4,})\b",
            ],
            "resource": [
                r"\bresource[:\s]+([A-Z0-9_]+)\b",
                r"\b(?:file|arquivo|database|queue|lock)[:\s]+([A-Z0-9_./]+)\b",
            ],
        }

    async def _get_kg(self):
        """Get or create Knowledge Graph instance."""
        if self._kg is None:
            from resync.knowledge.retrieval.graph import get_knowledge_graph

            # get_knowledge_graph may or may not be async
            kg = get_knowledge_graph()
            # If it's a coroutine, await it
            if hasattr(kg, "__await__"):
                self._kg = await kg
            else:
                self._kg = kg
        return self._kg

    def _get_feedback_store(self):
        """Get feedback store instance."""
        if self._feedback_store is None:
            self._feedback_store = get_feedback_store()
        return self._feedback_store

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract entities from text using patterns."""
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in self._entity_patterns.items():
            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found.update(m.upper() for m in matches)
            if found:
                entities[entity_type] = list(found)

        return entities

    def _classify_error_type(self, reason: str) -> ErrorRelationType:
        """Classify the type of error based on reason."""
        reason_lower = reason.lower()

        if any(w in reason_lower for w in ["confus", "mix", "wrong", "errado"]):
            return ErrorRelationType.CONFUSION_WITH

        if any(
            w in reason_lower
            for w in ["deprecated", "obsolete", "obsoleto", "old", "antigo", "outdated"]
        ):
            return ErrorRelationType.DEPRECATED_INFO

        if any(w in reason_lower for w in ["mislead", "enganos", "context"]):
            return ErrorRelationType.MISLEADING_CONTEXT

        if any(w in reason_lower for w in ["common", "frequent", "recurrent"]):
            return ErrorRelationType.COMMON_ERROR

        return ErrorRelationType.INCORRECT_ASSOCIATION

    async def process_audit_result(
        self,
        audit_result: AuditResult,
    ) -> dict[str, Any]:
        """
        Process an audit result and convert to KG knowledge.

        Args:
            audit_result: The audit finding to process

        Returns:
            Summary of actions taken
        """
        if not audit_result.is_incorrect:
            return {"status": "skipped", "reason": "not_incorrect"}

        if audit_result.confidence < self.min_confidence_for_kg:
            return {
                "status": "skipped",
                "reason": "low_confidence",
                "confidence": audit_result.confidence,
            }

        logger.info(
            "processing_audit_result",
            memory_id=audit_result.memory_id,
            confidence=audit_result.confidence,
        )

        # Extract triplets from the error
        triplets = self._extract_error_triplets(audit_result)

        # Add triplets to Knowledge Graph
        kg_entries_added = 0
        if triplets:
            kg_entries_added = await self._add_triplets_to_kg(triplets)

        # Penalize related documents in RAG
        docs_penalized = 0
        if self.auto_penalize_rag:
            docs_penalized = await self._penalize_rag_documents(audit_result)

        result = {
            "status": "processed",
            "memory_id": audit_result.memory_id,
            "triplets_extracted": len(triplets),
            "kg_entries_added": kg_entries_added,
            "docs_penalized": docs_penalized,
            "error_type": self._classify_error_type(audit_result.reason).value,
        }

        logger.info("audit_to_kg_complete", **result)

        return result

    def _extract_error_triplets(
        self,
        audit_result: AuditResult,
    ) -> list[ErrorTriplet]:
        """Extract error triplets from audit result."""
        triplets = []

        # Extract entities from query and response
        query_entities = self._extract_entities(audit_result.user_query)
        response_entities = self._extract_entities(audit_result.agent_response)

        # Classify error type
        error_type = self._classify_error_type(audit_result.reason)

        # Create triplets based on error patterns

        # Pattern 1: Job mentioned in query, wrong info in response
        query_jobs = query_entities.get("job", [])
        response_resources = response_entities.get("resource", [])
        response_workstations = response_entities.get("workstation", [])

        for job in query_jobs:
            # Job incorrectly associated with resources
            for resource in response_resources:
                triplets.append(
                    ErrorTriplet(
                        subject=job,
                        subject_type="job",
                        predicate=error_type,
                        object=resource,
                        object_type="resource",
                        confidence=audit_result.confidence,
                        error_reason=audit_result.reason,
                        source_query=audit_result.user_query,
                        source_memory_id=audit_result.memory_id,
                    )
                )

            # Job incorrectly associated with workstations
            for ws in response_workstations:
                triplets.append(
                    ErrorTriplet(
                        subject=job,
                        subject_type="job",
                        predicate=error_type,
                        object=ws,
                        object_type="workstation",
                        confidence=audit_result.confidence,
                        error_reason=audit_result.reason,
                        source_query=audit_result.user_query,
                        source_memory_id=audit_result.memory_id,
                    )
                )

        # Pattern 2: Error code with wrong resolution
        error_codes = query_entities.get("error_code", []) or response_entities.get(
            "error_code", []
        )

        for error_code in error_codes:
            triplets.append(
                ErrorTriplet(
                    subject=error_code,
                    subject_type="error_code",
                    predicate=ErrorRelationType.INCORRECT_ASSOCIATION,
                    object=self._summarize_response(audit_result.agent_response),
                    object_type="resolution",
                    confidence=audit_result.confidence,
                    error_reason=audit_result.reason,
                    source_query=audit_result.user_query,
                    source_memory_id=audit_result.memory_id,
                )
            )

        # Pattern 3: General error pattern (if no specific entities found)
        if not triplets:
            # Create a general error triplet
            triplets.append(
                ErrorTriplet(
                    subject=self._normalize_query(audit_result.user_query),
                    subject_type="query_pattern",
                    predicate=error_type,
                    object=self._summarize_response(audit_result.agent_response),
                    object_type="response_pattern",
                    confidence=audit_result.confidence,
                    error_reason=audit_result.reason,
                    source_query=audit_result.user_query,
                    source_memory_id=audit_result.memory_id,
                )
            )

        return triplets

    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        # Remove specific values, keep structure
        normalized = query.lower()
        normalized = re.sub(r"\b[A-Z][A-Z0-9_]{2,}\b", "<ENTITY>", normalized)
        normalized = re.sub(r"\d+", "<NUM>", normalized)
        return normalized[:100]

    def _summarize_response(self, response: str) -> str:
        """Create a short summary of response."""
        # Take first sentence or first 100 chars
        sentences = response.split(".")
        if sentences:
            return sentences[0][:100]
        return response[:100]

    async def _add_triplets_to_kg(
        self,
        triplets: list[ErrorTriplet],
    ) -> int:
        """Add error triplets to Knowledge Graph."""
        kg = await self._get_kg()
        added = 0

        for triplet in triplets:
            try:
                # Create unique ID for the error relationship
                hashlib.sha256(
                    f"{triplet.subject}:{triplet.predicate}:{triplet.object}:{triplet.source_memory_id}".encode()
                ).hexdigest()[:16]

                # Add as a special "error knowledge" edge
                # First ensure nodes exist
                await kg.add_node(
                    node_id=triplet.subject,
                    node_type=triplet.subject_type,
                    properties={"from_audit": True},
                )

                await kg.add_node(
                    node_id=triplet.object,
                    node_type=triplet.object_type,
                    properties={"from_audit": True},
                )

                # Add the error edge
                await kg.add_edge(
                    source=triplet.subject,
                    target=triplet.object,
                    edge_type=triplet.predicate.value,
                    properties={
                        "error_reason": triplet.error_reason,
                        "confidence": triplet.confidence,
                        "source_query": triplet.source_query,
                        "source_memory_id": triplet.source_memory_id,
                        "is_error_knowledge": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                added += 1

            except Exception as e:
                logger.warning(
                    "failed_to_add_triplet",
                    subject=triplet.subject,
                    error=str(e),
                )

        return added

    async def _penalize_rag_documents(
        self,
        audit_result: AuditResult,
    ) -> int:
        """Penalize RAG documents related to the error."""
        feedback_store = self._get_feedback_store()

        # We don't have direct document IDs from audit
        # But we can create a penalty entry that will affect
        # documents retrieved for similar queries

        # Create a special "audit penalty" feedback
        await feedback_store.record_feedback(
            query=audit_result.user_query,
            doc_id=f"audit_penalty_{audit_result.memory_id}",
            rating=-2,  # Strong negative
            user_id="system:audit_pipeline",
            response_text=audit_result.agent_response,
            metadata={
                "audit_reason": audit_result.reason,
                "audit_confidence": audit_result.confidence,
                "is_audit_penalty": True,
            },
        )

        # The feedback store will now penalize similar queries
        return 1

    async def get_error_patterns(
        self,
        entity: str | None = None,
        error_type: ErrorRelationType | None = None,
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve known error patterns from KG.

        Used by QueryRouter to avoid known bad associations.
        """
        kg = await self._get_kg()

        # Query KG for error edges
        patterns = []

        # Get all error-type edges
        error_edge_types = [e.value for e in ErrorRelationType]

        try:
            stats = await kg.get_statistics()

            # For now, return basic stats about error knowledge
            # Full implementation would query the graph for error edges
            patterns.append(
                {
                    "total_error_patterns": stats.get("edges", 0),
                    "error_types": error_edge_types,
                }
            )

        except Exception as e:
            logger.warning("failed_to_get_error_patterns", error=str(e))

        return patterns

    async def should_warn_about_query(
        self,
        query: str,
    ) -> tuple[bool, str | None]:
        """
        Check if a query is similar to known error patterns.

        Used by the response generator to add warnings.
        """
        # Extract entities from query
        entities = self._extract_entities(query)

        if not entities:
            return False, None

        # Check if any entity has known error associations
        await self._get_kg()

        for _entity_type, entity_list in entities.items():
            for _entity in entity_list:
                # Check if entity has INCORRECT_ASSOCIATION edges
                try:
                    # This would query the graph for error edges
                    # For now, return False (no known issues)
                    pass
                except Exception as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        return False, None


# Singleton instance
_pipeline: AuditToKGPipeline | None = None


def get_audit_to_kg_pipeline() -> AuditToKGPipeline:
    """Get global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AuditToKGPipeline()
    return _pipeline


async def process_audit_finding(
    memory_id: str,
    user_query: str,
    agent_response: str,
    is_incorrect: bool,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    """
    Convenience function to process an audit finding.

    Called by ia_auditor when it identifies an issue.
    """
    pipeline = get_audit_to_kg_pipeline()

    result = AuditResult(
        memory_id=memory_id,
        user_query=user_query,
        agent_response=agent_response,
        is_incorrect=is_incorrect,
        confidence=confidence,
        reason=reason,
    )

    return await pipeline.process_audit_result(result)
