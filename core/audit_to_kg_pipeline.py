"""
Audit to Knowledge Graph Pipeline.

Converts audit findings into structured knowledge that enriches the
Knowledge Graph. When the IA auditor identifies errors, this pipeline:

1. Extracts error patterns as triplets
2. Adds "INCORRECT_ASSOCIATION" relationships to KG
3. Notifies RAG to penalize related documents
4. Creates learning opportunities from failures

Key Features:
- Automatic triplet extraction from errors
- Negative knowledge representation (what NOT to do)
- Feedback propagation to RAG
- Human review integration for complex cases
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class ErrorType(str, Enum):
    """Types of errors identified by audit."""

    TECHNICAL_INACCURACY = "technical_inaccuracy"
    IRRELEVANT_RESPONSE = "irrelevant_response"
    CONTRADICTORY_INFO = "contradictory_info"
    OUTDATED_INFO = "outdated_info"
    MISSING_CONTEXT = "missing_context"
    WRONG_RECOMMENDATION = "wrong_recommendation"
    HALLUCINATION = "hallucination"


@dataclass
class AuditFinding:
    """Represents an audit finding to be processed."""

    memory_id: str
    user_query: str
    agent_response: str
    reason: str
    confidence: float
    error_type: ErrorType | None = None
    extracted_entities: dict[str, list[str]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ErrorTriplet:
    """A triplet representing an error pattern."""

    subject: str
    subject_type: str
    predicate: str  # Usually "INCORRECT_FOR", "SHOULD_NOT", etc.
    object: str
    object_type: str
    error_reason: str
    confidence: float
    source_memory_id: str


class AuditToKGPipeline:
    """
    Pipeline that converts audit findings into Knowledge Graph knowledge.

    Process:
    1. Receive audit finding
    2. Classify error type
    3. Extract relevant entities
    4. Generate error triplets
    5. Add to KG as negative knowledge
    6. Notify RAG for document penalization
    """

    # Error patterns for classification
    ERROR_PATTERNS = {
        ErrorType.TECHNICAL_INACCURACY: [
            r"(?:incorrect|wrong|invalid)\s+(?:syntax|command|parameter)",
            r"(?:doesn't|does not)\s+(?:work|exist|apply)",
            r"technical(?:ly)?\s+(?:incorrect|wrong|inaccurate)",
        ],
        ErrorType.IRRELEVANT_RESPONSE: [
            r"(?:not|doesn't)\s+(?:related|relevant|answer)",
            r"(?:off[\s-]?topic|unrelated)",
            r"(?:didn't|did not)\s+address",
        ],
        ErrorType.CONTRADICTORY_INFO: [
            r"contradict(?:s|ory|ion)",
            r"inconsistent\s+with",
            r"conflicts?\s+with",
        ],
        ErrorType.WRONG_RECOMMENDATION: [
            r"(?:wrong|bad|incorrect)\s+(?:suggestion|recommendation|advice)",
            r"should(?:n't| not)\s+(?:suggest|recommend)",
            r"(?:inappropriate|unsuitable)\s+(?:solution|approach)",
        ],
        ErrorType.HALLUCINATION: [
            r"(?:made|making)\s+up",
            r"(?:doesn't|does not)\s+exist",
            r"fabricat(?:ed|ion)",
            r"(?:non[\s-]?existent|fictional)",
        ],
    }

    # TWS-specific entity patterns
    ENTITY_PATTERNS = {
        "job": [
            r"\bjob[:\s]+([A-Z][A-Z0-9_]+)\b",
            r"\b([A-Z][A-Z0-9_]{2,})\s+(?:job|batch|process)\b",
        ],
        "workstation": [
            r"\b(?:workstation|ws|server|agent)[:\s]+([A-Z0-9_]+)\b",
            r"\b([A-Z]{2,3}\d{3,})\b",
        ],
        "error_code": [
            r"\b(?:error|code|rc)[:\s]+(\d{4,})\b",
            r"\bAWS[A-Z]{3}\d{4}[A-Z]?\b",
            r"\b[A-Z]{3,4}\d{3,4}\b",
        ],
        "command": [
            r"(?:conman|optman|planman|composer)\s+['\"]?(\w+)['\"]?",
            r"(?:run|execute|submit)\s+['\"]?([A-Za-z0-9_]+)['\"]?",
        ],
    }

    def __init__(self):
        self._kg = None
        self._rag_feedback = None
        self._llm = None
        self._processing_lock: asyncio.Lock | None = None

        # Statistics
        self._processed_count = 0
        self._triplets_created = 0
        self._errors_by_type: dict[ErrorType, int] = {}

    @property
    def processing_lock(self) -> asyncio.Lock:
        """Lazy initialization of processing lock."""
        if self._processing_lock is None:
            self._processing_lock = asyncio.Lock()
        return self._processing_lock

    async def _get_kg(self):
        """Get Knowledge Graph instance."""
        if self._kg is None:
            from resync.knowledge.retrieval.graph import get_kg_instance

            self._kg = await get_kg_instance()
        return self._kg

    async def _get_rag_feedback(self):
        """Get RAG feedback store."""
        if self._rag_feedback is None:
            from resync.knowledge.store.feedback_store import get_feedback_store

            self._rag_feedback = get_feedback_store()
            await self._rag_feedback.initialize()
        return self._rag_feedback

    def _get_llm(self):
        """Get LLM client."""
        if self._llm is None:
            from resync.services.llm_service import get_llm_service

            self._llm = get_llm_service()
        return self._llm

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    async def process_audit_finding(
        self,
        memory_id: str,
        user_query: str,
        agent_response: str,
        reason: str,
        confidence: float,
    ) -> dict[str, Any]:
        """
        Process an audit finding and convert to KG knowledge.

        Args:
            memory_id: ID of the audited memory
            user_query: Original user query
            agent_response: Agent's response that was flagged
            reason: Reason for flagging
            confidence: Audit confidence (0.0-1.0)

        Returns:
            Processing result with created triplets and actions
        """
        async with self.processing_lock:
            finding = AuditFinding(
                memory_id=memory_id,
                user_query=user_query,
                agent_response=agent_response,
                reason=reason,
                confidence=confidence,
            )

            # Step 1: Classify error type
            finding.error_type = self._classify_error_type(reason)

            # Step 2: Extract entities
            finding.extracted_entities = self._extract_entities(
                user_query + " " + agent_response + " " + reason
            )

            # Step 3: Generate triplets
            triplets = await self._generate_error_triplets(finding)

            # Step 4: Add to KG
            kg_results = await self._add_triplets_to_kg(triplets)

            # Step 5: Notify RAG
            rag_results = await self._penalize_rag_documents(finding)

            # Update statistics
            self._processed_count += 1
            self._triplets_created += len(triplets)
            self._errors_by_type[finding.error_type] = (
                self._errors_by_type.get(finding.error_type, 0) + 1
            )

            logger.info(
                "audit_finding_processed",
                memory_id=memory_id,
                error_type=finding.error_type.value if finding.error_type else "unknown",
                triplets_created=len(triplets),
                entities_found=sum(len(v) for v in finding.extracted_entities.values()),
            )

            return {
                "memory_id": memory_id,
                "error_type": finding.error_type.value if finding.error_type else None,
                "entities": finding.extracted_entities,
                "triplets_created": len(triplets),
                "triplets": [self._triplet_to_dict(t) for t in triplets],
                "kg_results": kg_results,
                "rag_penalized": rag_results,
            }

    # =========================================================================
    # ERROR CLASSIFICATION
    # =========================================================================

    def _classify_error_type(self, reason: str) -> ErrorType:
        """Classify error type based on audit reason."""
        reason_lower = reason.lower()

        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, reason_lower, re.IGNORECASE):
                    return error_type

        # Default to technical inaccuracy
        return ErrorType.TECHNICAL_INACCURACY

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract TWS-related entities from text."""
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found.update(m.upper() if isinstance(m, str) else m[0].upper() for m in matches)
            if found:
                entities[entity_type] = list(found)

        return entities

    # =========================================================================
    # TRIPLET GENERATION
    # =========================================================================

    async def _generate_error_triplets(self, finding: AuditFinding) -> list[ErrorTriplet]:
        """Generate error triplets from audit finding."""
        triplets: list[ErrorTriplet] = []

        # Generate triplets based on error type and entities
        if finding.error_type == ErrorType.WRONG_RECOMMENDATION:
            triplets.extend(self._generate_recommendation_triplets(finding))

        if finding.error_type == ErrorType.TECHNICAL_INACCURACY:
            triplets.extend(self._generate_technical_triplets(finding))

        if finding.error_type == ErrorType.IRRELEVANT_RESPONSE:
            triplets.extend(self._generate_relevance_triplets(finding))

        # Use LLM for complex cases with high confidence
        if finding.confidence > 0.8 and len(triplets) < 2:
            llm_triplets = await self._generate_triplets_with_llm(finding)
            triplets.extend(llm_triplets)

        return triplets

    def _generate_recommendation_triplets(self, finding: AuditFinding) -> list[ErrorTriplet]:
        """Generate triplets for wrong recommendation errors."""
        triplets = []

        jobs = finding.extracted_entities.get("job", [])
        commands = finding.extracted_entities.get("command", [])
        error_codes = finding.extracted_entities.get("error_code", [])

        # Job + Error Code = Wrong recommendation
        for job in jobs:
            for error_code in error_codes:
                triplets.append(
                    ErrorTriplet(
                        subject=job,
                        subject_type="job",
                        predicate="INCORRECT_SOLUTION_FOR",
                        object=error_code,
                        object_type="error_code",
                        error_reason=finding.reason,
                        confidence=finding.confidence,
                        source_memory_id=finding.memory_id,
                    )
                )

        # Command + Error = Wrong approach
        for command in commands:
            for error_code in error_codes:
                triplets.append(
                    ErrorTriplet(
                        subject=command,
                        subject_type="command",
                        predicate="SHOULD_NOT_USE_FOR",
                        object=error_code,
                        object_type="error_code",
                        error_reason=finding.reason,
                        confidence=finding.confidence,
                        source_memory_id=finding.memory_id,
                    )
                )

        return triplets

    def _generate_technical_triplets(self, finding: AuditFinding) -> list[ErrorTriplet]:
        """Generate triplets for technical inaccuracy errors."""
        triplets = []

        jobs = finding.extracted_entities.get("job", [])
        workstations = finding.extracted_entities.get("workstation", [])
        commands = finding.extracted_entities.get("command", [])

        # Job + Workstation = Incorrect association
        for job in jobs:
            for ws in workstations:
                triplets.append(
                    ErrorTriplet(
                        subject=job,
                        subject_type="job",
                        predicate="INCORRECT_ASSOCIATION",
                        object=ws,
                        object_type="workstation",
                        error_reason=finding.reason,
                        confidence=finding.confidence,
                        source_memory_id=finding.memory_id,
                    )
                )

        # Command + Job = Incorrect usage
        for command in commands:
            for job in jobs:
                triplets.append(
                    ErrorTriplet(
                        subject=command,
                        subject_type="command",
                        predicate="INCORRECTLY_APPLIED_TO",
                        object=job,
                        object_type="job",
                        error_reason=finding.reason,
                        confidence=finding.confidence,
                        source_memory_id=finding.memory_id,
                    )
                )

        return triplets

    def _generate_relevance_triplets(self, finding: AuditFinding) -> list[ErrorTriplet]:
        """Generate triplets for irrelevant response errors."""
        triplets = []

        # Find entities mentioned in response but not query
        for entity_type, entities in finding.extracted_entities.items():
            for entity in entities:
                if entity.lower() not in finding.user_query.lower():
                    triplets.append(
                        ErrorTriplet(
                            subject=entity,
                            subject_type=entity_type,
                            predicate="NOT_RELEVANT_TO",
                            object=finding.user_query[:100],  # Truncate query
                            object_type="query",
                            error_reason=finding.reason,
                            confidence=finding.confidence,
                            source_memory_id=finding.memory_id,
                        )
                    )

        return triplets

    async def _generate_triplets_with_llm(self, finding: AuditFinding) -> list[ErrorTriplet]:
        """Use LLM to extract additional triplets."""
        try:
            llm = self._get_llm()

            prompt = f"""Analyze this error identified by an AI auditor in a TWS (Workload Automation) system.

User Query: "{finding.user_query}"
Agent Response: "{finding.agent_response}"
Error Reason: "{finding.reason}"
Error Type: {finding.error_type.value if finding.error_type else "unknown"}

Extract INCORRECT associations that should be avoided.
Return JSON array of triplets:
[
    {{
        "subject": "entity name",
        "subject_type": "job|workstation|command|error_code",
        "predicate": "INCORRECT_FOR|SHOULD_NOT|WRONG_ASSOCIATION",
        "object": "related entity",
        "object_type": "job|workstation|command|error_code|concept"
    }}
]

Only return triplets you're confident about. Max 3 triplets.
Return only valid JSON, no markdown."""

            response = await llm.generate(
                prompt,
                max_tokens=500,
                temperature=0.1,  # Low temp for consistency
            )

            # Parse response
            triplet_data = json.loads(response.strip())

            triplets = []
            for t in triplet_data[:3]:  # Max 3
                triplets.append(
                    ErrorTriplet(
                        subject=t["subject"],
                        subject_type=t.get("subject_type", "concept"),
                        predicate=t["predicate"],
                        object=t["object"],
                        object_type=t.get("object_type", "concept"),
                        error_reason=finding.reason,
                        confidence=finding.confidence * 0.9,  # Slight discount for LLM
                        source_memory_id=finding.memory_id,
                    )
                )

            return triplets

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("llm_triplet_extraction_failed", error=str(e))
            return []

    # =========================================================================
    # KG INTEGRATION
    # =========================================================================

    async def _add_triplets_to_kg(self, triplets: list[ErrorTriplet]) -> dict[str, Any]:
        """Add error triplets to Knowledge Graph."""
        kg = await self._get_kg()

        added = 0
        failed = 0

        for triplet in triplets:
            try:
                # Add nodes if they don't exist
                await kg.add_node(
                    node_id=triplet.subject,
                    node_type=triplet.subject_type,
                    properties={"source": "audit_pipeline"},
                )

                await kg.add_node(
                    node_id=triplet.object,
                    node_type=triplet.object_type,
                    properties={"source": "audit_pipeline"},
                )

                # Add error edge with metadata
                await kg.add_edge(
                    source=triplet.subject,
                    target=triplet.object,
                    relation_type=triplet.predicate,
                    properties={
                        "is_error_knowledge": True,
                        "error_reason": triplet.error_reason,
                        "confidence": triplet.confidence,
                        "source_memory_id": triplet.source_memory_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                added += 1

            except Exception as e:
                logger.warning(
                    "triplet_add_failed",
                    subject=triplet.subject,
                    predicate=triplet.predicate,
                    object=triplet.object,
                    error=str(e),
                )
                failed += 1

        return {"added": added, "failed": failed}

    # =========================================================================
    # RAG INTEGRATION
    # =========================================================================

    async def _penalize_rag_documents(self, finding: AuditFinding) -> dict[str, Any]:
        """Penalize RAG documents related to the error."""
        try:
            rag_feedback = await self._get_rag_feedback()

            # Record negative feedback for the query
            # This will affect future retrievals for similar queries
            penalized = 0

            # Extract document references from response if any
            doc_patterns = [
                r"document[:\s]+([A-Za-z0-9_]+)",
                r"source[:\s]+([A-Za-z0-9_]+)",
                r"(?:from|see)\s+([A-Za-z0-9_]+\.(?:md|pdf|txt))",
            ]

            doc_refs = set()
            for pattern in doc_patterns:
                matches = re.findall(pattern, finding.agent_response, re.IGNORECASE)
                doc_refs.update(matches)

            # If no specific docs found, use a synthetic ID based on entities
            if not doc_refs:
                for entity_type, entities in finding.extracted_entities.items():
                    for entity in entities[:3]:  # Max 3 per type
                        doc_refs.add(f"entity_{entity_type}_{entity}")

            # Record negative feedback
            for doc_ref in doc_refs:
                success = await rag_feedback.record_feedback(
                    query=finding.user_query,
                    doc_id=doc_ref,
                    rating=-1,  # Negative
                    user_id="audit_pipeline",
                )
                if success:
                    penalized += 1

            return {"documents_penalized": penalized, "doc_refs": list(doc_refs)}

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("rag_penalization_failed", error=str(e))
            return {"documents_penalized": 0, "error": str(e)}

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _triplet_to_dict(self, triplet: ErrorTriplet) -> dict[str, Any]:
        """Convert triplet to dictionary."""
        return {
            "subject": triplet.subject,
            "subject_type": triplet.subject_type,
            "predicate": triplet.predicate,
            "object": triplet.object,
            "object_type": triplet.object_type,
            "error_reason": triplet.error_reason,
            "confidence": triplet.confidence,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "findings_processed": self._processed_count,
            "triplets_created": self._triplets_created,
            "errors_by_type": {k.value: v for k, v in self._errors_by_type.items()},
            "avg_triplets_per_finding": (
                self._triplets_created / self._processed_count if self._processed_count > 0 else 0.0
            ),
        }


# Global instance
_pipeline: AuditToKGPipeline | None = None


def get_audit_kg_pipeline() -> AuditToKGPipeline:
    """Get the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AuditToKGPipeline()
    return _pipeline


# =========================================================================
# INTEGRATION WITH IA_AUDITOR
# =========================================================================


async def process_audit_result_to_kg(
    memory_id: str,
    user_query: str,
    agent_response: str,
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    """
    Convenience function to process an audit result.

    Call this from ia_auditor when an error is detected.
    """
    pipeline = get_audit_kg_pipeline()
    return await pipeline.process_audit_finding(
        memory_id=memory_id,
        user_query=user_query,
        agent_response=agent_response,
        reason=reason,
        confidence=confidence,
    )
