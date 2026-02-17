"""
Continual Learning Orchestrator - Integrates all learning components.

This module provides a unified interface for the continual learning system,
orchestrating the flow between:

- Context Enrichment (before query)
- Feedback-Aware Retrieval (during query)
- Active Learning (after response)
- Audit-to-KG Pipeline (on errors)

Usage:
    orchestrator = get_continual_learning_orchestrator()

    # Process a query with full continual learning support
    result = await orchestrator.process_query(
        query="What is job ABC doing?",
        instance_id="tws-prod-01",
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from resync.core.continual_learning.active_learning import (
    ActiveLearningDecision,
    get_active_learning_manager,
)
from resync.core.continual_learning.audit_to_kg_pipeline import (
    AuditResult,
    get_audit_to_kg_pipeline,
)
from resync.core.continual_learning.context_enrichment import EnrichmentResult, get_context_enricher
from resync.core.continual_learning.feedback_retriever import FeedbackAwareRetriever
from resync.core.continual_learning.feedback_store import get_feedback_store
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContinualLearningResult:
    """Result from continual learning orchestration."""

    # Query info
    original_query: str
    enriched_query: str

    # Retrieval info
    documents_retrieved: int
    top_similarity_score: float
    feedback_adjustments_applied: int

    # Classification info
    classification_confidence: float
    detected_intent: str | None = None

    # Active learning
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    warning_message: str | None = None
    review_id: str | None = None

    # Context
    enrichment_context: list[str] = field(default_factory=list)
    entities_found: dict[str, list[str]] = field(default_factory=dict)

    # Metadata
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "enriched_query": self.enriched_query,
            "documents_retrieved": self.documents_retrieved,
            "top_similarity_score": self.top_similarity_score,
            "feedback_adjustments_applied": self.feedback_adjustments_applied,
            "classification_confidence": self.classification_confidence,
            "detected_intent": self.detected_intent,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "warning_message": self.warning_message,
            "review_id": self.review_id,
            "enrichment_context": self.enrichment_context,
            "entities_found": self.entities_found,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ContinualLearningOrchestrator:
    """
    Orchestrates all continual learning components.

    Provides a unified interface for:
    1. Pre-processing (context enrichment)
    2. Retrieval (feedback-aware)
    3. Post-processing (active learning check)
    4. Learning (feedback recording, audit pipeline)
    """

    def __init__(
        self,
        feedback_retriever: FeedbackAwareRetriever | None = None,
        enable_enrichment: bool = True,
        enable_active_learning: bool = True,
        enable_audit_pipeline: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            feedback_retriever: Pre-configured feedback-aware retriever
            enable_enrichment: Enable context enrichment
            enable_active_learning: Enable active learning checks
            enable_audit_pipeline: Enable audit-to-KG pipeline
        """
        self._feedback_retriever = feedback_retriever
        self.enable_enrichment = enable_enrichment
        self.enable_active_learning = enable_active_learning
        self.enable_audit_pipeline = enable_audit_pipeline

        # Lazy-loaded components
        self._context_enricher = None
        self._active_learning_manager = None
        self._audit_pipeline = None
        self._feedback_store = None

    def _get_context_enricher(self):
        if self._context_enricher is None:
            self._context_enricher = get_context_enricher()
        return self._context_enricher

    def _get_active_learning_manager(self):
        if self._active_learning_manager is None:
            self._active_learning_manager = get_active_learning_manager()
        return self._active_learning_manager

    def _get_audit_pipeline(self):
        if self._audit_pipeline is None:
            self._audit_pipeline = get_audit_to_kg_pipeline()
        return self._audit_pipeline

    def _get_feedback_store(self):
        if self._feedback_store is None:
            self._feedback_store = get_feedback_store()
        return self._feedback_store

    async def enrich_query(
        self,
        query: str,
        instance_id: str | None = None,
    ) -> EnrichmentResult:
        """
        Step 1: Enrich query with learned context.
        """
        if not self.enable_enrichment:
            return EnrichmentResult(
                original_query=query,
                enriched_query=query,
                context_added=[],
                entities_found={},
                enrichment_source="disabled",
            )

        enricher = self._get_context_enricher()
        return await enricher.enrich_query(query, instance_id)

    async def check_active_learning(
        self,
        query: str,
        response: str,
        classification_confidence: float,
        rag_similarity_score: float,
        entities_found: dict[str, list[str]],
    ) -> tuple[ActiveLearningDecision, str | None]:
        """
        Step 3: Check if response needs human review.

        Returns:
            Tuple of (decision, review_id if added to queue)
        """
        if not self.enable_active_learning:
            return ActiveLearningDecision(
                should_review=False,
                reasons=[],
                confidence_scores={},
                suggested_action="proceed_normally",
            ), None

        manager = self._get_active_learning_manager()

        decision = await manager.should_request_review(
            query=query,
            response=response,
            classification_confidence=classification_confidence,
            rag_similarity_score=rag_similarity_score,
            entities_found=entities_found,
        )

        review_id = None
        if decision.should_review:
            review_id = await manager.add_to_review_queue(
                query=query,
                response=response,
                reasons=decision.reasons,
                confidence_scores=decision.confidence_scores,
            )

        return decision, review_id

    async def record_feedback(
        self,
        query: str,
        doc_id: str,
        rating: int,
        user_id: str | None = None,
        response_text: str | None = None,
    ) -> str:
        """
        Step 4a: Record user feedback.
        """
        store = self._get_feedback_store()
        return await store.record_feedback(
            query=query,
            doc_id=doc_id,
            rating=rating,
            user_id=user_id,
            response_text=response_text,
        )

    async def process_audit_error(
        self,
        memory_id: str,
        user_query: str,
        agent_response: str,
        confidence: float,
        reason: str,
    ) -> dict[str, Any]:
        """
        Step 4b: Process an audit error and convert to knowledge.
        """
        if not self.enable_audit_pipeline:
            return {"status": "disabled"}

        pipeline = self._get_audit_pipeline()

        result = AuditResult(
            memory_id=memory_id,
            user_query=user_query,
            agent_response=agent_response,
            is_incorrect=True,
            confidence=confidence,
            reason=reason,
        )

        return await pipeline.process_audit_result(result)

    async def process_full_cycle(
        self,
        query: str,
        response: str,
        classification_confidence: float,
        rag_similarity_score: float,
        documents_retrieved: int,
        instance_id: str | None = None,
        user_id: str | None = None,
        detected_intent: str | None = None,
    ) -> ContinualLearningResult:
        """
        Process a complete query-response cycle through all learning stages.

        This is the main entry point for integrating continual learning
        into the response pipeline.
        """
        start_time = datetime.now(timezone.utc)

        # Step 1: Enrich query (already done by caller, but we can re-extract entities)
        enricher = self._get_context_enricher()
        entities = enricher.extract_entities(query)
        enrichment = await self.enrich_query(query, instance_id)

        # Step 2: Check active learning
        decision, review_id = await self.check_active_learning(
            query=query,
            response=response,
            classification_confidence=classification_confidence,
            rag_similarity_score=rag_similarity_score,
            entities_found=entities,
        )

        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        result = ContinualLearningResult(
            original_query=query,
            enriched_query=enrichment.enriched_query,
            documents_retrieved=documents_retrieved,
            top_similarity_score=rag_similarity_score,
            feedback_adjustments_applied=0,  # Would be filled by retriever
            classification_confidence=classification_confidence,
            detected_intent=detected_intent,
            needs_review=decision.should_review,
            review_reasons=[r.value for r in decision.reasons],
            warning_message=decision.warning_message,
            review_id=review_id,
            enrichment_context=enrichment.context_added,
            entities_found=entities,
            processing_time_ms=processing_time,
        )

        logger.info(
            "continual_learning_cycle_complete",
            query_len=len(query),
            needs_review=result.needs_review,
            enrichment_count=len(result.enrichment_context),
            processing_time_ms=round(processing_time, 2),
        )

        return result

    async def get_system_stats(self) -> dict[str, Any]:
        """Get statistics from all continual learning components."""
        stats = {}

        # Feedback stats
        store = self._get_feedback_store()
        await store.initialize()
        stats["feedback"] = await store.get_feedback_stats()

        # Active learning stats
        if self.enable_active_learning:
            manager = self._get_active_learning_manager()
            stats["active_learning"] = await manager.get_queue_stats()

        # Config
        stats["config"] = {
            "enrichment_enabled": self.enable_enrichment,
            "active_learning_enabled": self.enable_active_learning,
            "audit_pipeline_enabled": self.enable_audit_pipeline,
        }

        return stats


# Singleton instance
_orchestrator: ContinualLearningOrchestrator | None = None


def get_continual_learning_orchestrator() -> ContinualLearningOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContinualLearningOrchestrator()
    return _orchestrator


async def process_with_continual_learning(
    query: str,
    response: str,
    classification_confidence: float,
    rag_similarity_score: float,
    documents_retrieved: int = 0,
    instance_id: str | None = None,
) -> ContinualLearningResult:
    """
    Convenience function to process a query-response through continual learning.

    Use this in your response pipeline after generating a response.
    """
    orchestrator = get_continual_learning_orchestrator()
    return await orchestrator.process_full_cycle(
        query=query,
        response=response,
        classification_confidence=classification_confidence,
        rag_similarity_score=rag_similarity_score,
        documents_retrieved=documents_retrieved,
        instance_id=instance_id,
    )
