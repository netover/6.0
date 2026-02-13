"""
Continual Learning Engine - Unified Integration Layer.

This module integrates all the continual learning components:
1. Feedback Loop RAG - Learning from user feedback
2. Audit → KG Pipeline - Converting errors to knowledge
3. Context Enrichment - Enhancing queries with learned context
4. Active Learning - Human-in-the-loop for uncertain cases

The engine provides a unified interface for:
- Processing queries with all enhancements
- Recording feedback and propagating learning
- Managing the review queue
- Monitoring system health and improvement

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  Continual Learning Engine                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────────┐                                         │
│  │ Context Enrichment │ ◄── Learning Store + KG                 │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                         │
│  │ Feedback-Aware RAG │ ◄── Feedback Store                      │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                         │
│  │  Response + Score  │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│       ┌────┴────┐                                               │
│       │         │                                                │
│       ▼         ▼                                                │
│  ┌─────────┐  ┌────────────────┐                                │
│  │ Feedback│  │ Active Learning│ ──► Review Queue               │
│  │ Record  │  │ Evaluation     │                                │
│  └────┬────┘  └────────────────┘                                │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────┐                                         │
│  │ Audit Pipeline     │ ──► KG + RAG Penalization               │
│  └────────────────────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of query processing through the continual learning engine."""

    # Query info
    original_query: str
    enriched_query: str
    query_embedding: list[float] | None = None

    # RAG results
    rag_results: list[dict[str, Any]] = field(default_factory=list)
    rag_top_score: float = 0.0
    feedback_applied: bool = False

    # Classification
    classification_confidence: float = 0.0
    entities_found: dict[str, list[str]] = field(default_factory=dict)

    # Context enrichment
    enrichment_applied: bool = False
    enrichment_types: list[str] = field(default_factory=list)

    # Active learning
    review_requested: bool = False
    review_reasons: list[str] = field(default_factory=list)
    review_priority: str | None = None

    # Metadata
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "enriched_query": self.enriched_query,
            "rag_top_score": self.rag_top_score,
            "feedback_applied": self.feedback_applied,
            "classification_confidence": self.classification_confidence,
            "entities_found": self.entities_found,
            "enrichment_applied": self.enrichment_applied,
            "enrichment_types": self.enrichment_types,
            "review_requested": self.review_requested,
            "review_reasons": self.review_reasons,
            "review_priority": self.review_priority,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeedbackResult:
    """Result of feedback processing."""

    success: bool
    feedback_stored: bool = False
    audit_triggered: bool = False
    kg_updated: bool = False
    rag_updated: bool = False
    message: str = ""


class ContinualLearningEngine:
    """
    Unified engine for continual learning in the Resync system.

    Integrates:
    - Context Enrichment (query enhancement)
    - Feedback-Aware RAG (learning from feedback)
    - Audit → KG Pipeline (error → knowledge)
    - Active Learning (human-in-the-loop)
    """

    def __init__(
        self,
        enable_enrichment: bool = True,
        enable_feedback_rag: bool = True,
        enable_active_learning: bool = True,
        enable_audit_pipeline: bool = True,
        instance_id: str = "default",
    ):
        """
        Initialize the continual learning engine.

        Args:
            enable_enrichment: Enable context enrichment
            enable_feedback_rag: Enable feedback-aware retrieval
            enable_active_learning: Enable active learning
            enable_audit_pipeline: Enable audit-to-KG pipeline
            instance_id: TWS instance ID for learning store
        """
        self.enable_enrichment = enable_enrichment
        self.enable_feedback_rag = enable_feedback_rag
        self.enable_active_learning = enable_active_learning
        self.enable_audit_pipeline = enable_audit_pipeline
        self.instance_id = instance_id

        # Components (lazy loaded)
        self._enricher = None
        self._feedback_retriever = None
        self._active_learning = None
        self._audit_pipeline = None
        self._embedder = None

        # Statistics
        self._queries_processed = 0
        self._feedback_recorded = 0
        self._reviews_triggered = 0
        self._audit_processes = 0

    async def _get_enricher(self):
        """Get context enricher."""
        if self._enricher is None:
            from resync.core.context_enrichment import get_context_enricher

            self._enricher = get_context_enricher()
        return self._enricher

    async def _get_feedback_retriever(self):
        """Get feedback-aware retriever."""
        if self._feedback_retriever is None:
            from resync.knowledge.ingestion.embedding_service import get_embedder
            from resync.knowledge.retrieval.feedback_retriever import (
                create_feedback_aware_retriever,
            )
            from resync.knowledge.store.pgvector_store import get_vector_store

            embedder = get_embedder()
            store = get_vector_store()
            self._feedback_retriever = create_feedback_aware_retriever(
                embedder=embedder,
                store=store,
                adaptive=True,
            )
        return self._feedback_retriever

    async def _get_active_learning(self):
        """Get active learning manager."""
        if self._active_learning is None:
            from resync.core.active_learning import get_active_learning_manager

            self._active_learning = get_active_learning_manager()
        return self._active_learning

    async def _get_audit_pipeline(self):
        """Get audit-to-KG pipeline."""
        if self._audit_pipeline is None:
            from resync.core.audit_to_kg_pipeline import get_audit_kg_pipeline

            self._audit_pipeline = get_audit_kg_pipeline()
        return self._audit_pipeline

    async def _get_embedder(self):
        """Get embedder."""
        if self._embedder is None:
            from resync.knowledge.ingestion.embedding_service import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    # =========================================================================
    # QUERY PROCESSING
    # =========================================================================

    async def process_query(
        self,
        query: str,
        top_k: int = 10,
        user_id: str | None = None,
        classification_result: dict[str, Any] | None = None,
    ) -> ProcessingResult:
        """
        Process a query through the full continual learning pipeline.

        Args:
            query: User's query
            top_k: Number of RAG results to retrieve
            user_id: Optional user identifier
            classification_result: Pre-computed classification (optional)

        Returns:
            ProcessingResult with all processing details
        """
        start_time = datetime.now(timezone.utc)

        result = ProcessingResult(
            original_query=query,
            enriched_query=query,
        )

        # Get embedder for later use
        embedder = await self._get_embedder()
        query_embedding = await embedder.embed(query)
        result.query_embedding = query_embedding

        # Step 1: Context Enrichment
        if self.enable_enrichment:
            enrichment_result = await self._apply_enrichment(query)
            result.enriched_query = enrichment_result.enriched_query
            result.enrichment_applied = enrichment_result.was_enriched
            result.enrichment_types = [e.value for e in enrichment_result.enrichments_applied]
            result.entities_found = enrichment_result.entities_found

        # Step 2: Feedback-Aware RAG Retrieval
        if self.enable_feedback_rag:
            rag_results = await self._retrieve_with_feedback(result.enriched_query, top_k, user_id)
            result.rag_results = rag_results
            result.feedback_applied = True
            if rag_results:
                result.rag_top_score = rag_results[0].get("score", 0.0)

        # Step 3: Extract classification confidence if provided
        if classification_result:
            result.classification_confidence = classification_result.get("confidence", 0.0)
            if not result.entities_found:
                result.entities_found = classification_result.get("entities", {})

        # Step 4: Active Learning Evaluation
        if self.enable_active_learning:
            needs_review, review_request = await self._evaluate_for_review(
                query=result.original_query,
                response="",  # Will be filled after response generation
                classification_confidence=result.classification_confidence,
                rag_similarity=result.rag_top_score,
                entities=result.entities_found,
                query_embedding=query_embedding,
            )
            result.review_requested = needs_review
            if review_request:
                result.review_reasons = review_request.get("reasons", [])
                result.review_priority = review_request.get("priority")
                self._reviews_triggered += 1

        # Calculate processing time
        result.processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        self._queries_processed += 1

        logger.info(
            "query_processed",
            query_len=len(query),
            enriched=result.enrichment_applied,
            rag_results=len(result.rag_results),
            review_requested=result.review_requested,
            processing_time_ms=result.processing_time_ms,
        )

        return result

    async def _apply_enrichment(self, query: str):
        """Apply context enrichment."""
        enricher = await self._get_enricher()
        return await enricher.enrich_query(query, self.instance_id)

    async def _retrieve_with_feedback(
        self,
        query: str,
        top_k: int,
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        """Retrieve with feedback-aware reranking."""
        try:
            retriever = await self._get_feedback_retriever()
            return await retriever.retrieve(
                query=query,
                top_k=top_k,
                apply_feedback=True,
                user_id=user_id,
            )
        except Exception as e:
            logger.warning(f"Feedback retrieval failed, using basic: {e}")
            return []

    async def _evaluate_for_review(
        self,
        query: str,
        response: str,
        classification_confidence: float,
        rag_similarity: float,
        entities: dict[str, list[str]],
        query_embedding: list[float] | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Evaluate if query needs human review."""
        try:
            active_learning = await self._get_active_learning()
            needs_review, request = await active_learning.evaluate_for_review(
                query=query,
                response=response,
                classification_confidence=classification_confidence,
                rag_similarity_score=rag_similarity,
                entities_found=entities,
                query_embedding=query_embedding,
            )
            return needs_review, request.to_dict() if request else None
        except Exception as e:
            logger.warning(f"Active learning evaluation failed: {e}")
            return False, None

    # =========================================================================
    # FEEDBACK PROCESSING
    # =========================================================================

    async def record_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        doc_ids: list[str] | None = None,
        user_id: str | None = None,
        trigger_audit: bool = True,
    ) -> FeedbackResult:
        """
        Record user feedback and propagate learning.

        Args:
            query: The query that was asked
            response: The response that was given
            rating: User rating (-1 negative, 0 neutral, +1 positive)
            doc_ids: Document IDs that were used in response
            user_id: User identifier
            trigger_audit: Whether to trigger audit for negative feedback

        Returns:
            FeedbackResult with processing details
        """
        result = FeedbackResult(success=True)

        # Step 1: Store feedback in RAG
        if self.enable_feedback_rag and doc_ids:
            try:
                retriever = await self._get_feedback_retriever()
                for doc_id in doc_ids:
                    await retriever.record_feedback(
                        query=query,
                        doc_id=doc_id,
                        rating=rating,
                        user_id=user_id,
                    )
                result.feedback_stored = True
            except Exception as e:
                logger.warning(f"Feedback storage failed: {e}")

        # Step 2: Trigger audit pipeline for negative feedback
        if trigger_audit and rating < 0 and self.enable_audit_pipeline:
            try:
                pipeline = await self._get_audit_pipeline()
                audit_result = await pipeline.process_audit_finding(
                    memory_id=f"feedback_{datetime.now(timezone.utc).timestamp()}",
                    user_query=query,
                    agent_response=response,
                    reason="Negative user feedback",
                    confidence=0.8,  # User feedback is reliable
                )
                result.audit_triggered = True
                result.kg_updated = audit_result.get("triplets_created", 0) > 0
                result.rag_updated = (
                    audit_result.get("rag_penalized", {}).get("documents_penalized", 0) > 0
                )
                self._audit_processes += 1
            except Exception as e:
                logger.warning(f"Audit pipeline failed: {e}")

        self._feedback_recorded += 1
        result.message = "Feedback processed successfully"

        logger.info(
            "feedback_recorded",
            rating=rating,
            feedback_stored=result.feedback_stored,
            audit_triggered=result.audit_triggered,
            kg_updated=result.kg_updated,
        )

        return result

    async def record_implicit_feedback(
        self,
        query: str,
        selected_doc_id: str,
        shown_doc_ids: list[str],
        user_id: str | None = None,
    ) -> FeedbackResult:
        """
        Record implicit feedback from user selection.

        When user selects/uses one document from results, this is
        implicit positive feedback for that document.
        """
        result = FeedbackResult(success=True)

        if self.enable_feedback_rag:
            try:
                retriever = await self._get_feedback_retriever()
                count = await retriever.record_implicit_feedback(
                    query=query,
                    selected_doc_id=selected_doc_id,
                    shown_doc_ids=shown_doc_ids,
                    user_id=user_id,
                )
                result.feedback_stored = count > 0
                result.message = f"Recorded {count} implicit feedback records"
            except Exception as e:
                logger.warning(f"Implicit feedback failed: {e}")
                result.success = False
                result.message = str(e)

        return result

    # =========================================================================
    # AUDIT INTEGRATION
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
        Process an audit finding through the KG pipeline.

        This is typically called from ia_auditor when an error is detected.
        """
        if not self.enable_audit_pipeline:
            return {"status": "disabled"}

        try:
            pipeline = await self._get_audit_pipeline()
            result = await pipeline.process_audit_finding(
                memory_id=memory_id,
                user_query=user_query,
                agent_response=agent_response,
                reason=reason,
                confidence=confidence,
            )
            self._audit_processes += 1
            return result
        except Exception as e:
            logger.error(f"Audit processing failed: {e}")
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # REVIEW MANAGEMENT
    # =========================================================================

    async def get_pending_reviews(
        self,
        priority: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get pending review requests."""
        if not self.enable_active_learning:
            return []

        try:
            from resync.core.active_learning import ReviewPriority

            active_learning = await self._get_active_learning()
            priority_enum = ReviewPriority(priority) if priority else None
            requests = await active_learning.get_pending_reviews(
                priority=priority_enum,
                limit=limit,
            )
            return [r.to_dict() for r in requests]
        except Exception as e:
            logger.warning(f"Failed to get pending reviews: {e}")
            return []

    async def submit_review(
        self,
        request_id: str,
        reviewer_id: str,
        is_correct: bool,
        correction: str | None = None,
        feedback: str | None = None,
    ) -> bool:
        """Submit a human review."""
        if not self.enable_active_learning:
            return False

        try:
            active_learning = await self._get_active_learning()
            return await active_learning.submit_review(
                request_id=request_id,
                reviewer_id=reviewer_id,
                is_correct=is_correct,
                correction=correction,
                feedback=feedback,
            )
        except Exception as e:
            logger.error(f"Review submission failed: {e}")
            return False

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics from all components."""
        stats = {
            "engine": {
                "queries_processed": self._queries_processed,
                "feedback_recorded": self._feedback_recorded,
                "reviews_triggered": self._reviews_triggered,
                "audit_processes": self._audit_processes,
            },
            "config": {
                "enrichment_enabled": self.enable_enrichment,
                "feedback_rag_enabled": self.enable_feedback_rag,
                "active_learning_enabled": self.enable_active_learning,
                "audit_pipeline_enabled": self.enable_audit_pipeline,
            },
        }

        # Get component statistics - log errors but don't fail
        try:
            if self.enable_enrichment and self._enricher:
                stats["enrichment"] = self._enricher.get_statistics()
        except Exception as e:
            logger.debug("enrichment_stats_failed", error=str(e))
            stats["enrichment"] = {"error": str(e)}

        try:
            if self.enable_feedback_rag and self._feedback_retriever:
                stats["feedback_rag"] = self._feedback_retriever.get_retriever_stats()
                stats["feedback_store"] = await self._feedback_retriever.get_feedback_stats()
        except Exception as e:
            logger.debug("feedback_rag_stats_failed", error=str(e))
            stats["feedback_rag"] = {"error": str(e)}

        try:
            if self.enable_active_learning and self._active_learning:
                stats["active_learning"] = self._active_learning.get_statistics()
                stats["review_queue"] = await self._active_learning.get_queue_statistics()
        except Exception as e:
            logger.debug("active_learning_stats_failed", error=str(e))
            stats["active_learning"] = {"error": str(e)}

        try:
            if self.enable_audit_pipeline and self._audit_pipeline:
                stats["audit_pipeline"] = self._audit_pipeline.get_statistics()
        except Exception as e:
            logger.debug("audit_pipeline_stats_failed", error=str(e))
            stats["audit_pipeline"] = {"error": str(e)}

        return stats

    async def get_health(self) -> dict[str, Any]:
        """Get health status of all components."""
        health = {
            "status": "healthy",
            "components": {},
        }

        # Check enricher
        if self.enable_enrichment:
            try:
                await self._get_enricher()
                health["components"]["enrichment"] = "healthy"
            except Exception as e:
                health["components"]["enrichment"] = f"unhealthy: {e}"
                health["status"] = "degraded"

        # Check feedback retriever
        if self.enable_feedback_rag:
            try:
                await self._get_feedback_retriever()
                health["components"]["feedback_rag"] = "healthy"
            except Exception as e:
                health["components"]["feedback_rag"] = f"unhealthy: {e}"
                health["status"] = "degraded"

        # Check active learning
        if self.enable_active_learning:
            try:
                al = await self._get_active_learning()
                await al.initialize()
                health["components"]["active_learning"] = "healthy"
            except Exception as e:
                health["components"]["active_learning"] = f"unhealthy: {e}"
                health["status"] = "degraded"

        # Check audit pipeline
        if self.enable_audit_pipeline:
            try:
                await self._get_audit_pipeline()
                health["components"]["audit_pipeline"] = "healthy"
            except Exception as e:
                health["components"]["audit_pipeline"] = f"unhealthy: {e}"
                health["status"] = "degraded"

        return health


# Global instance
_engine: ContinualLearningEngine | None = None


def get_continual_learning_engine(
    instance_id: str = "default",
) -> ContinualLearningEngine:
    """Get the global continual learning engine."""
    global _engine
    if _engine is None:
        _engine = ContinualLearningEngine(instance_id=instance_id)
    return _engine


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================


async def process_query_with_learning(
    query: str,
    top_k: int = 10,
    user_id: str | None = None,
) -> ProcessingResult:
    """Convenience function to process a query."""
    engine = get_continual_learning_engine()
    return await engine.process_query(query, top_k, user_id)


async def record_user_feedback(
    query: str,
    response: str,
    rating: int,
    doc_ids: list[str] | None = None,
) -> FeedbackResult:
    """Convenience function to record feedback."""
    engine = get_continual_learning_engine()
    return await engine.record_feedback(query, response, rating, doc_ids)
