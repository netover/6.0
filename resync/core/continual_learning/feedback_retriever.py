"""
Feedback-Aware Retriever - RAG Retriever que aprende com feedback.

Este módulo wrapa o retriever base do RAG e aplica reranking baseado
em feedback histórico de usuários:
- Documentos com feedback positivo recebem boost
- Documentos com feedback negativo recebem penalty
- Query-specific feedback tem maior peso que global
"""

from typing import Any

from resync.core.continual_learning.feedback_store import (
    FeedbackRating,
    FeedbackStore,
    get_feedback_store,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class FeedbackAwareRetriever:
    """
    Retriever que aplica reranking baseado em feedback histórico.

    Wrapa qualquer retriever base e adiciona:
    1. Feedback-based score adjustment
    2. Temporal decay (feedback recente tem mais peso)
    3. Query-specific boosting
    4. Automatic feedback recording
    """

    def __init__(
        self,
        base_retriever: Any,
        feedback_store: FeedbackStore | None = None,
        feedback_weight: float = 0.3,
        enable_feedback: bool = True,
    ):
        """
        Initialize feedback-aware retriever.

        Args:
            base_retriever: Base retriever to wrap (RagRetriever)
            feedback_store: FeedbackStore instance (uses default if None)
            feedback_weight: How much feedback affects final score (0-1)
            enable_feedback: Whether to apply feedback adjustments
        """
        self.base_retriever = base_retriever
        self.feedback_store = feedback_store or get_feedback_store()
        self.feedback_weight = min(1.0, max(0.0, feedback_weight))
        self.enable_feedback = enable_feedback

        # Cache for query session
        self._last_query: str | None = None
        self._last_results: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        apply_feedback: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents with feedback-based reranking.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for base retriever
            apply_feedback: Override enable_feedback setting

        Returns:
            List of documents sorted by feedback-adjusted score
        """
        should_apply = (
            apply_feedback if apply_feedback is not None else self.enable_feedback
        )

        # Get base results (retrieve more to allow reranking)
        fetch_k = min(top_k * 2, 50) if should_apply else top_k
        base_results = await self.base_retriever.retrieve(
            query=query,
            top_k=fetch_k,
            filters=filters,
        )

        if not base_results:
            return []

        # Apply feedback adjustments
        if should_apply and self.feedback_weight > 0:
            base_results = await self._apply_feedback_reranking(
                query=query,
                results=base_results,
            )

        # Store for potential feedback recording
        self._last_query = query
        self._last_results = base_results[:top_k]

        logger.debug(
            "feedback_aware_retrieve",
            query_len=len(query),
            results_count=len(base_results),
            feedback_applied=should_apply,
        )

        return base_results[:top_k]

    async def _apply_feedback_reranking(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply feedback-based score adjustments."""
        if not results:
            return results

        # Extract document IDs
        doc_ids = [self._get_doc_id(doc) for doc in results]
        doc_ids = [d for d in doc_ids if d]  # Filter None

        if not doc_ids:
            return results

        # Get feedback scores
        feedback_scores = await self.feedback_store.get_query_document_scores(
            query=query,
            doc_ids=doc_ids,
        )

        # Apply adjustments
        for doc in results:
            doc_id = self._get_doc_id(doc)
            if not doc_id:
                continue

            # Get original score
            original_score = self._get_score(doc)

            # Get feedback adjustment (-0.5 to +0.5)
            feedback_adj = feedback_scores.get(doc_id, 0.0)

            # Calculate new score
            # feedback_adj is in [-0.5, +0.5], scale by weight
            score_adjustment = feedback_adj * self.feedback_weight

            # New score = original * (1 + adjustment)
            # This means:
            # - +0.15 adjustment → 15% boost
            # - -0.15 adjustment → 15% penalty
            new_score = original_score * (1 + score_adjustment)

            # Store adjusted score
            doc["_original_score"] = original_score
            doc["_feedback_adjustment"] = feedback_adj
            doc["score"] = new_score

        # Re-sort by adjusted score
        results.sort(key=lambda x: self._get_score(x), reverse=True)

        # Log significant adjustments
        adjusted_count = sum(
            1 for doc in results if abs(doc.get("_feedback_adjustment", 0)) > 0.1
        )
        if adjusted_count > 0:
            logger.info(
                "feedback_reranking_applied",
                total_docs=len(results),
                adjusted_docs=adjusted_count,
            )

        return results

    def _get_doc_id(self, doc: dict[str, Any]) -> str | None:
        """Extract document ID from result."""
        # Try common ID fields
        for field in ["id", "doc_id", "document_id", "_id", "payload.id"]:
            if "." in field:
                parts = field.split(".")
                val = doc
                for part in parts:
                    val = val.get(part, {}) if isinstance(val, dict) else None
                if val:
                    return str(val)
            elif field in doc:
                return str(doc[field])

        # Try payload
        payload = doc.get("payload", {})
        if isinstance(payload, dict):
            for field in ["id", "doc_id", "source"]:
                if field in payload:
                    return str(payload[field])

        return None

    def _get_score(self, doc: dict[str, Any]) -> float:
        """Extract score from result."""
        for field in ["score", "_score", "similarity", "relevance"]:
            if field in doc:
                try:
                    return float(doc[field])
                except (ValueError, TypeError) as e:
                    logger.debug(
                        "suppressed_exception", error=str(e), exc_info=True
                    )  # was: pass
        return 0.0

    async def record_feedback(
        self,
        rating: int,
        doc_index: int | None = None,
        doc_id: str | None = None,
        response_text: str | None = None,
        user_id: str | None = None,
    ) -> str | None:
        """
        Record feedback for a retrieved document.

        Args:
            rating: Rating from -2 (very bad) to +2 (very good)
            doc_index: Index of document in last results (0-based)
            doc_id: Direct document ID (alternative to doc_index)
            response_text: Generated response text
            user_id: User identifier

        Returns:
            Feedback ID if recorded, None otherwise
        """
        if not self._last_query:
            logger.warning("no_query_to_record_feedback")
            return None

        # Determine which document
        if doc_id is None:
            if doc_index is not None and doc_index < len(self._last_results):
                target_doc = self._last_results[doc_index]
                doc_id = self._get_doc_id(target_doc)
            elif self._last_results:
                # Default to first result
                target_doc = self._last_results[0]
                doc_id = self._get_doc_id(target_doc)

        if not doc_id:
            logger.warning("no_doc_id_for_feedback")
            return None

        # Record feedback
        return await self.feedback_store.record_feedback(
            query=self._last_query,
            doc_id=doc_id,
            rating=rating,
            user_id=user_id,
            response_text=response_text,
        )

    async def record_positive_feedback(
        self,
        doc_index: int = 0,
        user_id: str | None = None,
    ) -> str | None:
        """Shortcut to record positive feedback."""
        return await self.record_feedback(
            rating=FeedbackRating.POSITIVE,
            doc_index=doc_index,
            user_id=user_id,
        )

    async def record_negative_feedback(
        self,
        doc_index: int = 0,
        user_id: str | None = None,
    ) -> str | None:
        """Shortcut to record negative feedback."""
        return await self.record_feedback(
            rating=FeedbackRating.NEGATIVE,
            doc_index=doc_index,
            user_id=user_id,
        )

    async def get_retriever_stats(self) -> dict[str, Any]:
        """Get statistics about feedback-aware retrieval."""
        feedback_stats = await self.feedback_store.get_feedback_stats()

        return {
            "feedback_enabled": self.enable_feedback,
            "feedback_weight": self.feedback_weight,
            "feedback_stats": feedback_stats,
        }


def create_feedback_aware_retriever(
    embedder: Any,
    store: Any,
    feedback_weight: float = 0.3,
) -> FeedbackAwareRetriever:
    """
    Factory function to create feedback-aware retriever.

    Args:
        embedder: Embedder instance
        store: Vector store instance
        feedback_weight: How much feedback affects scores

    Returns:
        FeedbackAwareRetriever wrapping base RagRetriever
    """
    from resync.knowledge.retrieval.retriever import RagRetriever

    base_retriever = RagRetriever(embedder=embedder, store=store)

    return FeedbackAwareRetriever(
        base_retriever=base_retriever,
        feedback_weight=feedback_weight,
    )
