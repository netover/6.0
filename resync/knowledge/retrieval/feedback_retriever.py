"""
Feedback-Aware Retriever for Continual Learning RAG.

Extends the base retriever with feedback-based reranking to improve
retrieval quality based on user interactions.

Key Features:
- Integrates feedback scores into retrieval ranking
- Supports multiple reranking strategies
- Provides explanation for ranking decisions
- Maintains backward compatibility with base retriever
"""

import math
from dataclasses import dataclass
from typing import Any

from resync.core.structured_logger import get_logger
from resync.knowledge.config import CFG
from resync.knowledge.interfaces import Embedder, Retriever, VectorStore
from resync.knowledge.monitoring import query_seconds
from resync.knowledge.store.feedback_store import (
    FEEDBACK_NEGATIVE,
    FEEDBACK_POSITIVE,
    FeedbackStore,
    get_feedback_store,
)

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with feedback information."""

    doc_id: str
    content: str
    metadata: dict[str, Any]
    base_score: float
    feedback_score: float
    final_score: float
    feedback_boost: float
    has_feedback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "base_score": self.base_score,
            "feedback_score": self.feedback_score,
            "final_score": self.final_score,
            "feedback_boost": self.feedback_boost,
            "has_feedback": self.has_feedback,
        }


class FeedbackAwareRetriever(Retriever):
    """
    Retriever that incorporates user feedback into ranking.

    Implements a hybrid scoring system:
    final_score = base_score * (1 + feedback_weight * feedback_score)

    Where feedback_score is derived from:
    1. Exact query-document feedback (highest weight)
    2. Similar query feedback (medium weight)
    3. Document-level aggregate (lowest weight)
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        feedback_store: FeedbackStore | None = None,
        feedback_weight: float = 0.3,
        min_feedback_boost: float = -0.5,
        max_feedback_boost: float = 0.5,
    ):
        """
        Initialize feedback-aware retriever.

        Args:
            embedder: Embedding service for queries
            store: Vector store for document retrieval
            feedback_store: Feedback storage (uses global if None)
            feedback_weight: Weight of feedback in final score (0.0-1.0)
            min_feedback_boost: Minimum multiplier from feedback
            max_feedback_boost: Maximum multiplier from feedback
        """
        self.embedder = embedder
        self.store = store
        self.feedback_store = feedback_store or get_feedback_store()
        self.feedback_weight = max(0.0, min(1.0, feedback_weight))
        self.min_feedback_boost = min_feedback_boost
        self.max_feedback_boost = max_feedback_boost

        # Statistics
        self._retrieval_count = 0
        self._feedback_applied_count = 0

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        apply_feedback: bool = True,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents with feedback-based reranking.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            apply_feedback: Whether to apply feedback reranking
            user_id: Optional user ID for personalized feedback

        Returns:
            List of document dicts with scores
        """
        top_k = min(top_k, CFG.max_top_k)
        self._retrieval_count += 1

        # Get query embedding
        query_embedding = await self.embedder.embed(query)

        # Calculate ef_search dynamically
        ef = CFG.ef_search_base + int(math.log2(max(10, top_k)) * 8)
        ef = min(ef, CFG.ef_search_max)

        # Retrieve more candidates for reranking
        retrieve_k = min(top_k * 3, CFG.max_top_k) if apply_feedback else top_k

        with query_seconds.time():
            hits = await self.store.query(
                vector=query_embedding,
                top_k=retrieve_k,
                collection=CFG.collection_read,
                filters=filters,
                ef_search=ef,
                with_vectors=CFG.enable_rerank,
            )

        if not hits:
            return []

        # Apply feedback-based reranking
        if apply_feedback:
            hits = await self._apply_feedback_reranking(hits, query, query_embedding, user_id)
            self._feedback_applied_count += 1

        # Apply traditional reranking if enabled
        if CFG.enable_rerank and hits and "vector" in hits[0]:
            hits = self._apply_vector_rerank(hits, query_embedding)

        return hits[:top_k]

    async def _apply_feedback_reranking(
        self,
        hits: list[dict[str, Any]],
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Apply feedback-based score adjustments.

        Scoring formula:
        final_score = base_score * (1 + clamp(feedback_weight * feedback_score, min, max))
        """
        # Get document IDs
        doc_ids = [hit.get("id", hit.get("doc_id", "")) for hit in hits]

        # Batch fetch document scores
        doc_scores = await self.feedback_store.get_document_scores_batch(doc_ids)

        # Apply feedback to each hit
        for hit in hits:
            doc_id = hit.get("id", hit.get("doc_id", ""))
            base_score = hit.get("score", 0.0)

            # Get feedback score (query-specific or document-level)
            feedback_score = await self.feedback_store.get_query_feedback_score(
                query, doc_id, query_embedding
            )

            # Fall back to document-level if no query-specific feedback
            if feedback_score == 0.0:
                feedback_score = doc_scores.get(doc_id, 0.0)

            # Calculate boost (clamped)
            feedback_boost = self.feedback_weight * feedback_score
            feedback_boost = max(
                self.min_feedback_boost, min(self.max_feedback_boost, feedback_boost)
            )

            # Apply boost to score
            final_score = base_score * (1 + feedback_boost)

            # Update hit with feedback info
            hit["base_score"] = base_score
            hit["feedback_score"] = feedback_score
            hit["feedback_boost"] = feedback_boost
            hit["score"] = final_score
            hit["has_feedback"] = feedback_score != 0.0

        # Sort by new scores
        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        logger.debug(
            "feedback_reranking_applied",
            query_len=len(query),
            hits_count=len(hits),
            docs_with_feedback=sum(1 for h in hits if h.get("has_feedback")),
        )

        return hits

    def _apply_vector_rerank(
        self,
        hits: list[dict[str, Any]],
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        """Apply cosine similarity reranking."""

        def cosine(a: list[float], b: list[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        for hit in hits:
            if "vector" in hit:
                similarity = cosine(query_embedding, hit["vector"])
                # Combine with existing score
                hit["vector_similarity"] = similarity
                hit["score"] = hit.get("score", 0.0) * 0.7 + similarity * 0.3

        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits

    # =========================================================================
    # FEEDBACK RECORDING
    # =========================================================================

    async def record_feedback(
        self,
        query: str,
        doc_id: str,
        rating: int,
        user_id: str | None = None,
    ) -> bool:
        """
        Record user feedback for a query-document pair.

        Args:
            query: The search query
            doc_id: The document ID
            rating: -1 (not helpful), 0 (neutral), +1 (helpful)
            user_id: Optional user identifier

        Returns:
            True if feedback was recorded
        """
        query_embedding = await self.embedder.embed(query)
        return await self.feedback_store.record_feedback(
            query=query,
            doc_id=doc_id,
            rating=rating,
            user_id=user_id,
            query_embedding=query_embedding,
        )

    async def record_implicit_feedback(
        self,
        query: str,
        selected_doc_id: str,
        shown_doc_ids: list[str],
        user_id: str | None = None,
    ) -> int:
        """
        Record implicit feedback from user selection.

        When a user selects one document from results:
        - Selected document gets positive feedback
        - Top non-selected documents get slight negative feedback

        Args:
            query: The search query
            selected_doc_id: The document the user selected/used
            shown_doc_ids: All documents shown to user
            user_id: Optional user identifier

        Returns:
            Number of feedback records created
        """
        await self.embedder.embed(query)

        feedback_pairs = []
        for i, doc_id in enumerate(shown_doc_ids):
            if doc_id == selected_doc_id:
                # Selected document gets positive feedback
                feedback_pairs.append((doc_id, FEEDBACK_POSITIVE))
            elif i < 3 and doc_id != selected_doc_id:
                # Top 3 non-selected get slight negative
                feedback_pairs.append((doc_id, FEEDBACK_NEGATIVE))
            # Others get no feedback

        return await self.feedback_store.record_batch_feedback(
            query=query,
            doc_ratings=feedback_pairs,
            user_id=user_id,
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_retriever_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_retrievals": self._retrieval_count,
            "feedback_applied_count": self._feedback_applied_count,
            "feedback_application_rate": (
                self._feedback_applied_count / self._retrieval_count
                if self._retrieval_count > 0
                else 0.0
            ),
            "feedback_weight": self.feedback_weight,
            "min_boost": self.min_feedback_boost,
            "max_boost": self.max_feedback_boost,
        }

    async def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback store statistics."""
        return await self.feedback_store.get_statistics()


class AdaptiveFeedbackRetriever(FeedbackAwareRetriever):
    """
    Retriever with adaptive feedback weight based on feedback density.

    Automatically adjusts feedback weight based on:
    - Amount of feedback available
    - Quality of feedback (consistency)
    - Query novelty
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        feedback_store: FeedbackStore | None = None,
        base_feedback_weight: float = 0.3,
        min_feedback_for_full_weight: int = 10,
    ):
        super().__init__(
            embedder=embedder,
            store=store,
            feedback_store=feedback_store,
            feedback_weight=base_feedback_weight,
        )
        self.base_feedback_weight = base_feedback_weight
        self.min_feedback_for_full_weight = min_feedback_for_full_weight

    async def _apply_feedback_reranking(
        self,
        hits: list[dict[str, Any]],
        query: str,
        query_embedding: list[float],
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply adaptive feedback reranking."""
        # Get feedback statistics
        stats = await self.feedback_store.get_statistics()
        total_feedback = stats.get("total_feedback_records", 0)

        # Adapt weight based on feedback density
        if total_feedback < self.min_feedback_for_full_weight:
            # Reduce weight when little feedback available
            density_factor = total_feedback / self.min_feedback_for_full_weight
            self.feedback_weight = self.base_feedback_weight * density_factor
        else:
            self.feedback_weight = self.base_feedback_weight

        # Use parent's reranking logic
        return await super()._apply_feedback_reranking(hits, query, query_embedding, user_id)


# Factory function
def create_feedback_aware_retriever(
    embedder: Embedder, store: VectorStore, adaptive: bool = True, **kwargs
) -> FeedbackAwareRetriever:
    """
    Create a feedback-aware retriever.

    Args:
        embedder: Embedding service
        store: Vector store
        adaptive: Whether to use adaptive weight adjustment
        **kwargs: Additional arguments for retriever

    Returns:
        Configured retriever instance
    """
    if adaptive:
        return AdaptiveFeedbackRetriever(embedder=embedder, store=store, **kwargs)
    return FeedbackAwareRetriever(embedder=embedder, store=store, **kwargs)
