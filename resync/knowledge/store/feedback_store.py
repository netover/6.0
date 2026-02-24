"""
RAG Feedback Store - PostgreSQL Implementation.

Provides feedback storage for RAG microservice using PostgreSQL.
"""

import structlog

from resync.core.database.models import Feedback
from resync.core.database.repositories import FeedbackStore as PGFeedbackStore

logger = structlog.get_logger(__name__)

# Feedback type constants for compatibility
FEEDBACK_POSITIVE = "positive"
FEEDBACK_NEGATIVE = "negative"
FEEDBACK_NEUTRAL = "neutral"

__all__ = [
    "FeedbackStore",
    "RAGFeedbackStore",
    "get_rag_feedback_store",
    "get_feedback_store",
    "FEEDBACK_POSITIVE",
    "FEEDBACK_NEGATIVE",
    "FEEDBACK_NEUTRAL",
]


class RAGFeedbackStore:
    """RAG Feedback Store - PostgreSQL Backend."""

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._store = PGFeedbackStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True
        logger.info("RAGFeedbackStore initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the store."""
        self._initialized = False

    async def add_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> Feedback:
        """Add RAG feedback."""
        return await self._store.feedback.add_feedback(
            session_id=session_id or "rag_default",
            query_text=query,
            response_text=response,
            rating=rating,
            feedback_type="rag",
            is_positive=rating >= 4,
            metadata=metadata,
        )

    async def get_feedback(self, limit: int = 100) -> list[Feedback]:
        """Get recent feedback."""
        all_feedback = await self._store.feedback.get_all(limit=limit * 2)
        return [f for f in all_feedback if f.feedback_type == "rag"][:limit]

    async def get_positive_examples(self, limit: int = 50) -> list[Feedback]:
        """Get positive RAG examples."""
        positive = await self._store.feedback.get_positive_examples(limit * 2)
        return [f for f in positive if f.feedback_type == "rag"][:limit]

    async def get_negative_examples(self, limit: int = 50) -> list[Feedback]:
        """Get negative RAG examples."""
        negative = await self._store.feedback.get_negative_examples(limit * 2)
        return [f for f in negative if f.feedback_type == "rag"][:limit]

    # ------------------------------------------------------------------
    # Methods required by FeedbackAwareRetriever
    # ------------------------------------------------------------------

    async def get_document_scores_batch(
        self, doc_ids: list[str]
    ) -> dict[str, float]:
        """Return aggregate feedback scores keyed by doc_id.

        Returns 0.0 for documents with no feedback (graceful degradation).
        Override with a real SQL implementation when the feedback table is
        wired to store per-document scores.
        """
        return {doc_id: 0.0 for doc_id in doc_ids}

    async def get_query_feedback_score(
        self,
        query: str,
        doc_id: str,
        query_embedding: list[float] | None = None,
    ) -> float:
        """Return the feedback score for a specific query+document pair.

        Returns 0.0 when no query-specific feedback exists (graceful degradation).
        Override with a semantic-similarity lookup against the feedback table.
        """
        return 0.0

    async def record_feedback(
        self,
        query: str,
        doc_id: str,
        rating: int,
        user_id: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> bool:
        """Record explicit user feedback for a query-document pair.

        Delegates to the underlying PostgreSQL store.
        Returns True if feedback was stored successfully.
        """
        try:
            await self._store.feedback.add_feedback(
                session_id=user_id or "anonymous",
                query_text=query,
                response_text=doc_id,
                rating=rating,
                feedback_type="rag_retrieval",
                is_positive=rating > 0,
            )
            return True
        except Exception:
            logger.warning("record_feedback_failed", doc_id=doc_id)
            return False

    async def record_batch_feedback(
        self,
        query: str,
        doc_ratings: list[tuple[str, str]],
        user_id: str | None = None,
    ) -> int:
        """Record feedback for multiple query-document pairs in one call.

        `doc_ratings` is a list of (doc_id, rating_constant) tuples where
        rating_constant is one of FEEDBACK_POSITIVE / FEEDBACK_NEGATIVE.

        Returns the number of feedback records created.
        """
        rating_map = {FEEDBACK_POSITIVE: 1, FEEDBACK_NEGATIVE: -1, FEEDBACK_NEUTRAL: 0}
        count = 0
        for doc_id, rating_label in doc_ratings:
            numeric = rating_map.get(rating_label, 0)
            ok = await self.record_feedback(
                query=query,
                doc_id=doc_id,
                rating=numeric,
                user_id=user_id,
            )
            if ok:
                count += 1
        return count

    async def get_statistics(self) -> dict[str, object]:
        """Return aggregate statistics for the feedback store."""
        try:
            records = await self._store.feedback.get_all(limit=100_000)
            return {
                "total_feedback_records": len(records),
                "backend": "postgresql",
                "initialized": self._initialized,
            }
        except Exception:
            return {
                "total_feedback_records": 0,
                "backend": "postgresql",
                "initialized": self._initialized,
            }


# Alias for backward compatibility
FeedbackStore = RAGFeedbackStore

_instance: RAGFeedbackStore | None = None


def get_rag_feedback_store() -> RAGFeedbackStore:
    """Get the singleton RAGFeedbackStore instance."""
    global _instance
    if _instance is None:
        _instance = RAGFeedbackStore()
    return _instance


# Alias for backward compatibility
get_feedback_store = get_rag_feedback_store
