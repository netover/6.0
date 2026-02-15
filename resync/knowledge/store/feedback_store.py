"""
RAG Feedback Store - PostgreSQL Implementation.

Provides feedback storage for RAG microservice using PostgreSQL.
"""

import logging

from resync.core.database.models import Feedback
from resync.core.database.repositories import FeedbackStore as PGFeedbackStore

logger = logging.getLogger(__name__)

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
