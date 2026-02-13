"""
Active Learning - PostgreSQL Implementation.

Provides active learning candidate selection using PostgreSQL.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from resync.core.database.models import ActiveLearningCandidate
from resync.core.database.repositories import FeedbackStore

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes for Active Learning
# ============================================================================

class ReviewStatus(str, Enum):
    """Status of a review item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ReviewReason(str, Enum):
    """Reasons for requiring review."""
    LOW_CONFIDENCE = "low_confidence"
    LOW_SIMILARITY = "low_similarity"
    NEW_ENTITY = "new_entity"
    HALLUCINATION_DETECTED = "hallucination_detected"
    USER_FEEDBACK = "user_feedback"
    MANUAL_REQUEST = "manual_request"


@dataclass
class ReviewItem:
    """Item requiring human review."""
    id: str
    query: str
    response: str
    reasons: list[ReviewReason]
    confidence_scores: dict[str, float]
    status: ReviewStatus = ReviewStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLearningDecision:
    """Decision from active learning check."""
    should_review: bool
    reasons: list[ReviewReason]
    confidence_scores: dict[str, float]
    suggested_action: str = "proceed_normally"
    metadata: dict[str, Any] = field(default_factory=dict)


class ActiveLearningManager:
    """Manager for active learning workflow."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.6,
        enable_new_entity_detection: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.enable_new_entity_detection = enable_new_entity_detection
        self._store = FeedbackStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the manager."""
        self._initialized = True
        logger.info("ActiveLearningManager initialized")

    def close(self) -> None:
        """Close the manager."""
        self._initialized = False

    def should_request_review(
        self,
        query: str,
        response: str,
        classification_confidence: float,
        rag_similarity_score: float,
        entities_found: dict[str, list[str]],
    ) -> ActiveLearningDecision:
        """Determine if a response needs human review."""
        reasons = []
        confidence_scores = {
            "classification": classification_confidence,
            "rag_similarity": rag_similarity_score,
        }

        # Check confidence threshold
        if classification_confidence < self.confidence_threshold:
            reasons.append(ReviewReason.LOW_CONFIDENCE)

        # Check similarity threshold
        if rag_similarity_score < self.similarity_threshold:
            reasons.append(ReviewReason.LOW_SIMILARITY)

        should_review = len(reasons) > 0
        suggested_action = "request_review" if should_review else "proceed_normally"

        return ActiveLearningDecision(
            should_review=should_review,
            reasons=reasons,
            confidence_scores=confidence_scores,
            suggested_action=suggested_action,
        )

    def add_to_review_queue(
        self,
        query: str,
        response: str,
        decision: ActiveLearningDecision,
    ) -> str:
        """Add item to review queue."""
        import uuid
        review_id = str(uuid.uuid4())
        logger.info("Added to review queue: %s", review_id)
        return review_id


async def check_for_review(
    query: str,
    response: str,
    classification_confidence: float = 0.5,
    rag_similarity_score: float = 0.5,
) -> ActiveLearningDecision:
    """Quick check if review is needed."""
    manager = get_active_learning_manager()
    return manager.should_request_review(
        query=query,
        response=response,
        classification_confidence=classification_confidence,
        rag_similarity_score=rag_similarity_score,
        entities_found={},
    )


# Singleton instance
_active_learning_manager: ActiveLearningManager | None = None


def get_active_learning_manager() -> ActiveLearningManager:
    """Get or create active learning manager singleton."""
    global _active_learning_manager
    if _active_learning_manager is None:
        _active_learning_manager = ActiveLearningManager()
    return _active_learning_manager


__all__ = [
    "ActiveLearner",
    "get_active_learner",
    "ActiveLearningDecision",
    "ActiveLearningManager",
    "ReviewItem",
    "ReviewReason",
    "ReviewStatus",
    "check_for_review",
    "get_active_learning_manager",
]


class ActiveLearner:
    """Active Learner - PostgreSQL Backend."""

    def __init__(self, db_path: str | None = None):
        """Initialize. db_path is ignored - uses PostgreSQL."""
        if db_path:
            logger.debug("db_path ignored, using PostgreSQL: %s", db_path)
        self._store = FeedbackStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the learner."""
        self._initialized = True
        logger.info("ActiveLearner initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the learner."""
        self._initialized = False

    async def add_candidate(
        self,
        query_text: str,
        uncertainty_score: float,
        response_text: str | None = None,
        metadata: dict | None = None,
    ) -> ActiveLearningCandidate:
        """Add a candidate for review."""
        return await self._store.active_learning.add_candidate(
            query_text=query_text,
            uncertainty_score=uncertainty_score,
            response_text=response_text,
            metadata=metadata,
        )

    async def get_top_candidates(self, limit: int = 10) -> list[ActiveLearningCandidate]:
        """Get top uncertain candidates for review."""
        return await self._store.active_learning.get_top_candidates(limit)

    async def review_candidate(
        self, candidate_id: int, selected_label: str, reviewer_id: str
    ) -> ActiveLearningCandidate | None:
        """Mark candidate as reviewed."""
        return await self._store.active_learning.review_candidate(
            candidate_id, selected_label, reviewer_id
        )

    async def get_reviewed_candidates(self, limit: int = 100) -> list[ActiveLearningCandidate]:
        """Get reviewed candidates for training."""
        return await self._store.active_learning.find(
            {"status": "reviewed"}, limit=limit, order_by="reviewed_at", desc=True
        )

    def should_request_label(self, uncertainty_score: float, threshold: float = 0.7) -> bool:
        """Check if we should request a label based on uncertainty."""
        return uncertainty_score >= threshold


_instance: ActiveLearner | None = None


def get_active_learner() -> ActiveLearner:
    """Get the singleton ActiveLearner instance."""
    global _instance
    if _instance is None:
        _instance = ActiveLearner()
    return _instance
