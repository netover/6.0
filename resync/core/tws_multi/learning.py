"""
TWS Multi Learning - PostgreSQL Implementation.

Provides learning storage for TWS multi-instance using PostgreSQL.
"""

import logging

from resync.core.database.models import Feedback
from resync.core.database.repositories import FeedbackStore

logger = logging.getLogger(__name__)

__all__ = ["TWSLearningStore", "get_tws_learning_store"]


class TWSLearningStore:
    """TWS Learning Store - PostgreSQL Backend."""

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._store = FeedbackStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True
        logger.info("TWSLearningStore initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the store."""
        self._initialized = False

    async def record_outcome(
        self, job_name: str, action: str, outcome: str, details: dict | None = None
    ) -> Feedback:
        """Record an action outcome for learning."""
        return await self._store.feedback.add_feedback(
            session_id=f"tws_{job_name}",
            feedback_type="tws_outcome",
            query_text=action,
            response_text=outcome,
            is_positive=outcome == "success",
            metadata={"job_name": job_name, "details": details or {}},
        )

    async def get_job_outcomes(self, job_name: str, limit: int = 100) -> list[Feedback]:
        """Get outcomes for a job."""
        all_feedback = await self._store.feedback.get_all(limit=limit * 2)
        return [f for f in all_feedback if f.metadata_ and f.metadata_.get("job_name") == job_name][
            :limit
        ]

    async def get_success_rate(self, job_name: str, window: int = 100) -> float:
        """Get success rate for a job."""
        outcomes = await self.get_job_outcomes(job_name, limit=window)
        if not outcomes:
            return 0.0
        successes = sum(1 for o in outcomes if o.is_positive)
        return successes / len(outcomes)


_instance: TWSLearningStore | None = None


def get_tws_learning_store() -> TWSLearningStore:
    """Get the singleton TWSLearningStore instance."""
    global _instance
    if _instance is None:
        _instance = TWSLearningStore()
    return _instance
