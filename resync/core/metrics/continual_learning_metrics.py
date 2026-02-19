"""
Continual Learning Metrics Collector.

Collects metrics from all continual learning components:
- Feedback rates and scores
- Active learning queue
- Context enrichment usage
- Audit pipeline activity
- RAG performance with feedback

Designed to work with LightweightMetricsStore.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from resync.core.metrics.lightweight_store import (
    LightweightMetricsStore,
    MetricType,
    get_metrics_store,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# METRIC NAMES (centralized for consistency)
# =============================================================================


class MetricNames:
    """Centralized metric names for continual learning."""

    # Query metrics
    QUERY_TOTAL = "cl.query.total"
    QUERY_DURATION_MS = "cl.query.duration_ms"
    QUERY_WITH_ENRICHMENT = "cl.query.with_enrichment"

    # Feedback metrics
    FEEDBACK_TOTAL = "cl.feedback.total"
    FEEDBACK_POSITIVE = "cl.feedback.positive"
    FEEDBACK_NEGATIVE = "cl.feedback.negative"
    FEEDBACK_RATING_AVG = "cl.feedback.rating_avg"

    # Active learning metrics
    REVIEW_QUEUE_SIZE = "cl.review.queue_size"
    REVIEW_ADDED = "cl.review.added"
    REVIEW_COMPLETED = "cl.review.completed"
    REVIEW_APPROVED = "cl.review.approved"
    REVIEW_CORRECTED = "cl.review.corrected"
    REVIEW_REJECTED = "cl.review.rejected"

    # Context enrichment metrics
    ENRICHMENT_TOTAL = "cl.enrichment.total"
    ENRICHMENT_CONTEXT_ADDED = "cl.enrichment.context_added"
    ENRICHMENT_NO_CONTEXT = "cl.enrichment.no_context"
    ENRICHMENT_DURATION_MS = "cl.enrichment.duration_ms"

    # Audit pipeline metrics
    AUDIT_PROCESSED = "cl.audit.processed"
    AUDIT_TRIPLETS_CREATED = "cl.audit.triplets_created"
    AUDIT_KG_ENTRIES_ADDED = "cl.audit.kg_entries_added"
    AUDIT_DOCUMENTS_PENALIZED = "cl.audit.documents_penalized"

    # RAG metrics
    RAG_RETRIEVAL_TOTAL = "cl.rag.retrieval_total"
    RAG_WITH_FEEDBACK_RERANK = "cl.rag.with_feedback_rerank"
    RAG_SIMILARITY_SCORE = "cl.rag.similarity_score"
    RAG_RERANK_BOOST = "cl.rag.rerank_boost"

    # System metrics
    SYSTEM_MEMORY_MB = "cl.system.memory_mb"
    SYSTEM_CPU_PERCENT = "cl.system.cpu_percent"
    SYSTEM_DB_SIZE_MB = "cl.system.db_size_mb"


@dataclass
class QueryMetrics:
    """Metrics collected during a single query."""

    start_time: float
    query_length: int
    enrichment_applied: bool = False
    enrichment_context_count: int = 0
    enrichment_duration_ms: float = 0
    rag_similarity: float = 0
    feedback_rerank_applied: bool = False
    needs_review: bool = False
    review_reasons: int = 0
    total_duration_ms: float = 0


class ContinualLearningMetrics:
    """
    Metrics collector for continual learning system.

    Usage:
        metrics = ContinualLearningMetrics()

        # Record individual events
        await metrics.record_feedback(rating=1, doc_id="doc1")
        await metrics.record_review_added(reasons=["low_confidence"])

        # Track query lifecycle
        async with metrics.track_query("What is job ABC?") as qm:
            qm.enrichment_applied = True
            qm.enrichment_context_count = 3
            # ... process query ...
    """

    def __init__(self, store: LightweightMetricsStore | None = None):
        """
        Initialize metrics collector.

        Args:
            store: Optional metrics store. Uses global singleton if not provided.
        """
        self._store = store

    @property
    def store(self) -> LightweightMetricsStore:
        """Get metrics store (lazy initialization)."""
        if self._store is None:
            self._store = get_metrics_store()
        return self._store

    # =========================================================================
    # QUERY TRACKING
    # =========================================================================

    @asynccontextmanager
    async def track_query(self, query: str, instance_id: str | None = None):
        """
        Context manager to track query metrics.

        Usage:
            async with metrics.track_query("What is job ABC?") as qm:
                qm.enrichment_applied = True
                qm.rag_similarity = 0.85
        """
        qm = QueryMetrics(
            start_time=time.time(),
            query_length=len(query),
        )

        tags = {"instance": instance_id} if instance_id else {}

        try:
            yield qm
        finally:
            qm.total_duration_ms = (time.time() - qm.start_time) * 1000

            # Record metrics
            await self.store.increment(MetricNames.QUERY_TOTAL, tags=tags)
            await self.store.record(
                MetricNames.QUERY_DURATION_MS,
                qm.total_duration_ms,
                MetricType.HISTOGRAM,
                tags=tags,
            )

            if qm.enrichment_applied:
                await self.store.increment(MetricNames.QUERY_WITH_ENRICHMENT, tags=tags)

    # =========================================================================
    # FEEDBACK METRICS
    # =========================================================================

    async def record_feedback(
        self,
        rating: int,
        doc_id: str | None = None,
        query_type: str | None = None,
    ) -> None:
        """Record a feedback event."""
        tags = {}
        if query_type:
            tags["query_type"] = query_type

        await self.store.increment(MetricNames.FEEDBACK_TOTAL, tags=tags)

        if rating > 0:
            await self.store.increment(MetricNames.FEEDBACK_POSITIVE, tags=tags)
        elif rating < 0:
            await self.store.increment(MetricNames.FEEDBACK_NEGATIVE, tags=tags)

        await self.store.record(
            MetricNames.FEEDBACK_RATING_AVG,
            float(rating),
            MetricType.HISTOGRAM,
            tags=tags,
        )

    async def update_feedback_stats(
        self,
        total: int,
        positive: int,
        negative: int,
        avg_rating: float,
    ) -> None:
        """Update feedback statistics (from FeedbackStore)."""
        await self.store.set_gauge(MetricNames.FEEDBACK_TOTAL, float(total))
        await self.store.set_gauge(MetricNames.FEEDBACK_POSITIVE, float(positive))
        await self.store.set_gauge(MetricNames.FEEDBACK_NEGATIVE, float(negative))
        await self.store.set_gauge(MetricNames.FEEDBACK_RATING_AVG, avg_rating)

    # =========================================================================
    # ACTIVE LEARNING METRICS
    # =========================================================================

    async def record_review_added(
        self,
        reasons: list,
        confidence: float = 0,
    ) -> None:
        """Record when an item is added to review queue."""
        tags = {}
        if reasons:
            tags["primary_reason"] = reasons[0] if isinstance(reasons[0], str) else reasons[0].value

        await self.store.increment(MetricNames.REVIEW_ADDED, tags=tags)

    async def record_review_completed(
        self,
        status: str,
        wait_time_hours: float = 0,
    ) -> None:
        """Record when a review is completed."""
        tags = {"status": status}

        await self.store.increment(MetricNames.REVIEW_COMPLETED, tags=tags)

        if status == "approved":
            await self.store.increment(MetricNames.REVIEW_APPROVED)
        elif status == "corrected":
            await self.store.increment(MetricNames.REVIEW_CORRECTED)
        elif status == "rejected":
            await self.store.increment(MetricNames.REVIEW_REJECTED)

    async def update_review_queue_size(self, size: int) -> None:
        """Update the current review queue size gauge."""
        await self.store.set_gauge(MetricNames.REVIEW_QUEUE_SIZE, float(size))

    # =========================================================================
    # CONTEXT ENRICHMENT METRICS
    # =========================================================================

    async def record_enrichment(
        self,
        context_count: int,
        duration_ms: float,
        source: str | None = None,
    ) -> None:
        """Record a context enrichment event."""
        tags = {"source": source} if source else {}

        await self.store.increment(MetricNames.ENRICHMENT_TOTAL, tags=tags)

        if context_count > 0:
            await self.store.increment(MetricNames.ENRICHMENT_CONTEXT_ADDED, tags=tags)
            await self.store.record(
                "cl.enrichment.context_count",
                float(context_count),
                MetricType.HISTOGRAM,
                tags=tags,
            )
        else:
            await self.store.increment(MetricNames.ENRICHMENT_NO_CONTEXT, tags=tags)

        await self.store.record(
            MetricNames.ENRICHMENT_DURATION_MS,
            duration_ms,
            MetricType.HISTOGRAM,
            tags=tags,
        )

    # =========================================================================
    # AUDIT PIPELINE METRICS
    # =========================================================================

    async def record_audit_processed(
        self,
        triplets_created: int,
        kg_entries_added: int,
        documents_penalized: int,
        error_type: str | None = None,
    ) -> None:
        """Record an audit pipeline processing event."""
        tags = {"error_type": error_type} if error_type else {}

        await self.store.increment(MetricNames.AUDIT_PROCESSED, tags=tags)

        if triplets_created > 0:
            await self.store.record(
                MetricNames.AUDIT_TRIPLETS_CREATED,
                float(triplets_created),
                MetricType.COUNTER,
                tags=tags,
            )

        if kg_entries_added > 0:
            await self.store.record(
                MetricNames.AUDIT_KG_ENTRIES_ADDED,
                float(kg_entries_added),
                MetricType.COUNTER,
                tags=tags,
            )

        if documents_penalized > 0:
            await self.store.record(
                MetricNames.AUDIT_DOCUMENTS_PENALIZED,
                float(documents_penalized),
                MetricType.COUNTER,
                tags=tags,
            )

    # =========================================================================
    # RAG METRICS
    # =========================================================================

    async def record_rag_retrieval(
        self,
        similarity_score: float,
        feedback_rerank_applied: bool = False,
        boost_factor: float = 0,
        result_count: int = 0,
    ) -> None:
        """Record a RAG retrieval event."""
        await self.store.increment(MetricNames.RAG_RETRIEVAL_TOTAL)

        await self.store.record(
            MetricNames.RAG_SIMILARITY_SCORE,
            similarity_score,
            MetricType.HISTOGRAM,
        )

        if feedback_rerank_applied:
            await self.store.increment(MetricNames.RAG_WITH_FEEDBACK_RERANK)

            if boost_factor != 0:
                await self.store.record(
                    MetricNames.RAG_RERANK_BOOST,
                    boost_factor,
                    MetricType.HISTOGRAM,
                )

    # =========================================================================
    # SYSTEM METRICS
    # =========================================================================

    async def record_system_metrics(
        self,
        memory_mb: float,
        cpu_percent: float,
        db_size_mb: float,
    ) -> None:
        """Record system resource metrics."""
        await self.store.set_gauge(MetricNames.SYSTEM_MEMORY_MB, memory_mb)
        await self.store.set_gauge(MetricNames.SYSTEM_CPU_PERCENT, cpu_percent)
        await self.store.set_gauge(MetricNames.SYSTEM_DB_SIZE_MB, db_size_mb)

    # =========================================================================
    # DASHBOARD DATA
    # =========================================================================

    async def get_dashboard_data(self, hours: int = 24) -> dict[str, Any]:
        """
        Get data formatted for the dashboard.

        Returns:
            Dict with sections for each dashboard panel
        """
        await self.store.initialize()

        # Get summary
        summary = await self.store.get_summary()

        # Get time series for key metrics
        query_series = await self.store.get_aggregated(
            MetricNames.QUERY_TOTAL,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )

        feedback_series = await self.store.get_aggregated(
            MetricNames.FEEDBACK_TOTAL,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )

        response_time_series = await self.store.get_aggregated(
            MetricNames.QUERY_DURATION_MS,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )

        # Format for charts
        def format_series(series, value_key="sum_value"):
            return [
                {
                    "timestamp": m.period_start.isoformat(),
                    "value": getattr(m, value_key),
                }
                for m in series
            ]

        return {
            "summary": summary,
            "charts": {
                "queries": format_series(query_series),
                "feedback": format_series(feedback_series),
                "response_time": format_series(response_time_series, "avg_value"),
            },
            "current": {
                "review_queue_size": self.store.get_gauge(MetricNames.REVIEW_QUEUE_SIZE) or 0,
                "feedback_total": self.store.get_counter(MetricNames.FEEDBACK_TOTAL),
                "feedback_positive_rate": self._calc_positive_rate(),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _calc_positive_rate(self) -> float:
        """Calculate feedback positive rate."""
        total = self.store.get_counter(MetricNames.FEEDBACK_TOTAL)
        positive = self.store.get_counter(MetricNames.FEEDBACK_POSITIVE)

        if total == 0:
            return 0.0
        return positive / total


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_cl_metrics: ContinualLearningMetrics | None = None


def get_cl_metrics() -> ContinualLearningMetrics:
    """Get the global continual learning metrics instance."""
    global _cl_metrics

    if _cl_metrics is None:
        _cl_metrics = ContinualLearningMetrics()
    return _cl_metrics
