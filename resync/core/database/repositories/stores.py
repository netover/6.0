"""
Additional Store Repositories.

PostgreSQL implementations for Context, Audit, Analytics, Learning, and Metrics stores.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, Text
from sqlalchemy.ext.asyncio import async_sessionmaker

from resync.core.database.models import (
    ActiveLearningCandidate,
    # Audit
    AuditEntry,
    AuditQueueItem,
    ContextContent,
    # Context
    Conversation,
    # Learning
    Feedback,
    LearningThreshold,
    MetricAggregation,
    # Metrics
    MetricDataPoint,
    SessionHistory,
    # Analytics
    UserProfile,
)

from .base import BaseRepository, TimestampedRepository

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT REPOSITORIES
# =============================================================================


class ConversationRepository(TimestampedRepository[Conversation]):
    """Repository for conversation history."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(Conversation, session_factory)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_id: str | None = None,
    ) -> Conversation:
        """Add a conversation message."""
        return await self.create(
            session_id=session_id,
            role=role,
            content=content,
            user_id=user_id,
            metadata_=metadata or {},
            embedding_id=embedding_id,
        )

    async def get_session_history(
        self, session_id: str, limit: int = 100
    ) -> list[Conversation]:
        """Get conversation history for a session."""
        return await self.find(
            {"session_id": session_id},
            limit=limit,
            order_by="timestamp",
            desc=False,  # Chronological order
        )

    async def search_conversations(
        self, query: str, limit: int = 50
    ) -> list[Conversation]:
        """Search conversations by content (full-text search)."""
        async with self._get_session() as session:
            # PostgreSQL full-text search
            result = await session.execute(
                select(Conversation)
                .where(Conversation.content.ilike(f"%{query}%"))
                .order_by(Conversation.timestamp.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def flag_conversation(self, conv_id: int) -> Conversation | None:
        """Flag a conversation for review."""
        return await self.update(conv_id, is_flagged=True, is_approved=False)

    async def approve_conversation(self, conv_id: int) -> Conversation | None:
        """Approve a flagged conversation."""
        return await self.update(conv_id, is_approved=True)


class ContextContentRepository(BaseRepository[ContextContent]):
    """Repository for context content."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(ContextContent, session_factory)

    async def add_content(
        self,
        content_type: str,
        content: str,
        title: str | None = None,
        source: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_id: str | None = None,
    ) -> ContextContent:
        """Add context content."""
        return await self.create(
            content_type=content_type,
            content=content,
            title=title,
            source=source,
            summary=summary,
            metadata_=metadata or {},
            embedding_id=embedding_id,
        )

    async def search_content(
        self, query: str, content_type: str | None = None, limit: int = 50
    ) -> list[ContextContent]:
        """Search content."""
        async with self._get_session() as session:
            q = select(ContextContent).where(
                and_(
                    ContextContent.is_active,
                    or_(
                        ContextContent.content.ilike(f"%{query}%"),
                        ContextContent.title.ilike(f"%{query}%"),
                    ),
                )
            )

            if content_type:
                q = q.where(ContextContent.content_type == content_type)

            q = q.order_by(ContextContent.updated_at.desc()).limit(limit)
            result = await session.execute(q)
            return list(result.scalars().all())


class ContextStore:
    """Unified Context Store facade."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        self.conversations = ConversationRepository(session_factory)
        self.content = ContextContentRepository(session_factory)

    async def add_conversation(
        self, session_id: str, role: str, content: str, **kwargs
    ) -> Conversation:
        """Add a conversation message."""
        return await self.conversations.add_message(session_id, role, content, **kwargs)

    async def get_relevant_context(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get relevant context for a query."""
        convs = await self.conversations.search_conversations(query, limit=limit // 2)
        content = await self.content.search_content(query, limit=limit // 2)

        results = []
        for c in convs:
            results.append(
                {
                    "type": "conversation",
                    "content": c.content,
                    "timestamp": c.timestamp.isoformat(),
                    "session_id": c.session_id,
                }
            )
        for c in content:
            results.append(
                {
                    "type": c.content_type,
                    "content": c.content,
                    "title": c.title,
                    "source": c.source,
                }
            )

        return results


# =============================================================================
# AUDIT REPOSITORIES
# =============================================================================


class AuditEntryRepository(TimestampedRepository[AuditEntry]):
    """Repository for audit entries."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(AuditEntry, session_factory)

    async def log_action(
        self,
        action: str,
        user_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        old_value: dict | None = None,
        new_value: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """Log an audit action."""
        return await self.create(
            action=action,
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            old_value=old_value,
            new_value=new_value,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata_=metadata,
        )

    async def get_user_actions(
        self, user_id: str, limit: int = 100
    ) -> list[AuditEntry]:
        """Get actions by user."""
        return await self.find(
            {"user_id": user_id}, limit=limit, order_by="timestamp", desc=True
        )

    async def get_entity_history(
        self, entity_type: str, entity_id: str, limit: int = 100
    ) -> list[AuditEntry]:
        """Get audit history for an entity."""
        return await self.find(
            {"entity_type": entity_type, "entity_id": entity_id},
            limit=limit,
            order_by="timestamp",
            desc=True,
        )

    async def search(
        self,
        query: str,
        action: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Search audit entries with keyword matching.

        Searches in action, entity_type, and metadata fields.
        """
        async with self._get_session() as session:
            # Base query
            q = select(AuditEntry)

            # Apply filters
            filters = []
            if action:
                filters.append(AuditEntry.action == action)
            if entity_type:
                filters.append(AuditEntry.entity_type == entity_type)

            # Keyword search in action, entity_type and metadata (JSONB)
            keyword = f"%{query}%"
            search_filters = or_(
                AuditEntry.action.ilike(keyword),
                AuditEntry.entity_type.ilike(keyword),
                # Search in JSONB metadata by casting to text
                func.cast(AuditEntry.metadata_, Text).ilike(keyword),
            )
            filters.append(search_filters)

            if filters:
                q = q.where(and_(*filters))

            # Order and limit
            q = q.order_by(AuditEntry.timestamp.desc()).limit(limit)

            result = await session.execute(q)
            return list(result.scalars().all())


class AuditQueueRepository(BaseRepository[AuditQueueItem]):
    """Repository for audit queue."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(AuditQueueItem, session_factory)

    async def enqueue(
        self, action: str, payload: dict[str, Any], priority: int = 0
    ) -> AuditQueueItem:
        """Add item to queue."""
        return await self.create(action=action, payload=payload, priority=priority)

    async def get_pending(self, limit: int = 10) -> list[AuditQueueItem]:
        """Get pending items ordered by priority."""
        async with self._get_session() as session:
            result = await session.execute(
                select(AuditQueueItem)
                .where(AuditQueueItem.status == "pending")
                .order_by(
                    AuditQueueItem.priority.desc(), AuditQueueItem.created_at.asc()
                )
                .limit(limit)
            )
            return list(result.scalars().all())

    async def mark_processing(self, item_id: int) -> AuditQueueItem | None:
        """Mark item as processing."""
        return await self.update(item_id, status="processing")

    async def mark_completed(self, item_id: int) -> AuditQueueItem | None:
        """Mark item as completed."""
        return await self.update(
            item_id, status="completed", completed_at=datetime.now(timezone.utc)
        )

    async def mark_failed(
        self, item_id: int, error_message: str
    ) -> AuditQueueItem | None:
        """Mark item as failed."""
        async with self._get_session() as session:
            item = await session.get(AuditQueueItem, item_id)
            if item:
                item.retry_count += 1
                item.error_message = error_message
                if item.retry_count >= item.max_retries:
                    item.status = "failed"
                else:
                    item.status = "pending"
                await session.commit()
                await session.refresh(item)
            return item


# =============================================================================
# ANALYTICS (USER BEHAVIOR) REPOSITORIES
# =============================================================================


class UserProfileRepository(BaseRepository[UserProfile]):
    """Repository for user profiles."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(UserProfile, session_factory)

    async def get_or_create(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        profile = await self.find_one({"user_id": user_id})
        if not profile:
            profile = await self.create(user_id=user_id)
        return profile

    async def update_activity(
        self,
        user_id: str,
        increment_sessions: bool = False,
        increment_queries: bool = False,
    ) -> UserProfile:
        """Update user activity stats."""
        profile = await self.get_or_create(user_id)

        updates = {"last_active": datetime.now(timezone.utc)}
        if increment_sessions:
            updates["total_sessions"] = profile.total_sessions + 1
        if increment_queries:
            updates["total_queries"] = profile.total_queries + 1

        return await self.update(profile.id, **updates)

    async def update_preferences(
        self, user_id: str, preferences: dict[str, Any]
    ) -> UserProfile:
        """Update user preferences."""
        profile = await self.get_or_create(user_id)
        current_prefs = profile.preferences or {}
        current_prefs.update(preferences)
        return await self.update(profile.id, preferences=current_prefs)


class SessionHistoryRepository(TimestampedRepository[SessionHistory]):
    """Repository for session history."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(SessionHistory, session_factory)

    async def start_session(self, session_id: str, user_id: str) -> SessionHistory:
        """Start a new session."""
        return await self.create(session_id=session_id, user_id=user_id)

    async def end_session(self, session_id: str) -> SessionHistory | None:
        """End a session."""
        session = await self.find_one({"session_id": session_id})
        if session:
            ended_at = datetime.now(timezone.utc)
            duration = int((ended_at - session.started_at).total_seconds())
            return await self.update(
                session.id, ended_at=ended_at, duration_seconds=duration
            )
        return None

    async def increment_queries(self, session_id: str) -> SessionHistory | None:
        """Increment query count for session."""
        session = await self.find_one({"session_id": session_id})
        if session:
            return await self.update(session.id, query_count=session.query_count + 1)
        return None


class UserBehaviorStore:
    """Unified User Behavior Store facade."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        self.profiles = UserProfileRepository(session_factory)
        self.sessions = SessionHistoryRepository(session_factory)


# =============================================================================
# LEARNING REPOSITORIES
# =============================================================================


class FeedbackRepository(TimestampedRepository[Feedback]):
    """Repository for feedback."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(Feedback, session_factory)

    async def add_feedback(
        self,
        session_id: str,
        rating: int | None = None,
        feedback_type: str = "general",
        feedback_text: str | None = None,
        query_text: str | None = None,
        response_text: str | None = None,
        is_positive: bool | None = None,
        metadata: dict | None = None,
    ) -> Feedback:
        """Add feedback."""
        return await self.create(
            session_id=session_id,
            rating=rating,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            query_text=query_text,
            response_text=response_text,
            is_positive=is_positive,
            metadata_=metadata,
        )

    async def get_positive_examples(self, limit: int = 100) -> list[Feedback]:
        """Get positive feedback examples."""
        async with self._get_session() as session:
            result = await session.execute(
                select(Feedback)
                .where(or_(Feedback.is_positive, Feedback.rating >= 4))
                .order_by(Feedback.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_negative_examples(self, limit: int = 100) -> list[Feedback]:
        """Get negative feedback examples."""
        async with self._get_session() as session:
            result = await session.execute(
                select(Feedback)
                .where(or_(Feedback.is_positive.is_(False), Feedback.rating <= 2))
                .order_by(Feedback.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())


class LearningThresholdRepository(BaseRepository[LearningThreshold]):
    """Repository for learning thresholds."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(LearningThreshold, session_factory)

    async def get_threshold(self, name: str) -> float | None:
        """Get threshold value by name."""
        threshold = await self.find_one({"threshold_name": name})
        return threshold.current_value if threshold else None

    async def set_threshold(
        self, name: str, value: float, min_value: float = 0.0, max_value: float = 1.0
    ) -> LearningThreshold:
        """Set or create threshold."""
        threshold = await self.find_one({"threshold_name": name})
        if threshold:
            # Record history
            history = threshold.adjustment_history or {"changes": []}
            history["changes"].append(
                {
                    "old_value": threshold.current_value,
                    "new_value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            return await self.update(
                threshold.id,
                current_value=value,
                last_adjusted=datetime.now(timezone.utc),
                adjustment_history=history,
            )
        return await self.create(
            threshold_name=name,
            current_value=value,
            min_value=min_value,
            max_value=max_value,
        )


class ActiveLearningRepository(BaseRepository[ActiveLearningCandidate]):
    """Repository for active learning candidates."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(ActiveLearningCandidate, session_factory)

    async def add_candidate(
        self,
        query_text: str,
        uncertainty_score: float,
        response_text: str | None = None,
        metadata: dict | None = None,
    ) -> ActiveLearningCandidate:
        """Add a candidate for review."""
        return await self.create(
            query_text=query_text,
            uncertainty_score=uncertainty_score,
            response_text=response_text,
            metadata_=metadata,
        )

    async def get_top_candidates(
        self, limit: int = 10
    ) -> list[ActiveLearningCandidate]:
        """Get top uncertain candidates for review."""
        async with self._get_session() as session:
            result = await session.execute(
                select(ActiveLearningCandidate)
                .where(ActiveLearningCandidate.status == "pending")
                .order_by(ActiveLearningCandidate.uncertainty_score.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def review_candidate(
        self, candidate_id: int, selected_label: str, reviewer_id: str
    ) -> ActiveLearningCandidate | None:
        """Mark candidate as reviewed."""
        return await self.update(
            candidate_id,
            status="reviewed",
            selected_label=selected_label,
            reviewer_id=reviewer_id,
            reviewed_at=datetime.now(timezone.utc),
        )


class FeedbackStore:
    """Unified Feedback/Learning Store facade."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        self.feedback = FeedbackRepository(session_factory)
        self.thresholds = LearningThresholdRepository(session_factory)
        self.active_learning = ActiveLearningRepository(session_factory)


# =============================================================================
# METRICS REPOSITORIES
# =============================================================================


class MetricDataPointRepository(TimestampedRepository[MetricDataPoint]):
    """Repository for metric data points."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(MetricDataPoint, session_factory)

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> MetricDataPoint:
        """Record a metric data point."""
        return await self.create(
            metric_name=metric_name, value=value, unit=unit, tags=tags
        )

    async def get_metric_values(
        self, metric_name: str, start: datetime, end: datetime
    ) -> list[MetricDataPoint]:
        """Get metric values in time range."""
        return await self.find_in_range(
            start, end, filters={"metric_name": metric_name}
        )

    async def get_metric_stats(
        self, metric_name: str, start: datetime, end: datetime
    ) -> dict[str, float]:
        """Get aggregate stats for a metric."""
        async with self._get_session() as session:
            result = await session.execute(
                select(
                    func.min(MetricDataPoint.value).label("min"),
                    func.max(MetricDataPoint.value).label("max"),
                    func.avg(MetricDataPoint.value).label("avg"),
                    func.sum(MetricDataPoint.value).label("sum"),
                    func.count(MetricDataPoint.id).label("count"),
                ).where(
                    and_(
                        MetricDataPoint.metric_name == metric_name,
                        MetricDataPoint.timestamp >= start,
                        MetricDataPoint.timestamp <= end,
                    )
                )
            )
            row = result.one_or_none()
            if row:
                return {
                    "min": float(row.min) if row.min else 0,
                    "max": float(row.max) if row.max else 0,
                    "avg": float(row.avg) if row.avg else 0,
                    "sum": float(row.sum) if row.sum else 0,
                    "count": int(row.count) if row.count else 0,
                }
            return {"min": 0, "max": 0, "avg": 0, "sum": 0, "count": 0}


class MetricAggregationRepository(BaseRepository[MetricAggregation]):
    """Repository for pre-aggregated metrics."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        super().__init__(MetricAggregation, session_factory)


class MetricsStore:
    """Unified Metrics Store facade."""

    def __init__(self, session_factory: async_sessionmaker | None = None):
        self.data_points = MetricDataPointRepository(session_factory)
        self.aggregations = MetricAggregationRepository(session_factory)

    async def record(self, metric_name: str, value: float, **kwargs) -> MetricDataPoint:
        """Record a metric."""
        return await self.data_points.record_metric(metric_name, value, **kwargs)

    async def query(
        self, metric_name: str, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Query metric data."""
        points = await self.data_points.get_metric_values(metric_name, start, end)
        return [
            {"timestamp": p.timestamp.isoformat(), "value": p.value, "tags": p.tags}
            for p in points
        ]

    async def cleanup(self, days: int = 30) -> int:
        """Clean up old metrics."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return await self.data_points.delete_older_than(cutoff)
