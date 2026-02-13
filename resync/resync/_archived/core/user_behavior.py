"""
User Behavior Analytics - PostgreSQL Implementation.

Provides user behavior tracking and analytics using PostgreSQL.
Replaces the original SQLite implementation.
"""

import logging
from typing import Any

from resync.core.database.models import SessionHistory, UserProfile
from resync.core.database.repositories import UserBehaviorStore

logger = logging.getLogger(__name__)

__all__ = ["UserBehaviorAnalyzer", "get_user_behavior_analyzer"]


class UserBehaviorAnalyzer:
    """User Behavior Analyzer - PostgreSQL Backend."""

    def __init__(self, db_path: str | None = None):
        """Initialize. db_path is ignored - uses PostgreSQL."""
        if db_path:
            logger.debug(f"db_path ignored, using PostgreSQL: {db_path}")
        self._store = UserBehaviorStore()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the analyzer."""
        self._initialized = True
        logger.info("UserBehaviorAnalyzer initialized (PostgreSQL)")

    async def close(self) -> None:
        """Close the analyzer."""
        self._initialized = False

    # Profile Methods
    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        return await self._store.profiles.get_or_create(user_id)

    async def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Get user profile."""
        return await self._store.profiles.find_one({"user_id": user_id})

    async def update_preferences(self, user_id: str, preferences: dict[str, Any]) -> UserProfile:
        """Update user preferences."""
        return await self._store.profiles.update_preferences(user_id, preferences)

    async def record_activity(
        self, user_id: str, increment_sessions: bool = False, increment_queries: bool = False
    ) -> UserProfile:
        """Record user activity."""
        return await self._store.profiles.update_activity(
            user_id, increment_sessions, increment_queries
        )

    # Session Methods
    async def start_session(self, session_id: str, user_id: str) -> SessionHistory:
        """Start a new session."""
        # Update profile session count
        await self._store.profiles.update_activity(user_id, increment_sessions=True)
        return await self._store.sessions.start_session(session_id, user_id)

    async def end_session(self, session_id: str) -> SessionHistory | None:
        """End a session."""
        return await self._store.sessions.end_session(session_id)

    async def record_query(self, session_id: str, user_id: str) -> None:
        """Record a query in session."""
        await self._store.sessions.increment_queries(session_id)
        await self._store.profiles.update_activity(user_id, increment_queries=True)

    async def get_session_history(self, user_id: str, limit: int = 50) -> list[SessionHistory]:
        """Get user's session history."""
        return await self._store.sessions.find(
            {"user_id": user_id}, limit=limit, order_by="started_at", desc=True
        )

    # Analytics Methods
    async def get_user_stats(self, user_id: str) -> dict[str, Any]:
        """Get user statistics."""
        profile = await self.get_user_profile(user_id)
        if not profile:
            return {"error": "User not found"}

        return {
            "user_id": user_id,
            "total_sessions": profile.total_sessions,
            "total_queries": profile.total_queries,
            "skill_level": profile.skill_level,
            "last_active": profile.last_active.isoformat() if profile.last_active else None,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
        }

    async def get_behavior_patterns(self, user_id: str) -> dict[str, Any]:
        """Get user behavior patterns."""
        profile = await self.get_user_profile(user_id)
        return profile.behavior_patterns if profile and profile.behavior_patterns else {}

    async def update_behavior_patterns(self, user_id: str, patterns: dict[str, Any]) -> UserProfile:
        """Update user behavior patterns."""
        profile = await self.get_or_create_profile(user_id)
        current = profile.behavior_patterns or {}
        current.update(patterns)
        return await self._store.profiles.update(profile.id, behavior_patterns=current)


_instance: UserBehaviorAnalyzer | None = None


def get_user_behavior_analyzer() -> UserBehaviorAnalyzer:
    """Get the singleton UserBehaviorAnalyzer instance."""
    global _instance
    if _instance is None:
        _instance = UserBehaviorAnalyzer()
    return _instance


async def initialize_user_behavior_analyzer() -> UserBehaviorAnalyzer:
    """Initialize and return the UserBehaviorAnalyzer."""
    analyzer = get_user_behavior_analyzer()
    await analyzer.initialize()
    return analyzer
