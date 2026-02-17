"""
Context Store - PostgreSQL Implementation.

Provides storage for conversation context and related content.
Replaces the original SQLite implementation.
"""

import logging
from typing import Any

from resync.core.database.models import ContextContent, Conversation
from resync.core.database.repositories import ContextStore as PGContextStore

logger = logging.getLogger(__name__)

__all__ = ["ContextStore", "get_context_store"]


class ContextStore:
    """Context Store - PostgreSQL Backend."""

    def __init__(self) -> None:
        """Initialize - uses PostgreSQL backend."""
        self._store = PGContextStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True
        logger.info("ContextStore initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the store."""
        self._initialized = False

    async def add_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        metadata: dict | None = None,
        embedding_id: str | None = None,
    ) -> Conversation:
        """Add a conversation message."""
        return await self._store.add_conversation(
            session_id=session_id,
            role=role,
            content=content,
            user_id=user_id,
            metadata=metadata,
            embedding_id=embedding_id,
        )

    # Sync wrapper for backward compatibility
    def add_conversation_sync(self, *_args: Any, **_kwargs: Any) -> None:
        """Sync version - deprecated, use async version."""
        logger.warning("add_conversation_sync is deprecated, use async add_conversation")
        raise NotImplementedError("Sync method add_conversation_sync is deprecated, use async add_conversation")

    async def get_session_history(self, session_id: str, limit: int = 100) -> list[Conversation]:
        """Get conversation history for a session."""
        return await self._store.conversations.get_session_history(session_id, limit)

    async def add_content(
        self,
        content_type: str,
        content: str,
        title: str | None = None,
        source: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
        embedding_id: str | None = None,
    ) -> ContextContent:
        """Add context content."""
        return await self._store.content.add_content(
            content_type=content_type,
            content=content,
            title=title,
            source=source,
            summary=summary,
            metadata=metadata,
            embedding_id=embedding_id,
        )

    def add_content_sync(self) -> None:
        """Sync version - deprecated."""
        logger.warning("add_content_sync is deprecated, use async add_content")
        raise NotImplementedError("Sync method add_content_sync is deprecated, use async add_content")

    async def get_relevant_context(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get relevant context for a query."""
        return await self._store.get_relevant_context(query, limit)

    async def search_conversations(self, query: str, limit: int = 50) -> list[Conversation]:
        """Search conversations by content."""
        return await self._store.conversations.search_conversations(query, limit)

    def search_conversations_sync(self, _query: str, _limit: int = 50) -> None:
        """Sync version - deprecated."""
        logger.warning("search_conversations_sync is deprecated, use async search_conversations")
        raise NotImplementedError("Sync method search_conversations_sync is deprecated")

    async def search_similar_issues(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for similar issues/content."""
        return await self._store.get_relevant_context(query, limit)

    def search_similar_issues_sync(self, _query: str, _limit: int = 10) -> None:
        """Sync version - deprecated."""
        logger.warning("search_similar_issues_sync is deprecated, use async search_similar_issues")
        raise NotImplementedError("Sync method search_similar_issues_sync is deprecated")

    async def flag_memory(self, conversation_id: int) -> bool:
        """Flag a conversation for review."""
        result = await self._store.conversations.flag_conversation(conversation_id)
        return result is not None

    async def approve_memory(self, conversation_id: int) -> bool:
        """Approve a flagged conversation."""
        result = await self._store.conversations.approve_conversation(conversation_id)
        return result is not None

    async def is_memory_flagged(self, conversation_id: int) -> bool:
        """Check if a memory is flagged."""
        conv = await self._store.conversations.get_by_id(conversation_id)
        return conv.is_flagged if conv else False

    async def is_memory_approved(self, conversation_id: int) -> bool:
        """Check if a memory is approved."""
        conv = await self._store.conversations.get_by_id(conversation_id)
        return conv.is_approved if conv else False

    async def delete_memory(self, conversation_id: int) -> bool:
        """Delete a conversation."""
        return await self._store.conversations.delete(conversation_id)

    async def add_observations(self, session_id: str, observations: list[str]) -> None:
        """Add observations to a session."""
        for obs in observations:
            await self.add_content(
                content_type="observation", content=obs, metadata={"session_id": session_id}
            )

    async def add_solution_feedback(
        self, session_id: str, problem: str, solution: str, was_helpful: bool
    ) -> None:
        """Add solution feedback."""
        await self.add_content(
            content_type="solution",
            content=solution,
            summary=problem,
            metadata={"session_id": session_id, "was_helpful": was_helpful},
        )

    def add_solution_feedback_sync(self) -> None:
        """Sync version - deprecated."""
        logger.warning("add_solution_feedback_sync is deprecated, use async add_solution_feedback")
        raise NotImplementedError("Sync method add_solution_feedback_sync is deprecated")

    async def atomic_check_and_flag(self, conversation_id: int, expected_approved: bool) -> bool:
        """Atomically check and flag a memory."""
        conv = await self._store.conversations.get_by_id(conversation_id)
        if conv and conv.is_approved == expected_approved:
            await self.flag_memory(conversation_id)
            return True
        return False


_instance: ContextStore | None = None


def get_context_store() -> ContextStore:
    """Get the singleton ContextStore instance."""
    global _instance
    if _instance is None:
        _instance = ContextStore()
    return _instance


async def initialize_context_store() -> ContextStore:
    """Initialize and return the ContextStore."""
    store = get_context_store()
    store.initialize()
    return store
