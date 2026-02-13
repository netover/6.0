"""
Conversational Memory System for Resync v5.4.0

Provides persistent session-based memory for multi-turn conversations.
Enables follow-up questions like:
- "Show me job X" -> "What's the error?" -> "Restart it"

Storage backends:
- Redis (recommended for production)
- In-memory (fallback for development)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """
    Context from a conversation session.

    Tracks:
    - Recent messages
    - Referenced entities (jobs, workstations)
    - User preferences detected during conversation
    """

    session_id: str
    messages: list[Message] = field(default_factory=list)

    # Entities mentioned in conversation for anaphora resolution
    # e.g., "it" refers to the last mentioned job
    referenced_jobs: list[str] = field(default_factory=list)
    referenced_workstations: list[str] = field(default_factory=list)
    referenced_job_streams: list[str] = field(default_factory=list)

    # Conversation state
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0

    # User preferences detected in session
    user_preferences: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self.last_activity = time.time()
        self.turn_count += 1

        # Extract entity references from user messages
        if role == "user":
            self._extract_entities(content)

    def _extract_entities(self, content: str) -> None:
        """Extract TWS entity references from message content."""
        import re

        # Job name pattern (uppercase alphanumeric with underscore/hyphen)
        job_pattern = r"\b[A-Z][A-Z0-9_\-]{2,39}\b"
        jobs = re.findall(job_pattern, content.upper())

        for job in jobs:
            if job not in self.referenced_jobs:
                self.referenced_jobs.append(job)
                # Keep only last 10 references
                if len(self.referenced_jobs) > 10:
                    self.referenced_jobs.pop(0)

    def get_last_job(self) -> str | None:
        """Get the most recently mentioned job (for 'it' resolution)."""
        return self.referenced_jobs[-1] if self.referenced_jobs else None

    def get_recent_messages(self, n: int = 5) -> list[Message]:
        """Get the N most recent messages."""
        return self.messages[-n:]

    def get_context_for_prompt(self, max_messages: int = 5) -> str:
        """
        Format conversation context for LLM prompt injection.

        Returns a formatted string of recent conversation turns.
        """
        recent = self.get_recent_messages(max_messages)

        if not recent:
            return ""

        lines = ["<conversation_history>"]
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        lines.append("</conversation_history>")

        # Add entity context
        if self.referenced_jobs:
            lines.append("\n<referenced_entities>")
            lines.append(f"Recently mentioned jobs: {', '.join(self.referenced_jobs[-5:])}")
            if self.referenced_workstations:
                lines.append(f"Workstations: {', '.join(self.referenced_workstations[-3:])}")
            lines.append("</referenced_entities>")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "referenced_jobs": self.referenced_jobs,
            "referenced_workstations": self.referenced_workstations,
            "referenced_job_streams": self.referenced_job_streams,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "turn_count": self.turn_count,
            "user_preferences": self.user_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        ctx = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            turn_count=data.get("turn_count", 0),
        )
        ctx.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx.referenced_jobs = data.get("referenced_jobs", [])
        ctx.referenced_workstations = data.get("referenced_workstations", [])
        ctx.referenced_job_streams = data.get("referenced_job_streams", [])
        ctx.user_preferences = data.get("user_preferences", {})
        return ctx


# =============================================================================
# MEMORY STORE INTERFACE
# =============================================================================


class MemoryStore(ABC):
    """Abstract base class for conversation memory storage."""

    @abstractmethod
    def save_context(self, context: ConversationContext) -> None:
        """Save conversation context."""

    @abstractmethod
    def load_context(self, session_id: str) -> ConversationContext | None:
        """Load conversation context by session ID."""

    @abstractmethod
    def delete_context(self, session_id: str) -> bool:
        """Delete conversation context."""

    @abstractmethod
    def list_sessions(self, limit: int = 100) -> list[str]:
        """List active session IDs."""


# =============================================================================
# REDIS MEMORY STORE
# =============================================================================


class RedisMemoryStore(MemoryStore):
    """
    Redis-backed memory store for production use.

    Features:
    - Automatic TTL expiration
    - Cluster-compatible
    - High performance
    """

    def __init__(
        self,
        redis_url: str | None = None,
        key_prefix: str = "resync:memory:",
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._redis = None
        self._redis_url = redis_url

    async def _get_redis(self):
        """Lazy initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                redis_url = self._redis_url
                if not redis_url:
                    from resync.settings import settings

                    redis_url = settings.redis_url

                self._redis = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info("Redis memory store connected")

            except Exception as e:
                logger.error("Redis connection failed: %s", e)
                raise

        return self._redis

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.key_prefix}{session_id}"

    async def save_context(self, context: ConversationContext) -> None:
        """Save context to Redis with TTL."""
        redis = await self._get_redis()
        key = self._key(context.session_id)

        data = json.dumps(context.to_dict())
        await redis.setex(key, self.ttl_seconds, data)

        logger.debug("Saved context %s, %s messages", context.session_id, len(context.messages))

    async def load_context(self, session_id: str) -> ConversationContext | None:
        """Load context from Redis."""
        redis = await self._get_redis()
        key = self._key(session_id)

        data = await redis.get(key)
        if not data:
            return None

        try:
            return ConversationContext.from_dict(json.loads(data))
        except Exception as e:
            logger.error("Failed to parse context: %s", e)
            return None

    async def delete_context(self, session_id: str) -> bool:
        """Delete context from Redis."""
        redis = await self._get_redis()
        key = self._key(session_id)

        result = await redis.delete(key)
        return result > 0

    async def list_sessions(self, limit: int = 100) -> list[str]:
        """List active sessions."""
        redis = await self._get_redis()
        pattern = f"{self.key_prefix}*"

        keys = []
        async for key in redis.scan_iter(pattern, count=limit):
            session_id = key.replace(self.key_prefix, "")
            keys.append(session_id)
            if len(keys) >= limit:
                break

        return keys

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# IN-MEMORY STORE (FALLBACK)
# =============================================================================


class InMemoryStore(MemoryStore):
    """
    In-memory store for development/testing.

    WARNING: Data is lost on restart. Use Redis for production.
    """

    def __init__(self, max_sessions: int = 1000):
        self._store: dict[str, ConversationContext] = {}
        self._max_sessions = max_sessions

    def save_context(self, context: ConversationContext) -> None:
        """Save context to memory."""
        # Evict oldest sessions if at capacity
        if len(self._store) >= self._max_sessions:
            oldest = min(
                self._store.values(),
                key=lambda c: c.last_activity,
            )
            del self._store[oldest.session_id]

        self._store[context.session_id] = context

    def load_context(self, session_id: str) -> ConversationContext | None:
        """Load context from memory."""
        return self._store.get(session_id)

    def delete_context(self, session_id: str) -> bool:
        """Delete context from memory."""
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def list_sessions(self, limit: int = 100) -> list[str]:
        """List session IDs."""
        return list(self._store.keys())[:limit]


# =============================================================================
# MEMORY MANAGER
# =============================================================================


class ConversationMemory:
    """
    High-level conversation memory manager.

    Provides easy-to-use interface for managing conversation context
    across chat sessions.

    Usage:
        memory = ConversationMemory()

        # Start or resume session
        ctx = await memory.get_or_create_session(session_id)

        # Add user message
        ctx.add_message("user", "Show me job AWSBH001")

        # Get context for LLM
        prompt_context = ctx.get_context_for_prompt()

        # Add assistant response
        ctx.add_message("assistant", "Here's job AWSBH001...")

        # Save
        await memory.save_session(ctx)
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        max_messages_per_session: int = 50,
        session_timeout_seconds: int = 3600,
    ):
        self._store = store
        self._max_messages = max_messages_per_session
        self._timeout = session_timeout_seconds
        self._initialized = False

    async def _ensure_store(self) -> MemoryStore:
        """Initialize store if needed."""
        if self._store is not None:
            return self._store

        # Try Redis first, fall back to in-memory
        try:
            from resync.settings import settings

            if not getattr(settings, "disable_redis", False):
                self._store = RedisMemoryStore(
                    ttl_seconds=self._timeout,
                )
                # Test connection
                await self._store._get_redis()
                logger.info("Using Redis for conversation memory")
                return self._store

        except Exception as e:
            logger.warning("Redis not available for memory: %s", e)

        # Fall back to in-memory
        self._store = InMemoryStore()
        logger.info("Using in-memory store for conversation memory (development mode)")
        return self._store

    async def get_or_create_session(
        self,
        session_id: str | None = None,
    ) -> ConversationContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Session ID (generates new if None)

        Returns:
            ConversationContext for the session
        """
        store = await self._ensure_store()

        if session_id:
            ctx = store.load_context(session_id)
            if ctx:
                return ctx

        # Create new session
        new_id = session_id or str(uuid.uuid4())
        return ConversationContext(session_id=new_id)

    async def save_session(self, context: ConversationContext) -> None:
        """
        Save session context.

        Automatically trims old messages if over limit.
        """
        # Trim messages if needed
        if len(context.messages) > self._max_messages:
            context.messages = context.messages[-self._max_messages :]

        store = await self._ensure_store()
        store.save_context(context)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        store = await self._ensure_store()
        return store.delete_context(session_id)

    async def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict | None = None,
    ) -> ConversationContext:
        """
        Add a complete conversation turn (user + assistant).

        Convenience method for adding both messages and saving.
        """
        ctx = await self.get_or_create_session(session_id)

        ctx.add_message("user", user_message, metadata)
        ctx.add_message("assistant", assistant_response)

        await self.save_session(ctx)
        return ctx

    def resolve_reference(
        self,
        context: ConversationContext,
        query: str,
    ) -> str:
        """
        Resolve anaphoric references in query.

        Replaces pronouns like "it", "that job" with actual entity names.

        Args:
            context: Current conversation context
            query: User query with potential references

        Returns:
            Query with resolved references
        """
        # Simple pronoun resolution
        resolved = query

        last_job = context.get_last_job()
        if last_job:
            # Replace common references
            patterns = [
                (r"\bit\b", last_job),
                (r"\bthat job\b", f"job {last_job}"),
                (r"\bthe job\b", f"job {last_job}"),
                (r"\bthis job\b", f"job {last_job}"),
            ]

            import re

            for pattern, replacement in patterns:
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)

        if resolved != query:
            logger.debug("Resolved reference: '%s' -> '%s'", query, resolved)

        return resolved


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_memory_instance: ConversationMemory | None = None


def get_conversation_memory() -> ConversationMemory:
    """Get singleton conversation memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


__all__ = [
    "Message",
    "ConversationContext",
    "MemoryStore",
    "RedisMemoryStore",
    "InMemoryStore",
    "ConversationMemory",
    "get_conversation_memory",
]
