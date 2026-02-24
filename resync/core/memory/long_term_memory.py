# ruff: noqa: E501
# pylint: disable=all
# mypy: no-rerun
"""
Long-Term Memory System for Resync v5.2.3.26

Implements Google's Context Engineering principles:
1. Declarative Memory - Facts and preferences about the user
2. Procedural Memory - How the user works (behavior patterns)
3. LLM-driven Memory Extraction - Automatic insight crystallization
4. Provenance Tracking - Source, confidence, timestamps
5. Push vs Pull Retrieval - Proactive vs reactive memory loading

Reference: Google's "Agent Memory" whitepaper (70 pages)

Architecture:
    Session Conversation
           │
           ▼
    ┌──────────────────┐
    │ Memory Extractor │ ← LLM extracts insights
    │     (LLM)        │
    └────────┬─────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐    ┌─────────────┐
│Declarative│   │ Procedural  │
│  Memory   │   │   Memory    │
│ (facts)   │   │ (patterns)  │
└─────┬─────┘   └──────┬──────┘
      │                │
      └───────┬────────┘
              ▼
    ┌──────────────────┐
    │  Vector Store    │ ← Semantic search
    │  (ChromaDB)      │
    └──────────────────┘

Author: Resync Team
Version: 5.2.3.26
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class MemoryType(str, Enum):
    """Types of long-term memory."""

    DECLARATIVE = "declarative"  # Facts, preferences, static info
    PROCEDURAL = "procedural"  # Behavior patterns, workflows


class DeclarativeCategory(str, Enum):
    """Categories of declarative (factual) memory."""

    PREFERENCE = "preference"  # "Prefere respostas em português"
    RESPONSIBILITY = "responsibility"  # "Gerencia jobs do BATCH_NOTURNO"
    ENVIRONMENT = "environment"  # "Trabalha com ambiente PROD"
    EXPERTISE = "expertise"  # "Especialista em jobs DB2"
    CONSTRAINT = "constraint"  # "Só pode reiniciar jobs após 18h"
    RELATIONSHIP = "relationship"  # "Job X depende de Y"


class ProceduralCategory(str, Enum):
    """Categories of procedural (behavioral) memory."""

    TROUBLESHOOTING = "troubleshooting"  # "Verifica logs antes de reiniciar"
    DECISION_MAKING = "decision_making"  # "Prefere ver dependências primeiro"
    COMMUNICATION = "communication"  # "Gosta de respostas detalhadas"
    WORKFLOW = "workflow"  # "Sempre confirma antes de ações"
    INVESTIGATION = "investigation"  # "Começa análise pelo predecessor"


class RetrievalMode(str, Enum):
    """How to retrieve memories."""

    PROACTIVE = "proactive"  # Always include (push)
    REACTIVE = "reactive"  # On-demand semantic search (pull)


# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW = 0.3


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class MemoryProvenance:
    """
    Provenance metadata for a memory.

    Tracks where the memory came from and how reliable it is.
    Essential for debugging and trust.
    """

    source_session_id: str
    source_message: str  # The message that generated this memory
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)
    times_referenced: int = 0
    times_confirmed: int = 0  # User confirmed this is correct
    times_contradicted: int = 0  # User said this is wrong

    @property
    def confidence_adjustment(self) -> float:
        """Calculate confidence adjustment based on usage."""
        if self.times_referenced == 0:
            return 0.0

        # More references + confirmations = higher confidence
        # Contradictions reduce confidence
        positive = self.times_confirmed * 0.1
        negative = self.times_contradicted * 0.2
        usage_bonus = min(self.times_referenced * 0.02, 0.1)

        return positive - negative + usage_bonus

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_session_id": self.source_session_id,
            "source_message": self.source_message,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat(),
            "times_referenced": self.times_referenced,
            "times_confirmed": self.times_confirmed,
            "times_contradicted": self.times_contradicted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryProvenance:
        return cls(
            source_session_id=data.get("source_session_id", ""),
            source_message=data.get("source_message", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            last_verified=datetime.fromisoformat(data["last_verified"])
            if "last_verified" in data
            else datetime.now(timezone.utc),
            times_referenced=data.get("times_referenced", 0),
            times_confirmed=data.get("times_confirmed", 0),
            times_contradicted=data.get("times_contradicted", 0),
        )


@dataclass
class DeclarativeMemory:
    """
    Declarative Memory: Facts and preferences about the user.

    Examples:
    - "User prefers responses in Portuguese"
    - "User manages BATCH_NOTURNO job stream"
    - "User works with PROD environment"
    - "User is allergic to peanuts" (in medical context)

    These are relatively static facts that don't change often.
    """

    id: str
    user_id: str
    category: DeclarativeCategory
    content: str  # The actual fact/preference

    # Metadata
    confidence: float = 0.5  # 0.0 - 1.0
    retrieval_mode: RetrievalMode = RetrievalMode.REACTIVE

    # TWS-specific context
    related_jobs: list[str] = field(default_factory=list)
    related_workstations: list[str] = field(default_factory=list)
    environment: str | None = None  # PROD, DEV, TEST

    # Provenance
    provenance: MemoryProvenance | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None  # None = never expires

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic ID based on content."""
        content_hash = hashlib.md5(
            f"{self.user_id}:{self.category}:{self.content}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]
        return f"decl_{content_hash}"

    @property
    def effective_confidence(self) -> float:
        """Confidence with provenance adjustment."""
        base = self.confidence
        if self.provenance:
            base += self.provenance.confidence_adjustment
        return max(0.0, min(1.0, base))

    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_high_confidence(self) -> bool:
        """Check if memory is high confidence (should be proactive)."""
        return self.effective_confidence >= CONFIDENCE_HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": MemoryType.DECLARATIVE.value,
            "category": self.category.value,
            "content": self.content,
            "confidence": self.confidence,
            "retrieval_mode": self.retrieval_mode.value,
            "related_jobs": self.related_jobs,
            "related_workstations": self.related_workstations,
            "environment": self.environment,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeclarativeMemory:
        provenance = None
        if data.get("provenance"):
            provenance = MemoryProvenance.from_dict(data["provenance"])

        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            category=DeclarativeCategory(data.get("category", "preference")),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.5),
            retrieval_mode=RetrievalMode(data.get("retrieval_mode", "reactive")),
            related_jobs=data.get("related_jobs", []),
            related_workstations=data.get("related_workstations", []),
            environment=data.get("environment"),
            provenance=provenance,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )

    def to_prompt_text(self) -> str:
        """Format memory for inclusion in LLM prompt."""
        confidence_label = (
            "alta"
            if self.is_high_confidence
            else "média"
            if self.effective_confidence >= CONFIDENCE_MEDIUM
            else "baixa"
        )
        return f"[{self.category.value}] {self.content} (confiança: {confidence_label})"


@dataclass
class ProceduralMemory:
    """
    Procedural Memory: How the user works (behavior patterns).

    Examples:
    - "When debugging, user checks logs first"
    - "User prefers to see dependencies before restarting"
    - "User always asks for confirmation before critical actions"
    - "User investigates predecessors when a job fails"

    These capture dynamic behavior patterns learned over time.
    """

    id: str
    user_id: str
    category: ProceduralCategory
    pattern: str  # Description of the behavior pattern

    # Evidence
    examples: list[str] = field(default_factory=list)  # Specific instances
    trigger_conditions: list[str] = field(default_factory=list)  # When this applies

    # Metadata
    confidence: float = 0.5
    times_observed: int = 1
    retrieval_mode: RetrievalMode = RetrievalMode.REACTIVE

    # Provenance
    provenance: MemoryProvenance | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic ID based on content."""
        content_hash = hashlib.md5(
            f"{self.user_id}:{self.category}:{self.pattern}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]
        return f"proc_{content_hash}"

    @property
    def effective_confidence(self) -> float:
        """Confidence with observation count adjustment."""
        # More observations = higher confidence
        observation_bonus = min(self.times_observed * 0.05, 0.3)
        base = self.confidence + observation_bonus

        if self.provenance:
            base += self.provenance.confidence_adjustment

        return max(0.0, min(1.0, base))

    @property
    def is_high_confidence(self) -> bool:
        """Check if memory is high confidence (should be proactive)."""
        return self.effective_confidence >= CONFIDENCE_HIGH

    def observe(self) -> None:
        """Record another observation of this pattern."""
        self.times_observed += 1
        self.last_observed = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

        # Increase confidence with observations
        if self.confidence < 0.9:
            self.confidence = min(0.9, self.confidence + 0.05)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": MemoryType.PROCEDURAL.value,
            "category": self.category.value,
            "pattern": self.pattern,
            "examples": self.examples,
            "trigger_conditions": self.trigger_conditions,
            "confidence": self.confidence,
            "times_observed": self.times_observed,
            "retrieval_mode": self.retrieval_mode.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_observed": self.last_observed.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProceduralMemory:
        provenance = None
        if data.get("provenance"):
            provenance = MemoryProvenance.from_dict(data["provenance"])

        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            category=ProceduralCategory(data.get("category", "workflow")),
            pattern=data.get("pattern", ""),
            examples=data.get("examples", []),
            trigger_conditions=data.get("trigger_conditions", []),
            confidence=data.get("confidence", 0.5),
            times_observed=data.get("times_observed", 1),
            retrieval_mode=RetrievalMode(data.get("retrieval_mode", "reactive")),
            provenance=provenance,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(timezone.utc),
            last_observed=datetime.fromisoformat(data["last_observed"])
            if "last_observed" in data
            else datetime.now(timezone.utc),
        )

    def to_prompt_text(self) -> str:
        """Format memory for inclusion in LLM prompt."""
        return (
            f"[{self.category.value}] {self.pattern} (observado {self.times_observed}x)"
        )


# Type alias for any memory type
Memory = DeclarativeMemory | ProceduralMemory


# =============================================================================
# MEMORY STORE INTERFACE
# =============================================================================


class LongTermMemoryStore(ABC):
    """Abstract interface for long-term memory storage."""

    @abstractmethod
    def save_memory(self, memory: Memory) -> None:
        """Save a memory."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None:
        """Get a memory by ID."""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""

    @abstractmethod
    def get_user_memories(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Memory]:
        """Get all memories for a user with optional filters."""

    @abstractmethod
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[Memory]:
        """Semantic search for relevant memories."""

    @abstractmethod
    def get_proactive_memories(
        self,
        user_id: str,
    ) -> list[Memory]:
        """Get memories that should always be included (push)."""


# =============================================================================
# IN-MEMORY STORE (Development/Testing)
# =============================================================================


class InMemoryLongTermStore(LongTermMemoryStore):
    """
    In-memory implementation for development/testing.

    WARNING: Data is lost on restart.
    """

    def __init__(self):
        self._memories: dict[str, Memory] = {}
        self._user_index: dict[str, set[str]] = {}  # user_id -> memory_ids

    def save_memory(self, memory: Memory) -> None:
        self._memories[memory.id] = memory

        if memory.user_id not in self._user_index:
            self._user_index[memory.user_id] = set()
        self._user_index[memory.user_id].add(memory.id)

        logger.debug("Saved memory %s for user %s", memory.id, memory.user_id)

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id not in self._memories:
            return False

        memory = self._memories[memory_id]
        del self._memories[memory_id]

        if memory.user_id in self._user_index:
            self._user_index[memory.user_id].discard(memory_id)

        return True

    def get_user_memories(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Memory]:
        memory_ids = self._user_index.get(user_id, set())
        results = []

        for mid in memory_ids:
            memory = self._memories.get(mid)
            if not memory:
                continue

            # Filter by type
            if memory_type:
                if (
                    isinstance(memory, DeclarativeMemory)
                    and memory_type != MemoryType.DECLARATIVE
                ):
                    continue
                if (
                    isinstance(memory, ProceduralMemory)
                    and memory_type != MemoryType.PROCEDURAL
                ):
                    continue

            # Filter by category
            if category and memory.category.value != category:
                continue

            # Filter by confidence
            if memory.effective_confidence < min_confidence:
                continue

            # Filter expired
            if isinstance(memory, DeclarativeMemory) and memory.is_expired:
                continue

            results.append(memory)

        # Sort by confidence descending
        results.sort(key=lambda m: m.effective_confidence, reverse=True)
        return results

    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[Memory]:
        """Simple keyword search (production should use vector search)."""
        query_lower = query.lower()
        memories = await self.get_user_memories(user_id)

        scored = []
        for memory in memories:
            content = ""
            if isinstance(memory, DeclarativeMemory):
                content = memory.content.lower()
            elif isinstance(memory, ProceduralMemory):
                content = f"{memory.pattern} {' '.join(memory.examples)}".lower()

            # Simple relevance scoring
            score = 0
            for word in query_lower.split():
                if word in content:
                    score += 1

            if score > 0:
                scored.append((memory, score * memory.effective_confidence))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:limit]]

    async def get_proactive_memories(self, user_id: str) -> list[Memory]:
        """Get memories marked for proactive retrieval."""
        memories = await self.get_user_memories(user_id)
        return [
            m
            for m in memories
            if m.retrieval_mode == RetrievalMode.PROACTIVE or m.is_high_confidence
        ]


# =============================================================================
# REDIS STORE (Production)
# =============================================================================


class RedisLongTermStore(LongTermMemoryStore):
    """
    Redis-backed long-term memory store for production.

    Uses Redis for persistence and optional vector search.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        key_prefix: str = "resync:ltm:",
    ):
        self._redis = None
        self._redis_url = redis_url
        self._key_prefix = key_prefix

    async def _get_redis(self):
        """Lazy initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                redis_url = self._redis_url
                if not redis_url:
                    from resync.settings import settings

                    redis_url = getattr(settings, "redis_url", "redis://localhost:6379")

                self._redis = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info("Redis long-term memory store connected")
            except Exception as e:
                logger.error("Redis connection failed: %s", e)
                raise

        return self._redis

    def _memory_key(self, memory_id: str) -> str:
        return f"{self._key_prefix}memory:{memory_id}"

    def _user_index_key(self, user_id: str) -> str:
        return f"{self._key_prefix}user:{user_id}:memories"

    async def save_memory(self, memory: Memory) -> None:
        redis = await self._get_redis()

        data = json.dumps(memory.to_dict())
        await redis.set(self._memory_key(memory.id), data)
        await redis.sadd(self._user_index_key(memory.user_id), memory.id)

        logger.debug("Saved memory %s to Redis", memory.id)

    async def get_memory(self, memory_id: str) -> Memory | None:
        redis = await self._get_redis()

        data = await redis.get(self._memory_key(memory_id))
        if not data:
            return None

        try:
            parsed = json.loads(data)
            if parsed.get("type") == MemoryType.DECLARATIVE.value:
                return DeclarativeMemory.from_dict(parsed)
            return ProceduralMemory.from_dict(parsed)
        except Exception as e:
            logger.error("Failed to parse memory %s: %s", memory_id, e)
            return None

    async def delete_memory(self, memory_id: str) -> bool:
        redis = await self._get_redis()

        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        await redis.delete(self._memory_key(memory_id))
        await redis.srem(self._user_index_key(memory.user_id), memory_id)

        return True

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Memory]:
        redis = await self._get_redis()

        memory_ids = await redis.smembers(self._user_index_key(user_id))
        results = []

        for mid in memory_ids:
            memory = await self.get_memory(mid)
            if not memory:
                continue

            # Apply filters (same logic as InMemoryStore)
            if memory_type:
                if (
                    isinstance(memory, DeclarativeMemory)
                    and memory_type != MemoryType.DECLARATIVE
                ):
                    continue
                if (
                    isinstance(memory, ProceduralMemory)
                    and memory_type != MemoryType.PROCEDURAL
                ):
                    continue

            if category and memory.category.value != category:
                continue

            if memory.effective_confidence < min_confidence:
                continue

            if isinstance(memory, DeclarativeMemory) and memory.is_expired:
                continue

            results.append(memory)

        results.sort(key=lambda m: m.effective_confidence, reverse=True)
        return results

    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[Memory]:
        # For production, this should use Redis Search or external vector DB
        # For now, use simple keyword matching
        memories = await self.get_user_memories(user_id)
        query_lower = query.lower()

        scored = []
        for memory in memories:
            content = ""
            if isinstance(memory, DeclarativeMemory):
                content = memory.content.lower()
            elif isinstance(memory, ProceduralMemory):
                content = f"{memory.pattern} {' '.join(memory.examples)}".lower()

            score = sum(1 for word in query_lower.split() if word in content)
            if score > 0:
                scored.append((memory, score * memory.effective_confidence))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:limit]]

    async def get_proactive_memories(self, user_id: str) -> list[Memory]:
        memories = await self.get_user_memories(user_id)
        return [
            m
            for m in memories
            if m.retrieval_mode == RetrievalMode.PROACTIVE or m.is_high_confidence
        ]

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# LLM MEMORY EXTRACTOR
# =============================================================================


# Prompt for extracting memories from conversation
MEMORY_EXTRACTION_PROMPT = """Analise a conversa abaixo e extraia insights sobre o usuário que devem ser lembrados para futuras interações.

CONVERSA:
{conversation}

Extraia dois tipos de memórias:

1. DECLARATIVAS (fatos estáticos sobre o usuário):
   - Preferências (idioma, formato de resposta, etc.)
   - Responsabilidades (jobs, sistemas, ambientes que gerencia)
   - Expertise (áreas de conhecimento)
   - Restrições (limitações, horários, permissões)

2. PROCEDURAIS (padrões de comportamento):
   - Como o usuário faz troubleshooting
   - Como toma decisões
   - Preferências de comunicação
   - Fluxos de trabalho típicos

Responda APENAS em JSON válido no formato:
{{
  "declarative": [
    {{
      "category": "preference|responsibility|environment|expertise|constraint|relationship",
      "content": "descrição do fato",
      "confidence": 0.0-1.0,
      "related_jobs": ["JOB1", "JOB2"],
      "environment": "PROD|DEV|TEST|null"
    }}
  ],
  "procedural": [
    {{
      "category": "troubleshooting|decision_making|communication|workflow|investigation",
      "pattern": "descrição do padrão de comportamento",
      "trigger_conditions": ["quando isso acontece"],
      "confidence": 0.0-1.0
    }}
  ]
}}

Se não houver insights para extrair, retorne listas vazias.
Apenas extraia informações claramente demonstradas na conversa, não faça suposições.
"""


class MemoryExtractor:
    """
    LLM-powered memory extraction from conversations.

    Implements Google's "Extract → Consolidate → Load" pipeline.
    """

    def __init__(self, llm_caller: Any = None):
        """
        Initialize extractor.

        Args:
            llm_caller: Function to call LLM (async def call(prompt) -> str)
        """
        self._llm_caller = llm_caller

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for extraction."""
        if self._llm_caller:
            return await self._llm_caller(prompt)

        # Default: try to use project's LLM utility
        try:
            from resync.core.utils.llm import call_llm

            return await call_llm(
                prompt=prompt,
                model="gpt-4o-mini",  # Use smaller model for extraction
                max_tokens=1500,
                temperature=0.3,
            )
        except ImportError:
            logger.warning("No LLM caller available, returning empty extraction")
            return '{"declarative": [], "procedural": []}'

    async def extract_memories(
        self,
        user_id: str,
        conversation: list[dict[str, str]],
        session_id: str,
    ) -> list[Memory]:
        """
        Extract memories from a conversation.

        Args:
            user_id: User identifier
            conversation: List of messages [{"role": "user"|"assistant", "content": "..."}]
            session_id: Source session ID for provenance

        Returns:
            List of extracted memories
        """
        if not conversation:
            return []

        # Format conversation for prompt
        conv_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation]
        )

        prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            response = await self._call_llm(prompt)

            # Parse JSON response
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            memories: list[Memory] = []

            # Create provenance
            source_message = conversation[-1]["content"] if conversation else ""
            provenance = MemoryProvenance(
                source_session_id=session_id,
                source_message=source_message[:200],
            )

            # Process declarative memories
            for decl in data.get("declarative", []):
                try:
                    memory = DeclarativeMemory(
                        id="",  # Will be generated
                        user_id=user_id,
                        category=DeclarativeCategory(
                            decl.get("category", "preference")
                        ),
                        content=decl.get("content", ""),
                        confidence=float(decl.get("confidence", 0.5)),
                        related_jobs=decl.get("related_jobs", []),
                        environment=decl.get("environment"),
                        provenance=provenance,
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.warning("Failed to create declarative memory: %s", e)

            # Process procedural memories
            for proc in data.get("procedural", []):
                try:
                    memory = ProceduralMemory(
                        id="",  # Will be generated
                        user_id=user_id,
                        category=ProceduralCategory(proc.get("category", "workflow")),
                        pattern=proc.get("pattern", ""),
                        trigger_conditions=proc.get("trigger_conditions", []),
                        confidence=float(proc.get("confidence", 0.5)),
                        provenance=provenance,
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.warning("Failed to create procedural memory: %s", e)

            logger.info(f"Extracted {len(memories)} memories from session {session_id}")
            return memories

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            return []
        except Exception as e:
            logger.error("Memory extraction failed: %s", e)
            return []


# =============================================================================
# LONG-TERM MEMORY MANAGER
# =============================================================================


class LongTermMemoryManager:
    """
    High-level manager for long-term memory.

    Provides:
    - Memory extraction from sessions
    - Memory retrieval (push/pull)
    - Memory consolidation (deduplication, updates)
    - Context assembly for prompts

    Usage:
        manager = LongTermMemoryManager()

        # After a session ends, extract memories
        memories = await manager.extract_from_session(user_id, conversation, session_id)

        # Before processing a query, get relevant memories
        context = await manager.get_memory_context(user_id, query)

        # Include in prompt
        full_prompt = f"{context}\n\nUSER: {query}"
    """

    def __init__(
        self,
        store: LongTermMemoryStore | None = None,
        extractor: MemoryExtractor | None = None,
    ):
        self._store = store
        self._extractor = extractor or MemoryExtractor()
        self._initialized = False

    def _ensure_store(self) -> LongTermMemoryStore:
        """Initialize store if needed."""
        if self._store is not None:
            return self._store

        # Try Redis first
        try:
            from resync.settings import settings

            if not getattr(settings, "disable_redis", False):
                self._store = RedisLongTermStore()
                logger.info("Using Redis for long-term memory")
                return self._store
        except Exception as e:
            logger.warning("Redis not available for LTM: %s", e)

        # Fallback to in-memory
        self._store = InMemoryLongTermStore()
        logger.info("Using in-memory store for long-term memory (development mode)")
        return self._store

    # =========================================================================
    # MEMORY EXTRACTION
    # =========================================================================

    async def extract_from_session(
        self,
        user_id: str,
        conversation: list[dict[str, str]],
        session_id: str,
    ) -> list[Memory]:
        """
        Extract and store memories from a completed session.

        This should be called when a session ends or periodically
        during long sessions.

        Args:
            user_id: User identifier
            conversation: Session messages
            session_id: Session ID for provenance

        Returns:
            List of extracted and stored memories
        """
        # Extract memories using LLM
        memories = await self._extractor.extract_memories(
            user_id, conversation, session_id
        )

        if not memories:
            return []

        store = self._ensure_store()

        # Consolidate and store
        stored = []
        for memory in memories:
            # Check for existing similar memory
            existing = await self._find_similar_memory(user_id, memory)

            if existing:
                # Update existing memory
                await self._consolidate_memory(existing, memory)
                stored.append(existing)
            else:
                # Store new memory
                await store.save_memory(memory)
                stored.append(memory)

        logger.info(
            f"Stored {len(stored)} memories for user {user_id} from session {session_id}"
        )
        return stored

    async def _find_similar_memory(
        self,
        user_id: str,
        new_memory: Memory,
    ) -> Memory | None:
        """Find an existing memory similar to the new one."""
        store = self._ensure_store()

        # Get memories of same type and category
        memory_type = (
            MemoryType.DECLARATIVE
            if isinstance(new_memory, DeclarativeMemory)
            else MemoryType.PROCEDURAL
        )

        existing = await store.get_user_memories(
            user_id,
            memory_type=memory_type,
            category=new_memory.category.value,
        )

        # Check for content similarity
        for mem in existing:
            if isinstance(new_memory, DeclarativeMemory) and isinstance(
                mem, DeclarativeMemory
            ):
                if self._content_similar(new_memory.content, mem.content):
                    return mem
            elif isinstance(new_memory, ProceduralMemory) and isinstance(
                mem, ProceduralMemory
            ):
                if self._content_similar(new_memory.pattern, mem.pattern):
                    return mem

        return None

    def _content_similar(
        self, content1: str, content2: str, threshold: float = 0.7
    ) -> bool:
        """Check if two content strings are similar."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    async def _consolidate_memory(
        self,
        existing: Memory,
        new: Memory,
    ) -> None:
        """Consolidate new memory into existing one."""
        store = self._ensure_store()

        # Update confidence (weighted average toward higher)
        existing.confidence = max(existing.confidence, new.confidence)

        # Update timestamps
        existing.updated_at = datetime.now(timezone.utc)

        # For procedural, record observation
        if isinstance(existing, ProceduralMemory) and isinstance(new, ProceduralMemory):
            existing.observe()
            # Add new examples if any
            for example in new.examples:
                if example not in existing.examples:
                    existing.examples.append(example)
                    if len(existing.examples) > 10:
                        existing.examples.pop(0)

        # For declarative, update content if newer is more confident
        if isinstance(existing, DeclarativeMemory) and isinstance(
            new, DeclarativeMemory
        ):
            if new.confidence > existing.confidence:
                existing.content = new.content
            # Merge related entities
            for job in new.related_jobs:
                if job not in existing.related_jobs:
                    existing.related_jobs.append(job)

        await store.save_memory(existing)
        logger.debug("Consolidated memory %s", existing.id)

    # =========================================================================
    # MEMORY RETRIEVAL
    # =========================================================================

    async def get_memory_context(
        self,
        user_id: str,
        query: str | None = None,
        max_memories: int = 10,
    ) -> str:
        """
        Get formatted memory context for LLM prompt.

        Combines proactive (push) and reactive (pull) memories.

        Args:
            user_id: User identifier
            query: Current query for reactive retrieval (optional)
            max_memories: Maximum memories to include

        Returns:
            Formatted string for prompt injection
        """
        store = self._ensure_store()

        memories: list[Memory] = []

        # 1. Always include proactive memories (push)
        proactive = await store.get_proactive_memories(user_id)
        memories.extend(proactive)

        # 2. If query provided, search for reactive memories (pull)
        if query:
            remaining = max_memories - len(memories)
            if remaining > 0:
                reactive = await store.search_memories(user_id, query, limit=remaining)
                # Add only if not already included
                existing_ids = {m.id for m in memories}
                for mem in reactive:
                    if mem.id not in existing_ids:
                        memories.append(mem)

        # 3. Update reference counts
        for mem in memories:
            if mem.provenance:
                mem.provenance.times_referenced += 1
            await store.save_memory(mem)

        # 4. Format for prompt
        if not memories:
            return ""

        return self._format_memories_for_prompt(memories)

    def _format_memories_for_prompt(self, memories: list[Memory]) -> str:
        """Format memories as context block for LLM prompt."""
        lines = ["<user_memory>"]
        lines.append("Informações conhecidas sobre este usuário:")
        lines.append("")

        # Group by type
        declarative = [m for m in memories if isinstance(m, DeclarativeMemory)]
        procedural = [m for m in memories if isinstance(m, ProceduralMemory)]

        if declarative:
            lines.append("FATOS E PREFERÊNCIAS:")
            for mem in declarative:
                lines.append(f"  • {mem.to_prompt_text()}")
            lines.append("")

        if procedural:
            lines.append("PADRÕES DE COMPORTAMENTO:")
            for mem in procedural:
                lines.append(f"  • {mem.to_prompt_text()}")
            lines.append("")

        lines.append("Use estas informações para personalizar sua resposta.")
        lines.append("</user_memory>")

        return "\n".join(lines)

    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================

    async def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        category: str,
        confidence: float = 0.7,
        source_session: str = "manual",
    ) -> Memory:
        """
        Manually add a memory.

        Useful for explicit user preferences or admin input.
        """
        store = self._ensure_store()

        provenance = MemoryProvenance(
            source_session_id=source_session,
            source_message="[manually added]",
        )

        if memory_type == MemoryType.DECLARATIVE:
            memory = DeclarativeMemory(
                id="",
                user_id=user_id,
                category=DeclarativeCategory(category),
                content=content,
                confidence=confidence,
                provenance=provenance,
            )
        else:
            memory = ProceduralMemory(
                id="",
                user_id=user_id,
                category=ProceduralCategory(category),
                pattern=content,
                confidence=confidence,
                provenance=provenance,
            )

        await store.save_memory(memory)
        return memory

    async def confirm_memory(self, memory_id: str) -> bool:
        """User confirms a memory is correct."""
        store = self._ensure_store()
        memory = await store.get_memory(memory_id)

        if not memory or not memory.provenance:
            return False

        memory.provenance.times_confirmed += 1
        memory.provenance.last_verified = datetime.now(timezone.utc)

        # Increase confidence
        if memory.confidence < 0.95:
            memory.confidence = min(0.95, memory.confidence + 0.1)

        # Maybe promote to proactive
        if memory.effective_confidence >= CONFIDENCE_HIGH:
            memory.retrieval_mode = RetrievalMode.PROACTIVE

        await store.save_memory(memory)
        return True

    async def contradict_memory(self, memory_id: str) -> bool:
        """User says a memory is wrong."""
        store = self._ensure_store()
        memory = await store.get_memory(memory_id)

        if not memory or not memory.provenance:
            return False

        memory.provenance.times_contradicted += 1

        # Decrease confidence
        memory.confidence = max(0.1, memory.confidence - 0.2)

        # Demote from proactive if low confidence
        if memory.effective_confidence < CONFIDENCE_MEDIUM:
            memory.retrieval_mode = RetrievalMode.REACTIVE

        await store.save_memory(memory)
        return True

    async def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance)."""
        store = self._ensure_store()
        memories = await store.get_user_memories(user_id)

        count = 0
        for memory in memories:
            if await store.delete_memory(memory.id):
                count += 1

        logger.info("Deleted %s memories for user %s", count, user_id)
        return count

    async def get_statistics(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics for a user."""
        store = self._ensure_store()
        memories = await store.get_user_memories(user_id)

        declarative = [m for m in memories if isinstance(m, DeclarativeMemory)]
        procedural = [m for m in memories if isinstance(m, ProceduralMemory)]
        proactive = [m for m in memories if m.retrieval_mode == RetrievalMode.PROACTIVE]
        high_conf = [m for m in memories if m.effective_confidence >= CONFIDENCE_HIGH]

        return {
            "total_memories": len(memories),
            "declarative_count": len(declarative),
            "procedural_count": len(procedural),
            "proactive_count": len(proactive),
            "high_confidence_count": len(high_conf),
            "categories": {
                "declarative": [m.category.value for m in declarative],
                "procedural": [m.category.value for m in procedural],
            },
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_ltm_manager: LongTermMemoryManager | None = None


def get_long_term_memory() -> LongTermMemoryManager:
    """Get singleton LongTermMemoryManager instance."""
    global _ltm_manager
    if _ltm_manager is None:
        _ltm_manager = LongTermMemoryManager()
    return _ltm_manager


__all__ = [
    # Enums
    "MemoryType",
    "DeclarativeCategory",
    "ProceduralCategory",
    "RetrievalMode",
    # Data classes
    "MemoryProvenance",
    "DeclarativeMemory",
    "ProceduralMemory",
    "Memory",
    # Stores
    "LongTermMemoryStore",
    "InMemoryLongTermStore",
    "RedisLongTermStore",
    # Core classes
    "MemoryExtractor",
    "LongTermMemoryManager",
    # Singleton
    "get_long_term_memory",
]
