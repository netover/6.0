"""
Type definitions for Semantic Cache.

Separated from semantic_cache.py to allow lightweight imports and better testing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

@dataclass
class CacheEntry:
    """
    Represents a cached LLM response.

    Attributes:
        query: Original user query
        response: LLM's response
        embedding: Query embedding vector
        timestamp: When entry was created
        hit_count: How many times this entry was returned
        metadata: Additional info (model used, latency, etc.)
    """

    query: str
    response: str
    embedding: list[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "query": self.query,
            "query_text": self.query,
            "response": self.response,
            "embedding": json.dumps(self.embedding),
            "timestamp": self.timestamp.isoformat(),
            "hit_count": self.hit_count,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from Redis hash data."""
        query = data.get("query") or data.get("query_text") or ""

        raw_embedding = data.get("embedding", "[]")
        try:
            embedding = json.loads(raw_embedding)
            if not isinstance(embedding, list):
                embedding = []
        except (TypeError, json.JSONDecodeError):
            embedding = []

        raw_metadata = data.get("metadata", "{}")
        try:
            metadata = json.loads(raw_metadata)
            if not isinstance(metadata, dict):
                metadata = {}
        except (TypeError, json.JSONDecodeError):
            metadata = {}

        raw_timestamp = data.get("timestamp")
        try:
            timestamp = (
                datetime.fromisoformat(raw_timestamp) if raw_timestamp else _EPOCH_UTC
            )
        except (TypeError, ValueError):
            timestamp = _EPOCH_UTC

        try:
            hit_count = int(data.get("hit_count", 0))
        except (TypeError, ValueError):
            hit_count = 0

        return cls(
            query=query,
            response=data.get("response", ""),
            embedding=embedding,
            timestamp=timestamp,
            hit_count=hit_count,
            metadata=metadata,
        )

@dataclass
class CacheResult:
    """
    Result of a cache lookup.

    Attributes:
        hit: Whether cache was hit
        response: Cached response (None if miss)
        distance: Semantic distance from original query (0 = exact match)
        entry: Full cache entry (for metrics)
        lookup_time_ms: Time taken for lookup
        reranked: Whether cross-encoder reranking was applied
        rerank_score: Cross-encoder similarity score (if reranked)
    """

    hit: bool
    response: str | None = None
    distance: float = 1.0
    entry: CacheEntry | None = None
    lookup_time_ms: float = 0.0
    reranked: bool = False
    rerank_score: float | None = None
