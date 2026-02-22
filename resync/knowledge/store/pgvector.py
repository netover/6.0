# pylint: skip-file
# mypy: ignore-errors
"""
PgVector Service - PostgreSQL Vector Similarity Search.

This module provides backward-compatible interface to pgvector.
The actual implementation is in pgvector_store.py with Binary+Halfvec optimization.

For new code, use PgVectorStore directly from pgvector_store.py.

Author: Resync Team
Version: 5.9.0
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = structlog.get_logger(__name__)


class DistanceMetric(str, Enum):
    """Supported distance metrics for similarity search."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


@dataclass
class VectorDocument:
    """Document with embedding for vector storage."""

    document_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: int = 0


@dataclass
class SearchResult:
    """Result from similarity search."""

    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: int = 0

    @property
    def similarity(self) -> float:
        """Get similarity score (higher is better for cosine)."""
        return 1.0 - self.score


@dataclass
class CollectionStats:
    """Statistics for a collection."""

    name: str
    document_count: int
    unique_documents: int


class PgVectorService:
    """
    PostgreSQL-based vector service with Binary+Halfvec optimization.

    This class provides backward-compatible interface.
    Uses two-phase search: Binary HNSW → Halfvec rescoring.
    """

    def __init__(
        self,
        pool,
        default_collection: str = "documents",
        embedding_dimension: int = 1536,
    ):
        self._pool = pool
        self._default_collection = default_collection
        self._embedding_dimension = embedding_dimension
        self._initialized = False

    @classmethod
    async def create(
        cls,
        pool,
        default_collection: str = "documents",
        embedding_dimension: int = 1536,
    ) -> PgVectorService:
        """Create and initialize service."""
        service = cls(pool, default_collection, embedding_dimension)
        service.initialize()
        return service

    def initialize(self) -> None:
        """Initialize service (tables are created by migration)."""
        self._initialized = True
        logger.info(
            "pgvector_service_initialized",
            dimension=self._embedding_dimension,
            mode="binary_halfvec",
        )

    async def upsert(
        self,
        documents: list[VectorDocument],
        collection: str | None = None,
    ) -> int:
        """Upsert documents."""
        if not documents:
            return 0

        collection = collection or self._default_collection
        count = 0

        async with self._pool.acquire() as conn:
            for doc in documents:
                if doc.embedding is None:
                    logger.warning(
                        "document_missing_embedding", document_id=doc.document_id
                    )
                    continue

                embedding_str = f"[{','.join(str(x) for x in doc.embedding)}]"

                await conn.execute(
                    """
                    INSERT INTO document_embeddings
                    (collection_name, document_id, chunk_id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                    ON CONFLICT (collection_name, document_id, chunk_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    collection,
                    doc.document_id,
                    doc.chunk_id,
                    doc.content,
                    embedding_str,
                    doc.metadata,
                )
                count += 1

        logger.debug("upserted", collection=collection, count=count)
        return count

    async def search(
        self,
        query_embedding: list[float],
        collection: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[SearchResult]:
        """
        Search using optimized two-phase Binary+Halfvec search.
        """
        collection = collection or self._default_collection

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        binary_str = "".join("1" if v > 0 else "0" for v in query_embedding)

        # Build filter clause
        filter_clause = ""
        params: list[str | int] = [embedding_str, binary_str, collection]
        param_idx = 4

        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_clause += f" AND metadata->>'{key}' = ${param_idx}"
                params.append(str(value))
                param_idx += 1

        # Calculate candidates
        candidates = max(limit * 10, 50)

        # Threshold clause - validate numeric value to prevent SQL injection
        threshold_clause = ""
        if score_threshold is not None and metric == DistanceMetric.COSINE:
            if (
                not isinstance(score_threshold, (int, float))
                or score_threshold < 0
                or score_threshold > 1
            ):
                raise ValueError(
                    f"score_threshold must be a number between 0 and 1, got: {score_threshold}"
                )
            threshold_clause = f"WHERE similarity >= {float(score_threshold)}"

        query = f"""
            WITH binary_candidates AS (
                SELECT document_id, chunk_id, content, metadata, embedding_half
                FROM document_embeddings
                WHERE collection_name = $3
                {filter_clause}
                ORDER BY binary_quantize(embedding_half)::bit({self._embedding_dimension}) <~> $2::bit({self._embedding_dimension})
                LIMIT {candidates}
            ),
            rescored AS (
                SELECT
                    document_id, chunk_id, content, metadata,
                    1 - (embedding_half <=> $1::halfvec) AS similarity
                FROM binary_candidates
            )
            SELECT document_id, chunk_id, content, metadata, 1 - similarity AS distance
            FROM rescored
            {threshold_clause}
            ORDER BY similarity DESC
            LIMIT ${param_idx}
        """
        params.append(limit)  # int is valid for LIMIT in asyncpg

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            results.append(
                SearchResult(
                    document_id=row["document_id"],
                    content=row["content"],
                    score=float(row["distance"]),
                    metadata=dict(row["metadata"]) if row["metadata"] else {},
                    chunk_id=row["chunk_id"],
                )
            )

        logger.debug(
            "search_completed",
            collection=collection,
            results=len(results),
            mode="binary_halfvec",
        )

        return results

    async def delete(
        self,
        document_id: str,
        collection: str | None = None,
    ) -> bool:
        """Delete document."""
        collection = collection or self._default_collection

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE collection_name = $1 AND document_id = $2
                """,
                collection,
                document_id,
            )

        deleted = result.split()[-1]
        return int(deleted) > 0

    async def get_collection_stats(
        self, collection: str | None = None
    ) -> CollectionStats:
        """Get collection statistics."""
        collection = collection or self._default_collection

        async with self._pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as document_count,
                    COUNT(DISTINCT document_id) as unique_documents
                FROM document_embeddings
                WHERE collection_name = $1
                """,
                collection,
            )

        return CollectionStats(
            name=collection,
            document_count=stats["document_count"],
            unique_documents=stats["unique_documents"],
        )


# =============================================================================
# Singleton accessor
# =============================================================================

_pool = None
_vector_service: PgVectorService | None = None


async def get_vector_service() -> PgVectorService:
    """Get singleton vector service instance."""
    global _pool, _vector_service

    if _vector_service is not None:
        return _vector_service

    import os

    import asyncpg

    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/resync")
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    _pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
    )

    _vector_service = await PgVectorService.create(_pool)
    return _vector_service
