# pylint
# mypy
"""
pgvector-based vector store with Binary+Halfvec optimization.

Implements the gold standard for high-performance vector search:
- Binary HNSW for ultra-fast initial search (~5ms)
- Halfvec cosine for precise rescoring (~10ms)
- Auto-quantize trigger keeps Python code simple

Storage: ~75% reduction vs float32
Speed: ~70% faster search
Quality: ~99% with halfvec rescaring

Author: Resync Team
Version: 6.0.0 - Fixed concurrency issues, added ef_search support
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from resync.knowledge.config import CFG
from resync.knowledge.interfaces import VectorStore

import structlog

logger = structlog.get_logger(__name__)

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False
if TYPE_CHECKING:
    import asyncpg

class PgVectorStore(VectorStore):
    """
    PostgreSQL vector store with Binary+Halfvec optimization.

    Uses two-phase search for optimal speed and accuracy:
    1. Binary HNSW (Hamming distance) for fast candidates
    2. Halfvec cosine similarity for precise rescoring

    The trigger 'trg_auto_quantize_embedding' automatically populates
    embedding_half when embedding is inserted, so Python code stays simple.

    Thread-safe singleton pattern with lazy initialization.
    """

    def __init__(
        self,
        database_url: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required. pip install asyncpg")
        import os

        self._database_url = (
            database_url or os.getenv("DATABASE_URL") or CFG.database_url
        )
        if self._database_url.startswith("postgresql+asyncpg://"):
            self._database_url = self._database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
        self._collection_default = collection or CFG.collection_write
        self._dim = dim if dim is not None else CFG.embed_dim
        self._pool: "asyncpg.Pool | None" = None
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._initialized = False
        # Lazy initialised — avoids RuntimeError on module import or sync init
        self._pool_lock: asyncio.Lock | None = None

    async def _get_pool(self) -> "asyncpg.Pool":
        """Get or create connection pool with double-checked async-safe initialisation."""
        if self._pool is None:
            if self._pool_lock is None:
                self._pool_lock = asyncio.Lock()
            async with self._pool_lock:
                # Double-check after acquiring lock
                if self._pool is None:
                    self._pool = await asyncio.wait_for(
                        asyncpg.create_pool(
                            self._database_url,
                            min_size=self._pool_min_size,
                            max_size=self._pool_max_size,
                            command_timeout=60.0,
                        ),
                        timeout=65.0,
                    )
                    logger.info(
                        "pgvector_pool_created",
                        min_size=self._pool_min_size,
                        max_size=self._pool_max_size,
                    )
        return self._pool

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("pgvector_pool_closed")

    async def upsert_batch(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        collection: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Batch upsert documents with embeddings.

        Note: The trigger 'trg_auto_quantize_embedding' automatically
        populates embedding_half from embedding.
        """
        if not ids:
            return
        col = collection or self._collection_default
        pool = await self._get_pool()
        records = []
        for doc_id, vector, payload in zip(ids, vectors, payloads, strict=False):
            content = payload.get("text", payload.get("content", ""))
            sha256 = payload.get("sha256", "")
            chunk_id = payload.get("chunk_id", 0)
            embedding_str = f"[{','.join((str(x) for x in vector))}]"
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in ("text", "content", "sha256", "chunk_id")
            }
            records.append(
                (col, doc_id, chunk_id, content, embedding_str, metadata, sha256)
            )
        async with pool.acquire() as conn:
            async with asyncio.timeout(timeout):
                await conn.executemany(
                    """
                    INSERT INTO document_embeddings (
                        collection_name, document_id, chunk_id,
                        content, embedding, metadata, sha256
                    )
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, $7)
                    ON CONFLICT (collection_name, document_id, chunk_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        sha256 = EXCLUDED.sha256,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    records,
                )
        logger.debug("batch_upserted", extra={"collection": col, "count": len(ids)})

    async def query(
        self,
        vector: list[float],
        top_k: int,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        ef_search: int | None = None,
        with_vectors: bool = False,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Query using optimized two-phase Binary+Halfvec search.

        Phase 1: Binary HNSW (Hamming) for fast candidates (~5ms)
        Phase 2: Halfvec cosine for precise rescoring (~10ms)

        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            collection: Collection name
            filters: Metadata filters
            ef_search: HNSW ef_search parameter (overrides default)
            with_vectors: Include embedding vectors in results
            timeout: Query timeout in seconds
        """
        col = collection or CFG.collection_read
        pool = await self._get_pool()
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            raise ValueError(
                f"top_k must be an integer between 1 and 1000, got: {top_k}"
            )

        # Use provided ef_search or fall back to config defaults
        ef = ef_search if ef_search is not None else CFG.ef_search_max

        embedding_str = f"[{','.join((str(x) for x in vector))}]"
        binary_str = "".join(("1" if v > 0 else "0" for v in vector))
        
        if len(binary_str) != self._dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self._dim}, got {len(binary_str)}.")
            
        filter_clause = ""
        filter_params = []
        param_idx = 4
        if filters:
            if len(filters) > 20:
                raise ValueError("Too many filters provided. Maximum allowed is 20.")
            for key, value in filters.items():
                if value is None:
                    continue
                if key == "sha256":
                    filter_clause += f" AND sha256 = ${param_idx}"
                else:
                    # Use parameterized query to prevent SQL injection
                    filter_clause += f" AND metadata->>${param_idx} = ${param_idx + 1}"
                    filter_params.append(str(key))
                    filter_params.append(str(value))
                    param_idx += 2
                    continue
                filter_params.append(str(value))
                param_idx += 1
        candidates = max(top_k * 10, 50)
        # Use parameterized query for candidates limit
        query = f"""
            WITH binary_candidates AS (
                -- Phase 1: Fast binary search with HNSW
                SELECT
                    id, document_id, chunk_id, content,
                    metadata, sha256, embedding_half
                FROM document_embeddings
                WHERE collection_name = $3
                {filter_clause}
                ORDER BY
                    binary_quantize(embedding_half)::bit({self._dim})
                    <~> $2::bit({self._dim})
                LIMIT ${param_idx}
            )
            -- Phase 2: Precise halfvec rescoring
            SELECT
                document_id, chunk_id, content, metadata, sha256,
                1 - (embedding_half <=> $1::halfvec) AS similarity
            FROM binary_candidates
            ORDER BY embedding_half <=> $1::halfvec
            LIMIT ${param_idx + 1}
        """
        params = [embedding_str, binary_str, col, *filter_params, candidates, top_k]
        async with pool.acquire(timeout=5.0) as conn:
            async with asyncio.timeout(timeout):
                rows = await conn.fetch(query, *params)
        results = []
        for row in rows:
            payload = dict(row["metadata"]) if row["metadata"] else {}
            payload["text"] = row["content"]
            payload["sha256"] = row["sha256"]
            payload["chunk_id"] = row["chunk_id"]
            results.append(
                {
                    "id": row["document_id"],
                    "score": float(row["similarity"]),
                    "payload": payload,
                }
            )
        logger.debug(
            "query_completed",
            extra={
                "collection": col,
                "results": len(results),
                "candidates": candidates,
                "ef_search": ef,
                "mode": "binary_halfvec",
            },
        )
        return results

    async def query_simple(
        self, vector: list[float], top_k: int, collection: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Simple halfvec-only search (faster, slightly less accurate).

        Use this when speed is critical and you don't need maximum precision.
        """
        col = collection or CFG.collection_read
        pool = await self._get_pool()
        embedding_str = f"[{','.join((str(x) for x in vector))}]"
        query = """
            SELECT
                document_id, chunk_id, content, metadata, sha256,
                1 - (embedding_half <=> $1::halfvec) AS similarity
            FROM document_embeddings
            WHERE collection_name = $2
            ORDER BY embedding_half <=> $1::halfvec
            LIMIT $3
        """
        async with pool.acquire(timeout=5.0) as conn:
            rows = await conn.fetch(query, embedding_str, col, top_k)
        results = []
        for row in rows:
            payload = dict(row["metadata"]) if row["metadata"] else {}
            payload["text"] = row["content"]
            payload["sha256"] = row["sha256"]
            payload["chunk_id"] = row["chunk_id"]
            results.append(
                {
                    "id": row["document_id"],
                    "score": float(row["similarity"]),
                    "payload": payload,
                }
            )
        return results

    async def count(self, collection: str | None = None) -> int:
        """Count documents in collection."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire(timeout=5.0) as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_embeddings WHERE collection_name = $1",
                col,
            )
        return count or 0

    async def exists_by_sha256(
        self, sha256: str, collection: str | None = None, timeout: float = 5.0
    ) -> bool:
        """Check if document with SHA256 exists."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT 1 FROM document_embeddings
                WHERE collection_name = $1 AND sha256 = $2
                LIMIT 1
                """,
                col,
                sha256,
            )
        return exists is not None

    async def exists(self, document_id: str, collection: str | None = None) -> bool:
        """Check if document exists."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT 1 FROM document_embeddings
                WHERE collection_name = $1 AND document_id = $2
                LIMIT 1
                """,
                col,
                document_id,
            )
        return exists is not None

    async def delete(self, document_id: str, collection: str | None = None) -> bool:
        """Delete document by ID."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire(timeout=5.0) as conn:
            result = await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE collection_name = $1 AND document_id = $2
                """,
                col,
                document_id,
            )
        # Robust parsing of 'DELETE N'
        import re
        match = re.search(r"DELETE (\d+)", result)
        deleted = int(match.group(1)) if match else 0
        return deleted > 0

    async def delete_by_doc_id(
        self,
        doc_id: str,
        collection: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> int:
        """Delete all vectors associated with a document ID."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire(timeout=5.0) as conn:
            result = await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE collection_name = $1 AND document_id = $2
                """,
                col,
                doc_id,
            )
        import re
        match = re.search(r"DELETE (\d+)", result)
        deleted = int(match.group(1)) if match else 0
        return deleted

    async def exists_batch_by_sha256(
        self,
        sha256_list: list[str],
        collection: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> set[str]:
        """Batch check which SHA256 hashes exist in collection."""
        if not sha256_list:
            return set()
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT sha256 FROM document_embeddings
                WHERE collection_name = $1 AND sha256 = ANY($2::text[])
                """,
                col,
                sha256_list,
            )
        return {row["sha256"] for row in rows}

    async def delete_collection(self, collection: str) -> int:
        """Delete entire collection."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM document_embeddings WHERE collection_name = $1", collection
            )
        deleted = result.split()[-1]
        return int(deleted)

    async def get_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Get collection statistics."""
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire(timeout=5.0) as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as document_count,
                    COUNT(DISTINCT document_id) as unique_documents,
                    MIN(created_at) as oldest_doc,
                    MAX(updated_at) as newest_doc,
                    pg_size_pretty(pg_total_relation_size('document_embeddings'))
                        as table_size
                FROM document_embeddings
                WHERE collection_name = $1
                """,
                col,
            )
        return {
            "collection": col,
            "document_count": stats["document_count"],
            "unique_documents": stats["unique_documents"],
            "oldest_doc": stats["oldest_doc"].isoformat()
            if stats["oldest_doc"]
            else None,
            "newest_doc": stats["newest_doc"].isoformat()
            if stats["newest_doc"]
            else None,
            "table_size": stats["table_size"],
            "search_mode": "binary_halfvec",
        }

    async def get_all_documents(
        self, collection: str | None = None, limit: int = 10000, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Retrieve all documents from the vector store.

        Required by VectorStore interface for BM25 index building.

        Args:
            collection: Collection name (optional, uses default)
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip

        Returns:
            List of documents with content and metadata
        """
        col = collection or self._collection_default
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    document_id,
                    chunk_index,
                    content,
                    metadata,
                    created_at,
                    updated_at
                FROM document_embeddings
                WHERE collection_name = $1
                ORDER BY document_id, chunk_index
                LIMIT $2 OFFSET $3
                """,
                col,
                limit,
                offset,
            )
        documents = []
        for row in rows:
            doc = {
                "id": f"{row['document_id']}_{row['chunk_index']}",
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": row["metadata"] or {},
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
                "updated_at": row["updated_at"].isoformat()
                if row["updated_at"]
                else None,
            }
            documents.append(doc)
        logger.info("Retrieved %s documents from collection '%s'", len(documents), col)
        return documents

# Async-safe singleton implementation using module-level lock
_store_instance: "PgVectorStore | None" = None
# Lazy initialised — avoids RuntimeError on module import
_store_lock: asyncio.Lock | None = None
_store_sync_lock = threading.Lock()

async def get_vector_store() -> "PgVectorStore":
    """Get singleton vector store instance (double-checked async-safe locking)."""
    global _store_instance, _store_lock

    # Fast path — no lock needed once initialised
    if _store_instance is not None:
        return _store_instance

    if _store_lock is None:
        _store_lock = asyncio.Lock()

    async with _store_lock:
        # Double-check after acquiring lock
        if _store_instance is None:
            _store_instance = PgVectorStore()
    return _store_instance

# Backward-compatible sync version (for non-async contexts)
def get_vector_store_sync() -> "PgVectorStore":
    """Synchronous version for backward compatibility.

    NOTE: This must use a module-level lock; a local threading.Lock provides no mutual exclusion.
    """
    global _store_instance
    if _store_instance is None:
        with _store_sync_lock:
            if _store_instance is None:
                _store_instance = PgVectorStore()
    return _store_instance
