"""
Semantic Cache for LLM Responses.

v5.3.17 - Semantic caching with:
- Vector similarity search using Redis Stack (when available)
- Python-based fallback for standard Redis OSS
- Configurable similarity threshold
- TTL-based expiration
- Hit/miss metrics
- Conditional cross-encoder reranking for gray zone queries (NEW)

Architecture (learned from decades of cache design):
1. Query comes in
2. Generate embedding for query
3. Search for similar cached queries (within threshold)
4. If in "gray zone" (uncertain match): apply cross-encoder reranking
5. If found: return cached response (HIT)
6. If not found: return None (MISS) - caller should query LLM
7. After LLM response: store in cache for future queries

Performance targets:
- Cache lookup: <100ms (including embedding generation)
- Cache lookup with reranking: <150ms (only for uncertain matches)
- Hit rate: >60% after warm-up period
- False positive rate: <2% (with reranking enabled)
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from resync.models.cache import CacheEntry, CacheResult
from .embedding_model import (
    cosine_distance,
    generate_embedding,
    get_embedding_dimension,
)
from .redis_config import (
    RedisDatabase,
    check_redis_stack_available,
    get_redis_client,
    get_redis_config,
)
from .reranker import (
    get_reranker_info,
    is_reranker_available,
    rerank_pair,
    should_rerank,
)

logger = logging.getLogger(__name__)




class SemanticCache:
    """
    Semantic cache for LLM responses.

    Uses embedding similarity to find cached responses that match
    semantically similar queries, even if wording is different.

    Example:
        cache = SemanticCache()
        await cache.initialize()

        # Check cache
        result = await cache.get("How do I restart a job?")
        if result.hit:
            return result.response  # Fast path

        # Cache miss - call LLM
        llm_response = await call_llm(query)

        # Store for future
        await cache.set(query, llm_response)
    """

    # Redis key prefixes
    KEY_PREFIX = "semantic_cache:"
    INDEX_NAME = "idx:semantic_cache"
    STATS_KEY = "semantic_cache:stats"

    def __init__(
        self,
        threshold: float | None = None,
        default_ttl: int | None = None,
        max_entries: int | None = None,
        enable_reranking: bool = True,
    ):
        """
        Initialize semantic cache.

        Args:
            threshold: Cosine distance threshold for cache hit (0-1, lower = stricter)
            default_ttl: Default TTL in seconds for cache entries
            max_entries: Maximum entries before LRU eviction
            enable_reranking: Whether to use cross-encoder for gray zone queries
        """
        config = get_redis_config()

        self.threshold = threshold or config.semantic_cache_threshold
        self.default_ttl = default_ttl or config.semantic_cache_ttl
        self.max_entries = max_entries or config.semantic_cache_max_entries
        self.enable_reranking = enable_reranking and is_reranker_available()

        self._redis_stack_available: bool | None = None
        self._index_created: bool = False
        self._initialized: bool = False

        # In-memory stats (periodically synced to Redis)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "total_lookup_time_ms": 0.0,
            "reranks": 0,
            "rerank_rejections": 0,  # False positives caught by reranker
        }

        logger.info(
            f"SemanticCache initialized with threshold={self.threshold}, "
            f"ttl={self.default_ttl}s, max_entries={self.max_entries}, "
            f"reranking={'enabled' if self.enable_reranking else 'disabled'}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the cache (create index if needed).

        Must be called before using get/set methods.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Check Redis Stack availability
            stack_info = await check_redis_stack_available()
            self._redis_stack_available = stack_info.get("search", False)

            if self._redis_stack_available:
                await self._create_redisearch_index()
                logger.info("SemanticCache using Redis Stack with RediSearch")
            else:
                logger.info(
                    "SemanticCache using fallback mode (no RediSearch). "
                    "Consider installing Redis Stack for better performance."
                )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize SemanticCache: %s", e)
            return False

    async def _create_redisearch_index(self) -> None:
        """
        Create RediSearch index for vector similarity search.

        Index schema:
        - query_hash: TAG (for deduplication)
        - embedding: VECTOR (for similarity search)
        - query_text: TEXT (for debugging)
        - timestamp: NUMERIC (for TTL filtering)
        """
        if self._index_created:
            return

        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
        dim = get_embedding_dimension()

        try:
            # Check if index already exists
            try:
                await client.execute_command("FT.INFO", self.INDEX_NAME)
                logger.info("RediSearch index %s already exists", self.INDEX_NAME)
                self._index_created = True
                return
            except Exception as exc:
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

            # Create index with vector similarity
            # Using HNSW algorithm for approximate nearest neighbor search
            await client.execute_command(
                "FT.CREATE",
                self.INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self.KEY_PREFIX,
                "SCHEMA",
                "query_text",
                "TEXT",
                "query_hash",
                "TAG",
                "timestamp",
                "NUMERIC",
                "SORTABLE",
                "hit_count",
                "NUMERIC",
                "SORTABLE",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dim),
                "DISTANCE_METRIC",
                "COSINE",
            )

            logger.info("Created RediSearch index %s with %s dimensions", self.INDEX_NAME, dim)
            self._index_created = True

        except Exception as e:
            logger.error("Failed to create RediSearch index: %s", e)
            # Don't fail completely - we can use fallback
            self._redis_stack_available = False

    def _hash_query(self, query: str) -> str:
        """
        Generate hash for query (for deduplication and key naming).

        Uses MD5 because:
        - Fast
        - Collision probability is acceptable for cache keys
        - Deterministic
        - NOT used for security purposes (usedforsecurity=False)
        """
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _make_key(self, query_hash: str) -> str:
        """Generate Redis key from query hash."""
        return f"{self.KEY_PREFIX}{query_hash}"

    async def get(self, query: str) -> CacheResult:
        """
        Look up query in semantic cache.

        Applies conditional reranking for uncertain matches (gray zone):
        - Distance < 0.20 → Clear HIT, skip reranking
        - Distance > 0.35 → Clear MISS, skip reranking
        - Distance 0.20-0.35 → Apply cross-encoder to confirm

        Args:
            query: User's query text

        Returns:
            CacheResult with hit status and response if found
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()

        try:
            # Generate embedding for query
            embedding = generate_embedding(query)

            # Search for similar entries
            if self._redis_stack_available:
                result = await self._search_redisearch(query, embedding)
            else:
                result = await self._search_fallback(query, embedding)

            # Apply conditional reranking for gray zone matches
            if result.hit and self.enable_reranking and should_rerank(result.distance):
                result = self._apply_reranking(query, result)

            # Update stats
            lookup_time = (time.perf_counter() - start_time) * 1000
            result.lookup_time_ms = lookup_time

            if result.hit:
                self._stats["hits"] += 1
                # Increment hit count asynchronously
                await create_tracked_task(self._increment_hit_count(result.entry), name="increment_hit_count")
            else:
                self._stats["misses"] += 1

            self._stats["total_lookup_time_ms"] += lookup_time

            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Cache lookup failed: %s", e)
            return CacheResult(hit=False)

    def _apply_reranking(self, query: str, result: CacheResult) -> CacheResult:
        """
        Apply cross-encoder reranking to confirm a gray zone match.

        If the reranker says the queries are not similar,
        convert the HIT to a MISS (false positive prevention).

        Args:
            query: New user query
            result: Initial cache result from embedding search

        Returns:
            Updated CacheResult (may change hit=True to hit=False)
        """
        if not result.entry or not result.entry.query:
            return result

        self._stats["reranks"] += 1

        # Run reranking (synchronous but fast ~20-50ms)
        rerank_result = rerank_pair(query, result.entry.query)

        result.reranked = True
        result.rerank_score = rerank_result.score

        if rerank_result.is_similar:
            logger.debug(
                "Reranker CONFIRMED: distance={result.distance:.3f}, "
                "rerank_score={rerank_result.score:.3f}, "
                f"query='{query[:40]}...'"
            )
            return result
        # Reranker says NOT similar - convert to MISS
        self._stats["rerank_rejections"] += 1
        logger.info(
            "Reranker REJECTED false positive: distance={result.distance:.3f}, "
            "rerank_score={rerank_result.score:.3f}, "
            f"query='{query[:40]}...' vs cached='{result.entry.query[:40]}...'"
        )
        return CacheResult(
            hit=False,
            response=None,
            distance=result.distance,
            entry=None,
            reranked=True,
            rerank_score=rerank_result.score,
        )

    async def _search_redisearch(self, query: str, embedding: list[float]) -> CacheResult:
        """
        Search using RediSearch vector similarity.

        This is the fast path when Redis Stack is available.
        """
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

        # Convert embedding to bytes for RediSearch
        import struct

        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        try:
            # KNN search for nearest neighbor
            results = await client.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                "*=>[KNN 1 @embedding $vec AS distance]",
                "PARAMS",
                "2",
                "vec",
                embedding_bytes,
                "SORTBY",
                "distance",
                "RETURN",
                "6",
                "query_text",
                "response",
                "distance",
                "timestamp",
                "hit_count",
                "metadata",
                "DIALECT",
                "2",
            )

            # Parse results
            # Format: [total, key1, [field1, value1, ...], key2, [...], ...]
            if results and results[0] > 0:
                # Get first result
                results[1]
                fields = results[2]

                # Convert flat list to dict
                data = {}
                for i in range(0, len(fields), 2):
                    data[fields[i]] = fields[i + 1]

                distance = float(data.get("distance", 1.0))

                if distance <= self.threshold:
                    entry = CacheEntry(
                        query=data.get("query_text", ""),
                        response=data.get("response", ""),
                        embedding=embedding,  # Use current embedding
                        timestamp=datetime.fromisoformat(
                            data.get("timestamp", datetime.now(timezone.utc).isoformat())
                        ),
                        hit_count=int(data.get("hit_count", 0)),
                        metadata=json.loads(data.get("metadata", "{}")),
                    )

                    logger.debug("Cache HIT: distance={distance:.4f}, query='{query[:50]}...'")

                    return CacheResult(
                        hit=True,
                        response=entry.response,
                        distance=distance,
                        entry=entry,
                    )

            logger.debug("Cache MISS: no similar entries found for '%s...'", query[:50])
            return CacheResult(hit=False)

        except Exception as e:
            logger.error("RediSearch query failed: %s", e)
            # Fall back to Python-based search
            return await self._search_fallback(query, embedding)

    async def _search_fallback(self, query: str, embedding: list[float]) -> CacheResult:
        """
        Search using Python-based brute-force similarity.

        This is slower but works with standard Redis.
        For production with many entries, Redis Stack is recommended.
        """
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

        try:
            # Get all cache keys
            keys = []
            async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*", count=100):
                keys.append(key)
                if len(keys) >= self.max_entries:
                    break

            if not keys:
                return CacheResult(hit=False)

            # Find best match
            best_distance = float("inf")
            best_entry = None

            for key in keys:
                try:
                    data = await client.hgetall(key)
                    if not data:
                        continue

                    stored_embedding = json.loads(data.get("embedding", "[]"))
                    if not stored_embedding:
                        continue

                    distance = cosine_distance(embedding, stored_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_entry = CacheEntry.from_dict(data)

                except Exception as e:
                    logger.warning("Error processing cache key %s: %s", key, e)
                    continue

            if best_entry and best_distance <= self.threshold:
                best_entry.embedding = embedding  # Update with current embedding
                return CacheResult(
                    hit=True,
                    response=best_entry.response,
                    distance=best_distance,
                    entry=best_entry,
                )

            return CacheResult(hit=False)

        except Exception as e:
            logger.error("Fallback search failed: %s", e)
            return CacheResult(hit=False)

    async def set(
        self,
        query: str,
        response: str,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store query-response pair in cache.

        Args:
            query: User's original query
            response: LLM's response
            ttl: Time-to-live in seconds (None = use default)
            metadata: Additional info to store (model, latency, etc.)

        Returns:
            True if stored successfully
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate embedding
            embedding = generate_embedding(query)

            # Create entry
            entry = CacheEntry(
                query=query,
                response=response,
                embedding=embedding,
                metadata=metadata or {},
            )

            # Store in Redis
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

            query_hash = self._hash_query(query)
            key = self._make_key(query_hash)

            # Store as hash (for RediSearch compatibility)
            data = entry.to_dict()
            data["query_hash"] = query_hash

            # For RediSearch, store embedding as binary
            if self._redis_stack_available:
                import struct

                data["embedding"] = struct.pack(f"{len(embedding)}f", *embedding)

            await client.hset(key, mapping=data)

            # Set TTL
            effective_ttl = ttl or self.default_ttl
            await client.expire(key, effective_ttl)

            self._stats["sets"] += 1

            logger.debug(
                f"Cached response for '{query[:50]}...' "
                f"(ttl={effective_ttl}s, key={query_hash[:8]})"
            )

            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to cache response: %s", e)
            return False

    async def _increment_hit_count(self, entry: CacheEntry | None) -> None:
        """Increment hit count for a cache entry (background task)."""
        if not entry:
            return

        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            query_hash = self._hash_query(entry.query)
            key = self._make_key(query_hash)
            await client.hincrby(key, "hit_count", 1)
        except Exception as e:
            logger.warning("Failed to increment hit count: %s", e)

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate (remove) a specific cache entry.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            query_hash = self._hash_query(query)
            key = self._make_key(query_hash)
            deleted = await client.delete(key)

            if deleted:
                logger.info("Invalidated cache entry for '%s...'", query[:50])
            return deleted > 0

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to invalidate cache entry: %s", e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all entries matching a pattern.

        Args:
            pattern: Glob pattern to match against query text

        Returns:
            Number of entries invalidated
        """
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

            count = 0
            async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                try:
                    query = await client.hget(key, "query_text")
                    if query and pattern.lower() in query.lower():
                        await client.delete(key)
                        count += 1
                except Exception:
                    continue

            logger.info("Invalidated %s cache entries matching '%s'", count, pattern)
            return count

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to invalidate by pattern: %s", e)
            return 0

    async def clear(self) -> bool:
        """
        Clear all cache entries.

        USE WITH CAUTION - this removes all cached responses!

        Returns:
            True if cleared successfully
        """
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

            # Delete all keys with our prefix
            count = 0
            async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                await client.delete(key)
                count += 1

            logger.warning("Cleared %s semantic cache entries", count)

            # Reset stats
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "errors": 0,
                "total_lookup_time_ms": 0.0,
                "reranks": 0,
                "rerank_rejections": 0,
            }

            return True

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to clear cache: %s", e)
            return False

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hit rate, total entries, memory usage, reranking stats, etc.
        """
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

            # Count entries
            count = 0
            async for _ in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                count += 1

            # Calculate hit rate
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests * 100 if total_requests > 0 else 0.0

            # Average lookup time
            avg_lookup_time = (
                self._stats["total_lookup_time_ms"] / total_requests if total_requests > 0 else 0.0
            )

            # Reranking stats
            rerank_rejection_rate = (
                self._stats["rerank_rejections"] / self._stats["reranks"] * 100
                if self._stats["reranks"] > 0
                else 0.0
            )

            # Get memory info
            info = await client.info("memory")

            # Get reranker info
            reranker_info = get_reranker_info()

            return {
                "entries": count,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "errors": self._stats["errors"],
                "hit_rate_percent": round(hit_rate, 2),
                "avg_lookup_time_ms": round(avg_lookup_time, 2),
                "threshold": self.threshold,
                "default_ttl": self.default_ttl,
                "max_entries": self.max_entries,
                "redis_stack_available": self._redis_stack_available,
                "used_memory_human": info.get("used_memory_human", "unknown"),
                # Reranking stats
                "reranking_enabled": self.enable_reranking,
                "reranks_total": self._stats["reranks"],
                "rerank_rejections": self._stats["rerank_rejections"],
                "rerank_rejection_rate_percent": round(rerank_rejection_rate, 2),
                "reranker_model": reranker_info.get("model"),
                "reranker_available": reranker_info.get("available", False),
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to get stats: %s", e)
            return {"error": str(e)}

    def update_threshold(self, new_threshold: float) -> None:
        """
        Update similarity threshold at runtime.

        Args:
            new_threshold: New threshold (0-1, lower = stricter)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info("Updated cache threshold: %s -> %s", old_threshold, new_threshold)

    def set_reranking_enabled(self, enabled: bool) -> bool:
        """
        Enable or disable cross-encoder reranking.

        Args:
            enabled: Whether to enable reranking

        Returns:
            Actual state after setting (may be False if reranker unavailable)
        """
        if enabled and not is_reranker_available():
            logger.warning("Cannot enable reranking - model not available")
            self.enable_reranking = False
            return False

        old_state = self.enable_reranking
        self.enable_reranking = enabled
        logger.info("Reranking %s (was: %s)", 'enabled' if enabled else 'disabled', old_state)
        return enabled


# Singleton instance
_cache_instance: SemanticCache | None = None
_cache_lock = None  # lazy-initialized asyncio.Lock (gunicorn --preload safe)
_cache_lock_loop = None


def _get_cache_lock() -> asyncio.Lock:
    """Return a process-global asyncio.Lock bound to the current event loop."""
    global _cache_lock, _cache_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _cache_lock = asyncio.Lock()
        _cache_lock_loop = None
        return _cache_lock

    if _cache_lock is None or _cache_lock_loop is not loop:
        _cache_lock = asyncio.Lock()
        _cache_lock_loop = loop
    return _cache_lock
async def get_semantic_cache() -> SemanticCache:
    """
    Get singleton SemanticCache instance.

    Thread-safe with async lock.
    """
    global _cache_instance

    if _cache_instance is not None:
        return _cache_instance

    async with _get_cache_lock():
        if _cache_instance is not None:
            return _cache_instance

        _cache_instance = SemanticCache()
        await _cache_instance.initialize()
        return _cache_instance


__all__ = [
    "CacheEntry",
    "CacheResult",
    "SemanticCache",
    "get_semantic_cache",
]
