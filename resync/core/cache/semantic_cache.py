"""
Semantic Cache for LLM Responses.

v6.4.1 - RedisVL Implementation Refined:
- Unified vector search via RedisVL
- Standardized SearchIndex and Schema
- Custom Vectorizer (ResyncVectorizer)
- Integrated Cross-Encoder Reranking
- Hit/miss metrics and TTL support
- Full API Parity (invalidate, threshold updates, etc.)
- Graceful Fallback for non-Redis Stack environments
"""

import asyncio
import hashlib
import json
import logging
import struct
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery

    REDISVL_AVAILABLE = True
except Exception:
    SearchIndex = None  # type: ignore[assignment]
    VectorQuery = None  # type: ignore[assignment]
    REDISVL_AVAILABLE = False

from resync.models.cache import CacheEntry, CacheResult
from .embedding_model import (
    cosine_distance,
)
from .redis_config import (
    RedisDatabase,
    get_redis_client,
    get_redis_config,
    check_redis_stack_available,
)
from .redisvl_adapter import ResyncVectorizer
from .reranker import (
    get_reranker_info,
    is_reranker_available,
    rerank_pair,
    should_rerank,
)

logger = logging.getLogger(__name__)

# RedisVL Index Schema Definition
SCHEMA: dict[str, Any] = {
    "index": {
        "name": "idx:semantic_cache_v2",
        "prefix": "semantic_cache_v2:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "query_text", "type": "text"},
        {"name": "query_hash", "type": "tag"},
        {"name": "response", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "algorithm": "hnsw",
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
        {"name": "timestamp", "type": "numeric"},
        {"name": "hit_count", "type": "numeric"},
        {"name": "metadata", "type": "text"},
        {"name": "user_id", "type": "tag"},  # SECURITY: User isolation field
    ],
}


class SemanticCache:
    """
    Enhanced Semantic Cache using RedisVL with full API parity and fallback support.
    """

    KEY_PREFIX = str(SCHEMA["index"]["prefix"])
    INDEX_NAME = str(SCHEMA["index"]["name"])

    def __init__(
        self,
        threshold: float | None = None,
        default_ttl: int | None = None,
        max_entries: int | None = None,
        enable_reranking: bool = True,
    ):
        config = get_redis_config()
        self.threshold = threshold or config.semantic_cache_threshold
        self.default_ttl = default_ttl or config.semantic_cache_ttl
        self.max_entries = max_entries or config.semantic_cache_max_entries
        self.enable_reranking = enable_reranking and is_reranker_available()

        # Initialize Vectorizer and Index (best-effort if RedisVL is available)
        self.vectorizer = ResyncVectorizer()
        self.index = SearchIndex.from_dict(SCHEMA) if REDISVL_AVAILABLE else None

        self._initialized = False
        self._redis_stack_available: bool | None = None

        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "total_lookup_time_ms": 0.0,
            "reranks": 0,
            "rerank_rejections": 0,
        }
        self._memory_only = not REDISVL_AVAILABLE
        self._memory_store: OrderedDict[str, tuple[CacheEntry, float, str | None]] = (
            OrderedDict()
        )
        self._memory_lock = asyncio.Lock()
        self._last_redis_check = 0.0

    async def initialize(self) -> bool:
        """Initialize connection and verify index/stack availability."""
        if self._initialized:
            return True

        if not REDISVL_AVAILABLE:
            self._memory_only = True
            self._redis_stack_available = False
            self._initialized = True
            logger.warning("RedisVL unavailable; semantic cache running in memory-only mode")
            return True

        try:
            redis_client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            try:
                await redis_client.ping()
            except Exception as e:
                self._memory_only = True
                self._redis_stack_available = False
                self._initialized = True
                logger.warning("SemanticCache Redis unavailable: %s", e)
                return True

            stack_info = await check_redis_stack_available()
            self._redis_stack_available = stack_info.get("search", False)  # type: ignore[assignment]

            if self._redis_stack_available:
                self.index.set_client(redis_client)

                if not await self.index.exists():
                    await self.index.create(overwrite=False)
                    logger.info("Created new RedisVL index: %s", self.index.name)
            else:
                logger.info(
                    "Redis Stack not available. Using fallback mode (brute force)."
                )

            self._initialized = True
            return True
        except Exception as e:
            logger.error("Failed to initialize SemanticCache: %s", e)
            self._redis_stack_available = False  # Force fallback if init fails
            self._initialized = True  # Mark as initialized to allow fallback
            self._memory_only = True
            return True

    def _enter_memory_only(self, reason: str) -> None:
        if not self._memory_only:
            logger.warning("Semantic cache switching to memory-only mode: %s", reason)
        self._memory_only = True
        self._redis_stack_available = False

    async def _try_restore_redis(self) -> None:
        if not self._memory_only:
            return
        now = time.monotonic()
        if now - self._last_redis_check < 5.0:
            return
        self._last_redis_check = now
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            await client.ping()
        except Exception:
            return
        try:
            stack_info = await check_redis_stack_available()
            self._redis_stack_available = stack_info.get("search", False)  # type: ignore[assignment]
        except Exception:
            self._redis_stack_available = False
        self._memory_only = not REDISVL_AVAILABLE

    async def _memory_cleanup_locked(self) -> None:
        now = time.monotonic()
        keys_to_delete = [
            k for k, (_, exp, _) in self._memory_store.items() if exp <= now
        ]
        for key in keys_to_delete:
            del self._memory_store[key]

    async def _memory_set_entry(
        self, key: str, entry: CacheEntry, ttl: int, user_id: str | None
    ) -> None:
        expire_at = time.monotonic() + ttl
        async with self._memory_lock:
            if key in self._memory_store:
                del self._memory_store[key]
            self._memory_store[key] = (entry, expire_at, user_id)
            await self._memory_cleanup_locked()
            while len(self._memory_store) > self.max_entries:
                self._memory_store.popitem(last=False)

    async def _memory_invalidate_query(self, query: str) -> int:
        async with self._memory_lock:
            await self._memory_cleanup_locked()
            keys_to_delete = [
                k
                for k, (entry, _, _) in self._memory_store.items()
                if entry.query == query
            ]
            for key in keys_to_delete:
                del self._memory_store[key]
            return len(keys_to_delete)

    async def _memory_invalidate_pattern(self, pattern: str) -> int:
        async with self._memory_lock:
            await self._memory_cleanup_locked()
            keys_to_delete = [
                k
                for k, (entry, _, _) in self._memory_store.items()
                if pattern.lower() in entry.query.lower()
            ]
            for key in keys_to_delete:
                del self._memory_store[key]
            return len(keys_to_delete)

    async def _search_memory(
        self, query: str, embedding: List[float], user_id: str | None = None
    ) -> CacheResult:
        best_distance = float("inf")
        best_entry: CacheEntry | None = None
        async with self._memory_lock:
            await self._memory_cleanup_locked()
            for _, (entry, _, entry_user_id) in self._memory_store.items():
                if user_id and entry_user_id != user_id:
                    continue
                distance = cosine_distance(embedding, entry.embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_entry = entry
        if best_entry and best_distance <= self.threshold:
            return CacheResult(
                hit=True,
                response=best_entry.response,
                distance=best_distance,
                entry=best_entry,
            )
        return CacheResult(hit=False)

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        normalized = query.strip().lower()
        return hashlib.md5(
            normalized.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

    def _build_cache_key(self, query_text: str, user_id: str | None = None) -> str:
        """
        Build a cache key that includes user_id for per-user caching.

        SECURITY: This prevents cross-user data leakage by scoping cache entries
        to specific users.

        Args:
            query_text: The user's message/query
            user_id: Optional user identifier

        Returns:
            Cache key text with user scoping
        """
        if user_id:
            # Scope cache key to user to prevent cross-user leakage
            return f"user:{user_id}:{query_text}"
        return query_text

    async def get(self, query: str, user_id: str | None = None) -> CacheResult:
        """Look up query in cache with fallback and reranking support.

        SECURITY: user_id parameter ensures proper user isolation at query time.

        Args:
            query: The query text to look up
            user_id: The user identifier for filtering (prevents cross-user data leakage)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        try:
            # Generate embedding vector (use user-scoped cache key for embedding)
            cache_key_text = self._build_cache_key(query, user_id)
            embedding = self.vectorizer.embed(cache_key_text)

            if self._memory_only:
                await self._try_restore_redis()

            if self._memory_only:
                result = await self._search_memory(query, embedding, user_id)
            elif self._redis_stack_available:
                result = await self._search_redisvl(query, embedding, user_id)
            else:
                result = await self._search_fallback(query, embedding, user_id)

            # Apply conditional reranking for gray zone matches
            if result.hit and self.enable_reranking and should_rerank(result.distance):
                result = self._apply_reranking(query, result)

            lookup_time = (time.perf_counter() - start_time) * 1000
            result.lookup_time_ms = lookup_time

            if result.hit:
                self._stats["hits"] += 1
                # Update hit count asynchronously
                asyncio.create_task(self._increment_hit_count(result.entry))
            else:
                self._stats["misses"] += 1

            self._stats["total_lookup_time_ms"] += lookup_time
            return result

        except Exception as e:
            logger.error("Cache lookup failed: %s", e)
            self._stats["errors"] += 1
            return CacheResult(hit=False)

    async def _search_redisvl(
        self, query: str, embedding: List[float], user_id: str | None = None
    ) -> CacheResult:
        """Search using RedisVL VectorQuery with user_id filtering.

        SECURITY: user_id filter ensures only entries belonging to the same user are returned.

        Args:
            query: The query text (used for embedding)
            embedding: The pre-computed embedding vector
            user_id: The user identifier for filtering (prevents cross-user data leakage)
        """
        # Create filter for user_id if provided
        filter_expr = None
        if user_id:
            # Use RedisVL FilterExpression for proper user isolation at query time
            # This is more secure than embedding user_id in the query text
            from redisvl.query.filter import Tag, FilterExpression

            filter_expr = FilterExpression(Tag("user_id").equals(user_id))

        v_query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            num_results=1,
            return_fields=[
                "query_text",
                "response",
                "timestamp",
                "hit_count",
                "metadata",
                "user_id",
            ],
            return_score=True,
            filter=filter_expr,
        )

        results = await self.index.query(v_query)
        if results:
            match = results[0]
            distance = float(match.get("vector_distance", 1.0))

            if distance <= self.threshold:
                entry = CacheEntry(
                    query=match["query_text"],
                    response=match["response"],
                    embedding=embedding,
                    timestamp=datetime.fromtimestamp(
                        float(match.get("timestamp", 0)), tz=timezone.utc
                    ),
                    hit_count=int(match.get("hit_count", 0)),
                    metadata=json.loads(match.get("metadata", "{}")),
                )
                return CacheResult(
                    hit=True, response=entry.response, distance=distance, entry=entry
                )

        return CacheResult(hit=False)

    async def _search_fallback(
        self, query: str, embedding: List[float], user_id: str | None = None
    ) -> CacheResult:
        """Fallback search using Python-based brute-force similarity with user_id filtering.

        SECURITY: user_id filter ensures only entries belonging to the same user are returned.

        Args:
            query: The query text (used for embedding)
            embedding: The pre-computed embedding vector
            user_id: The user identifier for filtering (prevents cross-user data leakage)
        """
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
        best_distance = float("inf")
        best_entry = None

        # SECURITY: Use user-scoped key pattern for lookup
        if user_id:
            key_pattern = f"{self.KEY_PREFIX}user:{user_id}:*"
        else:
            key_pattern = f"{self.KEY_PREFIX}*"

        try:
            async for key in client.scan_iter(match=key_pattern, count=100):
                data = await client.hgetall(key)
                if not data or "embedding" not in data:
                    continue

                # SECURITY: Additional check for user_id field in data (for backward compatibility)
                stored_user_id = data.get("user_id") or ""
                if user_id and stored_user_id != user_id:
                    # Skip entries that don't belong to the requesting user
                    continue

                # Decode binary embedding if stored as bytes (RedisVL format)
                raw_emb = data["embedding"]
                if isinstance(raw_emb, bytes):
                    stored_embedding = list(
                        struct.unpack(f"{len(raw_emb) // 4}f", raw_emb)
                    )
                else:
                    stored_embedding = json.loads(raw_emb)

                distance = cosine_distance(embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_entry = CacheEntry.from_dict(data)

            if best_entry and best_distance <= self.threshold:
                return CacheResult(
                    hit=True,
                    response=best_entry.response,
                    distance=best_distance,
                    entry=best_entry,
                )
        except Exception as e:
            logger.warning("Fallback search encountered error: %s", e)
            self._enter_memory_only("fallback_search_failed")
            return await self._search_memory(query, embedding, user_id)

        return CacheResult(hit=False)

    def _apply_reranking(self, query: str, result: CacheResult) -> CacheResult:
        """Apply cross-encoder reranking to confirm a gray zone match."""
        if not result.entry or not result.entry.query:
            return result

        self._stats["reranks"] += 1
        rerank_res = rerank_pair(query, result.entry.query)
        result.reranked = True
        result.rerank_score = rerank_res.score

        if rerank_res.is_similar:
            return result

        self._stats["rerank_rejections"] += 1
        return CacheResult(hit=False, reranked=True, rerank_score=rerank_res.score)

    async def set(
        self,
        query: str,
        response: str,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Store entry in cache with user isolation.

        SECURITY: user_id is stored as a tag field and used for filtering at query time.

        Args:
            query: The query text to cache
            response: The response to cache
            ttl: Optional TTL override
            metadata: Optional metadata dict
            user_id: The user identifier for isolation (prevents cross-user data leakage)
        """
        if not self._initialized:
            await self.initialize()

        metadata_payload = metadata or {}
        if user_id is not None:
            metadata_payload = {**metadata_payload, "user_id": user_id}
        cache_key_text = self._build_cache_key(query, user_id)
        query_hash = self._hash_query(cache_key_text)
        embedding: list[float] = []
        try:
            embedding = self.vectorizer.embed(cache_key_text)

            if self._memory_only:
                await self._try_restore_redis()

            if self._memory_only:
                entry = CacheEntry(
                    query=query,
                    response=response,
                    embedding=embedding,
                    metadata=metadata_payload,
                )
                await self._memory_set_entry(
                    f"{user_id or ''}:{query_hash}",
                    entry,
                    ttl or self.default_ttl,
                    user_id,
                )
                self._stats["sets"] += 1
                return True

            data = {
                "query_text": query,
                "query_hash": query_hash,
                "response": response,
                "embedding": embedding,
                "timestamp": datetime.now(timezone.utc).timestamp(),
                "hit_count": 0,
                "metadata": json.dumps(metadata_payload),
                "user_id": user_id or "",
            }

            if self._redis_stack_available:
                keys = await self.index.load([data])
                key = keys[0] if keys else None
            else:
                client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
                if user_id:
                    key = f"{self.KEY_PREFIX}user:{user_id}:{query_hash}"
                else:
                    key = f"{self.KEY_PREFIX}{query_hash}"
                data["embedding"] = struct.pack(f"{len(embedding)}f", *embedding)
                await client.hset(key, mapping=data)  # type: ignore[arg-type]

            if key:
                effective_ttl = ttl or self.default_ttl
                client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
                await client.expire(key, effective_ttl)
                self._stats["sets"] += 1
                return True
            return False

        except Exception as e:
            logger.error("Failed to set cache entry: %s", e)
            self._stats["errors"] += 1
            self._enter_memory_only("set_failed")
            entry = CacheEntry(
                query=query,
                response=response,
                embedding=embedding,
                metadata=metadata_payload,
            )
            await self._memory_set_entry(
                f"{user_id or ''}:{query_hash}", entry, ttl or self.default_ttl, user_id
            )
            self._stats["sets"] += 1
            return True

    async def _increment_hit_count(self, entry: Optional[CacheEntry]) -> None:
        """Increment hit count for a cache entry."""
        if not entry:
            return
        if self._memory_only:
            entry.hit_count += 1
            return
        try:
            user_id = None
            if isinstance(entry.metadata, dict):
                user_id = entry.metadata.get("user_id")
            cache_key_text = self._build_cache_key(entry.query, user_id)
            query_hash = self._hash_query(cache_key_text)
            if user_id:
                key = f"{self.KEY_PREFIX}user:{user_id}:{query_hash}"
            else:
                key = f"{self.KEY_PREFIX}{query_hash}"
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            await client.hincrby(key, "hit_count", 1)
        except Exception:
            self._enter_memory_only("increment_hit_count_failed")
            entry.hit_count += 1

    async def invalidate(self, query: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            if self._memory_only:
                return bool(await self._memory_invalidate_query(query))
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            query_hash = self._hash_query(query)
            deleted = await client.delete(f"{self.KEY_PREFIX}{query_hash}")
            return bool(deleted)
        except Exception as e:
            logger.error("Failed to invalidate cache entry: %s", e)
            self._enter_memory_only("invalidate_failed")
            return bool(await self._memory_invalidate_query(query))

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern."""
        try:
            if self._memory_only:
                return await self._memory_invalidate_pattern(pattern)
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            count = 0
            async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                query_text = await client.hget(key, "query_text")
                if query_text and pattern.lower() in query_text.lower():
                    await client.delete(key)
                    count += 1
            return count
        except Exception as e:
            logger.error("Failed to invalidate pattern: %s", e)
            self._enter_memory_only("invalidate_pattern_failed")
            return await self._memory_invalidate_pattern(pattern)

    async def clear(self) -> bool:
        """Clear the entire cache."""
        try:
            if self._memory_only:
                async with self._memory_lock:
                    self._memory_store.clear()
                self._stats = {k: 0 for k in self._stats}
                return True
            if self._redis_stack_available and self._initialized:
                await self.index.clear()
            else:
                client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
                async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                    await client.delete(key)

            self._stats = {k: 0 for k in self._stats}
            return True
        except Exception as e:
            logger.error("Failed to clear cache: %s", e)
            self._enter_memory_only("clear_failed")
            async with self._memory_lock:
                self._memory_store.clear()
            self._stats = {k: 0 for k in self._stats}
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive cache performance statistics."""
        try:
            if self._memory_only:
                async with self._memory_lock:
                    await self._memory_cleanup_locked()
                    count = len(self._memory_store)
                total_requests = self._stats["hits"] + self._stats["misses"]
                hit_rate = (
                    (self._stats["hits"] / total_requests * 100)
                    if total_requests > 0
                    else 0
                )
                avg_lookup_time = (
                    (self._stats["total_lookup_time_ms"] / total_requests)
                    if total_requests > 0
                    else 0
                )
                reranker_info = get_reranker_info()
                return {
                    "version": "redisvl-6.4.1",
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
                    "redis_stack_available": False,
                    "used_memory_human": "memory-only",
                    "reranking_enabled": self.enable_reranking,
                    "reranker_model": reranker_info.get("model"),
                    "reranker_available": reranker_info.get("available", False),
                    "reranks_total": self._stats["reranks"],
                    "rerank_rejections": self._stats["rerank_rejections"],
                }
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

            count = 0
            async for _ in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                count += 1

            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_requests * 100)
                if total_requests > 0
                else 0
            )
            avg_lookup_time = (
                (self._stats["total_lookup_time_ms"] / total_requests)
                if total_requests > 0
                else 0
            )

            info = await client.info("memory")
            reranker_info = get_reranker_info()

            return {
                "version": "redisvl-6.4.1",
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
                "reranking_enabled": self.enable_reranking,
                "reranker_model": reranker_info.get("model"),
                "reranker_available": reranker_info.get("available", False),
                "reranks_total": self._stats["reranks"],
                "rerank_rejections": self._stats["rerank_rejections"],
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            self._enter_memory_only("get_stats_failed")
            return await self.get_stats()

    def update_threshold(self, new_threshold: float) -> None:
        """Update similarity threshold at runtime."""
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            logger.info("Updated cache threshold to %s", new_threshold)

    def set_reranking_enabled(self, enabled: bool) -> bool:
        """Enable or disable cross-encoder reranking."""
        if enabled and not is_reranker_available():
            self.enable_reranking = False
            return False
        self.enable_reranking = enabled
        return enabled

    async def check_intent(
        self,
        query_text: str,
        user_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a query's intent is cached (Router Cache).

        Returns the cached intent data (intent, entities, confidence) if found
        with similarity above threshold, otherwise None.

        This is used by router_node to skip LLM classification for known queries.

        SECURITY: The cache key now includes user_id to prevent cross-user data leakage.
        Only intents from the same user can access cached data.

        Args:
            query_text: The user's message/query
            user_id: The user identifier to scope the cache per-user

        Returns:
            Dict with keys: intent, entities, confidence if cache hit, else None
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate embedding for the query with user scoping
            # SECURITY FIX: Include user_id in cache key to prevent cross-user leakage
            cache_key_text = self._build_cache_key(query_text, user_id)
            embedding = self.vectorizer.embed(cache_key_text)

            if self._memory_only:
                await self._try_restore_redis()

            if self._memory_only:
                result = await self._search_memory(cache_key_text, embedding, user_id)
            elif self._redis_stack_available:
                result = await self._search_redisvl(cache_key_text, embedding, user_id)
            else:
                result = await self._search_fallback(cache_key_text, embedding, user_id)

            if not result.hit:
                return None

            # Parse the cached response as JSON (intent data)
            try:
                response_str = result.response
                if response_str is None:
                    return None
                cached_data = json.loads(response_str)

                # Validate that it has the expected intent cache structure
                if "intent" in cached_data and "entities" in cached_data:
                    logger.debug(
                        "Intent cache hit: intent=%s, distance=%s",
                        cached_data.get("intent"),
                        result.distance,
                    )
                    return cached_data
                else:
                    # Not an intent cache entry, ignore
                    return None

            except json.JSONDecodeError:
                response_preview = result.response[:100] if result.response else ""
                logger.warning(
                    "Intent cache decode error: %s",
                    response_preview,
                )
                return None

        except Exception as e:
            logger.warning("Intent cache check failed: %s", e)
            return None

    async def store_intent(
        self,
        query_text: str,
        intent_data: Dict[str, Any],
        ttl: int | None = None,
        user_id: str | None = None,
    ) -> bool:
        """
        Store the router's intent classification in cache.

        This caches the UNDERSTANDING of the query (intent + entities),
        NOT the final response. This allows us to skip expensive LLM calls
        while still executing real-time data queries.

        SECURITY: The cache key now includes user_id to prevent cross-user data leakage.
        Only intents from the same user can access cached data.

        Args:
            query_text: The user's message/query
            intent_data: Dict containing intent, entities, confidence
            ttl: Optional TTL override (default: self.default_ttl)
            user_id: The user identifier to scope the cache per-user

        Returns:
            True if successfully stored, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Serialize intent data as JSON
            intent_json = json.dumps(intent_data, ensure_ascii=False)

            # SECURITY FIX: Include user_id in cache key to prevent cross-user leakage

            # Use the existing set() method with metadata to mark as intent cache
            metadata = {
                "type": "router_cache",
                "intent": intent_data.get("intent"),
                "confidence": intent_data.get("confidence"),
                "user_id": user_id,  # Track which user cached this
            }

            success = await self.set(
                query=query_text,
                response=intent_json,
                ttl=ttl,
                metadata=metadata,
                user_id=user_id,
            )

            return success

        except Exception as e:
            logger.error("Intent cache store failed: %s", e)
            return False

    def get_detailed_metrics(self) -> dict[str, Any]:
        """Return detailed cache metrics."""
        return {
            "threshold": self.threshold,
            "default_ttl": self.default_ttl,
            "max_entries": self.max_entries,
            "enable_reranking": self.enable_reranking,
            "memory_only": self._memory_only,
            "redis_stack_available": self._redis_stack_available,
        }


# Singleton Management

_cache_instance: SemanticCache | None = None
_cache_lock = None
_cache_lock_loop = None


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock, _cache_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Lock()

    if _cache_lock is None or _cache_lock_loop is not loop:
        _cache_lock = asyncio.Lock()
        _cache_lock_loop = loop
    return _cache_lock


async def get_semantic_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    async with _get_cache_lock():
        if _cache_instance is not None:
            return _cache_instance
        _cache_instance = SemanticCache()
        await _cache_instance.initialize()
        return _cache_instance


__all__ = ["CacheEntry", "CacheResult", "SemanticCache", "get_semantic_cache"]
