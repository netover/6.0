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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from resync.models.cache import CacheEntry, CacheResult
from .embedding_model import (
    cosine_distance,
    generate_embedding,
    get_embedding_dimension,
)
from .redis_config import (
    RedisDatabase,
    get_redis_client,
    get_redis_config,
    check_redis_stack_available
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
SCHEMA = {
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
    ],
}

class SemanticCache:
    """
    Enhanced Semantic Cache using RedisVL with full API parity and fallback support.
    """

    KEY_PREFIX = SCHEMA["index"]["prefix"]
    INDEX_NAME = SCHEMA["index"]["name"]

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
        
        # Initialize Vectorizer and Index
        self.vectorizer = ResyncVectorizer()
        self.index = SearchIndex.from_dict(SCHEMA)
        
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

    async def initialize(self) -> bool:
        """Initialize connection and verify index/stack availability."""
        if self._initialized:
            return True
            
        try:
            # Detect Redis Stack first
            stack_info = await check_redis_stack_available()
            self._redis_stack_available = stack_info.get("search", False)
            
            # Get standard async client
            redis_client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            
            if self._redis_stack_available:
                # RedisVL 0.3.1 workaround: it might need specific client type check
                # but SearchIndex.set_client works if we provide the right flavor.
                self.index.set_client(redis_client)
                
                # Check if index exists, create if not
                if not await self.index.exists():
                    await self.index.create(overwrite=False)
                    logger.info("Created new RedisVL index: %s", self.index.name)
            else:
                logger.info("Redis Stack not available. Using fallback mode (brute force).")
            
            self._initialized = True
            return True
        except Exception as e:
            logger.error("Failed to initialize SemanticCache: %s", e)
            self._redis_stack_available = False # Force fallback if init fails
            self._initialized = True # Mark as initialized to allow fallback
            return False

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()

    async def get(self, query: str) -> CacheResult:
        """Look up query in cache with fallback and reranking support."""
        if not self._initialized:
            await self.initialize()
            
        start_time = time.perf_counter()
        try:
            # Generate embedding vector
            embedding = self.vectorizer.embed(query)
            
            if self._redis_stack_available:
                result = await self._search_redisvl(query, embedding)
            else:
                result = await self._search_fallback(query, embedding)
                
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

    async def _search_redisvl(self, query: str, embedding: List[float]) -> CacheResult:
        """Search using RedisVL VectorQuery."""
        v_query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            num_results=1,
            return_fields=["query_text", "response", "timestamp", "hit_count", "metadata"],
            return_score=True
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
                    timestamp=datetime.fromtimestamp(float(match.get("timestamp", 0)), tz=timezone.utc),
                    hit_count=int(match.get("hit_count", 0)),
                    metadata=json.loads(match.get("metadata", "{}"))
                )
                return CacheResult(hit=True, response=entry.response, distance=distance, entry=entry)
        
        return CacheResult(hit=False)

    async def _search_fallback(self, query: str, embedding: List[float]) -> CacheResult:
        """Fallback search using Python-based brute-force similarity."""
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
        best_distance = float("inf")
        best_entry = None
        best_key = None

        try:
            async for key in client.scan_iter(match=f"{self.KEY_PREFIX}*", count=100):
                data = await client.hgetall(key)
                if not data or "embedding" not in data:
                    continue
                
                # Decode binary embedding if stored as bytes (RedisVL format)
                raw_emb = data["embedding"]
                if isinstance(raw_emb, bytes):
                    stored_embedding = list(struct.unpack(f"{len(raw_emb)//4}f", raw_emb))
                else:
                    stored_embedding = json.loads(raw_emb)
                
                distance = cosine_distance(embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_entry = CacheEntry.from_dict(data)
                    best_key = key

            if best_entry and best_distance <= self.threshold:
                return CacheResult(hit=True, response=best_entry.response, distance=best_distance, entry=best_entry)
        except Exception as e:
            logger.warning("Fallback search encountered error: %s", e)
            
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
    ) -> bool:
        """Store entry in cache."""
        if not self._initialized:
            await self.initialize()
            
        try:
            query_hash = self._hash_query(query)
            embedding = self.vectorizer.embed(query)
            
            data = {
                "query_text": query,
                "query_hash": query_hash,
                "response": response,
                "embedding": embedding,
                "timestamp": datetime.now(timezone.utc).timestamp(),
                "hit_count": 0,
                "metadata": json.dumps(metadata or {})
            }
            
            if self._redis_stack_available:
                keys = await self.index.load([data])
                key = keys[0] if keys else None
            else:
                # Manual HSET for fallback mode
                client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
                key = f"{self.KEY_PREFIX}{query_hash}"
                # Store embedding as binary for consistency
                data["embedding"] = struct.pack(f"{len(embedding)}f", *embedding)
                await client.hset(key, mapping=data)
            
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
            return False

    async def _increment_hit_count(self, entry: Optional[CacheEntry]) -> None:
        """Increment hit count for a cache entry."""
        if not entry:
            return
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            query_hash = self._hash_query(entry.query)
            await client.hincrby(f"{self.KEY_PREFIX}{query_hash}", "hit_count", 1)
        except Exception:
            pass

    async def invalidate(self, query: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            query_hash = self._hash_query(query)
            deleted = await client.delete(f"{self.KEY_PREFIX}{query_hash}")
            return bool(deleted)
        except Exception as e:
            logger.error("Failed to invalidate cache entry: %s", e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern."""
        try:
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
            return 0

    async def clear(self) -> bool:
        """Clear the entire cache."""
        try:
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
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive cache performance statistics."""
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            
            # Count total entries
            count = 0
            async for _ in client.scan_iter(match=f"{self.KEY_PREFIX}*"):
                count += 1
                
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            avg_lookup_time = (self._stats["total_lookup_time_ms"] / total_requests) if total_requests > 0 else 0
            
            # Memory and system info
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
                "rerank_rejections": self._stats["rerank_rejections"]
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {"error": str(e)}

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
