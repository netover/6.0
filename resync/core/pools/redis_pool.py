"""
Redis connection pool implementation for the Resync project.
Separated to follow Single Responsibility Principle.
"""

import logging

# Soft import for redis (optional dependency)
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
except ImportError:
    redis = None
    AsyncRedis = None
    RedisConnectionError = None
    RedisError = None

import os

from resync.core.redis_init import get_redis_client, is_redis_available

logger = logging.getLogger(__name__)


class RedisPool:
    """Connection pool for redis resources."""

    def __init__(self, url: str | None = None):
        self._url = url
        self._client = None

    @property
    def client(self):
        disable = os.getenv("RESYNC_DISABLE_REDIS", "0")
        if disable is not None and disable.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError("Redis disabled by RESYNC_DISABLE_REDIS")
        if not is_redis_available():
            raise RuntimeError("redis-py not installed")
        if self._client is None:
            if self._url:
                # usa URL explícita
                import redis.asyncio as redis  # type: ignore

                self._client = redis.from_url(
                    self._url, encoding="utf-8", decode_responses=True
                )
                logger.info("Initialized Redis client from explicit URL (lazy).")
            else:
                # usa factory centralizada
                self._client = get_redis_client()
        return self._client


# Adicionar alias para compatibilidade
RedisConnectionPool = RedisPool
