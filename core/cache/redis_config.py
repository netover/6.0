"""
Centralized Redis Configuration for Semantic Caching.

v5.3.16 - Redis configuration with:
- Database separation by purpose
- Connection pooling with lazy initialization
- Support for Redis Stack (RediSearch) when available
- Graceful fallback to Redis OSS

Design decisions (30 years of experience speaking):
1. Never hardcode passwords - always from environment
2. Lazy initialization - don't connect until needed
3. Connection pooling - reuse TCP connections
4. Graceful degradation - system works even if Redis fails
"""

import logging
import os
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import redis.asyncio as redis_async

logger = logging.getLogger(__name__)


class RedisDatabase(IntEnum):
    """
    Redis database separation by purpose.

    Why separate DBs?
    - Isolation: different TTLs, different eviction policies
    - Debugging: easier to inspect specific data
    - Performance: FLUSHDB affects only one purpose

    Note: Redis supports DBs 0-15 by default.
    """

    CONNECTIONS = 0  # Connection pools, health checks (existing usage)
    SESSIONS = 1  # User sessions, rate limiting (existing usage)
    CACHE = 2  # General application cache
    SEMANTIC_CACHE = 3  # Semantic cache for LLM responses (NEW)
    IDEMPOTENCY = 4  # Idempotency keys for request deduplication
    # DBs 5-15: Reserved for future use


class RedisConfig:
    """
    Configuration holder for Redis connections.

    All values come from environment variables with sensible defaults.
    Never expose passwords in logs or error messages.
    """

    def __init__(self) -> None:
        # v5.9.7: Prefer consolidated app settings (supports APP_REDIS_URL + legacy REDIS_URL).
        # Environment variables still take precedence for backwards compatibility.
        try:
            from resync.settings import settings as app_settings
        except Exception:  # pragma: no cover
            app_settings = None  # type: ignore

        redis_url = None
        if app_settings is not None:
            redis_url = getattr(app_settings, "redis_url", None)
        redis_url = redis_url or os.getenv("REDIS_URL") or os.getenv("APP_REDIS_URL")
        parsed = urlparse(redis_url) if redis_url else None

        # Base connection settings
        self.host: str = os.getenv("REDIS_HOST") or (parsed.hostname if parsed and parsed.hostname else "localhost")
        self.port: int = int(os.getenv("REDIS_PORT") or (str(parsed.port) if parsed and parsed.port else "6379"))
        env_password = os.getenv("REDIS_PASSWORD")
        self.password: str | None = env_password or (parsed.password if parsed and parsed.password else None)

        # Connection pool settings
        default_pool_min = str(getattr(app_settings, "redis_pool_min_size", 5)) if app_settings else "5"
        default_pool_max = str(getattr(app_settings, "redis_pool_max_size", 20)) if app_settings else "20"
        default_socket_timeout = str(getattr(app_settings, "redis_timeout", 5.0)) if app_settings else "5.0"
        default_connect_timeout = str(getattr(app_settings, "redis_pool_connect_timeout", 5.0)) if app_settings else "5.0"

        self.pool_min_connections: int = int(os.getenv("REDIS_POOL_MIN", default_pool_min))
        self.pool_max_connections: int = int(os.getenv("REDIS_POOL_MAX", default_pool_max))
        self.socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", default_socket_timeout))
        self.socket_connect_timeout: float = float(os.getenv("REDIS_CONNECT_TIMEOUT", default_connect_timeout))

        # Retry settings
        self.retry_on_timeout: bool = True
        default_health = str(getattr(app_settings, "redis_health_check_interval", 30)) if app_settings else "30"
        self.health_check_interval: int = int(os.getenv("REDIS_HEALTH_INTERVAL", default_health))

        # Semantic cache specific
        self.semantic_cache_ttl: int = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))  # 24h default
        self.semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.25"))
        self.semantic_cache_max_entries: int = int(
            os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "100000")
        )

    def get_url(self, db: RedisDatabase = RedisDatabase.CONNECTIONS) -> str:
        """
        Build Redis URL for specific database.

        Format: redis://[:password@]host:port/db
        """
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{db.value}"

    def __repr__(self) -> str:
        """Safe repr that never shows password."""
        return (
            f"RedisConfig(host={self.host}, port={self.port}, "
            f"password={'***' if self.password else None})"
        )


# Global config instance (singleton pattern)
@lru_cache(maxsize=1)
def get_redis_config() -> RedisConfig:
    """
    Get singleton Redis configuration.

    Uses lru_cache for thread-safe singleton pattern.
    """
    return RedisConfig()


# Connection pool cache (one per database)
_connection_pools: dict[RedisDatabase, Any] = {}


def get_redis_client(
    db: RedisDatabase = RedisDatabase.CONNECTIONS,
    decode_responses: bool = True,
) -> "redis_async.Redis":
    """
    Get Redis client for specific database with connection pooling.

    Args:
        db: Which Redis database to connect to
        decode_responses: If True, return strings instead of bytes

    Returns:
        Async Redis client with connection pool

    Raises:
        RuntimeError: If redis-py not installed
        ConnectionError: If Redis is unreachable
    """
    try:
        import redis.asyncio as redis_async
    except ImportError as e:
        raise RuntimeError("redis-py not installed. Run: pip install redis[hiredis]") from e

    config = get_redis_config()

    # Check if we already have a pool for this DB
    if db not in _connection_pools:
        logger.info("Creating Redis connection pool for DB %s (%s)", db.name, db.value)

        pool = redis_async.ConnectionPool(
            host=config.host,
            port=config.port,
            password=config.password,
            db=db.value,
            max_connections=config.pool_max_connections,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            retry_on_timeout=config.retry_on_timeout,
            health_check_interval=config.health_check_interval,
            decode_responses=decode_responses,
        )
        _connection_pools[db] = pool

    return redis_async.Redis(connection_pool=_connection_pools[db])


async def check_redis_stack_available() -> dict[str, bool]:
    """
    Check if Redis Stack modules are available.

    Returns dict with module availability:
    - search: RediSearch (required for semantic cache)
    - json: ReJSON
    - bloom: RedisBloom
    - timeseries: RedisTimeSeries

    Why this matters:
    - With Redis Stack: Use native vector search (faster, more features)
    - Without: Fallback to Python-based similarity (works but slower)
    """
    result = {
        "search": False,
        "json": False,
        "bloom": False,
        "timeseries": False,
        "redis_version": "unknown",
    }

    try:
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)

        # Get Redis version
        info = await client.info("server")
        result["redis_version"] = info.get("redis_version", "unknown")

        # Check loaded modules
        try:
            modules = await client.module_list()
            module_names = {m.get("name", "").lower() for m in modules}

            result["search"] = "search" in module_names or "ft" in module_names
            result["json"] = "rejson" in module_names or "json" in module_names
            result["bloom"] = "bf" in module_names or "bloom" in module_names
            result["timeseries"] = "timeseries" in module_names

        except Exception:
            # MODULE LIST not available (older Redis or disabled)
            # Try specific commands to detect modules
            try:
                await client.execute_command("FT._LIST")
                result["search"] = True
            except Exception as exc:
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

            try:
                await client.execute_command("JSON.DEBUG", "MEMORY", "__test__")
            except Exception as e:
                if "unknown command" not in str(e).lower():
                    result["json"] = True

        logger.info(
            f"Redis Stack check: version={result['redis_version']}, "
            f"search={result['search']}, json={result['json']}"
        )

    except Exception as e:
        logger.warning("Failed to check Redis Stack availability: %s", e)

    return result


async def close_all_pools() -> None:
    """
    Close all Redis connection pools.

    Call this during application shutdown to release resources cleanly.
    """
    for db, pool in _connection_pools.items():
        try:
            await pool.disconnect()
            logger.info("Closed Redis pool for DB %s", db.name)
        except Exception as e:
            logger.warning("Error closing Redis pool for DB %s: %s", db.name, e)

    _connection_pools.clear()


# Health check utility
async def redis_health_check(db: RedisDatabase = RedisDatabase.CONNECTIONS) -> dict[str, Any]:
    """
    Perform health check on specific Redis database.

    Returns:
        Dict with status, latency, and error info if any
    """
    import time

    result = {
        "status": "unhealthy",
        "database": db.name,
        "latency_ms": -1,
        "error": None,
    }

    try:
        client = get_redis_client(db)

        start = time.perf_counter()
        pong = await client.ping()
        latency = (time.perf_counter() - start) * 1000

        if pong:
            result["status"] = "healthy"
            result["latency_ms"] = round(latency, 2)

            # Get some stats
            info = await client.info("memory")
            result["used_memory_human"] = info.get("used_memory_human", "unknown")
            result["connected_clients"] = (await client.info("clients")).get(
                "connected_clients", -1
            )

    except Exception as e:
        result["error"] = str(e)
        logger.error("Redis health check failed for %s: %s", db.name, e)

    return result


__all__ = [
    "RedisDatabase",
    "RedisConfig",
    "get_redis_config",
    "get_redis_client",
    "check_redis_stack_available",
    "close_all_pools",
    "redis_health_check",
]
