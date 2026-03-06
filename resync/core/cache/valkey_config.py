"""
Centralized Valkey Configuration for Semantic Caching.

v5.3.16 - Valkey configuration with:
- Database separation by purpose
- Connection pooling with lazy initialization
- Support for Valkey Stack (RediSearch) when available
- Graceful fallback to Valkey OSS

Design decisions (30 years of experience speaking):
1. Never hardcode passwords - always from environment
2. Lazy initialization - don't connect until needed
3. Connection pooling - reuse TCP connections
4. Graceful degradation - system works even if Valkey fails
"""

import logging
import os
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import valkey.asyncio as valkey_async

logger = logging.getLogger(__name__)

class ValkeyDatabase(IntEnum):
    """
    Valkey database separation by purpose.

    Why separate DBs?
    - Isolation: different TTLs, different eviction policies
    - Debugging: easier to inspect specific data
    - Performance: FLUSHDB affects only one purpose

    Note: Valkey supports DBs 0-15 by default.
    """

    CONNECTIONS = 0  # Connection pools, health checks (existing usage)
    SESSIONS = 1  # User sessions, rate limiting (existing usage)
    CACHE = 2  # General application cache
    SEMANTIC_CACHE = 3  # Semantic cache for LLM responses (NEW)
    IDEMPOTENCY = 4  # Idempotency keys for request deduplication
    # DBs 5-15: Reserved for future use

class ValkeyConfig:
    """
    Configuration holder for Valkey connections.

    All values come from environment variables with sensible defaults.
    Never expose passwords in logs or error messages.
    """

    def __init__(self) -> None:
        # v5.9.7: Prefer consolidated app settings (supports APP_VALKEY_URL + legacy VALKEY_URL).
        # Environment variables still take precedence for backwards compatibility.
        try:
            from resync.settings import settings as app_settings
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):  # pragma: no cover
            app_settings = None  # type: ignore

        valkey_url = None
        if app_settings is not None:
            valkey_url = getattr(app_settings, "valkey_url", None)
            # Handle Pydantic SecretStr if applicable
            if hasattr(valkey_url, "get_secret_value"):
                valkey_url = valkey_url.get_secret_value()  # type: ignore[union-attr]

        valkey_url = valkey_url or os.getenv("APP_VALKEY_URL") or "valkey://localhost:6379/0"
        parsed = urlparse(valkey_url) if valkey_url else None

        # Base connection settings
        self.host: str = os.getenv("APP_VALKEY_HOST") or (
            parsed.hostname if parsed and parsed.hostname else "localhost"
        )
        self.port: int = int(
            os.getenv("APP_VALKEY_PORT")
            or (str(parsed.port) if parsed and parsed.port else "6379")
        )
        env_password = os.getenv("APP_VALKEY_PASSWORD")
        password_from_parsed = parsed.password if parsed and parsed.password else None

        self.password: str | None = env_password or password_from_parsed

        # Ensure password is a string if it's a SecretStr
        if hasattr(self.password, "get_secret_value"):
            self.password = self.password.get_secret_value()  # type: ignore[union-attr]

        # Connection pool settings
        default_pool_min = (
            str(getattr(app_settings, "valkey_pool_min_size", 5))
            if app_settings
            else "5"
        )
        default_pool_max = (
            str(getattr(app_settings, "valkey_pool_max_size", 20))
            if app_settings
            else "20"
        )
        default_socket_timeout = (
            str(getattr(app_settings, "valkey_timeout", 5.0)) if app_settings else "5.0"
        )
        default_connect_timeout = (
            str(getattr(app_settings, "valkey_pool_connect_timeout", 5.0))
            if app_settings
            else "5.0"
        )

        self.pool_min_connections: int = int(
            os.getenv("APP_VALKEY_POOL_MIN", default_pool_min)
        )
        self.pool_max_connections: int = int(
            os.getenv("APP_VALKEY_POOL_MAX", default_pool_max)
        )
        self.socket_timeout: float = float(
            os.getenv("APP_VALKEY_SOCKET_TIMEOUT", default_socket_timeout)
        )
        self.socket_connect_timeout: float = float(
            os.getenv("APP_VALKEY_CONNECT_TIMEOUT", default_connect_timeout)
        )

        # Retry settings
        self.retry_on_timeout: bool = True
        default_health = (
            str(getattr(app_settings, "valkey_health_check_interval", 30))
            if app_settings
            else "30"
        )
        self.health_check_interval: int = int(
            os.getenv("APP_VALKEY_HEALTH_INTERVAL", default_health)
        )

        # Semantic cache specific
        self.semantic_cache_ttl: int = int(
            os.getenv("SEMANTIC_CACHE_TTL", "86400")
        )  # 24h default
        self.semantic_cache_threshold: float = float(
            os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.25")
        )
        self.semantic_cache_max_entries: int = int(
            os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "100000")
        )

    def get_url(self, db: ValkeyDatabase = ValkeyDatabase.CONNECTIONS) -> str:
        """
        Build Valkey URL for specific database.

        Format: valkey://[:password@]host:port/db
        """
        auth = f":{self.password}@" if self.password else ""
        return f"valkey://{auth}{self.host}:{self.port}/{db.value}"

    def __repr__(self) -> str:
        """Safe repr that never shows password."""
        return (
            f"ValkeyConfig(host={self.host}, port={self.port}, "
            f"password={'***' if self.password else None})"
        )

# Global config instance (singleton pattern)
@lru_cache(maxsize=1)
def get_valkey_config() -> ValkeyConfig:
    """
    Get singleton Valkey configuration.

    Uses lru_cache for thread-safe singleton pattern.
    """
    return ValkeyConfig()

# Connection pool cache (one per database)
_connection_pools: dict[ValkeyDatabase, Any] = {}

def get_valkey_client(
    db: ValkeyDatabase = ValkeyDatabase.CONNECTIONS,
    decode_responses: bool = True,
) -> "valkey_async.Valkey":
    """
    Get Valkey client for specific database with connection pooling.

    Args:
        db: Which Valkey database to connect to
        decode_responses: If True, return strings instead of bytes

    Returns:
        Async Valkey client with connection pool

    Raises:
        RuntimeError: If valkey-py not installed
        ConnectionError: If Valkey is unreachable
    """
    try:
        from valkey.asyncio import from_url
    except ImportError as e:
        raise RuntimeError(
            "valkey not installed. Run: pip install valkey[hiredis]"
        ) from e

    config = get_valkey_config()
    url = config.get_url(db)

    # Check cache to avoid creating a new pool for every request
    if db in _connection_pools:
        return _connection_pools[db]

    client = from_url(
        url,
        decode_responses=decode_responses,
        max_connections=config.pool_max_connections,
        socket_timeout=config.socket_timeout,
        socket_connect_timeout=config.socket_connect_timeout,
        retry_on_timeout=config.retry_on_timeout,
        health_check_interval=config.health_check_interval,
    )
    _connection_pools[db] = client
    return client

async def check_valkey_stack_available() -> dict[str, bool | str]:
    """
    Check if Valkey Stack modules are available.

    Returns dict with module availability:
    - search: RediSearch (required for semantic cache)
    - json: ReJSON
    - bloom: Valkey Bloom
    - timeseries: Valkey TimeSeries

    Why this matters:
    - With Valkey Stack: Use native vector search (faster, more features)
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
        client = get_valkey_client(ValkeyDatabase.SEMANTIC_CACHE)

        # Get Valkey version
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

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            # MODULE LIST not available (older Valkey or disabled)
            # Try specific commands to detect modules
            try:
                await client.execute_command("FT._LIST")
                result["search"] = True
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.debug(
                    "suppressed_exception", extra={"error": str(exc)}, exc_info=True
                )

            try:
                await client.execute_command("JSON.DEBUG", "MEMORY", "__test__")
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                if "unknown command" not in str(exc).lower():
                    result["json"] = True

        logger.info(
            f"Valkey Stack check: version={result['redis_version']}, "
            f"search={result['search']}, json={result['json']}"
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        logger.warning("Failed to check Valkey Stack availability", exc_info=True)

    return result  # type: ignore[return-value]

async def close_all_pools() -> None:
    """
    Close all Valkey connection pools.

    Call this during application shutdown to release resources cleanly.
    """
    for db, pool in _connection_pools.items():
        try:
            await pool.disconnect()
            logger.info("Closed Valkey pool", extra={"db": db.name})
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.warning(
                "Error closing Valkey pool", extra={"db": db.name, "error": str(e)}
            )

    _connection_pools.clear()

# Health check utility
async def valkey_health_check(
    db: ValkeyDatabase = ValkeyDatabase.CONNECTIONS,
) -> dict[str, Any]:
    """
    Perform health check on specific Valkey database.

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
        client = get_valkey_client(db)

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

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        result["error"] = str(e)
        logger.error(
            "Valkey health check failed", extra={"db": db.name, "error": str(e)}
        )

    return result

__all__ = [
    "ValkeyDatabase",
    "ValkeyConfig",
    "get_valkey_config",
    "get_valkey_client",
    "check_valkey_stack_available",
    "close_all_pools",
    "valkey_health_check",
]
