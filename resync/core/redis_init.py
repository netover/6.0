"""
Redis initialization and connection management.

This module provides Redis client initialization with connection pooling,
distributed locking, health checks, and proper error handling.
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
import logging
import os
import socket
from contextlib import suppress
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from resync.core.idempotency.manager import IdempotencyManager

from resync.settings import settings

try:  # pragma: no cover
    import redis.asyncio as redis  # type: ignore

    # Import correct Redis exceptions
    from redis.exceptions import (
        AuthenticationError,
        BusyLoadingError,
        RedisError,
    )
    from redis.exceptions import (
        ConnectionError as RedisConnError,
    )
    from redis.exceptions import (
        TimeoutError as RedisTimeoutError,
    )
except ImportError:  # redis opcional
    # If redis is not installed, define placeholder types for exceptions to avoid NameError
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore
    BusyLoadingError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    RedisConnError = Exception  # type: ignore
    RedisTimeoutError = Exception  # type: ignore

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    """Return True when an env var is set to a truthy value.

    Accepts common truthy strings ("1", "true", "yes", "on") and treats
    unset/empty/"0"/"false" as False.

    We intentionally support both ``RESYNC_DISABLE_REDIS=1`` and
    ``RESYNC_DISABLE_REDIS=true`` because operators commonly use boolean
    values in .env / Kubernetes manifests.
    """

    raw = os.getenv(name, default)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class RedisInitError(RuntimeError):
    """Erro de inicialização do Redis."""


_REDIS_CLIENT: Optional["redis.Redis"] = None  # type: ignore

_IDEMPOTENCY_MANAGER: Optional["IdempotencyManager"] = None


def is_redis_available() -> bool:
    """Check if Redis library is available."""
    return redis is not None


def get_redis_client() -> "redis.Redis":  # type: ignore
    """Return the canonical Redis client.

    In production (gunicorn --preload), creating network clients during import
    can lead to subtle bugs (master process connects, workers inherit state).

    Therefore, **by default** this accessor only returns a client that was
    initialized during startup (lifespan) via :class:`RedisInitializer`.

    - Set **RESYNC_REDIS_LAZY_INIT=1** to allow the legacy "create-on-first-use"
      behavior (useful for one-off scripts).
    - Set **RESYNC_DISABLE_REDIS=1** (or ``true``) to disable Redis entirely (tests/CI).
    """

    if _env_flag("RESYNC_DISABLE_REDIS"):
        raise RuntimeError("Redis disabled by RESYNC_DISABLE_REDIS")
    if redis is None:
        raise RuntimeError("redis-py not installed (redis.asyncio).")

    global _REDIS_CLIENT  # pylint: disable=W0603

    if _REDIS_CLIENT is None:
        lazy = os.getenv("RESYNC_REDIS_LAZY_INIT", "0").strip().lower() in {"1", "true", "yes"}
        if not lazy:
            raise RuntimeError(
                "Redis client not initialized. Ensure RedisInitializer.initialize() runs "
                "during application startup (lifespan). To allow lazy init (legacy), "
                "set RESYNC_REDIS_LAZY_INIT=1."
            )

        # Legacy path (scripts/dev only)
        _url = getattr(settings, "redis_url", None)
        if hasattr(_url, "get_secret_value"):
            url = _url.get_secret_value()
        else:
            url = _url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
        _REDIS_CLIENT = redis.from_url(url, encoding="utf-8", decode_responses=True)
        logger.warning("Initialized Redis client via lazy init (RESYNC_REDIS_LAZY_INIT=1).")

    return _REDIS_CLIENT


async def close_redis_client() -> None:
    """Close the global Redis client if it exists.

    This is the public API for shutting down the module-level client.
    External modules should use this instead of importing ``_REDIS_CLIENT``.
    """
    global _REDIS_CLIENT  # pylint: disable=W0603
    if _REDIS_CLIENT is not None:
        try:
            await _REDIS_CLIENT.close()
        except Exception:
            pass  # best-effort; logged by caller
        _REDIS_CLIENT = None



def get_idempotency_manager() -> "IdempotencyManager":
    """Return the global idempotency manager if initialized.

    This accessor is primarily for non-HTTP contexts. In the HTTP path, use
    FastAPI dependencies wired via lifespan/app.state.
    """
    from resync.core.idempotency.manager import IdempotencyManager

    global _IDEMPOTENCY_MANAGER  # pylint: disable=W0603
    if _IDEMPOTENCY_MANAGER is None:
        # Best-effort lazy init using the lazy redis client accessor.
        client = get_redis_client()
        _IDEMPOTENCY_MANAGER = IdempotencyManager(client)
    return _IDEMPOTENCY_MANAGER
class RedisInitializer:
    """
    Thread-safe Redis initialization with connection pooling.
    """

    UNLOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
      return redis.call("del", KEYS[1])
    else
      return 0
    end
    """

    def __init__(self):
        self._lock: asyncio.Lock | None = None
        self._initialized = False
        self._client: redis.Redis | None = None  # type: ignore
        self._health_task: asyncio.Task | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def initialized(self) -> bool:
        """Retorna se o initializer está inicializado. Para compatibilidade."""
        return self._initialized

    async def initialize(
        self,
        max_retries: int = 3,
        base_backoff: float = 0.1,  # pylint: disable=W0613
        max_backoff: float = 10.0,  # pylint: disable=W0613
        health_check_interval: int = 5,
        fatal_on_fail: bool = False,  # pylint: disable=W0613
        redis_url: str | None = None,
    ) -> "redis.Redis":  # type: ignore
        """
        Inicializa cliente Redis com:
        - Lock concorrente
        - Lock distribuído seguro (unlock com verificação)
        - Teste RW consistente
        - Health check em background
        """
        if _env_flag("RESYNC_DISABLE_REDIS"):
            raise RedisInitError("Redis disabled by RESYNC_DISABLE_REDIS")
        if redis is None:
            raise RedisInitError("redis-py (redis.asyncio) not installed.")

        async with self.lock:
            if self._initialized and self._client:
                try:
                    await asyncio.wait_for(self._client.ping(), timeout=1.0)
                    return self._client
                except (RedisError, asyncio.TimeoutError):
                    logger.warning("Existing Redis connection lost, reinitializing")
                    self._initialized = False

            lock_key = "resync:init:lock"
            lock_val = f"instance-{os.getpid()}"
            lock_timeout = 30  # seconds

            for attempt in range(max_retries):
                try:
                    redis_client = self._create_client_with_pool(redis_url)

                    acquired = await redis_client.set(lock_key, lock_val, nx=True, ex=lock_timeout)
                    if not acquired:
                        logger.info(
                            "Another instance is initializing Redis, waiting... (attempt %s/%s)",
                            attempt + 1,
                            max_retries,
                        )
                        # Avoid leaking clients/pools when the init lock is held elsewhere.
                        try:
                            await redis_client.close()
                        except Exception:
                            logger.debug("redis_client_close_failed", exc_info=True)
                        await asyncio.sleep(2)
                        continue

                    try:
                        # Conectividade básica
                        await asyncio.wait_for(redis_client.ping(), timeout=2.0)

                        # Teste RW coerente com decode_responses=True
                        test_key = f"resync:health:test:{os.getpid()}"
                        await redis_client.set(test_key, "ok", ex=60)
                        test_value = await redis_client.get(test_key)
                        if test_value != "ok":
                            raise RedisInitError("Redis read/write test failed")
                        await redis_client.delete(test_key)

                        # Idempotency manager
                        self._initialize_idempotency(redis_client)

                        self._client = redis_client
                        self._initialized = True

                        # Keep module-level lazy accessors consistent.
                        # This avoids having multiple Redis client instances
                        # in the same worker process (lifespan vs lazy access).
                        global _REDIS_CLIENT  # pylint: disable=W0603
                        _REDIS_CLIENT = redis_client

                        logger.info(
                            "Redis initialized successfully",
                            extra={
                                "attempt": attempt + 1,
                                "pool_size": getattr(
                                    redis_client.connection_pool,
                                    "max_connections",
                                    None,
                                ),
                            },
                        )

                        # Health check (encerra se já houver uma task antiga)
                        if self._health_task and not self._health_task.done():
                            self._health_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await self._health_task
                        self._health_task = await create_tracked_task(
                            self._health_check_loop(health_check_interval)
                        )

                        return redis_client

                    finally:
                        # Unlock seguro (só remove se ainda for nosso lock)
                        # SECURITY NOTE: This is legitimate Redis EVAL usage for atomic distributed locking
                        # The Lua script is hardcoded and performs only safe Redis operations
                        with suppress(RedisError, ConnectionError):
                            await redis_client.eval(self.UNLOCK_SCRIPT, 1, lock_key, lock_val)

                except AuthenticationError as e:
                    msg = f"Redis authentication failed: {e}"
                    logger.critical(msg)
                    raise RedisInitError(msg) from e

                except Exception as e:  # pylint: disable=W0705
                    msg = "Unexpected error during Redis initialization"
                    logger.critical(msg, exc_info=True)
                    raise RedisInitError(f"{msg}: {e}") from e

        raise RedisInitError("Redis initialization failed - unexpected fallthrough") from None

    def _create_client_with_pool(self, redis_url: str | None = None) -> "redis.Redis":  # type: ignore
        if redis is None:
            raise RedisInitError("redis not installed.")
        keepalive_opts = {}
        # Opções portáveis
        for name in ("TCP_KEEPIDLE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            if hasattr(socket, name):
                keepalive_opts[getattr(socket, name)] = {  # type: ignore[arg-type]
                    "TCP_KEEPIDLE": 60,
                    "TCP_KEEPINTVL": 10,
                    "TCP_KEEPCNT": 3,
                }[name]
        # v5.9.7: Use consolidated settings field names
        _url = redis_url or getattr(settings, "redis_url", "redis://localhost:6379/0")
        if hasattr(_url, "get_secret_value"):
            url = _url.get_secret_value()
        else:
            url = str(_url)
            
        max_conns = getattr(settings, "redis_pool_max_size", None) or getattr(settings, "redis_max_connections", 50)
        socket_connect_timeout = getattr(settings, "redis_pool_connect_timeout", 5)
        socket_timeout = getattr(settings, "redis_timeout", 30)
        health_interval = getattr(settings, "redis_health_check_interval", 30)

        return redis.Redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=max_conns,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            socket_keepalive=True,
            socket_keepalive_options=keepalive_opts or None,
            health_check_interval=health_interval,
            retry_on_timeout=True,
            retry_on_error=[RedisConnError, RedisTimeoutError, BusyLoadingError],
        )

    def _initialize_idempotency(self, redis_client: "redis.Redis") -> None:  # type: ignore
        """Initialize the global idempotency manager using the provided Redis client."""
        from resync.core.idempotency.manager import IdempotencyManager

        global _IDEMPOTENCY_MANAGER  # pylint: disable=W0603
        _IDEMPOTENCY_MANAGER = IdempotencyManager(redis_client)
        logger.info("Idempotency manager initialized")

    async def _health_check_loop(self, interval: int) -> None:
        while self._initialized:
            await asyncio.sleep(interval)
            try:
                if self._client:
                    await asyncio.wait_for(self._client.ping(), timeout=2.0)
            except (RedisError, asyncio.TimeoutError):
                logger.error("Redis health check failed - connection may be lost", exc_info=True)
                self._initialized = False
                break
            except (OSError, ValueError) as e:
                logger.error("Unexpected error in Redis health check: %s", e, exc_info=True)

    async def close(self) -> None:
        """Close the Redis initializer and cleanup resources."""
        self._initialized = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task
        if self._client:
            with suppress(RedisError, ConnectionError):
                await self._client.close()
            with suppress(RedisError, ConnectionError):
                await self._client.connection_pool.disconnect()


# Global Redis initializer instance - lazy initialization
_redis_initializer: RedisInitializer | None = None


def get_redis_initializer() -> RedisInitializer:
    """
    Retorna instância global do initializer.
    Nota: se houver alta concorrência de criação, considere um lock.
    """
    global _redis_initializer  # pylint: disable=W0603
    if _redis_initializer is None:
        _redis_initializer = RedisInitializer()
    return _redis_initializer