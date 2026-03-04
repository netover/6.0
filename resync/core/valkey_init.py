"""
Valkey initialization and connection management (Valkey 9 migration).

This module provides Valkey client initialization with connection pooling,
distributed locking, health checks, and proper error handling.
"""

import asyncio
import inspect
import logging
import os
import threading
import socket
from collections.abc import Awaitable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from resync.core.task_tracker import create_tracked_task

if TYPE_CHECKING:
    from resync.core.idempotency.manager import IdempotencyManager

from resync.settings import settings
try:
    import valkey.asyncio as valkey  # type: ignore
    from valkey.exceptions import (
        AuthenticationError,
        BusyLoadingError,
        ValkeyError,
    )
    from valkey.exceptions import (
        ConnectionError as ValkeyConnError,
    )
    from valkey.exceptions import (
        TimeoutError as ValkeyTimeoutError,
    )
    # Alias for backward compatibility
    ValkeyError = ValkeyError  # type: ignore
except ImportError:
    valkey = None  # type: ignore
    ValkeyError = Exception  # type: ignore
    BusyLoadingError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    ValkeyConnError = Exception  # type: ignore
    ValkeyTimeoutError = Exception  # type: ignore

# Legacy aliases for backward compatibility
RedisError = ValkeyError
RedisConnError = ValkeyConnError
RedisTimeoutError = ValkeyTimeoutError

logger = logging.getLogger(__name__)


async def _ensure_awaitable_bool(result: Awaitable[bool] | bool) -> bool:
    """Normalize redis methods that may return bool or awaitable bool."""
    if inspect.isawaitable(result):
        return await result
    return result


async def _ensure_awaitable_str(result: Awaitable[str] | str) -> str:
    """Normalize redis eval that may return str or awaitable str."""
    if inspect.isawaitable(result):
        return await result
    return result


def _resolve_redis_url(url_value: object) -> str:
    """Normalize Valkey/Redis url value from str/SecretStr-like objects."""
    getter = getattr(url_value, "get_secret_value", None)
    if callable(getter):
        secret = getter()
        return str(secret)
    if isinstance(url_value, str):
        return url_value
    if url_value is None:
        return "valkey://localhost:6379/0"
    return str(url_value)


def _env_flag(name: str, default: str = "0") -> bool:
    """Return True when an env var is set to a truthy value.

    Accepts common truthy strings ("1", "true", "yes", "on") and treats
    unset/empty/"0"/"false" as False.

    We intentionally support both ``RESYNC_DISABLE_VALKEY=1`` and
    ``RESYNC_DISABLE_VALKEY=true`` because operators commonly use boolean
    values in .env / Kubernetes manifests.
    """

    raw = os.getenv(name, default)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ValkeyInitError(RuntimeError):
    """Erro de inicialização do Valkey."""


# Legacy alias for backward compatibility
RedisInitError = ValkeyInitError


_REDIS_CLIENT: "valkey.Redis" | None = None  # type: ignore
_REDIS_CLIENT_INIT_LOCK = threading.Lock()

_IDEMPOTENCY_MANAGER: "IdempotencyManager" | None = None


def is_valkey_available() -> bool:
    """Check if Valkey library is available."""
    return valkey is not None


def get_valkey_client() -> "valkey.Redis":  # type: ignore
    """Return the canonical Valkey client.

    In production (gunicorn --preload), creating network clients during import
    can lead to subtle bugs (master process connects, workers inherit state).

    Therefore, **by default** this accessor only returns a client that was
    initialized during startup (lifespan) via :class:`ValkeyInitializer`.

    - Set **RESYNC_VALKEY_LAZY_INIT=1** to allow the legacy "create-on-first-use"
      behavior (useful for one-off scripts).
    - Set **RESYNC_DISABLE_VALKEY=1** (or ``true``) to disable Valkey entirely (tests/CI).
    """

    if _env_flag("RESYNC_DISABLE_VALKEY"):
        raise RuntimeError("Valkey disabled by RESYNC_DISABLE_VALKEY")
    if valkey is None:
        raise RuntimeError("valkey-py not installed (valkey.asyncio).")

    global _REDIS_CLIENT  # pylint

    if _REDIS_CLIENT is None:
        # Lazy init is supported only for legacy/dev scripts. In async servers,
        # concurrent calls may happen (e.g., via threadpool). Guard with a lock.
        with _REDIS_CLIENT_INIT_LOCK:
            if _REDIS_CLIENT is None:
                lazy_valkey = os.getenv("RESYNC_VALKEY_LAZY_INIT")
                lazy_redis = os.getenv("RESYNC_REDIS_LAZY_INIT", "0")
                lazy = (lazy_valkey or lazy_redis).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                }
                if not lazy:
                    raise RuntimeError(
                        "Valkey client not initialized. Ensure "
                        "ValkeyInitializer.initialize() runs "
                        "during application startup (lifespan). To allow lazy init (legacy), "
                        "set RESYNC_VALKEY_LAZY_INIT=1."
                    )

                # Legacy path (scripts/dev only)
                _url = getattr(settings, "valkey_url", None)
                url = (
                    _resolve_redis_url(_url)
                    if _url is not None
                    else os.getenv("APP_VALKEY_URL") or "valkey://localhost:6379/0"
                )

                _REDIS_CLIENT = valkey.from_url(url, encoding="utf-8", decode_responses=True)
                logger.warning("Initialized Valkey client via lazy init (RESYNC_VALKEY_LAZY_INIT=1).")

    return _REDIS_CLIENT


async def close_valkey_client() -> None:
    """Close the global Valkey client if it exists.

    This is the public API for shutting down the module-level client.
    External modules should use this instead of importing ``_REDIS_CLIENT``.
    """
    global _REDIS_CLIENT  # pylint
    if _REDIS_CLIENT is not None:
        try:
            await _REDIS_CLIENT.close()
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error

            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.debug("valkey_close_error_ignored: %s", str(exc))  # best-effort on shutdown
        _REDIS_CLIENT = None


# Legacy aliases for backward compatibility
def get_redis_client() -> "valkey.Redis":
    """Alias for get_valkey_client() for Redis migration."""
    return get_valkey_client()


def is_redis_available() -> bool:
    """Alias for is_valkey_available() for Redis migration."""
    return is_valkey_available()


async def close_redis_client() -> None:
    """Alias for close_valkey_client() for Redis migration."""
    await close_valkey_client()


def get_idempotency_manager() -> "IdempotencyManager":
    """Return the global idempotency manager if initialized.

    This accessor is primarily for non-HTTP contexts. In the HTTP path, use
    FastAPI dependencies wired via lifespan/app.state.
    """
    from resync.core.idempotency.manager import IdempotencyManager

    global _IDEMPOTENCY_MANAGER  # pylint
    if _IDEMPOTENCY_MANAGER is None:
        # Best-effort lazy init using the lazy redis client accessor.
        client = get_redis_client()
        _IDEMPOTENCY_MANAGER = IdempotencyManager(client)
    return _IDEMPOTENCY_MANAGER


class ValkeyInitializer:
    """
    Thread-safe Valkey initialization with connection pooling.
    """

    UNLOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
      return redis.call("del", KEYS[1])
    else
      return 0
    end
    """

    def __init__(self) -> None:
        # P0 fix: Initialize lock eagerly to prevent race condition
        # Lock is still lazily bound to event loop but with proper initialization
        self._lock: asyncio.Lock = asyncio.Lock()
        self._initialized = False
        self._client: valkey.Redis | None = None  # type: ignore
        self._health_task: asyncio.Task[Any] | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Return the async lock (eagerly initialized)."""
        return self._lock

    @property
    def initialized(self) -> bool:
        """Retorna se o initializer está inicializado. Para compatibilidade."""
        return self._initialized

    async def initialize(
        self,
        max_retries: int = 3,
        base_backoff: float = 0.1,  # pylint
        max_backoff: float = 10.0,  # pylint
        health_check_interval: int = 5,
        fatal_on_fail: bool = False,  # pylint
        redis_url: str | None = None,
    ) -> "valkey.Redis":  # type: ignore
        """
        Inicializa cliente Redis com:
        - Lock concorrente
        - Lock distribuído seguro (unlock com verificação)
        - Teste RW consistente
        - Health check em background
        """
        if _env_flag("RESYNC_DISABLE_VALKEY"):
            raise ValkeyInitError("Valkey disabled by RESYNC_DISABLE_VALKEY")
        if valkey is None:
            raise ValkeyInitError("valkey-py (valkey.asyncio) not installed.")

        async with self.lock:
            if self._initialized and self._client:
                try:
                    await asyncio.wait_for(_ensure_awaitable_bool(self._client.ping()), timeout=1.0)
                    return self._client
                except (RedisError, asyncio.TimeoutError):
                    logger.warning("Existing Valkey connection lost, reinitializing")
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
                            ("Another instance is initializing Valkey, waiting... (attempt %s/%s)"),
                            attempt + 1,
                            max_retries,
                        )
                        # Avoid leaking clients/pools
                        # when init lock is held elsewhere.
                        try:
                            await redis_client.close()
                        except (
                            OSError,
                            ValueError,
                            TypeError,
                            KeyError,
                            AttributeError,
                            RuntimeError,
                            TimeoutError,
                            ConnectionError,
                        ):
                            logger.debug("redis_client_close_failed", exc_info=True)
                        await asyncio.sleep(2)
                        continue

                    try:
                        # Conectividade básica
                        await asyncio.wait_for(
                            _ensure_awaitable_bool(redis_client.ping()), timeout=2.0
                        )

                        # Teste RW coerente com decode_responses=True
                        test_key = f"resync:health:test:{os.getpid()}"
                        await redis_client.set(test_key, "ok", ex=60)
                        test_value = await redis_client.get(test_key)
                        if test_value != "ok":
                            raise ValkeyInitError("Valkey read/write test failed")
                        await redis_client.delete(test_key)

                        # Idempotency manager
                        self._initialize_idempotency(redis_client)

                        self._client = redis_client
                        self._initialized = True

                        # Keep module-level lazy accessors consistent.
                        # This avoids having multiple Redis client instances
                        # in the same worker process (lifespan vs lazy access).
                        global _REDIS_CLIENT  # pylint
                        _REDIS_CLIENT = redis_client

                        logger.info(
                            "Valkey initialized successfully",
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
                        self._health_task = create_tracked_task(
                            self._health_check_loop(health_check_interval)
                        )

                        return redis_client

                    finally:
                        # Unlock seguro (só remove se ainda for nosso lock)
                        # SECURITY NOTE: legit Redis EVAL usage
                        # for atomic distributed locking
                        # Lua script is hardcoded and
                        # performs only safe Redis operations
                        with suppress(RedisError, ConnectionError):
                            await _ensure_awaitable_str(
                                redis_client.eval(self.UNLOCK_SCRIPT, 1, lock_key, lock_val)
                            )

                except AuthenticationError as e:
                    msg = f"Valkey authentication failed: {e}"
                    logger.critical(msg)
                    raise ValkeyInitError(msg) from e

                except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                    TimeoutError,
                    ConnectionError,
                ) as e:  # pylint
                    msg = "Unexpected error during Valkey initialization"
                    logger.critical(msg, exc_info=True)
                    raise ValkeyInitError(f"{msg}: {e}") from e

        raise ValkeyInitError("Valkey initialization failed - unexpected fallthrough") from None

    def _create_client_with_pool(self, redis_url: str | None = None) -> "valkey.Redis":  # type: ignore
        if valkey is None:
            raise ValkeyInitError("valkey not installed.")
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
        _url = redis_url or getattr(settings, "valkey_url", "valkey://localhost:6379/0")
        url = _resolve_redis_url(_url)

        max_conns = getattr(settings, "valkey_pool_max_size", None) or getattr(
            settings, "valkey_max_connections", 50
        )
        socket_connect_timeout = getattr(settings, "valkey_pool_connect_timeout", 5)
        socket_timeout = getattr(settings, "valkey_timeout", 30)
        health_interval = getattr(settings, "valkey_health_check_interval", 30)

        return valkey.Redis.from_url(
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

    def _initialize_idempotency(self, redis_client: "valkey.Redis") -> None:  # type: ignore
        """Initialize the global idempotency manager using the provided Redis client."""
        from resync.core.idempotency.manager import IdempotencyManager

        global _IDEMPOTENCY_MANAGER  # pylint
        _IDEMPOTENCY_MANAGER = IdempotencyManager(redis_client)
        logger.info("Idempotency manager initialized")

    async def _health_check_loop(self, interval: int) -> None:
        while self._initialized:
            await asyncio.sleep(interval)
            try:
                if self._client:
                    await asyncio.wait_for(_ensure_awaitable_bool(self._client.ping()), timeout=2.0)
            except (RedisError, asyncio.TimeoutError):
                logger.error("Valkey health check failed - attempting reconnect", exc_info=True)
                self._initialized = False
                try:
                    await self.initialize(max_retries=2, fatal_on_fail=False)
                    logger.info("Valkey health-check reconnect succeeded")
                    continue
                except ValkeyInitError:
                    logger.critical("Valkey reconnect failed; stopping health check", exc_info=True)
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


# Global Valkey initializer instance - lazy initialization
_valkey_initializer: ValkeyInitializer | None = None
_valkey_initializer_create_lock = threading.Lock()


def get_valkey_initializer() -> ValkeyInitializer:
    """
    Retorna instância global do initializer com double-checked locking
    para evitar race conditions em ambientes de alta concorrência.
    """
    global _valkey_initializer  # pylint
    if _valkey_initializer is None:
        with _valkey_initializer_create_lock:
            if _valkey_initializer is None:
                _valkey_initializer = ValkeyInitializer()
    return _valkey_initializer

# Legacy aliases for backward compatibility
RedisInitializer = ValkeyInitializer
get_redis_initializer = get_valkey_initializer
_redis_initializer = _valkey_initializer
