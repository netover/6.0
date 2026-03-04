"""
Valkey connection pool implementation for the Resync project.
Separated to follow Single Responsibility Principle.
"""

import logging

# Soft import for valkey (optional dependency)
try:
    import valkey.asyncio as valkey
    from valkey.asyncio import Valkey as AsyncValkey
    from valkey.exceptions import ConnectionError as ValkeyConnectionError
    from valkey.exceptions import ValkeyError
except ImportError:
    valkey = None
    AsyncValkey = None
    ValkeyConnectionError = None
    ValkeyError = None

import os

from resync.core.valkey_init import get_valkey_client, is_valkey_available

logger = logging.getLogger(__name__)

class ValkeyPool:
    """Connection pool for valkey resources."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url
        self._client = None

    @property
    def client(self):
        disable = os.getenv("RESYNC_DISABLE_VALKEY", "0")
        if disable is not None and disable.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError("Valkey disabled by RESYNC_DISABLE_VALKEY")
        if not is_valkey_available():
            raise RuntimeError("valkey-py not installed")
        if self._client is None:
            if self._url:
                # usa URL explícita
                import valkey.asyncio as valkey  # type: ignore

                self._client = valkey.from_url(
                    self._url, encoding="utf-8", decode_responses=True
                )
                logger.info("Initialized Valkey client from explicit URL (lazy).")
            else:
                # usa factory centralizada
                self._client = get_valkey_client()
        return self._client

# Adicionar alias para compatibilidade
ValkeyConnectionPool = ValkeyPool
RedisPool = ValkeyPool
RedisConnectionPool = ValkeyPool
