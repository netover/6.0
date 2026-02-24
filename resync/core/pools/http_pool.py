# ruff: noqa: E501
# pylint
# mypy
"""
HTTP connection pool implementation for the Resync project.
Separated to follow Single Responsibility Principle.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from resync.core.exceptions import TWSConnectionError
from resync.core.pools.base_pool import ConnectionPool, ConnectionPoolConfig

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class HTTPConnectionPool(ConnectionPool[httpx.AsyncClient]):
    """HTTP connection pool for external API calls."""

    def __init__(
        self, config: ConnectionPoolConfig, base_url: str, **client_kwargs: Any
    ) -> None:
        super().__init__(config)
        self.base_url = base_url
        self.client_kwargs = client_kwargs
        self._client: httpx.AsyncClient | None = None

    def _setup_pool(self) -> None:
        """Setup HTTP connection pool using httpx."""
        try:
            # Configure httpx client with connection pooling
            limits = httpx.Limits(
                max_connections=self.config.max_size,
                max_keepalive_connections=max(self.config.min_size, 10),
                keepalive_expiry=self.config.idle_timeout,
            )

            timeout = httpx.Timeout(
                connect=self.config.connection_timeout,
                read=self.config.connection_timeout,
                write=self.config.connection_timeout,
                pool=self.config.connection_timeout,
            )

            transport = httpx.AsyncHTTPTransport(limits=limits, trust_env=True)

            # Create the httpx client with connection pooling
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                transport=transport,
                **self.client_kwargs,
            )

            logger.info(
                f"HTTP connection pool '{self.config.pool_name}' initialized with {self.config.min_size}-{self.config.max_size} connections"
            )
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to setup HTTP connection pool: %s", e)
            raise TWSConnectionError(
                f"Failed to setup HTTP connection pool: {e}"
            ) from e

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get an HTTP connection from the pool."""
        if not self._initialized or self._shutdown:
            raise TWSConnectionError("HTTP pool not initialized or shutdown")

        start_time = time.time()

        try:
            # Record pool request
            await self.increment_stat("pool_hits")

            # Get connection (httpx handles pooling)
            if not self._client:
                raise TWSConnectionError("HTTP client not available")

            yield self._client

        except Exception as e:
            await self.increment_stat("pool_misses")
            logger.error("Failed to get HTTP connection: %s", e)
            raise TWSConnectionError(f"Failed to acquire HTTP connection: {e}") from e
        finally:
            wait_time = time.time() - start_time
            await self.update_wait_time(wait_time)

            # Record connection metrics
            logger.debug(
                "HTTP connection acquired in {wait_time:.3f}s for pool {self.config.pool_name}"
            )

    async def _close_pool(self) -> None:
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            logger.info("HTTP connection pool '%s' closed", self.config.pool_name)
