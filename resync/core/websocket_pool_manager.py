import asyncio
import contextlib
import json

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from resync.core.metrics import runtime_metrics
from resync.core.task_tracker import create_tracked_task

# --- Logging Setup ---
logger = logging.getLogger(__name__)

def _get_settings():
    """Lazy import of settings to avoid circular imports."""
    from resync.settings import settings

    return settings

@dataclass
class WebSocketConnectionInfo:
    """Information about a WebSocket connection."""

    client_id: str
    websocket: WebSocket
    connected_at: datetime
    last_activity: datetime
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    is_healthy: bool = True
    connection_errors: int = 0

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def mark_error(self) -> None:
        """Mark connection as having an error."""
        self.connection_errors += 1
        if self.connection_errors > 5:
            self.is_healthy = False

@dataclass
class WebSocketPoolStats:
    """Statistics for WebSocket connection pool."""

    total_connections: int = 0
    active_connections: int = 0
    healthy_connections: int = 0
    unhealthy_connections: int = 0
    peak_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    connection_errors: int = 0
    cleanup_cycles: int = 0
    last_cleanup: datetime | None = None

class WebSocketPoolManager:
    """Enhanced WebSocket connection manager with pooling capabilities."""

    def __init__(self):
        self.connections: dict[str, WebSocketConnectionInfo] = {}
        self.stats = WebSocketPoolStats()
        # P0 fix: Initialize lock eagerly to prevent race condition
        self._lock: asyncio.Lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._initialized = False
        self._shutdown = False

        # Do NOT rely on GIL atomicity; protect shared snapshot state for sync readers.
        self._health_lock: threading.Lock = threading.Lock()
        self._cached_health_ok: bool = True

        # Number of accepts currently in-flight. This prevents thundering-herd
        # accept/close storms when many clients connect concurrently.
        self._pending_accepts: int = 0

        # Backpressure & latency guards
        self._send_timeout_seconds = float(os.getenv("WS_SEND_TIMEOUT_SECONDS", "5"))
        self._broadcast_concurrency = int(os.getenv("WS_BROADCAST_CONCURRENCY", "100"))

    async def initialize(self) -> None:
        """Initialize the WebSocket pool manager.

        P0-03 fix: Task creation moved inside lock to prevent double
        cleanup task when two concurrent calls race past the early-return.
        """
        async with self._lock:
            if self._initialized or self._shutdown:
                return
            self._initialized = True
            self._cleanup_task = create_tracked_task(
                self._cleanup_loop(), name="cleanup_loop"
            )

        await self._recompute_health_snapshot()

        logger.info("WebSocket pool manager initialized")
        runtime_metrics.record_gauge("websocket_pool.initialized", 1)

    async def shutdown(self) -> None:
        """Shutdown the WebSocket pool manager."""
        if self._shutdown:
            return

        self._shutdown = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        await self._close_all_connections()

        async with self._lock:
            self._initialized = False

            logger.info("WebSocket pool manager shutdown")
            runtime_metrics.record_gauge("websocket_pool.initialized", 0)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop for WebSocket connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(_get_settings().WS_POOL_CLEANUP_INTERVAL)
                await self._cleanup_connections()
                await self._recompute_health_snapshot()
            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Error in WebSocket cleanup loop: %s", e)

    async def _cleanup_connections(self) -> None:
        """Clean up stale and unhealthy WebSocket connections."""
        current_time = datetime.now(timezone.utc)
        connections_to_remove = []

        # Collect connections to remove under lock
        async with self._lock:
            for client_id, conn_info in self.connections.items():
                # Check for stale connections (no activity for timeout period)
                time_since_activity = (
                    current_time - conn_info.last_activity
                ).total_seconds()
                if time_since_activity > _get_settings().WS_CONNECTION_TIMEOUT:
                    connections_to_remove.append(client_id)
                    logger.warning("Removing stale WebSocket connection: %s", client_id)

                # Check for unhealthy connections
                elif not conn_info.is_healthy:
                    connections_to_remove.append(client_id)
                    logger.warning(
                        "Removing unhealthy WebSocket connection: %s", client_id
                    )

                # Additional check: enforce max connection duration
                # to prevent long-lived connections
                else:
                    try:
                        connection_duration = (
                            current_time - conn_info.connected_at
                        ).total_seconds()
                        max_duration = getattr(
                            _get_settings(), "WS_MAX_CONNECTION_DURATION", 3600
                        )  # Default 1 hour
                        if connection_duration > max_duration:
                            connections_to_remove.append(client_id)
                            logger.info(
                                "Removing long-lived WebSocket connection: "
                                f"{client_id} (duration: {connection_duration:.1f}s, "
                                f"max: {max_duration}s)"
                            )
                    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                        import sys as _sys
                        from resync.core.exception_guard import maybe_reraise_programming_error
                        _exc_type, _exc, _tb = _sys.exc_info()
                        maybe_reraise_programming_error(_exc, _tb)

                        logger.error(
                            "Error checking connection duration for %s: %s",
                            client_id,
                            e,
                        )
                        connections_to_remove.append(client_id)

            # Update stats under lock
            self.stats.cleanup_cycles += 1
            self.stats.last_cleanup = current_time

        # Remove connections OUTSIDE the lock to avoid deadlock
        for client_id in connections_to_remove:
            try:
                await self._remove_connection_safe(client_id)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Error removing connection during cleanup: %s", e)

        # Record cleanup metrics (after cleanup completes)
        runtime_metrics.record_counter(
            "websocket_pool.cleanup_cycles",
            len(connections_to_remove),
        )

    async def _close_all_connections(self) -> None:
        """Close all WebSocket connections."""
        # Get copy of connections under lock
        async with self._lock:
            connections_to_close = list(self.connections.keys())

        # Close each connection outside the lock
        for client_id in connections_to_close:
            try:
                await self._remove_connection_safe(client_id)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Error closing WebSocket connection %s: %s", client_id, e)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept a new WebSocket connection and add it to the pool.

        FIX P1-05: Original code checked capacity under lock, then released the
        lock, called websocket.accept() outside, and re-acquired the lock for the
        insert.  Under concurrent load, two coroutines could both pass the capacity
        check before either inserted, causing active_connections > WS_POOL_MAX_SIZE.

        New approach:
          1. Optimistic fast-path check WITHOUT lock (cheap, avoids blocking accept).
          2. Call websocket.accept() — blocking I/O, must be outside lock.
          3. Re-acquire lock for FINAL authoritative check + insert atomically.
             If the slot was taken while we were accepting, close & abort.

        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for the client
        """
        if self._shutdown:
            raise RuntimeError("WebSocket pool manager is shutdown")

        max_size: int = _get_settings().WS_POOL_MAX_SIZE

        # P1-05 (improved): reserve capacity BEFORE calling accept().
        # This prevents accept/close storms (and temporary over-capacity) when
        # many clients connect concurrently.
        old_conn: WebSocketConnectionInfo | None = None
        reject = False
        async with self._lock:
            # Replace duplicates deterministically.
            old_conn = self.connections.pop(client_id, None)
            if old_conn is not None:
                self.stats.active_connections = len(self.connections)
                logger.warning("Client %s already connected. Replacing connection.", client_id)

            if (len(self.connections) + self._pending_accepts) >= max_size:
                reject = True
                self.stats.connection_errors += 1
            else:
                self._pending_accepts += 1

        # Close the old connection OUTSIDE the lock.
        if old_conn is not None:
            try:
                if old_conn.websocket.client_state != WebSocketState.DISCONNECTED:
                    await old_conn.websocket.close(code=1000)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning("old_ws_close_failed", client_id=client_id, error=str(e))

        if reject:
            # To avoid ASGI state errors, accept first, then close with 1013.
            await websocket.accept()
            await websocket.close(code=1013, reason="Server at capacity")
            return

        # Accept the network connection (I/O, outside lock).
        try:
            await websocket.accept()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            async with self._lock:
                self._pending_accepts = max(self._pending_accepts - 1, 0)
            raise

        current_time = datetime.now(timezone.utc)
        conn_info = WebSocketConnectionInfo(
            client_id=client_id,
            websocket=websocket,
            connected_at=current_time,
            last_activity=current_time,
        )

        # Finalize insert under lock.
        async with self._lock:
            self._pending_accepts = max(self._pending_accepts - 1, 0)
            if self._shutdown or len(self.connections) >= max_size:
                self.stats.connection_errors += 1
                conn_info = None
            else:
                self.connections[client_id] = conn_info
                self.stats.total_connections += 1
                self.stats.active_connections = len(self.connections)
                self.stats.healthy_connections += 1
                if self.stats.active_connections > self.stats.peak_connections:
                    self.stats.peak_connections = self.stats.active_connections

        if conn_info is None:
            await websocket.close(code=1013, reason="Server at capacity")
            return

        logger.info("WebSocket connection accepted: %s", client_id)
        logger.info(
            "Total active WebSocket connections: %s", self.stats.active_connections
        )

        # Record connection metrics — no dynamic labels to avoid cardinality explosion
        runtime_metrics.record_counter(
            "websocket_pool.connections_accepted", 1
        )
        runtime_metrics.record_gauge(
            "websocket_pool.active_connections", self.stats.active_connections
        )

    async def disconnect(self, client_id: str) -> None:
        """
        Remove a WebSocket connection from the pool.

        Args:
            client_id: Unique identifier for the client
        """
        await self._remove_connection(client_id)

    async def _remove_connection_safe(self, client_id: str) -> None:
        """Alias for _remove_connection (kept for backward compat; both acquire lock).

        .. deprecated:: Use _remove_connection() directly.
        """
        await self._remove_connection(client_id)

    async def _remove_connection(self, client_id: str) -> None:
        """Internal method to remove a connection.

        Stability fix:
        - Updates pool stats under a single lock acquisition (avoids TOCTOU/races).
        - Closes the WebSocket outside the lock to prevent head-of-line blocking.
        """
        async with self._lock:
            conn_info = self.connections.pop(client_id, None)
            if conn_info is None:
                return

            # Update statistics atomically with the removal.
            if conn_info.is_healthy:
                self.stats.healthy_connections = max(0, self.stats.healthy_connections - 1)
            else:
                self.stats.unhealthy_connections = max(0, self.stats.unhealthy_connections - 1)

            self.stats.active_connections = len(self.connections)

        # Close WebSocket OUTSIDE the lock
        try:
            if conn_info.websocket.client_state != WebSocketState.DISCONNECTED:
                await conn_info.websocket.close()
        except asyncio.CancelledError:
            raise
        except (WebSocketDisconnect, ConnectionError, OSError, RuntimeError) as exc:
            logger.warning("Error closing WebSocket for %s: %s", client_id, exc)

        logger.info("WebSocket connection removed: %s", client_id)
        logger.info("Total active WebSocket connections: %s", self.stats.active_connections)

        runtime_metrics.record_counter("websocket_pool.connections_closed", 1)
        runtime_metrics.record_gauge("websocket_pool.active_connections", self.stats.active_connections)

    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """Send a message to a specific client.

        P1 fix: Access connection info under lock to prevent race condition
        with concurrent disconnect/cleanup operations.

        Args:
            message: The message to send
            client_id: The target client ID

        Returns:
            True if message was sent successfully, False otherwise
        """
        async with self._lock:
            conn_info = self.connections.get(client_id)
            if not conn_info:
                logger.warning("Client %s not found for personal message", client_id)
                return False
            ws_state = conn_info.websocket.client_state

        # Check WebSocket state before sending (ASGI compliance)
        if ws_state != WebSocketState.CONNECTED:
            logger.warning(
                "Client %s WebSocket is not connected (state: %s)",
                client_id,
                ws_state,
            )
            await self.disconnect(client_id)
            return False

        try:
            await asyncio.wait_for(
                conn_info.websocket.send_text(message),
                timeout=self._send_timeout_seconds,
            )
            msg_bytes = len(message.encode("utf-8"))

            # Update stats under lock
            async with self._lock:
                conn_info.update_activity()
                conn_info.message_count += 1
                conn_info.bytes_sent += msg_bytes
                self.stats.total_messages_sent += 1
                self.stats.total_bytes_sent += msg_bytes

            # Record message metrics — no dynamic labels
            runtime_metrics.record_counter("websocket_pool.messages_sent", 1)
            runtime_metrics.record_counter("websocket_pool.bytes_sent", msg_bytes)

            return True
        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket send timed out for client %s (timeout=%ss) — disconnecting",
                client_id,
                self._send_timeout_seconds,
            )
            await self._remove_connection(client_id)
            return False
        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket broadcast send timed out for client %s (timeout=%ss) — removing connection",
                client_id,
                self._send_timeout_seconds,
            )
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning("Connection issue sending message to %s: %s", client_id, e)
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("Unexpected error sending message to %s: %s", client_id, e)
            conn_info.mark_error()
            return False
    async def broadcast(self, message: str) -> int:
        """Send a message to all connected clients.

        Snapshots connection objects atomically under lock before releasing it,
        preventing TOCTOU races with concurrent _cleanup_loop / disconnect calls.
        Uses asyncio.gather with a semaphore for true concurrent sends.

        Args:
            message: The message to broadcast

        Returns:
            Number of clients that received the message
        """
        # Atomically snapshot both client IDs and their conn_info objects so that
        # concurrent _remove_connection calls cannot invalidate the references
        # between the snapshot and the actual send.
        async with self._lock:
            if not self.connections:
                logger.info("Broadcast requested, but no active WebSocket connections")
                return 0
            snapshot = list(self.connections.items())  # [(client_id, conn_info), ...]

        logger.info("Broadcasting message to %s WebSocket clients", len(snapshot))

        sem = asyncio.Semaphore(self._broadcast_concurrency)  # Limit concurrent sends

        async def _send_with_sem(cid: str, conn_info: WebSocketConnectionInfo) -> bool:
            async with sem:
                return await self._send_message_with_error_handling(
                    cid, conn_info, message
                )

        results = await asyncio.gather(
            *[_send_with_sem(cid, ci) for cid, ci in snapshot],
            return_exceptions=True,
        )

        client_ids = [cid for cid, _ in snapshot]
        successful_sends = sum(1 for r in results if r is True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    "Error in broadcast task for %s: %s", client_ids[i], r
                )

        logger.info("Message successfully broadcast to %s clients", successful_sends)
        return successful_sends

    async def _send_message_with_error_handling(
        self, client_id: str, conn_info: WebSocketConnectionInfo, message: str
    ) -> bool:
        """Send message to a pre-snapshotted connection with proper error handling.

        Accepts ``conn_info`` directly from a locked snapshot so there is no
        second dict lookup — eliminating the TOCTOU race between broadcast()
        collecting client IDs and this method fetching the connection object.
        """
        # Check WebSocket state before sending (ASGI compliance)
        if conn_info.websocket.client_state != WebSocketState.CONNECTED:
            logger.warning(
                "Client %s WebSocket is not connected during broadcast (state: %s)",
                client_id,
                conn_info.websocket.client_state,
            )
            await self._remove_connection(client_id)
            return False

        try:
            await asyncio.wait_for(conn_info.websocket.send_text(message), timeout=self._send_timeout_seconds)
            conn_info.update_activity()
            conn_info.message_count += 1
            conn_info.bytes_sent += len(message.encode("utf-8"))
            return True

        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket broadcast send timed out for client %s (timeout=%ss) — removing connection",
                client_id,
                self._send_timeout_seconds,
            )
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning("Connection issue during broadcast to %s: %s", client_id, e)
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False
        except RuntimeError as e:
            if "websocket state" in str(e).lower():
                logger.warning(
                    "WebSocket in wrong state during broadcast to %s: %s", client_id, e
                )
                conn_info.mark_error()
                return False
            logger.error("Runtime error during broadcast to %s: %s", client_id, e)
            return False
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TimeoutError, ConnectionError, ValueError) as e:
            logger.error("Unexpected error during broadcast to %s: %s", client_id, e)
            return False
        except Exception as e:
            logger.exception("Unexpected non-runtime error during broadcast to %s", client_id)
            return False

    async def broadcast_json(self, data: dict[str, Any]) -> int:
        """Send JSON data to all connected clients.

        Uses atomic snapshot under lock (same TOCTOU fix as broadcast()).
        asyncio.gather with semaphore for true concurrent sends.

        Args:
            data: The JSON data to broadcast

        Returns:
            Number of clients that received the data
        """
        async with self._lock:
            if not self.connections:
                logger.info("JSON broadcast requested, but no active WebSocket connections")
                return 0
            snapshot = list(self.connections.items())  # [(client_id, conn_info), ...]

        logger.info("Broadcasting JSON data to %s WebSocket clients", len(snapshot))

        sem = asyncio.Semaphore(100)

        async def _send_with_sem(cid: str, conn_info: WebSocketConnectionInfo) -> bool:
            async with sem:
                return await self._send_json_with_error_handling(cid, conn_info, data)

        results = await asyncio.gather(
            *[_send_with_sem(cid, ci) for cid, ci in snapshot],
            return_exceptions=True,
        )

        client_ids = [cid for cid, _ in snapshot]
        successful_sends = sum(1 for r in results if r is True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    "Error in JSON broadcast task for %s: %s", client_ids[i], r
                )

        logger.info("JSON data successfully broadcast to %s clients", successful_sends)
        return successful_sends

    async def _send_json_with_error_handling(
        self, client_id: str, conn_info: WebSocketConnectionInfo, data: dict[str, Any]
    ) -> bool:
        """Send JSON data to a pre-snapshotted connection with error handling."""
        # Check WebSocket state before sending (ASGI compliance)
        if conn_info.websocket.client_state != WebSocketState.CONNECTED:
            logger.warning(
                "Client %s WebSocket is not connected during JSON broadcast (state: %s)",
                client_id,
                conn_info.websocket.client_state,
            )
            await self._remove_connection(client_id)
            return False

        try:
            await conn_info.websocket.send_json(data)
            conn_info.update_activity()
            conn_info.message_count += 1
            json_str = (orjson.dumps(data).decode('utf-8') if orjson is not None else json.dumps(data))
            conn_info.bytes_sent += len(json_str.encode("utf-8"))
            return True

        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket broadcast send timed out for client %s (timeout=%ss) — removing connection",
                client_id,
                self._send_timeout_seconds,
            )
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning(
                "Connection issue during JSON broadcast to %s: %s", client_id, e
            )
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False
        except ValueError as e:
            logger.error(
                "JSON serialization error during broadcast to %s: %s", client_id, e
            )
            return False
        except RuntimeError as e:
            if "websocket state" in str(e).lower():
                logger.warning(
                    "WebSocket in wrong state during JSON broadcast to %s: %s",
                    client_id, e,
                )
                conn_info.mark_error()
                return False
            logger.error("Runtime error during JSON broadcast to %s: %s", client_id, e)
            return False
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "Unexpected error during JSON broadcast to %s: %s", client_id, e
            )
            return False

    def get_connection_info(self, client_id: str) -> WebSocketConnectionInfo | None:
        """Get information about a specific connection."""
        return self.connections.get(client_id)

    def get_all_connections(self) -> dict[str, WebSocketConnectionInfo]:
        """Get information about all connections."""
        return self.connections.copy()

    def get_stats(self) -> WebSocketPoolStats:
        """Get WebSocket pool statistics."""
        return self.stats

    async def _recompute_health_snapshot(self) -> None:
        """Compute and store a cached health boolean for sync readers.

        Do NOT rely on CPython GIL "atomicity". This remains safe for
        free-threaded builds. Writers run in the event loop; readers are sync.
        """
        async with self._lock:
            if self._shutdown:
                ok = False
            else:
                unhealthy_ratio = self.stats.unhealthy_connections / max(
                    1, self.stats.active_connections
                )
                ok = unhealthy_ratio <= 0.5

        # Publish snapshot for sync callers.
        with self._health_lock:
            self._cached_health_ok = ok

    def health_check(self) -> bool:
        """Return last computed health snapshot.

        This is synchronous and safe even without the GIL because it reads a
        cached boolean protected by a threading.Lock.
        """
        if self._shutdown:
            return False
        with self._health_lock:
            return bool(self._cached_health_ok)

# Global WebSocket pool manager instance
_websocket_pool_manager: WebSocketPoolManager | None = None
# P0-04 fix: Module-level lock is safe on Python 3.10+ (no longer bound to loop)
_global_lock: asyncio.Lock = asyncio.Lock()

async def get_websocket_pool_manager() -> WebSocketPoolManager:
    """Get the global WebSocket pool manager instance."""
    global _websocket_pool_manager

    if _websocket_pool_manager is None:
        async with _global_lock:
            if _websocket_pool_manager is None:
                _websocket_pool_manager = WebSocketPoolManager()
                await _websocket_pool_manager.initialize()

    return _websocket_pool_manager

async def shutdown_websocket_pool_manager() -> None:
    """Shutdown the global WebSocket pool manager."""
    global _websocket_pool_manager

    if _websocket_pool_manager:
        await _websocket_pool_manager.shutdown()
        _websocket_pool_manager = None
