import asyncio
from resync.core.task_tracker import create_tracked_task
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from resync.core.metrics import runtime_metrics

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
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._initialized = False
        self._shutdown = False

    async def initialize(self) -> None:
        """Initialize the WebSocket pool manager."""
        if self._initialized or self._shutdown:
            return

        self._cleanup_task = await create_tracked_task(self._cleanup_loop(), name="cleanup_loop")

        async with self._lock:
            if self._initialized:
                return
            self._initialized = True

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
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in WebSocket cleanup loop: %s", e)
    async def _cleanup_connections(self) -> None:
        """Clean up stale and unhealthy WebSocket connections."""
        current_time = datetime.now(timezone.utc)
        connections_to_remove = []

        # Collect connections to remove under lock
        async with self._lock:
            for client_id, conn_info in self.connections.items():
                # Check for stale connections (no activity for timeout period)
                time_since_activity = (current_time - conn_info.last_activity).total_seconds()
                if time_since_activity > _get_settings().WS_CONNECTION_TIMEOUT:
                    connections_to_remove.append(client_id)
                    logger.warning("Removing stale WebSocket connection: %s", client_id)

                # Check for unhealthy connections
                elif not conn_info.is_healthy:
                    connections_to_remove.append(client_id)
                    logger.warning("Removing unhealthy WebSocket connection: %s", client_id)

                # Additional check: enforce max connection duration to prevent long-lived connections
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
                                f"Removing long-lived WebSocket connection: {client_id} (duration: {connection_duration:.1f}s, max: {max_duration}s)"
                            )
                    except Exception as e:
                        logger.error("Error checking connection duration for %s: %s", client_id, e)
                        connections_to_remove.append(client_id)

            # Update stats under lock
            self.stats.cleanup_cycles += 1
            self.stats.last_cleanup = current_time

        # Remove connections OUTSIDE the lock to avoid deadlock
        for client_id in connections_to_remove:
            try:
                await self._remove_connection_safe(client_id)
            except Exception as e:
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
            except Exception as e:
                logger.error("Error closing WebSocket connection %s: %s", client_id, e)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept a new WebSocket connection and add it to the pool.

        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for the client
        """
        if self._shutdown:
            raise RuntimeError("WebSocket pool manager is shutdown")

        # Check pool size limit
        if len(self.connections) >= _get_settings().WS_POOL_MAX_SIZE:
            logger.warning("WebSocket pool at capacity. Rejecting connection for %s", client_id)
            await websocket.close(code=1013, reason="Server at capacity")
            self.stats.connection_errors += 1
            return

        await websocket.accept()

        current_time = datetime.now(timezone.utc)
        conn_info = WebSocketConnectionInfo(
            client_id=client_id,
            websocket=websocket,
            connected_at=current_time,
            last_activity=current_time,
        )

        async with self._lock:
            self.connections[client_id] = conn_info
            self.stats.total_connections += 1
            self.stats.active_connections = len(self.connections)
            self.stats.healthy_connections += 1

            if self.stats.active_connections > self.stats.peak_connections:
                self.stats.peak_connections = self.stats.active_connections

        logger.info("WebSocket connection accepted: %s", client_id)
        logger.info("Total active WebSocket connections: %s", self.stats.active_connections)

        # Record connection metrics
        runtime_metrics.record_counter(
            "websocket_pool.connections_accepted", 1, {"client_id": client_id}
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
        """
        Safe method to remove a connection - does NOT acquire lock.
        Used internally to avoid deadlocks when called from methods that already hold the lock.
        """
        # Get connection info under lock, then remove and close outside
        async with self._lock:
            if client_id not in self.connections:
                return
            conn_info = self.connections.pop(client_id)
            is_healthy = conn_info.is_healthy

        # Close WebSocket OUTSIDE the lock (async I/O)
        try:
            if conn_info.websocket.client_state != WebSocketState.DISCONNECTED:
                await conn_info.websocket.close()
        except Exception as e:
            logger.error("Error closing WebSocket for %s: %s", client_id, e)

        # Update statistics under lock
        async with self._lock:
            if is_healthy:
                self.stats.healthy_connections = max(0, self.stats.healthy_connections - 1)
            else:
                self.stats.unhealthy_connections = max(0, self.stats.unhealthy_connections - 1)
            self.stats.active_connections = len(self.connections)

        logger.info("WebSocket connection removed: %s", client_id)
        logger.info("Total active WebSocket connections: %s", self.stats.active_connections)

        # Record disconnection metrics
        runtime_metrics.record_counter(
            "websocket_pool.connections_closed", 1
        )
        runtime_metrics.record_gauge(
            "websocket_pool.active_connections", self.stats.active_connections
        )

    async def _remove_connection(self, client_id: str) -> None:
        """Internal method to remove a connection - acquires lock."""
        async with self._lock:
            if client_id not in self.connections:
                return
            conn_info = self.connections.pop(client_id)
            is_healthy = conn_info.is_healthy

        # Close WebSocket OUTSIDE the lock
        try:
            if conn_info.websocket.client_state != WebSocketState.DISCONNECTED:
                await conn_info.websocket.close()
        except Exception as e:
            logger.error("Error closing WebSocket for %s: %s", client_id, e)

        # Update statistics
        async with self._lock:
            if is_healthy:
                self.stats.healthy_connections = max(0, self.stats.healthy_connections - 1)
            else:
                self.stats.unhealthy_connections = max(0, self.stats.unhealthy_connections - 1)
            self.stats.active_connections = len(self.connections)

        logger.info("WebSocket connection removed: %s", client_id)
        logger.info("Total active WebSocket connections: %s", self.stats.active_connections)

        # Record disconnection metrics
        runtime_metrics.record_counter(
            "websocket_pool.connections_closed", 1
        )
        runtime_metrics.record_gauge(
            "websocket_pool.active_connections", self.stats.active_connections
        )

    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """
        Send a message to a specific client.

        Args:
            message: The message to send
            client_id: The target client ID

        Returns:
            True if message was sent successfully, False otherwise
        """
        conn_info = self.connections.get(client_id)
        if not conn_info:
            logger.warning("Client %s not found for personal message", client_id)
            return False

        try:
            await conn_info.websocket.send_text(message)
            conn_info.update_activity()
            conn_info.message_count += 1
            conn_info.bytes_sent += len(message.encode("utf-8"))

            # Update statistics
            self.stats.total_messages_sent += 1
            self.stats.total_bytes_sent += len(message.encode("utf-8"))

            # Record message metrics
            runtime_metrics.record_counter(
                "websocket_pool.messages_sent", 1, {"client_id": client_id}
            )
            runtime_metrics.record_counter(
                "websocket_pool.bytes_sent",
                len(message.encode("utf-8")),
                {"client_id": client_id},
            )

            return True

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning("Connection issue sending message to %s: %s", client_id, e)
            conn_info.mark_error()
            await self._remove_connection(client_id)
            return False
        except Exception as e:
            logger.error("Unexpected error sending message to %s: %s", client_id, e)
            conn_info.mark_error()
            return False

    async def broadcast(self, message: str) -> int:
        """
        Send a message to all connected clients.

        Args:
            message: The message to broadcast

        Returns:
            Number of clients that received the message
        """
        if not self.connections:
            logger.info("Broadcast requested, but no active WebSocket connections")
            return 0

        logger.info("Broadcasting message to %s WebSocket clients", len(self.connections))

        # Create tasks to send messages concurrently
        tasks = []
        client_ids = list(self.connections.keys())

        for client_id in client_ids:
            task = await create_tracked_task(self._send_message_with_error_handling(client_id, message), name="send_message_with_error_handling")
            tasks.append((client_id, task))

        # Wait for all tasks to complete
        successful_sends = 0
        for client_id, task in tasks:
            try:
                success = await task
                if success:
                    successful_sends += 1
            except Exception as e:
                logger.error("Error in broadcast task for %s: %s", client_id, e)

        logger.info("Message successfully broadcast to %s clients", successful_sends)
        return successful_sends

    async def _send_message_with_error_handling(self, client_id: str, message: str) -> bool:
        """Send message with proper error handling and connection cleanup."""
        conn_info = self.connections.get(client_id)
        if not conn_info:
            return False

        try:
            await conn_info.websocket.send_text(message)
            conn_info.update_activity()
            conn_info.message_count += 1
            conn_info.bytes_sent += len(message.encode("utf-8"))
            return True

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning("Connection issue during broadcast to %s: %s", client_id, e)
            conn_info.mark_error()
            # Remove connection after disconnection
            await self._remove_connection(client_id)
            return False
        except RuntimeError as e:
            if "websocket state" in str(e).lower():
                logger.warning("WebSocket in wrong state during broadcast to %s: %s", client_id, e)
                conn_info.mark_error()
                return False
            logger.error("Runtime error during broadcast to %s: %s", client_id, e)
            return False
        except Exception as e:
            logger.error("Unexpected error during broadcast to %s: %s", client_id, e)
            return False

    async def broadcast_json(self, data: dict[str, Any]) -> int:
        """
        Send JSON data to all connected clients.

        Args:
            data: The JSON data to broadcast

        Returns:
            Number of clients that received the data
        """
        if not self.connections:
            logger.info("JSON broadcast requested, but no active WebSocket connections")
            return 0

        logger.info("Broadcasting JSON data to %s WebSocket clients", len(self.connections))

        # Create tasks to send JSON data concurrently
        tasks = []
        client_ids = list(self.connections.keys())

        for client_id in client_ids:
            task = await create_tracked_task(self._send_json_with_error_handling(client_id, data), name="send_json_with_error_handling")
            tasks.append((client_id, task))

        # Wait for all tasks to complete
        successful_sends = 0
        for client_id, task in tasks:
            try:
                success = await task
                if success:
                    successful_sends += 1
            except Exception as e:
                logger.error("Error in JSON broadcast task for %s: %s", client_id, e)

        logger.info("JSON data successfully broadcast to %s clients", successful_sends)
        return successful_sends

    async def _send_json_with_error_handling(self, client_id: str, data: dict[str, Any]) -> bool:
        """Send JSON data with proper error handling and connection cleanup."""
        conn_info = self.connections.get(client_id)
        if not conn_info:
            return False

        try:
            await conn_info.websocket.send_json(data)
            conn_info.update_activity()
            conn_info.message_count += 1

            # Estimate bytes sent (rough approximation)
            import json

            json_str = json.dumps(data)
            conn_info.bytes_sent += len(json_str.encode("utf-8"))
            return True

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning("Connection issue during JSON broadcast to %s: %s", client_id, e)
            conn_info.mark_error()
            # Remove connection after disconnection
            await self._remove_connection(client_id)
            return False
        except ValueError as e:
            logger.error("JSON serialization error during broadcast to %s: %s", client_id, e)
            return False
        except RuntimeError as e:
            if "websocket state" in str(e).lower():
                logger.warning(
                    f"WebSocket in wrong state during JSON broadcast to {client_id}: {e}"
                )
                conn_info.mark_error()
                return False
            logger.error("Runtime error during JSON broadcast to %s: %s", client_id, e)
            return False
        except Exception as e:
            logger.error("Unexpected error during JSON broadcast to %s: %s", client_id, e)
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

    def health_check(self) -> bool:
        """Perform health check on the WebSocket pool."""
        if self._shutdown:
            return False

        try:
            # Check if we have too many unhealthy connections
            unhealthy_ratio = self.stats.unhealthy_connections / max(
                1, self.stats.active_connections
            )
            if unhealthy_ratio > 0.5:  # More than 50% unhealthy connections
                logger.warning(
                    f"WebSocket pool unhealthy: {unhealthy_ratio:.1%} connections are unhealthy"
                )
                return False

            # Check for connection leaks (connections that should be closed but aren't)
            stale_connections = 0
            current_time = datetime.now(timezone.utc)

            for _client_id, conn_info in self.connections.items():
                time_since_activity = (current_time - conn_info.last_activity).total_seconds()
                if time_since_activity > _get_settings().WS_CONNECTION_TIMEOUT * 2:
                    stale_connections += 1

            if stale_connections > 0:
                logger.warning("WebSocket pool has %s stale connections", stale_connections)

            return stale_connections < self.stats.active_connections * 0.1  # Less than 10% stale

        except Exception as e:
            logger.error("WebSocket pool health check failed: %s", e)
            return False


# Global WebSocket pool manager instance
_websocket_pool_manager: WebSocketPoolManager | None = None


async def get_websocket_pool_manager() -> WebSocketPoolManager:
    """Get the global WebSocket pool manager instance."""
    global _websocket_pool_manager

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
