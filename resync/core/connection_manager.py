import asyncio
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from resync.core.websocket_pool_manager import get_websocket_pool_manager

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Constants ---
LOCK_TIMEOUT_SECONDS = 5.0


class ConnectionManager:
    """
    Manages active WebSocket connections for real-time updates.
    Enhanced with connection pooling and monitoring capabilities.
    Thread-safe implementation with asyncio.Lock for concurrent access.
    """

    def __init__(self) -> None:
        """Initializes the ConnectionManager with connection pooling support."""
        self.active_connections: dict[str, WebSocket] = {}
        self._pool_manager = None
        self._pool_manager_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        logger.info("ConnectionManager initialized with pooling support.")

    async def _get_pool_manager(self):
        """Get or create the WebSocket pool manager (thread-safe)."""
        if self._pool_manager is None:
            async with self._pool_manager_lock:
                if self._pool_manager is None:
                    self._pool_manager = await get_websocket_pool_manager()
        return self._pool_manager

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accepts a new WebSocket connection and adds it to the active list.
        Integrates with the WebSocket pool manager for enhanced monitoring.
        Thread-safe: uses lock to protect active_connections modifications.
        """
        pool_manager = await self._get_pool_manager()
        await pool_manager.connect(websocket, client_id)

        # Maintain backward compatibility with existing dictionary
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info("New WebSocket connection accepted: %s", client_id)
        logger.info("Total active connections: %d", len(self.active_connections))

    async def disconnect(self, client_id: str) -> None:
        """
        Removes a WebSocket connection from the active list.
        Integrates with the WebSocket pool manager for cleanup.
        Thread-safe: uses lock to protect active_connections modifications.
        """
        async with self._lock:
            if client_id not in self.active_connections:
                return
            websocket = self.active_connections.pop(client_id)

        # Close websocket outside lock to avoid potential deadlocks
        try:
            await websocket.close()
        except Exception as e:
            logger.warning("Error closing websocket for client %s: %s", client_id, e)

        if self._pool_manager:
            await self._pool_manager.disconnect(client_id)

        logger.info("WebSocket connection closed: %s", client_id)
        logger.info("Total active connections: %d", len(self.active_connections))

    async def send_personal_message(self, message: str, client_id: str) -> None:
        """
        Sends a message to a specific client.
        Integrates with the WebSocket pool manager for enhanced delivery.
        """
        if self._pool_manager:
            success = await self._pool_manager.send_personal_message(message, client_id)
            if success:
                return

        async with self._lock:
            websocket = self.active_connections.get(client_id)
            if not websocket:
                return

            try:
                await websocket.send_text(message)
            except (WebSocketDisconnect, ConnectionError) as e:
                logger.warning(
                    "Connection error sending to client %s: %s", client_id, e
                )
                self.active_connections.pop(client_id, None)
            except RuntimeError as e:
                if "websocket state" in str(e).lower():
                    logger.warning(
                        "WebSocket in wrong state for client %s: %s", client_id, e
                    )
                else:
                    logger.error("Runtime error sending to client %s: %s", client_id, e)
            except Exception as e:
                logger.error("Unexpected error sending to client %s: %s", client_id, e)

    async def broadcast(self, message: str) -> None:
        """
        Sends a plain text message to all connected clients.
        Uses the WebSocket pool manager for enhanced broadcasting.
        Thread-safe: iterates over a copy of connections to avoid modification during iteration.
        """
        # Use pool manager for enhanced broadcasting with monitoring
        if self._pool_manager and self._pool_manager.connections:
            successful_sends = await self._pool_manager.broadcast(message)
            logger.info(
                "broadcast_completed",
                successful_sends=successful_sends,
                message="clients received the message",
            )
            return

        # Fallback to legacy broadcasting for backward compatibility
        async with self._lock:
            if not self.active_connections:
                logger.info("Broadcast requested, but no active connections.")
                return
            # Create a copy of the connections to iterate safely
            connections = list(self.active_connections.values())

        logger.info("broadcasting_message", client_count=len(connections))
        # HARDENING [P0]: asyncio.TaskGroup previne Head-of-Line Blocking e garante concorrência estruturada.
        # Um cliente lento ou com erro de rede não atrasa o envio para os demais.
        tasks = []
        try:
            async with asyncio.TaskGroup() as tg:
                for connection in connections:
                    tasks.append(tg.create_task(connection.send_text(message)))
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.warning("broadcast_connection_error", error=str(exc))

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """
        Sends a JSON payload to all connected clients.
        Uses the WebSocket pool manager for enhanced broadcasting.
        Thread-safe: iterates over a copy of connections to avoid modification during iteration.
        """
        # Use pool manager for enhanced JSON broadcasting with monitoring
        if self._pool_manager and self._pool_manager.connections:
            successful_sends = await self._pool_manager.broadcast_json(data)
            logger.info(
                "json_broadcast_completed",
                successful_sends=successful_sends,
                message="clients received the data",
            )
            return

        # Fallback to legacy JSON broadcasting for backward compatibility
        async with self._lock:
            if not self.active_connections:
                logger.info("JSON broadcast requested, but no active connections.")
                return
            # Create a copy of the connections to iterate safely
            connections = list(self.active_connections.values())

        logger.info("Broadcasting JSON data to %d clients.", len(connections))
        # HARDENING [P0]: TaskGroup para concorrência real e isolamento de falhas estruturado
        tasks = []
        try:
            async with asyncio.TaskGroup() as tg:
                for connection in connections:
                    tasks.append(tg.create_task(connection.send_json(data)))
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.warning("json_broadcast_connection_error", error=str(exc))

    def get_connection_stats(self) -> dict[str, Any]:
        """
        Get WebSocket connection statistics from the pool manager.

        Returns:
            Dictionary containing connection statistics
        """
        if self._pool_manager:
            stats = self._pool_manager.get_stats()
            return {
                "total_connections": stats.total_connections,
                "active_connections": stats.active_connections,
                "healthy_connections": stats.healthy_connections,
                "unhealthy_connections": stats.unhealthy_connections,
                "peak_connections": stats.peak_connections,
                "total_messages_sent": stats.total_messages_sent,
                "total_messages_received": stats.total_messages_received,
                "connection_errors": stats.connection_errors,
                "cleanup_cycles": stats.cleanup_cycles,
                "last_cleanup": (
                    stats.last_cleanup.isoformat() if stats.last_cleanup else None
                ),
            }
        # Fallback to basic stats if pool manager not available
        return {
            "active_connections": len(self.active_connections),
            "total_connections": len(self.active_connections),
        }
