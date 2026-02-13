import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from resync.core.websocket_pool_manager import get_websocket_pool_manager

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections for real-time updates.
    Enhanced with connection pooling and monitoring capabilities.
    """

    def __init__(self) -> None:
        """Initializes the ConnectionManager with connection pooling support."""
        self.active_connections: dict[str, WebSocket] = {}
        self._pool_manager = None
        logger.info("ConnectionManager initialized with pooling support.")

    async def _get_pool_manager(self):
        """Get or create the WebSocket pool manager."""
        if self._pool_manager is None:
            self._pool_manager = await get_websocket_pool_manager()
        return self._pool_manager

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accepts a new WebSocket connection and adds it to the active list.
        Integrates with the WebSocket pool manager for enhanced monitoring.
        """
        pool_manager = await self._get_pool_manager()
        await pool_manager.connect(websocket, client_id)

        # Maintain backward compatibility with existing dictionary
        self.active_connections[client_id] = websocket
        logger.info("New WebSocket connection accepted: %s", client_id)
        logger.info("Total active connections: %d", len(self.active_connections))

    async def disconnect(self, client_id: str) -> None:
        """
        Removes a WebSocket connection from the active list.
        Integrates with the WebSocket pool manager for cleanup.
        """
        if client_id in self.active_connections:
            # Remove from pool manager first
            if self._pool_manager:
                await self._pool_manager.disconnect(client_id)
            # Maintain backward compatibility
            del self.active_connections[client_id]
            logger.info("WebSocket connection closed: %s", client_id)
            logger.info("Total active connections: %d", len(self.active_connections))

    async def send_personal_message(self, message: str, client_id: str) -> None:
        """
        Sends a message to a specific client.
        Integrates with the WebSocket pool manager for enhanced delivery.
        """
        # Try pool manager first for enhanced features
        if self._pool_manager:
            success = await self._pool_manager.send_personal_message(message, client_id)
            if success:
                return

        # Fallback to direct WebSocket delivery for backward compatibility
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        """
        Sends a plain text message to all connected clients.
        Uses the WebSocket pool manager for enhanced broadcasting.
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
        if not self.active_connections:
            logger.info("Broadcast requested, but no active connections.")
            return

        logger.info("broadcasting_message", client_count=len(self.active_connections))
        # Create a list of tasks to send messages concurrently
        tasks = [connection.send_text(message) for connection in self.active_connections.values()]
        # In a high-load scenario, you might want to handle exceptions here
        # for failed sends, but for now, we keep it simple.
        for task in tasks:
            try:
                await task
            except (WebSocketDisconnect, ConnectionError) as e:
                # Client disconnected or connection lost during broadcast
                logger.warning("Connection issue during broadcast: %s", e)
            except RuntimeError as e:  # pragma: no cover
                # WebSocket in wrong state
                if "websocket state" in str(e).lower():
                    logger.warning("WebSocket in wrong state during broadcast: %s", e)
                else:
                    logger.warning("Runtime error during broadcast: %s", e)
            except Exception as _e:
                # Log error but don't stop broadcasting to other clients
                logger.error("Unexpected error during broadcast.", exc_info=True)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """
        Sends a JSON payload to all connected clients.
        Uses the WebSocket pool manager for enhanced broadcasting.
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
        if not self.active_connections:
            logger.info("JSON broadcast requested, but no active connections.")
            return

        logger.info("Broadcasting JSON data to %d clients.", len(self.active_connections))
        tasks = [connection.send_json(data) for connection in self.active_connections.values()]
        for task in tasks:
            try:
                await task
            except (WebSocketDisconnect, ConnectionError) as e:
                # Client disconnected or connection lost during broadcast
                logger.warning("Connection issue during JSON broadcast: %s", e)
            except RuntimeError as e:  # pragma: no cover
                # WebSocket in wrong state
                if "websocket state" in str(e).lower():
                    logger.warning("WebSocket in wrong state during JSON broadcast: %s", e)
                else:
                    logger.error("Runtime error during JSON broadcast: %s", e)
            except ValueError as e:
                # JSON serialization error
                logger.error("JSON serialization error during broadcast: %s", e, exc_info=True)
            except Exception as _e:
                logger.error("Unexpected error during JSON broadcast.", exc_info=True)

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
                "last_cleanup": (stats.last_cleanup.isoformat() if stats.last_cleanup else None),
            }
        # Fallback to basic stats if pool manager not available
        return {
            "active_connections": len(self.active_connections),
            "total_connections": len(self.active_connections),
        }
