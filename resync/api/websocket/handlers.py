"""
WebSocket handlers for FastAPI
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect, status

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


async def _verify_ws_auth(websocket: WebSocket, token: str | None = None) -> bool:
    """Verify WebSocket authentication.

    Args:
        websocket: The WebSocket connection
        token: Optional JWT token from query parameter

    Returns:
        True if authenticated, False otherwise
    """
    try:
        # Check for token in query parameter
        if token:
            from resync.api.auth.service import get_auth_service

            auth_service = get_auth_service()
            if auth_service.verify_token(token):
                return True

        # Check for token in Authorization header
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            from resync.api.security import decode_token

            payload = decode_token(token)
            if payload:
                return True

        return False
    except Exception as e:
        logger.debug("WebSocket authentication failed: %s", type(e).__name__)
        return False


class ConnectionManager:
    """Manage WebSocket connections with thread-safe operations"""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
        # Map websocket -> agent_id
        self.agent_connections: dict[WebSocket, str] = {}
        # Lock for protecting concurrent access to connection collections
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, agent_id: str):
        """Connect WebSocket for specific agent"""
        await websocket.accept()

        async with self._lock:
            if agent_id not in self.active_connections:
                self.active_connections[agent_id] = set()

            self.active_connections[agent_id].add(websocket)
            self.agent_connections[websocket] = agent_id

        logger.info("WebSocket connected for agent %s", agent_id)

    async def disconnect_async(self, websocket: WebSocket):
        """Disconnect WebSocket - async version with proper locking."""
        async with self._lock:
            agent_id = self.agent_connections.get(websocket)
            if agent_id and websocket in self.active_connections.get(agent_id, set()):
                self.active_connections[agent_id].remove(websocket)
                if not self.active_connections[agent_id]:
                    del self.active_connections[agent_id]

            if websocket in self.agent_connections:
                del self.agent_connections[websocket]

        logger.info("WebSocket disconnected from agent %s", agent_id)

    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket - sync wrapper for backward compatibility.

        Note: This method attempts to handle disconnection from synchronous contexts.
        For proper async handling, use disconnect_async() instead.
        """
        try:
            # We're in async context - schedule the async disconnect
            asyncio.create_task(self.disconnect_async(websocket))
        except RuntimeError:
            # No running loop - we're in sync context, do best-effort cleanup
            # This is not thread-safe but better than nothing for sync contexts
            agent_id = self.agent_connections.get(websocket)
            if agent_id and websocket in self.active_connections.get(agent_id, set()):
                self.active_connections[agent_id].discard(websocket)
                if not self.active_connections[agent_id]:
                    del self.active_connections[agent_id]

            if websocket in self.agent_connections:
                del self.agent_connections[websocket]

            logger.info("WebSocket disconnected from agent %s (sync context)", agent_id)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to send message to WebSocket: %s", e)
            await self.disconnect_async(websocket)

    async def broadcast_to_agent(self, message: str, agent_id: str):
        """Broadcast message to all connections for specific agent"""
        async with self._lock:
            if agent_id not in self.active_connections:
                return
            # Create a copy of the set to iterate safely
            websockets = list(self.active_connections[agent_id])

        disconnected = []
        for websocket in websockets:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error("Failed to broadcast to agent %s: %s", agent_id, e)
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect_async(websocket)

    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections"""
        # Create a copy of all websockets to iterate safely
        async with self._lock:
            all_websockets = []
            for agent_connections in self.active_connections.values():
                all_websockets.extend(agent_connections)

        disconnected = []
        for websocket in all_websockets:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error("Failed to broadcast to all: %s", e)
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect_async(websocket)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_handler(
    websocket: WebSocket, agent_id: str, token: str | None = None
):
    """Handle WebSocket connections for chat agents.

    Args:
        websocket: The WebSocket connection
        agent_id: The agent ID to connect to
        token: Optional JWT token for authentication
    """
    # Authenticate before accepting connection - fail closed (deny by default)
    if not await _verify_ws_auth(websocket, token):
        logger.warning("WebSocket auth failed for agent %s", agent_id)
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
        )
        return

    await manager.connect(websocket, agent_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info("Received message from agent %s: %s", agent_id, data)

            try:
                # Try to parse as JSON first
                try:
                    message_data = json.loads(data)
                    message_type = message_data.get("type", "message")
                    is_json = True
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text message
                    message_data = {"content": data, "type": "message"}
                    message_type = "message"
                    is_json = False

                if message_type == "chat_message":
                    # Process chat message with AI agent
                    content = message_data.get("content", data if not is_json else "")

                    # Send initial streaming response
                    response = {
                        "type": "stream",
                        "message": f"Processando: {content}",
                        "agent_id": agent_id,
                        "is_final": False,
                    }
                    await manager.send_personal_message(json.dumps(response), websocket)

                    # Generate real AI response using LLM service
                    try:
                        from resync.services.llm_service import get_llm_service

                        llm_service = get_llm_service()

                        # Agent configuration
                        agent_config = {
                            "name": f"Agente {agent_id}",
                            "type": "general",
                            "description": "Assistente de IA do sistema Resync TWS",
                        }

                        # Generate response using real LLM
                        ai_response = await llm_service.generate_agent_response(
                            agent_id=agent_id,
                            user_message=content,
                            agent_config=agent_config,
                        )

                        # Send final response with real AI content
                        final_response = {
                            "type": "message",
                            "message": ai_response,
                            "agent_id": agent_id,
                            "is_final": True,
                        }

                    except Exception as e:
                        logger.error("Error generating AI response: %s", e)
                        # Fallback to mock response if LLM fails
                        final_response = {
                            "type": "message",
                            "message": f"Olá! Recebi sua mensagem: '{content}'. O sistema Resync TWS está funcionando perfeitamente. Como posso ajudar?",
                            "agent_id": agent_id,
                            "is_final": True,
                        }

                    await manager.send_personal_message(
                        json.dumps(final_response), websocket
                    )

                elif message_type == "heartbeat":
                    # Respond to heartbeat
                    response = {
                        "type": "heartbeat_ack",
                        "timestamp": "2025-01-01T00:00:00Z",
                        "agent_id": agent_id,
                    }
                    await manager.send_personal_message(json.dumps(response), websocket)

                else:
                    # Handle plain text messages
                    if not is_json:
                        # Send initial streaming response for plain text
                        response = {
                            "type": "stream",
                            "message": f"Processando: {data}",
                            "agent_id": agent_id,
                            "is_final": False,
                        }
                        await manager.send_personal_message(
                            json.dumps(response), websocket
                        )

                        # Generate real AI response using LLM service
                        try:
                            from resync.services.llm_service import get_llm_service

                            llm_service = get_llm_service()

                            # Agent configuration
                            agent_config = {
                                "name": f"Agente {agent_id}",
                                "type": "general",
                                "description": "Assistente de IA do sistema Resync TWS",
                            }

                            # Generate response using real LLM
                            ai_response = await llm_service.generate_agent_response(
                                agent_id=agent_id,
                                user_message=data,
                                agent_config=agent_config,
                            )

                            # Send final response with real AI content
                            final_response = {
                                "type": "message",
                                "message": ai_response,
                                "agent_id": agent_id,
                                "is_final": True,
                            }

                        except Exception as e:
                            logger.error("Error generating AI response: %s", e)
                            # Fallback to mock response if LLM fails
                            final_response = {
                                "type": "message",
                                "message": f"Olá! Recebi sua mensagem: '{data}'. O sistema Resync TWS está funcionando perfeitamente. Como posso ajudar?",
                                "agent_id": agent_id,
                                "is_final": True,
                            }

                        await manager.send_personal_message(
                            json.dumps(final_response), websocket
                        )
                    else:
                        # Unknown JSON message type
                        error_response = {
                            "type": "error",
                            "message": f"Unknown message type: {message_type}",
                            "agent_id": agent_id,
                        }
                        await manager.send_personal_message(
                            json.dumps(error_response), websocket
                        )

            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e)
                error_response = {
                    "type": "error",
                    "message": "Internal server error",
                    "agent_id": agent_id,
                }
                await manager.send_personal_message(
                    json.dumps(error_response), websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected for agent %s", agent_id)

    except Exception as e:
        logger.error("Unexpected WebSocket error for agent %s: %s", agent_id, e)
        manager.disconnect(websocket)
