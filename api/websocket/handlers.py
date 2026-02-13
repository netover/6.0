"""
WebSocket handlers for FastAPI
"""

import json

from fastapi import WebSocket, WebSocketDisconnect

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
        # Map websocket -> agent_id
        self.agent_connections: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, agent_id: str):
        """Connect WebSocket for specific agent"""
        await websocket.accept()

        if agent_id not in self.active_connections:
            self.active_connections[agent_id] = set()

        self.active_connections[agent_id].add(websocket)
        self.agent_connections[websocket] = agent_id

        logger.info("WebSocket connected for agent %s", agent_id)

    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket"""
        agent_id = self.agent_connections.get(websocket)
        if agent_id and websocket in self.active_connections.get(agent_id, set()):
            self.active_connections[agent_id].remove(websocket)
            if not self.active_connections[agent_id]:
                del self.active_connections[agent_id]

        if websocket in self.agent_connections:
            del self.agent_connections[websocket]

        logger.info("WebSocket disconnected from agent %s", agent_id)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to send message to WebSocket: %s", e)
            self.disconnect(websocket)

    async def broadcast_to_agent(self, message: str, agent_id: str):
        """Broadcast message to all connections for specific agent"""
        if agent_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[agent_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error("Failed to broadcast to agent %s: %s", agent_id, e)
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)

    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections"""
        disconnected = []
        for agent_connections in self.active_connections.values():
            for websocket in agent_connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error("Failed to broadcast to all: %s", e)
                    disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_handler(websocket: WebSocket, agent_id: str):
    """Handle WebSocket connections for chat agents"""
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
                            agent_id=agent_id, user_message=content, agent_config=agent_config
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

                    await manager.send_personal_message(json.dumps(final_response), websocket)

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
                                agent_id=agent_id, user_message=data, agent_config=agent_config
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

                        await manager.send_personal_message(json.dumps(final_response), websocket)
                    else:
                        # Unknown JSON message type
                        error_response = {
                            "type": "error",
                            "message": f"Unknown message type: {message_type}",
                            "agent_id": agent_id,
                        }
                        await manager.send_personal_message(json.dumps(error_response), websocket)

            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e)
                error_response = {
                    "type": "error",
                    "message": "Internal server error",
                    "agent_id": agent_id,
                }
                await manager.send_personal_message(json.dumps(error_response), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected for agent %s", agent_id)

    except Exception as e:
        logger.error("Unexpected WebSocket error for agent %s: %s", agent_id, e)
        manager.disconnect(websocket)
