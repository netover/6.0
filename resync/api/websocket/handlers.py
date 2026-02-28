# pylint
"""
WebSocket handlers for FastAPI
"""

import asyncio
import os
import time
import orjson

from fastapi import WebSocket, WebSocketDisconnect, status
from resync.core.security.rate_limiter_v2 import ws_allow_connect

from resync.core.context import set_trace_id, set_user_id
from resync.core.langfuse.trace_utils import hash_user_id, normalize_trace_id
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

try:
    from langfuse.decorators import langfuse_context

    LANGFUSE_AVAILABLE = True
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
    import sys as _sys
    from resync.core.exception_guard import maybe_reraise_programming_error
    _exc_type, _exc, _tb = _sys.exc_info()
    maybe_reraise_programming_error(_exc, _tb)

    LANGFUSE_AVAILABLE = False
    langfuse_context = None
    logger.warning("langfuse_ws_context_unavailable reason=%s", type(exc).__name__)

async def _verify_ws_auth(websocket: WebSocket, token: str | None = None) -> str | None:
    """Verify WebSocket authentication and return user_id.

    Args:
        websocket: The WebSocket connection
        token: Optional JWT token from query parameter

    Returns:
        User ID if authenticated, None otherwise
    """
    try:
        # Check for token in query parameter
        if token:
            from resync.api.auth.service import get_auth_service

            auth_service = get_auth_service()
            payload = auth_service.verify_token(token)
            if payload:
                return str(payload.sub)

        # Check for token in Authorization header
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            from resync.api.security import decode_token

            payload = decode_token(token)
            if payload and "sub" in payload:
                return str(payload["sub"])

        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.debug("WebSocket authentication failed: %s", type(e).__name__)
        return None

class AgentConnectionManager:
    """Manage WebSocket connections with thread-safe operations.

    Security fixes applied:
    - Eager lock initialization to prevent race conditions (P0)
    """

    def __init__(self):
        self._send_timeout_seconds = float(os.getenv("WS_SEND_TIMEOUT_SECONDS", "5"))
        self._broadcast_concurrency = int(os.getenv("WS_BROADCAST_CONCURRENCY", "100"))
        self.active_connections: dict[str, set[WebSocket]] = {}
        # Map websocket -> agent_id
        self.agent_connections: dict[WebSocket, str] = {}
        # P0 fix: Initialize lock eagerly to prevent race condition
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, agent_id: str):
        """Connect WebSocket for specific agent"""
        # Rate limit connection attempts per client IP
        client_ip = getattr(getattr(websocket, "client", None), "host", None) or "unknown"
        allowed, retry_after = await ws_allow_connect(client_ip)
        if not allowed:
            # 1013: Try Again Later
            # Accept first for better ASGI/WebSocket compliance across servers.
            await websocket.accept()
            await websocket.close(code=1013, reason=f"Rate limited. Retry after ~{retry_after}s")
            return
        await websocket.accept()

        async with self._lock:
            if agent_id not in self.active_connections:
                self.active_connections[agent_id] = set()

            self.active_connections[agent_id].add(websocket)
            self.agent_connections[websocket] = agent_id

        logger.info("WebSocket connected for agent %s", agent_id)

    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket - async version (replaces sync wrapper).
        
        Args:
            websocket: The WebSocket connection to remove
        """
        async with self._lock:
            agent_id = self.agent_connections.get(websocket)
            if agent_id and websocket in self.active_connections.get(agent_id, set()):
                self.active_connections[agent_id].remove(websocket)
                if not self.active_connections[agent_id]:
                    del self.active_connections[agent_id]

            if websocket in self.agent_connections:
                del self.agent_connections[websocket]

        if agent_id:
            logger.info("WebSocket disconnected from agent %s", agent_id)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to send message to WebSocket: %s", e)
            await self.disconnect(websocket)

    async def _safe_send_text(self, websocket: WebSocket, message: str, *, agent_id: str | None = None) -> bool:
        """Send text with timeout and basic error handling.

        Returns True on success; False if the socket should be disconnected.
        """
        try:
            await asyncio.wait_for(websocket.send_text(message), timeout=self._send_timeout_seconds)
            return True
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timed out (timeout=%ss) for agent=%s", self._send_timeout_seconds, agent_id)
            return False
        except asyncio.CancelledError:
            raise
        except (WebSocketDisconnect, ConnectionError, OSError, RuntimeError) as exc:
            logger.warning("WebSocket send failed for agent=%s: %s", agent_id, exc)
            return False

    async def broadcast_to_agent(self, message: str, agent_id: str):
        """Broadcast message to all connections for a specific agent.

        Performance fix:
        - Concurrent sends with a semaphore (avoids head-of-line blocking).
        - Per-send timeout.
        """
        async with self._lock:
            websockets = list(self.active_connections.get(agent_id, set()))

        if not websockets:
            return

        sem = asyncio.Semaphore(self._broadcast_concurrency)

        async def _send(ws: WebSocket) -> tuple[WebSocket, bool]:
            async with sem:
                ok = await self._safe_send_text(ws, message, agent_id=agent_id)
                return ws, ok

        results = await asyncio.gather(*(_send(ws) for ws in websockets), return_exceptions=False)
        for ws, ok in results:
            if not ok:
                await self.disconnect(ws)

    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections.

        Performance fix:
        - Concurrent sends with a semaphore.
        - Per-send timeout.
        """
        async with self._lock:
            all_websockets: list[WebSocket] = []
            for agent_connections in self.active_connections.values():
                all_websockets.extend(agent_connections)

        if not all_websockets:
            return

        sem = asyncio.Semaphore(self._broadcast_concurrency)

        async def _send(ws: WebSocket) -> tuple[WebSocket, bool]:
            async with sem:
                ok = await self._safe_send_text(ws, message)
                return ws, ok

        results = await asyncio.gather(*(_send(ws) for ws in all_websockets), return_exceptions=False)
        for ws, ok in results:
            if not ok:
                await self.disconnect(ws)

# Global connection manager instance
manager = AgentConnectionManager()

# Backward-compatible alias (deprecated): prefer AgentConnectionManager.
ConnectionManager = AgentConnectionManager


def _build_agent_config(agent_id: str) -> dict[str, str]:
    """Build canonical agent configuration payload for LLM responses."""
    return {
        "name": f"Agente {agent_id}",
        "type": "general",
        "description": "Assistente de IA do sistema Resync TWS",
    }

async def _generate_llm_response(agent_id: str, content: str) -> str:
    """Helper to generate AI response with fallback handling."""
    try:
        from resync.services.llm_service import get_llm_service

        llm_service = await get_llm_service()
        agent_config = _build_agent_config(agent_id)

        return await llm_service.generate_agent_response(
            agent_id=agent_id,
            user_message=content,
            agent_config=agent_config,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("Error generating AI response: %s", e)
        # P1 fix: Do NOT reflect raw user content in the fallback response.
        # If rendered by a frontend without escaping, this enables XSS.
        return (
            "Olá! Recebi sua mensagem. "
            "O sistema Resync TWS está funcionando perfeitamente. Como posso ajudar?"
        )

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
    user_id = await _verify_ws_auth(websocket, token)
    if not user_id:
        logger.warning("WebSocket auth failed for agent %s", agent_id)
        # Starlette requires accept() before close() in newer versions.
        await websocket.accept()
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
        )
        return

    # Set trace context
    trace_id = normalize_trace_id(
        websocket.headers.get("x-correlation-id")
        or websocket.headers.get("x-request-id")
    )
    set_trace_id(trace_id)
    set_user_id(user_id)

    # Update Langfuse context with secure user ID
    hashed_user = hash_user_id(user_id)
    if LANGFUSE_AVAILABLE and langfuse_context is not None:
        langfuse_context.update_current_trace(user_id=hashed_user)

    await manager.connect(websocket, agent_id)

    # Maximum allowed payload per message (env: WS_MAX_MESSAGE_SIZE, default 256 KiB).
    # Prevents DoS via unbounded memory allocation from a single client.
    _WS_MAX_MESSAGE_BYTES: int = int(
        os.environ.get("WS_MAX_MESSAGE_SIZE", str(256 * 1024))
    )

    # Idle timeout to prevent abandoned connections from living forever (env: WS_IDLE_TIMEOUT_SECONDS).
    _WS_IDLE_TIMEOUT_SECONDS: float = float(os.environ.get("WS_IDLE_TIMEOUT_SECONDS", "300"))

    try:
        while True:
            # Receive message from client
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=_WS_IDLE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.info("ws_idle_timeout", agent_id=agent_id)
                await websocket.close(code=1000, reason="Idle timeout")
                return

            # Reject oversized messages (close code 1009 = "Message Too Big").
            if len(data.encode("utf-8")) > _WS_MAX_MESSAGE_BYTES:
                logger.warning(
                    "ws_message_too_large",
                    agent_id=agent_id,
                    size=len(data),
                    max_size=_WS_MAX_MESSAGE_BYTES,
                )
                await websocket.close(code=1009, reason="Message Too Big")
                return

            # P1 fix: truncate logged data to prevent log injection & flooding
            logger.info(
                "Received message from agent %s: %s",
                agent_id,
                data[:200] if len(data) > 200 else data,
            )

            try:
                # Try to parse as JSON first
                try:
                    message_data = orjson.loads(data)
                    message_type = message_data.get("type", "message")
                    is_json = True
                except orjson.JSONDecodeError:
                    # If not JSON, treat as plain text message
                    message_data = {"content": data, "type": "message"}
                    message_type = "message"
                    is_json = False

                if message_type == "chat_message" or not is_json:
                    # Process message with AI agent
                    content = message_data.get("content", data if not is_json else "")

                    # Send initial streaming response
                    response = {
                        "type": "stream",
                        # Do NOT echo raw user content — reflected XSS risk if
                        # the frontend renders this without HTML escaping.
                        "message": "Processando sua mensagem...",
                        "agent_id": agent_id,
                        "is_final": False,
                    }
                    await manager.send_personal_message(orjson.dumps(response).decode("utf-8"), websocket)

                    # Generate AI response
                    ai_response = await _generate_llm_response(agent_id, content)

                    # Send final response
                    final_response = {
                        "type": "message",
                        "message": ai_response,
                        "agent_id": agent_id,
                        "is_final": True,
                    }
                    await manager.send_personal_message(
                        orjson.dumps(final_response).decode("utf-8"), websocket
                    )

                elif message_type == "heartbeat":
                    # P1 fix: Respond to heartbeat with pong instead of error
                    pong_response = {
                        "type": "heartbeat_ack",
                        "agent_id": agent_id,
                        "timestamp": time.time(),
                    }
                    await manager.send_personal_message(
                        orjson.dumps(pong_response).decode("utf-8"), websocket
                    )

                else:
                    # Unknown JSON message type
                    error_response = {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "agent_id": agent_id,
                    }
                    await manager.send_personal_message(
                        orjson.dumps(error_response).decode("utf-8"), websocket
                    )

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Error processing WebSocket message: %s", e)
                error_response = {
                    "type": "error",
                    "message": "Internal server error",
                    "agent_id": agent_id,
                }
                await manager.send_personal_message(
                    orjson.dumps(error_response).decode("utf-8"), websocket
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for agent %s", agent_id)
    except asyncio.CancelledError:
        # P0-06 fix: Must re-raise CancelledError for proper shutdown
        logger.info("WebSocket cancelled for agent %s", agent_id)
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("Unexpected WebSocket error for agent %s: %s", agent_id, e)
    finally:
        # P1 fix: Guaranteed cleanup on all exit paths
        await manager.disconnect(websocket)
