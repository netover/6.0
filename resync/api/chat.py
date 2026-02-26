"""Chat and agent interaction API endpoints.

This module provides WebSocket endpoints for real-time chat interactions
with AI agents, supporting both streaming and non-streaming responses.
It handles agent management, conversation context, and error handling
for chat-based interactions.

v6.0 REFACTORING:
- Now uses HybridRouter as single source of truth for routing
- Uses enterprise_state_from_app to access dependencies
- Removes double "is_final" response issue
- Proper ContextStore integration for conversation persistence
- Normalized payload per WebSocketMessage validation model
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import weakref
from datetime import datetime, timezone
from typing import Any, Protocol

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from resync.core.exceptions import (
    AgentExecutionError,
    AuditError,
    DatabaseError,
    KnowledgeGraphError,
    LLMError,
    ToolExecutionError,
)
from resync.core.ia_auditor import analyze_and_flag_memories
from resync.core.interfaces import IAgentManager
from resync.core.security import SafeAgentID, sanitize_input
from resync.core.task_tracker import create_tracked_task
from resync.core.types.app_state import enterprise_state_from_app

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- APIRouter Initialization ---
chat_router = APIRouter()

# Optional: track background tasks for observability (non-blocking)
_bg_tasks: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()

class SupportsAgentMeta(Protocol):
    """Minimal contract used by this module for agent-like objects."""

    name: str | None
    description: str | None
    # Some agents expose 'llm_model', others 'model'
    llm_model: Any | None  # type: ignore[assignment]
    model: Any | None  # type: ignore[assignment]

def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

async def send_error_message(
    websocket: WebSocket, message: str, agent_id: str, session_id: str
) -> None:
    """
    Helper function to send error messages to the client.
    Handles exceptions if the WebSocket connection is already closed.
    """
    try:
        await websocket.send_json(
            {
                "type": "error",
                "sender": "system",
                "message": message,
                "agent_id": agent_id,
                "session_id": session_id,
                "is_final": True,
                "timestamp": _now_iso(),
                "metadata": {},
            }
        )
    except WebSocketDisconnect:
        logger.debug("Failed to send error message, WebSocket disconnected.")
    except RuntimeError as exc:
        # This typically happens when the WebSocket is already closed
        logger.debug("Failed to send error message, WebSocket runtime error: %s", exc)
    except ConnectionError as exc:
        logger.debug("Failed to send error message, connection error: %s", exc)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as _e:  # pylint
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        # Last resort to prevent the application from crashing if sending fails.
        logger.warning(
            "Failed to send error message due to an unexpected error.", exc_info=True
        )

async def run_auditor_safely() -> None:
    """
    Executes the IA auditor in a safe context, catching and logging any exceptions
    to prevent the background task from dying silently.
    """
    try:
        await analyze_and_flag_memories()
    except asyncio.TimeoutError:
        logger.error("IA Auditor timed out during execution.", exc_info=True)
    except KnowledgeGraphError:
        logger.error("IA Auditor encountered a knowledge graph error.", exc_info=True)
    except DatabaseError:
        logger.error("IA Auditor encountered a database error.", exc_info=True)
    except AuditError:
        logger.error("IA Auditor encountered an audit-specific error.", exc_info=True)
    except asyncio.CancelledError:  # pylint
        # Propagate task cancellation correctly
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as _e:  # pylint
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical(
            "IA Auditor background task failed with an unhandled exception.",
            exc_info=True,
        )

async def _handle_agent_interaction(
    websocket: WebSocket,
    agent_id: SafeAgentID,
    data: str,
) -> None:
    """
    Handles the core logic of agent interaction using HybridRouter.

    v6.0 REFACTORING:
    - Uses HybridRouter as single source of truth
    - Gets dependencies from enterprise_state (not Depends injection)
    - Sends only ONE final response (no double is_final)
    - Proper ContextStore integration with correct signature
    - Normalized payload per WebSocketMessage validation model
    """
    from resync.core.context_store import ContextStore

    sanitized = sanitize_input(data)

    # Get enterprise state (includes hybrid_router and knowledge_graph)
    st = enterprise_state_from_app(websocket.app)
    router = st.hybrid_router
    kg: ContextStore = st.knowledge_graph  # Now properly typed as ContextStore

    # Get or create session_id from query params
    session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
    agent_id_str = str(agent_id)

    # Send user's message back to UI for display
    # Per WebSocketMessage model:
    # type, sender, message, agent_id, session_id, timestamp, metadata
    await websocket.send_json(
        {
            "type": "message",
            "sender": "user",
            "message": sanitized,
            "agent_id": agent_id_str,
            "session_id": session_id,
            "is_final": False,
            "timestamp": _now_iso(),
            "metadata": {},
        }
    )

    try:
        # Persist user turn (best-effort, don't block on failure)
        try:
            await kg.add_conversation(
                session_id=session_id,
                role="user",
                content=sanitized,
                metadata={"agent_id": agent_id_str},
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Failed to persist user message: %s", e)

        # Route via HybridRouter (single source of truth)
        start_time = time.time()

        result = await router.route(
            sanitized,
            context={"agent_id": agent_id_str, "session_id": session_id},
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Send final response ONCE with metadata in correct location
        # Per WebSocketMessage: routing info goes in metadata, not at top level
        await websocket.send_json(
            {
                "type": "message",
                "sender": "agent",
                "message": result.response,
                "agent_id": agent_id_str,
                "session_id": session_id,
                "is_final": True,
                "timestamp": _now_iso(),
                "correlation_id": result.trace_id,
                "metadata": {
                    # Routing info
                    "routing_mode": result.routing_mode.value,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "handler": result.handler,
                    "tools_used": result.tools_used,
                    "entities": result.entities,
                    "processing_time_ms": processing_time_ms,
                    "requires_approval": result.requires_approval,
                    "approval_id": result.approval_id,
                    # Backward compatibility aliases
                    "query_type": result.intent,
                },
            }
        )

        logger.info(
            "Agent '%s' response: mode=%s, intent=%s, tools=%s",
            agent_id,
            result.routing_mode.value,
            result.intent,
            result.tools_used,
        )

        # Persist assistant turn (best-effort, don't block on failure)
        try:
            await kg.add_conversation(
                session_id=session_id,
                role="assistant",
                content=result.response,
                metadata={
                    "agent_id": agent_id_str,
                    "routing_mode": result.routing_mode.value,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "tools_used": result.tools_used,
                    "entities": result.entities,
                    "processing_time_ms": processing_time_ms,
                },
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Failed to persist assistant message: %s", e)

        # Schedule the IA Auditor to run in the background
        logger.info("Scheduling IA Auditor to run in the background.")
        task = create_tracked_task(
            run_auditor_safely(), name="run_auditor_safely"
        )
        _bg_tasks.add(task)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("Error in agent interaction: %s", e, exc_info=True)
        await send_error_message(
            websocket,
            "Erro ao processar sua mensagem. Tente novamente.",
            agent_id_str,
            session_id,
        )

async def _setup_websocket_session(
    websocket: WebSocket, agent_id: SafeAgentID
) -> tuple[SupportsAgentMeta | Any, str]:
    """Handles WebSocket connection setup and agent retrieval."""
    logger.info("WebSocket connection established for agent %s", agent_id)

    # WebSocket endpoints cannot use FastAPI Depends(get_agent_manager) which
    # requires a Request object.  Access the singleton directly from app state.
    st = enterprise_state_from_app(websocket.app)
    agent_manager: IAgentManager = st.agent_manager

    # Get agent from the manager (supports sync or async implementations).
    # This is the merged version that handles both sync and async agent managers.
    maybe_agent = agent_manager.get_agent(agent_id)
    agent = await maybe_agent if inspect.isawaitable(maybe_agent) else maybe_agent

    agent_id_str = str(agent_id)
    session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"

    if not agent:
        logger.warning("Agent '%s' not found.", agent_id)
        await send_error_message(
            websocket, f"Agente '{agent_id}' não encontrado.", agent_id_str, session_id
        )
        raise WebSocketDisconnect(code=1008, reason=f"Agent '{agent_id}' not found")

    # Welcome message - per WebSocketMessage model with type="system"
    welcome_data = {
        "type": "system",
        "sender": "system",
        "message": (
            f"Conectado ao agente: {getattr(agent, 'name', 'Unknown Agent')}. "
            "Digite sua mensagem..."
        ),
        "agent_id": agent_id_str,
        "session_id": session_id,
        "is_final": False,
        "timestamp": _now_iso(),
        "metadata": {},
    }
    await websocket.send_json(welcome_data)
    logger.info(
        "Agent '%s' ready for WebSocket communication",
        getattr(agent, "name", "Unknown Agent"),
    )
    return agent, session_id

async def _message_processing_loop(
    websocket: WebSocket,
    agent: SupportsAgentMeta | Any,
    agent_id: SafeAgentID,
    session_id: str,
) -> None:
    """Main loop for receiving and processing messages from the client."""
    while True:
        raw_data = await websocket.receive_text()
        logger.info("Received message for agent '%s': %s...", agent_id, raw_data[:200])

        validation = await _validate_input(raw_data, agent_id, websocket)
        if not validation["is_valid"]:
            continue

        await _handle_agent_interaction(websocket, agent_id, raw_data)

@chat_router.websocket("/ws/{agent_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: SafeAgentID,
    token: str | None = None,
) -> None:
    """
    Main WebSocket endpoint for real-time chat with an agent.

    v6.0 REFACTORING:
    - Removed knowledge_graph dependency injection (now gets from enterprise_state)
    - Uses HybridRouter as single source of truth
    - Normalized payload per WebSocketMessage validation model

    Authentication via query parameter: ws://host/ws/{agent_id}?token=JWT_TOKEN
    """
    # Accept first to follow report guidance (some ASGI servers also allow close-before-accept to deny with 403).
    await websocket.accept()

    # Verify token after accepting connection - FAIL CLOSED
    try:
        from resync.api.auth.service import get_auth_service

        auth_service = get_auth_service()
        if not token or not auth_service.verify_token(token):
            await websocket.close(code=1008, reason="Authentication required")
            return
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        # P0 fix: FAIL CLOSED — deny connection when auth service is unavailable.
        # Previous code fell through and allowed unauthenticated connections.
        logger.warning(
            "WebSocket auth check failed - rejecting connection "
            "(auth service unavailable)"
        )
        await websocket.close(code=1008, reason="Authentication service unavailable")
        return

    try:
        agent, session_id = await _setup_websocket_session(websocket, agent_id)
        # Note: No knowledge_graph passed - it's now obtained from enterprise_state
        await _message_processing_loop(websocket, agent, agent_id, session_id)
    except WebSocketDisconnect:
        code = getattr(websocket.state, "code", "unknown")
        reason = getattr(websocket.state, "reason", "unknown")
        logger.info(
            "Client disconnected from agent '%s'. Reason: %s (Code: %s)",
            agent_id,
            reason,
            code,
        )
    except (LLMError, ToolExecutionError, AgentExecutionError) as exc:
        logger.error(
            "Agent-related error in WebSocket for agent '%s': %s",
            agent_id,
            exc,
            exc_info=True,
        )
        agent_id_str = str(agent_id)
        session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
        await send_error_message(
            websocket,
            "Ocorreu um erro com o agente. Tente novamente.",
            agent_id_str,
            session_id,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as _e:  # pylint: disable=broad-exception-caught
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical(
            "Unhandled exception in WebSocket for agent '%s'", agent_id, exc_info=True
        )
        agent_id_str = str(agent_id)
        session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
        await send_error_message(
            websocket,
            "Ocorreu um erro inesperado no servidor.",
            agent_id_str,
            session_id,
        )

async def _validate_input(
    raw_data: str, agent_id: SafeAgentID, websocket: WebSocket
) -> dict[str, bool]:
    """Validate input data for size and potential injection attempts."""
    # Input validation and size check
    if len(raw_data) > 10000:  # Limit message size to 10KB
        agent_id_str = str(agent_id)
        session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
        await send_error_message(
            websocket,
            "Mensagem muito longa. Máximo de 10.000 caracteres permitido.",
            agent_id_str,
            session_id,
        )
        return {"is_valid": False}

    # Additional validation: check for potential injection attempts
    if "<script>" in raw_data or "javascript:" in raw_data.lower():
        logger.warning(
            "Potential injection attempt detected from agent '%s': %s...",
            agent_id,
            raw_data[:100],
        )
        agent_id_str = str(agent_id)
        session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
        await send_error_message(
            websocket, "Conteúdo não permitido detectado.", agent_id_str, session_id
        )
        return {"is_valid": False}

    return {"is_valid": True}