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
import contextlib
import inspect
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Protocol

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from resync.core.exception_guard import maybe_reraise_programming_error
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
from resync.core.security import SafeAgentID, sanitize_input, validate_input
from resync.core.task_tracker import create_tracked_task
from resync.core.types.app_state import enterprise_state_from_app
from resync.core.trace_utils import hash_user_id

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- APIRouter Initialization ---
chat_router = APIRouter()

# Optional: track background tasks for observability (non-blocking)
_bg_tasks: set[asyncio.Task[Any]] = set()

# Bound concurrent writes per conversation to preserve ordering across sockets
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_SESSION_LOCKS_MAXSIZE = int(os.getenv("WS_SESSION_LOCKS_MAXSIZE", "5000"))

# Bounded in-process audit scheduling to avoid unbounded fire-and-forget growth
_AUDITOR_QUEUE_MAXSIZE = 256
_AUDITOR_WORKERS = 2
_auditor_queue: asyncio.Queue[None] | None = None
_auditor_workers: set[asyncio.Task[Any]] = set()
_auditor_init_lock = asyncio.Lock()


INFRA_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    ConnectionError,
)

INFRA_ERRORS_NON_TIMEOUT = tuple(exc for exc in INFRA_ERRORS if exc is not TimeoutError)


class SupportsAgentMeta(Protocol):
    """Minimal contract used by this module for agent-like objects."""

    name: str | None
    description: str | None
    # Some agents expose 'llm_model', others 'model'
    llm_model: Any | None  # type: ignore[assignment]
    model: Any | None  # type: ignore[assignment]


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _session_id_for_websocket(websocket: WebSocket) -> str:
    """Return stable server-side session id for this websocket connection."""
    existing = getattr(websocket.state, "session_id", None)
    if existing:
        return str(existing)

    user_id = getattr(websocket.state, "user_id", None)
    if user_id:
        return f"ws:{hash_user_id(str(user_id))}:{id(websocket)}"
    return f"ws:{id(websocket)}"


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
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
        logger.debug("Failed to send error message: %s", exc)
    except (TypeError, KeyError, AttributeError, IndexError):
        raise


async def run_auditor_safely() -> None:
    """
    Executes the IA auditor in a safe context, catching and logging any exceptions
    to prevent the background task from dying silently.
    """
    try:
        await analyze_and_flag_memories()
    except TimeoutError:
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
    except INFRA_ERRORS_NON_TIMEOUT as _e:  # pylint
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical(
            "IA Auditor background task failed with an unhandled exception.",
            exc_info=True,
        )


async def _auditor_worker() -> None:
    """Consume queued audit triggers with bounded worker concurrency.

    Important: ``asyncio.Queue.task_done()`` must only be called for items that
    were successfully retrieved via ``get()``. If the task is cancelled while
    blocked on ``get()``, calling ``task_done()`` would corrupt the queue's
    unfinished-task counter and can deadlock ``queue.join()``.
    """
    global _auditor_queue
    if _auditor_queue is None:
        return

    while True:
        got_item: bool = False
        try:
            await _auditor_queue.get()
            got_item = True
            await run_auditor_safely()
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError):
            logger.error("IA Auditor worker failed while processing queue item", exc_info=True)
        finally:
            if got_item:
                _auditor_queue.task_done()


async def _ensure_auditor_workers() -> None:
    """Initialize bounded auditor worker pool lazily and safely."""
    global _auditor_queue
    async with _auditor_init_lock:
        if _auditor_queue is None:
            _auditor_queue = asyncio.Queue(maxsize=_AUDITOR_QUEUE_MAXSIZE)
        if _auditor_workers:
            return
        for idx in range(_AUDITOR_WORKERS):
            task = create_tracked_task(_auditor_worker(), name=f"ia_auditor_worker_{idx}")
            _auditor_workers.add(task)
            task.add_done_callback(_auditor_workers.discard)


async def _schedule_auditor_run() -> None:
    """Queue an auditor execution request without unbounded task creation."""
    await _ensure_auditor_workers()
    if _auditor_queue is None:
        return
    try:
        _auditor_queue.put_nowait(None)
    except asyncio.QueueFull:
        logger.warning("IA Auditor queue is full; dropping trigger to preserve service stability")


async def _handle_agent_interaction(
    websocket: WebSocket,
    agent_id: SafeAgentID,
    data: str,
) -> None:
    """
    P0-CRITICAL FIX: Use UnifiedAgent.chat() instead of HybridRouter.
    
    This ensures WebSocket uses the same LLM tool calling logic as REST API.
    HybridRouter is now deprecated in favor of direct LLM tool calling.
    """
    from resync.core.context_store import ContextStore
    from resync.core.agent_manager import get_unified_agent

    sanitized = sanitize_input(data)

    # Get enterprise state
    st = enterprise_state_from_app(websocket.app)
    kg: ContextStore = st.knowledge_graph

    # Get or create server-side session_id
    session_id = _session_id_for_websocket(websocket)
    agent_id_str = str(agent_id)

    # Send user's message back to UI for display
    await websocket.send_json(
        {
            "type": "message",
            "sender": "user",
            "message": sanitized,
            "agent_id": agent_id_str,
            "session_id": session_id,
            "is_final": True,
            "timestamp": _now_iso(),
            "metadata": {},
        }
    )

    try:
        # Serialize session to preserve conversation ordering
        async with _session_locks[session_id]:
            # Persist user turn (best-effort)
            try:
                await kg.add_conversation(
                    session_id=session_id,
                    role="user",
                    content=sanitized,
                    metadata={"agent_id": agent_id_str},
                )
            except INFRA_ERRORS as e:
                _exc_type, _exc, _tb = sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)
                logger.warning("Failed to persist user message: %s", e)

            # P0-FIX: Use UnifiedAgent instead of HybridRouter
            start_time = time.time()
            
            unified_agent = get_unified_agent()
            response = await unified_agent.chat(
                message=sanitized,
                agent_id=agent_id_str,
                include_history=True,
                conversation_id=session_id,  # Use session_id for history isolation
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Send final response with simplified metadata (no routing_mode)
            response_payload = {
                "type": "message",
                "sender": "agent",
                "message": response,
                "agent_id": agent_id_str,
                "session_id": session_id,
                "is_final": True,
                "timestamp": _now_iso(),
                "correlation_id": None,  # Add trace ID if available
                "metadata": {
                    "processing_time_ms": processing_time_ms,
                    "handler": "UnifiedAgent",
                    "tools_used": [],  # TODO: Extract from LLM tool_calls
                },
            }
            logger.info("Sending response to WebSocket: %s", response_payload)
            
            try:
                await websocket.send_json(response_payload)
                logger.info("Response sent successfully to WebSocket")
            except Exception as e:
                logger.error("Failed to send response to WebSocket: %s", e, exc_info=True)
                await send_error_message(
                    websocket,
                    "Erro ao enviar resposta.",
                    agent_id_str,
                    session_id,
                )

            logger.info(
                "Agent '%s' response via UnifiedAgent, processing_time=%dms",
                agent_id,
                processing_time_ms,
            )

            # Persist assistant turn (best-effort)
            try:
                await kg.add_conversation(
                    session_id=session_id,
                    role="assistant",
                    content=response,
                    metadata={
                        "agent_id": agent_id_str,
                        "processing_time_ms": processing_time_ms,
                        "handler": "UnifiedAgent",
                    },
                )
            except INFRA_ERRORS as e:
                _exc_type, _exc, _tb = sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)
                logger.warning("Failed to persist assistant message: %s", e)

            # Schedule auditor
            logger.info("Queueing IA Auditor execution trigger.")
            await _schedule_auditor_run()

    except INFRA_ERRORS as e:
        _exc_type, _exc, _tb = sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)
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
    session_id = _session_id_for_websocket(websocket)

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



async def _ws_allow_message_checked(websocket: WebSocket) -> tuple[bool, int]:
    """Best-effort per-message WebSocket rate limiting.

    Returns:
        (allowed, retry_after_seconds)

    Behavior:
      - Fails **open** on infrastructure/network errors (to preserve availability).
      - Re-raises programming errors (TypeError/AttributeError/etc.) so they are
        not silently swallowed and do not accidentally bypass rate limiting.
    """
    from resync.core.security.rate_limiter_v2 import ws_allow_message

    client_ip: str = getattr(getattr(websocket, "client", None), "host", None) or "unknown"
    try:
        return await ws_allow_message(client_ip)
    except asyncio.CancelledError:
        raise
    except (OSError, ConnectionError, TimeoutError) as rate_err:
        logger.warning("ws_rate_limiter_infra_error", error=str(rate_err)[:200])
        return True, 0


async def _message_processing_loop(
    websocket: WebSocket,
    agent: SupportsAgentMeta | Any,
    agent_id: SafeAgentID,
    session_id: str,
) -> None:
    """Main loop for receiving and processing messages from the client.

    Resilience/DoS hardening:
    - Enforces inactivity timeout using settings.ws_connection_timeout.
    - Enforces maximum connection duration using settings.ws_max_connection_duration.
    - Avoids logging full user payloads (PII/secrets) by truncating to 200 chars.
    """
    from resync.settings import get_settings

    settings = get_settings()
    inactivity_timeout = float(settings.ws_connection_timeout)
    max_duration = float(settings.ws_max_connection_duration)
    started_at = time.monotonic()

    while True:
        # Enforce max connection duration (defense-in-depth against leaked connections)
        if (time.monotonic() - started_at) > max_duration:
            await send_error_message(
                websocket,
                "Sessão expirada por tempo máximo de conexão.",
                str(agent_id),
                session_id,
            )
            raise WebSocketDisconnect(code=1001, reason="Max connection duration exceeded")

        try:
            raw_data = await asyncio.wait_for(websocket.receive_text(), timeout=inactivity_timeout)
        except TimeoutError:
            # Inactivity timeout: close to free server resources
            await send_error_message(
                websocket,
                "Conexão encerrada por inatividade.",
                str(agent_id),
                session_id,
            )
            raise WebSocketDisconnect(code=1001, reason="Inactivity timeout") from None

        logger.info("Received message for agent '%s': %s...", agent_id, raw_data[:200])

        # Per-message rate limiting (defense-in-depth; protects LLM cost)
        allowed, retry_after = await _ws_allow_message_checked(websocket)
        if not allowed:
            await send_error_message(
                websocket,
                f"Rate limit excedido. Tente novamente em ~{retry_after}s.",
                str(agent_id),
                session_id,
            )
            continue

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
    # Verify token before accepting connection - FAIL CLOSED
    # Prefer Authorization header (avoids leaking tokens via query strings).
    # Backward-compat: still accept `?token=` for older clients.
    if token is None:
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip() or None
        
        if not token:
            token = websocket.cookies.get("access_token")
    try:
        if not token:
            await websocket.close(code=1008, reason="Authentication required")
            return

        try:
            from resync.api.core.security import verify_token_async

            payload = await verify_token_async(token)
            is_valid = bool(payload)
            if is_valid:
                subject = payload.get("sub") if isinstance(payload, dict) else getattr(payload, "sub", None)
                if subject:
                    websocket.state.user_id = str(subject)
        except INFRA_ERRORS:
            from resync.api.auth.service import get_auth_service

            auth_service = get_auth_service()
            verified_payload = await asyncio.to_thread(auth_service.verify_token, token)
            is_valid = bool(verified_payload)
            if is_valid:
                subject = verified_payload.get("sub") if isinstance(verified_payload, dict) else getattr(verified_payload, "sub", None)
                if subject:
                    websocket.state.user_id = str(subject)

        if not is_valid:
            await websocket.close(code=1008, reason="Authentication required")
            return
    except INFRA_ERRORS:
        logger.warning(
            "WebSocket auth check failed - rejecting connection (auth service unavailable)"
        )
        await websocket.close(code=1008, reason="Authentication service unavailable")
        return

    await websocket.accept()

    agent_id_str: str = str(agent_id)
    websocket.state.session_id = _session_id_for_websocket(websocket)
    # Compute once to avoid race conditions / wrong cleanup after disconnects
    session_id_for_cleanup: str = _session_id_for_websocket(websocket)


    try:
        agent, session_id = await _setup_websocket_session(websocket, agent_id)
        session_id_for_cleanup = session_id
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
        await send_error_message(
            websocket,
            "Ocorreu um erro com o agente. Tente novamente.",
            agent_id_str,
            session_id_for_cleanup,
        )
    except INFRA_ERRORS_NON_TIMEOUT as _e:  # pylint: disable=broad-exception-caught
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical("Unhandled exception in WebSocket for agent '%s'", agent_id, exc_info=True)
        await send_error_message(
            websocket,
            "Ocorreu um erro inesperado no servidor.",
            agent_id_str,
            session_id_for_cleanup,
        )
    finally:
        # Best-effort cleanup: prevent unbounded growth if clients set random session_ids.
        try:
            _session_locks.pop(session_id_for_cleanup, None)
            # Opportunistic pruning if under attack
            if len(_session_locks) > _SESSION_LOCKS_MAXSIZE:
                # Drop arbitrary oldest-ish keys (dict order in Py3.7+ preserves insertion).
                for k in list(_session_locks.keys())[: max(1, len(_session_locks) - _SESSION_LOCKS_MAXSIZE)]:
                    _session_locks.pop(k, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("ws_session_lock_cleanup_failed", exc_info=True)

        # Guarantee WebSocket is closed, even after unexpected errors
        with contextlib.suppress(
            RuntimeError,
            WebSocketDisconnect,
            ConnectionError,
            OSError,
        ):
            await websocket.close()


async def _validate_input(
    raw_data: str, agent_id: SafeAgentID, websocket: WebSocket
) -> dict[str, bool]:
    """Validate input data for size and potential injection attempts."""
    # Input validation and size check
    if len(raw_data) > 10000:  # Limit message size to 10KB
        agent_id_str = str(agent_id)
        session_id = _session_id_for_websocket(websocket)
        await send_error_message(
            websocket,
            "Mensagem muito longa. Máximo de 10.000 caracteres permitido.",
            agent_id_str,
            session_id,
        )
        return {"is_valid": False}

    result = validate_input(raw_data, max_length=10000)
    if not result.is_valid:
        agent_id_str = str(agent_id)
        session_id = _session_id_for_websocket(websocket)
        await send_error_message(websocket, "Conteúdo inválido.", agent_id_str, session_id)
        return {"is_valid": False}

    return {"is_valid": True}
