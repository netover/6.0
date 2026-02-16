"""Chat and agent interaction API endpoints.

This module provides WebSocket endpoints for real-time chat interactions
with AI agents, supporting both streaming and non-streaming responses.
It handles agent management, conversation context, and error handling
for chat-based interactions.
"""

from __future__ import annotations

import asyncio
from resync.core.task_tracker import create_tracked_task
import logging
import weakref
from typing import Any, Protocol

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from resync.core.exceptions import (
    AgentExecutionError,
    AuditError,
    DatabaseError,
    KnowledgeGraphError,
    LLMError,
    ToolExecutionError,
)
from resync.core.fastapi_di import get_agent_manager, get_knowledge_graph
from resync.core.ia_auditor import analyze_and_flag_memories
from resync.core.interfaces import IAgentManager, IKnowledgeGraph
from resync.core.security import SafeAgentID, sanitize_input

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# Module-level dependencies to avoid B008 errors
agent_manager_dependency = Depends(get_agent_manager)
knowledge_graph_dependency = Depends(get_knowledge_graph)

# --- APIRouter Initialization ---
chat_router = APIRouter()

# Optional: track background tasks for observability (non-blocking)
_bg_tasks: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()


class SupportsAgentMeta(Protocol):
    """Minimal contract used by this module for agent-like objects."""

    name: str | None = None
    description: str | None = None
    # Some agents expose 'llm_model', others 'model'
    llm_model: Any | None = None  # type: ignore[assignment]
    model: Any | None = None  # type: ignore[assignment]


async def send_error_message(websocket: WebSocket, message: str) -> None:
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
            }
        )
    except WebSocketDisconnect:
        logger.debug("Failed to send error message, WebSocket disconnected.")
    except RuntimeError as exc:
        # This typically happens when the WebSocket is already closed
        logger.debug("Failed to send error message, WebSocket runtime error: %s", exc)
    except ConnectionError as exc:
        logger.debug("Failed to send error message, connection error: %s", exc)
    except Exception as _e:  # pylint: disable=broad-exception-caught
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        # Last resort to prevent the application from crashing if sending fails.
        logger.warning("Failed to send error message due to an unexpected error.", exc_info=True)


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
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
        # Propagate task cancellation correctly
        raise
    except Exception as _e:  # pylint: disable=broad-exception-caught
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical(
            "IA Auditor background task failed with an unhandled exception.",
            exc_info=True,
        )


async def _finalize_and_store_interaction(
    websocket: WebSocket,
    knowledge_graph: IKnowledgeGraph,
    agent: SupportsAgentMeta | Any,
    agent_id: SafeAgentID,
    sanitized_query: str,
    full_response: str,
) -> None:
    """Sends the final message, stores the conversation, and schedules the auditor."""
    # Send a final message indicating the stream has ended
    await websocket.send_json(
        {
            "type": "message",
            "sender": "agent",
            "message": full_response,
            "is_final": True,
        }
    )
    logger.info("Agent '%s' full response: %s", agent_id, full_response)

    # Safe access to agent attributes - FIXED
    agent_name = getattr(agent, "name", "Unknown Agent")
    agent_description = getattr(agent, "description", "No description")
    agent_model = getattr(agent, "llm_model", getattr(agent, "model", "Unknown Model"))

    # Store the interaction in the Knowledge Graph
    await knowledge_graph.add_conversation(
        user_query=sanitized_query,
        agent_response=full_response,
        agent_id=agent_id,
        context={
            "agent_name": agent_name,
            "agent_description": agent_description,
            "model_used": str(agent_model),
        },
    )

    # Schedule the IA Auditor to run in the background
    logger.info("Scheduling IA Auditor to run in the background.")
    task = await create_tracked_task(run_auditor_safely(), name="run_auditor_safely")
    _bg_tasks.add(task)


async def _handle_agent_interaction(
    websocket: WebSocket,
    agent: SupportsAgentMeta | Any,
    agent_id: SafeAgentID,
    knowledge_graph: IKnowledgeGraph,
    data: str,
) -> None:
    """Handles the core logic of agent interaction with structured query processing."""
    from resync.core.graphrag_integration import get_graphrag_integration
    from resync.core.query_processor import QueryProcessor
    from resync.services.llm_service import get_llm_service

    sanitized_data = sanitize_input(data)
    # Send the user's message back to the UI for display
    await websocket.send_json({"type": "message", "sender": "user", "message": sanitized_data})

    try:
        # 1. Processar query de forma estruturada
        llm = get_llm_service()
        processor = QueryProcessor(llm, knowledge_graph)

        structured_query = await processor.process_query(sanitized_data)

        logger.info(
            f"Query structured: type={structured_query.query_type}, "
            f"entities={structured_query.entities}, "
            f"intent={structured_query.intent}"
        )

        # 2. Enriquecer contexto com GraphRAG (se disponível)
        graphrag = get_graphrag_integration()
        if graphrag and structured_query.entities.get("job_names"):
            # Use subgraph retrieval for job-related queries
            job_name = structured_query.entities["job_names"][0]
            try:
                subgraph_context = await graphrag.get_enriched_context(job_name)

                # Add subgraph context to messages
                if subgraph_context and subgraph_context.get("dependencies"):
                    context_text = graphrag.subgraph_retriever.format_for_llm(subgraph_context)
                    structured_query.context_requirements["subgraph"] = context_text

                    logger.info("GraphRAG context added for %s", job_name)
            except Exception as e:
                logger.warning("GraphRAG context failed, continuing: %s", e)

        # 3. Formatar para LLM
        messages = processor.format_for_llm(structured_query)

        # 4. Gerar resposta (com tools se disponíveis)
        try:
            response = await llm.generate_response_with_tools(
                messages=messages,
                user_role="operator",
                session_id=str(id(websocket)),  # Usar websocket id como session
                max_tool_iterations=3,  # Limitar para não travar
            )
        except Exception as e:
            logger.warning("Tool-based generation failed, falling back to simple: %s", e)
            # Fallback para geração simples sem tools
            response = await llm.generate_response(messages=messages, max_tokens=1000)

        # 5. Enviar resposta
        await websocket.send_json(
            {
                "type": "message",
                "sender": "agent",
                "message": response,
                "query_type": structured_query.query_type.value,
                "entities": structured_query.entities,
                "is_final": True,
            }
        )

        # 6. Armazenar interação
        await _finalize_and_store_interaction(
            websocket=websocket,
            knowledge_graph=knowledge_graph,
            agent=agent,
            agent_id=agent_id,
            sanitized_query=sanitized_data,
            full_response=response,
        )

    except Exception as e:
        logger.error("Error in agent interaction: %s", e, exc_info=True)
        await send_error_message(websocket, "Erro ao processar sua mensagem. Tente novamente.")


async def _setup_websocket_session(
    websocket: WebSocket, agent_id: SafeAgentID
) -> SupportsAgentMeta | Any:
    """Handles WebSocket connection setup and agent retrieval."""
    await websocket.accept()
    logger.info("WebSocket connection established for agent %s", agent_id)

    # WebSocket endpoints cannot use FastAPI Depends(get_agent_manager) which
    # requires a Request object.  Access the singleton directly from app state.
    from resync.core.types.app_state import enterprise_state_from_app

    st = enterprise_state_from_app(websocket.app)
    agent_manager: IAgentManager = st.agent_manager
    agent = await agent_manager.get_agent(agent_id)

    if not agent:
        logger.warning("Agent '%s' not found.", agent_id)
        await send_error_message(websocket, f"Agente '{agent_id}' não encontrado.")
        raise WebSocketDisconnect(code=1008, reason=f"Agent '{agent_id}' not found")

    welcome_data = {
        "type": "info",
        "sender": "system",
        "message": (
            f"Conectado ao agente: {getattr(agent, 'name', 'Unknown Agent')}. "
            "Digite sua mensagem..."
        ),
    }
    await websocket.send_json(welcome_data)
    logger.info(
        "Agent '%s' ready for WebSocket communication",
        getattr(agent, "name", "Unknown Agent"),
    )
    return agent


async def _message_processing_loop(
    websocket: WebSocket,
    agent: SupportsAgentMeta | Any,
    agent_id: SafeAgentID,
    knowledge_graph: IKnowledgeGraph,
) -> None:
    """Main loop for receiving and processing messages from the client."""
    while True:
        raw_data = await websocket.receive_text()
        logger.info("Received message for agent '%s': %s...", agent_id, raw_data[:200])

        validation = await _validate_input(raw_data, agent_id, websocket)
        if not validation["is_valid"]:
            continue

        await _handle_agent_interaction(websocket, agent, agent_id, knowledge_graph, raw_data)


@chat_router.websocket("/ws/{agent_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: SafeAgentID,
    token: str | None = None,
    knowledge_graph: IKnowledgeGraph = knowledge_graph_dependency,
) -> None:
    """Main WebSocket endpoint for real-time chat with an agent.

    Authentication via query parameter: ws://host/ws/{agent_id}?token=JWT_TOKEN
    """
    # Verify token before accepting connection
    try:
        from resync.api.auth.service import get_auth_service

        auth_service = get_auth_service()
        if not token or not auth_service.verify_token(token):
            await websocket.close(code=1008, reason="Authentication required")
            return
    except Exception:
        logger.warning("WebSocket auth check failed, allowing connection (auth service unavailable)")

    try:
        agent = await _setup_websocket_session(websocket, agent_id)
        await _message_processing_loop(websocket, agent, agent_id, knowledge_graph)
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
            "Agent-related error in WebSocket for agent '%s': %s", agent_id, exc, exc_info=True
        )
        await send_error_message(websocket, "Ocorreu um erro com o agente. Tente novamente.")
    except Exception as _e:  # pylint: disable=broad-exception-caught
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.critical("Unhandled exception in WebSocket for agent '%s'", agent_id, exc_info=True)
        await send_error_message(websocket, "Ocorreu um erro inesperado no servidor.")


async def _validate_input(
    raw_data: str, agent_id: SafeAgentID, websocket: WebSocket
) -> dict[str, bool]:
    """Validate input data for size and potential injection attempts."""
    # Input validation and size check
    if len(raw_data) > 10000:  # Limit message size to 10KB
        await send_error_message(
            websocket, "Mensagem muito longa. Máximo de 10.000 caracteres permitido."
        )
        return {"is_valid": False}

    # Additional validation: check for potential injection attempts
    if "<script>" in raw_data or "javascript:" in raw_data.lower():
        logger.warning(
            "Potential injection attempt detected from agent '%s': %s...",
            agent_id,
            raw_data[:100],
        )
        await send_error_message(websocket, "Conteúdo não permitido detectado.")
        return {"is_valid": False}

    return {"is_valid": True}
