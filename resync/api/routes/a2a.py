"""A2A (Agent-to-Agent) Protocol Routes.

Implements discovery and JSON-RPC communication for agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse
import asyncio
import json
import re

from resync.core.fastapi_di import get_agent_manager, get_a2a_handler
from resync.models.a2a import AgentCard, JsonRpcRequest, JsonRpcResponse
from resync.settings import settings

import structlog

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from resync.core.a2a_handler import A2AHandler
    from resync.core.agent_manager import AgentManager

router = APIRouter(tags=["A2A Protocol"])

def check_a2a_enabled():
    """Verify if the A2A protocol is enabled in settings."""
    if not settings.a2a_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A2A Protocol is disabled",
        )

AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

def validate_agent_id(agent_id: str) -> str:
    """Validate agent_id format to prevent injection attacks."""
    if not AGENT_ID_PATTERN.match(agent_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid agent_id format. Only alphanumeric, underscore, and hyphen allowed.",
        )
    return agent_id

@router.get("/.well-known/agent.json", response_model=AgentCard)
async def a2a_well_known_discovery(
    request: Request,
    agent_manager: Annotated[AgentManager, Depends(get_agent_manager)],
):
    """A2A standardized discovery for the primary orchestrator."""
    check_a2a_enabled()
    base_url = str(request.base_url).rstrip("/")
    card = await agent_manager.get_agent_card("tws-general", base_url=base_url)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Orchestrator card not found"
        )
    return card

@router.get("/api/v1/a2a/agents", response_model=list[AgentCard])
async def list_a2a_agents(
    request: Request,
    agent_manager: Annotated[AgentManager, Depends(get_agent_manager)],
):
    """List all agents available via A2A."""
    check_a2a_enabled()
    base_url = str(request.base_url).rstrip("/")
    return await agent_manager.export_a2a_cards(base_url=base_url)

@router.get("/api/v1/a2a/{agent_id}/card", response_model=AgentCard)
async def get_specific_agent_card(
    agent_id: str,
    request: Request,
    agent_manager: Annotated[AgentManager, Depends(get_agent_manager)],
):
    """Get A2A card for a specific specialist agent."""
    check_a2a_enabled()
    validate_agent_id(agent_id)
    base_url = str(request.base_url).rstrip("/")
    card = await agent_manager.get_agent_card(agent_id, base_url=base_url)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent card for '{agent_id}' not found",
        )
    return card

# --- JSON-RPC & SSE Implementation ---

@router.post("/api/v1/a2a/{agent_id}/jsonrpc", response_model=JsonRpcResponse)
async def agent_jsonrpc_endpoint(
    agent_id: str,
    request_data: JsonRpcRequest,
    a2a_handler: Annotated[A2AHandler, Depends(get_a2a_handler)],
):
    """Standard A2A JSON-RPC 2.0 communication endpoint."""
    check_a2a_enabled()
    validate_agent_id(agent_id)
    return await a2a_handler.handle_request(agent_id, request_data)

@router.get("/api/v1/a2a/events")
async def a2a_events_endpoint(
    a2a_handler: Annotated[A2AHandler, Depends(get_a2a_handler)],
):
    """Server-Sent Events (SSE) stream for A2A task updates."""
    check_a2a_enabled()

    async def event_generator():
        try:
            async for event in a2a_handler.get_event_stream():
                try:
                    yield f"data: {json.dumps(event)}\n\n"
                except (TypeError, ValueError) as e:
                    logger.warning("event_serialization_failed", error=str(e))
                    continue
        except asyncio.CancelledError:
            pass
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("event_stream_error", error=str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
