import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from resync.core.exceptions import (
    NotFoundError,
)
from resync.core.fastapi_di import get_agent_manager
from resync.core.security import SafeAgentID

agents_router = APIRouter()

logger = logging.getLogger(__name__)

@agents_router.get("/all")
async def list_all_agents(
    request: Request,
    agent_manager: Any = Depends(get_agent_manager),
) -> list[dict[str, Any]]:
    """
    Lists the configuration of all available agents.

    Raises:
        ServiceUnavailableError: If there's an infrastructure error.
    """
    logger.info("list_all_agents endpoint called")
    agents = await agent_manager.get_all_agents()
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "goal": agent.goal,
            "model": agent.model_name,
            "tools": agent.tools,
        }
        for agent in agents
    ]

@agents_router.get("/{agent_id}")
async def get_agent_details(
    agent_id: SafeAgentID,
    request: Request,
    agent_manager: Any = Depends(get_agent_manager),
):
    """
    Retrieves the detailed configuration of a specific agent by its ID.

    Raises:
        NotFoundError: If no agent with the specified ID is found.
        ServiceUnavailableError: If there's an infrastructure error.
    """
    logger.info("get_agent_details endpoint called with agent_id: %s", agent_id)

    # FIX: Don't mask database/infrastructure errors as NotFoundError
    # Let the global exception handler deal with proper error codes
    agent_config = await agent_manager.get_agent_config(agent_id)

    if agent_config is None:
        raise NotFoundError(f"Agent with ID '{agent_id}' not found.")

    return {
        "id": agent_config.id,
        "name": agent_config.name,
        "role": agent_config.role,
        "goal": agent_config.goal,
        "backstory": agent_config.backstory,
        "tools": agent_config.tools,
        "model": agent_config.model_name,
        "memory": agent_config.memory,
    }
