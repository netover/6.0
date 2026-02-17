import logging
from typing import Any

from fastapi import APIRouter, Depends, Request

from resync.core.exceptions import NotFoundError
from resync.core.fastapi_di import get_agent_manager
from resync.core.security import SafeAgentID

# Module-level dependency for agent manager to avoid B008 error
agent_manager_dependency = Depends(get_agent_manager)

agents_router = APIRouter()

logger = logging.getLogger(__name__)


@agents_router.get("/all")
async def list_all_agents(
    request: Request, agent_manager=agent_manager_dependency
) -> list[dict[str, Any]]:
    """
    Lists the configuration of all available agents.
    """
    logger.info("list_all_agents endpoint called")
    try:
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
    except Exception as e:
        logger.error("Error listing agents: %s", e)
        return []


@agents_router.get("/{agent_id}")
async def get_agent_details(
    agent_id: SafeAgentID, request: Request, agent_manager=agent_manager_dependency
):
    """
    Retrieves the detailed configuration of a specific agent by its ID.

    Raises:
        NotFoundError: If no agent with the specified ID is found.
    """
    logger.info("get_agent_details endpoint called with agent_id: %s", agent_id)
    try:
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
    except NotFoundError:
        raise
    except Exception as e:
        logger.error("Error getting agent details for %s: %s", agent_id, e)
        raise NotFoundError(f"Agent with ID '{agent_id}' not found.") from None
