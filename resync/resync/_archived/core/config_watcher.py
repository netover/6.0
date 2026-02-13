import json
import logging
from typing import cast

logger = logging.getLogger(__name__)


# Lazy imports to avoid circular dependencies
def _get_container():
    """Lazy import of container."""
    from resync.core.di_container import container

    return container


def _get_interfaces():
    """Lazy import of interfaces."""
    from resync.core.interfaces import IAgentManager, IConnectionManager

    return IAgentManager, IConnectionManager


async def handle_config_change() -> None:
    """
    Handles the reloading of agent configurations and notifies clients.
    """
    try:
        # Resolve dependencies from the DI container (lazy imports)
        from resync.core.agent_manager import AgentManager
        from resync.core.connection_manager import ConnectionManager

        container = _get_container()
        i_agent_manager, i_connection_manager = _get_interfaces()

        agent_manager = cast(AgentManager, container.get(i_agent_manager))
        connection_manager = cast(ConnectionManager, container.get(i_connection_manager))
    except Exception as e:
        logger.error(
            "Failed to resolve dependencies from DI container", error=str(e), exc_info=True
        )
        return

    logger.info("Configuration change detected. Reloading agents...")
    try:
        # Trigger the agent manager to reload its configuration
        await agent_manager.load_agents_from_config()
        logger.info("Agent configurations reloaded successfully.")

        # Get the updated list of agents
        agents = await agent_manager.get_all_agents()
        agent_list = [{"id": agent.id, "name": agent.name} for agent in agents]

        # Notify all connected WebSocket clients about the change
        await connection_manager.broadcast(
            json.dumps(
                {
                    "type": "config_update",
                    "message": "A configuração do agente foi atualizada. A lista de agentes foi recarregada.",
                    "agents": agent_list,
                }
            )
        )
        logger.info("Broadcasted config update to all clients.")

    except Exception as e:
        logger.error("error_handling_config_change", error=str(e), exc_info=True)
