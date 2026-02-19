"""AI Agent management and orchestration.

Provides agent lifecycle management, configuration via YAML, execution
orchestration, and a unified chat interface.

v6.1.2 — Removed singleton anti-pattern, fixed DI, per-session history,
          tools filtered by config, structured logging throughout.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
import yaml
import aiofiles
from pydantic import BaseModel, Field

from resync.core.exceptions import AgentError
from resync.core.metrics import runtime_metrics
from resync.models.agents import AgentConfig, AgentType
from resync.models.a2a import AgentCard, AgentCapabilities, AgentContact
from resync.settings import settings
from resync.tools.definitions.tws import (
    tws_status_tool,
    tws_troubleshooting_tool,
)
from .global_utils import get_environment_tags, get_global_correlation_id

agent_logger = structlog.get_logger("resync.agent_manager")
logger = structlog.get_logger(__name__)


# =============================================================================
# NATIVE AGENT IMPLEMENTATION (v5.2.3.24 — replaces Agno)
# =============================================================================


class Agent:
    """Lightweight LLM agent backed by LiteLLM.

    Provides an ``arun`` / ``run`` interface that is compatible with the
    legacy Agno agent surface, so callers can swap in transparently.
    """

    def __init__(
        self,
        tools: Any = None,
        model: Any = None,
        instructions: Any = None,
        name: str = "TWS Agent",
        description: str = "TWS operations agent",
        **kwargs: Any,
    ) -> None:
        self.tools = tools or []
        self.model = model or "ollama/qwen2.5:3b"
        self.llm_model = self.model
        self.instructions = (
            instructions
            if isinstance(instructions, str)
            else "\n".join(instructions or [])
        )
        self.name = name
        self.description = description
        self.role = name
        self.goal = "Assist with TWS operations"
        self.backstory = description

    async def arun(self, message: str, skill_context: str = "") -> str:
        """Process *message* via LiteLLM (async)."""
        try:
            import litellm

            litellm.suppress_debug_info = True

            # Injeção dinâmica da Skill no prompt do sistema
            full_instructions = self.instructions
            if skill_context:
                full_instructions += (
                    f"\n\n--- CONHECIMENTO ESPECÍFICO APLICÁVEL ---\n{skill_context}"
                )

            system_prompt = (
                f"You are {self.name}.\n"
                f"{full_instructions}\n\n"  # Use full_instructions em vez de self.instructions
                f"Available tools: "
                f"{', '.join(str(t) for t in self.tools) if self.tools else 'None'}\n\n"
                "Respond in Portuguese (Brazilian) unless the user writes in English."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                timeout=30.0,
            )

            return response.choices[0].message.content

        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            agent_logger.error("agent_arun_error", error=str(exc), agent=self.name)
            return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        """Provide a keyword-based fallback when the LLM is unavailable."""
        try:
            msg = message.lower()

            if "job" in msg and ("abend" in msg or "erro" in msg):
                return (
                    "Jobs em estado ABEND encontrados. Recomendo investigar "
                    "o log do job e verificar dependências."
                )
            if "status" in msg or "workstation" in msg:
                return (
                    "Para verificar o status, use o comando 'conman' ou "
                    "consulte a interface web do TWS."
                )
            if "tws" in msg:
                return (
                    f"Como {self.name}, posso ajudar com questões "
                    "relacionadas ao TWS. O que você precisa?"
                )
            return (
                f"Entendi sua mensagem. Como {self.name}, estou aqui para "
                "ajudar com operações TWS. "
                "(Nota: LLM temporariamente indisponível)"
            )
        except Exception as e:
            agent_logger.error("fallback_response_error", error=str(e), agent=self.name)
            return (
                "Ocorreu um erro inesperado ao tentar gerar uma resposta alternativa."
            )

    def run(self, message: str) -> str:
        """Synchronous wrapper around :meth:`arun`."""
        import asyncio as _asyncio

        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            return _asyncio.run(self.arun(message))

        raise RuntimeError(
            "Agent.run() cannot be called from an async event loop; "
            "use `await agent.arun()` instead."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the agent for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "model": str(self.model) if self.model else None,
            "llm_model": str(self.llm_model) if self.llm_model else None,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": [str(t) for t in self.tools] if self.tools else [],
        }


# Backward compatibility aliases
MockAgent = Agent
AGNO_AVAILABLE = True


# --- Pydantic models --------------------------------------------------------
class AgentsConfig(BaseModel):
    """Container for multiple agent configurations."""

    agents: list[AgentConfig] = Field(default_factory=list)


# =============================================================================
# AGENT MANAGER
# =============================================================================


def _discover_tools() -> dict[str, Any]:
    """Return the map of tool-name -> callable for agent injection."""
    try:
        return {
            "get_tws_status": tws_status_tool.get_tws_status,
            "analyze_tws_failures": tws_troubleshooting_tool.analyze_failures,
        }
    except (ImportError, AttributeError) as exc:
        logger.warning("tool_discovery_failed", error=str(exc))
        return {}


class AgentManager:
    """Manages agent lifecycle: creation, caching, configuration, and tools.

    **Not a singleton.**  Create via :func:`initialize_agent_manager` (called
    by the app lifespan in ``wiring.py``) and retrieve with
    :func:`get_agent_manager`.

    Args:
        settings_module: Application settings (used for model defaults).
        tws_client_factory: Zero-arg callable that returns an ``ITWSClient``
            instance (sync).  Provided by ``wiring.py`` so the manager never
            needs to call ``get_service()`` directly.
    """

    def __init__(
        self,
        settings_module: Any = settings,
        tws_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "agent_manager",
                "operation": "init",
                "global_correlation": get_global_correlation_id(),
                "environment": get_environment_tags(),
            }
        )

        try:
            logger.info(
                "initializing_agent_manager",
                native_agent=True,
                has_factory=tws_client_factory is not None,
                correlation_id=correlation_id,
            )
            runtime_metrics.record_health_check("agent_manager", "initializing")

            self.settings = settings_module
            self.agents: dict[str, Agent] = {}

            # Default agent configurations (overridden by YAML if available)
            self.agent_configs: list[AgentConfig] = [
                AgentConfig(
                    id="tws-troubleshooting",
                    name="TWS Troubleshooting Agent",
                    agent_type=AgentType.TASK,
                    role="TWS Troubleshooting Specialist",
                    goal="Help users identify and resolve TWS system issues",
                    backstory=(
                        "I am an expert AI assistant specialized in "
                        "IBM Workload Automation (TWS) troubleshooting "
                        "and system monitoring."
                    ),
                    tools=["get_tws_status", "analyze_tws_failures"],
                    model_name="tongyi-deepresearch",
                ),
                AgentConfig(
                    id="tws-general",
                    name="TWS General Assistant",
                    agent_type=AgentType.CHAT,
                    role="TWS General Assistant",
                    goal="Provide general assistance for TWS operations",
                    backstory=(
                        "I am a helpful AI assistant for IBM Workload "
                        "Automation (TWS) operations, providing "
                        "information about system status and job execution."
                    ),
                    tools=["get_tws_status", "analyze_tws_failures"],
                    model_name="openrouter-fallback",
                ),
            ]

            self.tools: dict[str, Any] = _discover_tools()
            self._tws_client_factory = tws_client_factory
            self.tws_client: Any = None

            # Lazy-created async primitives (must not be created at import
            # time when there is no running event loop).
            self._tws_init_lock: asyncio.Lock | None = None
            self._agent_creation_lock: asyncio.Lock | None = None

            runtime_metrics.record_health_check("agent_manager", "healthy")
            logger.info(
                "agent_manager_initialized",
                agents_count=len(self.agent_configs),
                tools_count=len(self.tools),
                correlation_id=correlation_id,
            )

        except AgentError as exc:
            runtime_metrics.record_health_check(
                "agent_manager", "failed", {"error": str(exc)}
            )
            logger.critical(
                "agent_manager_init_failed",
                error=str(exc),
                correlation_id=correlation_id,
                exc_info=True,
            )
            raise
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    # -----------------------------------------------------------------
    # Lazy lock helpers (avoids creating asyncio primitives at import)
    # -----------------------------------------------------------------

    def _get_tws_lock(self) -> asyncio.Lock:
        if self._tws_init_lock is None:
            self._tws_init_lock = asyncio.Lock()
        return self._tws_init_lock

    def _get_agent_lock(self) -> asyncio.Lock:
        if self._agent_creation_lock is None:
            self._agent_creation_lock = asyncio.Lock()
        return self._agent_creation_lock

    # -----------------------------------------------------------------
    # YAML configuration loader
    # -----------------------------------------------------------------

    async def load_agents_from_config(self, config_path: str | None = None) -> None:
        """Load agent configurations from a YAML file.

        If the file is missing or invalid the hardcoded defaults are kept.
        """
        if config_path is None:
            search_paths = [
                Path("config/agents.yaml"),
                Path(__file__).parent.parent.parent / "config" / "agents.yaml",
                Path("/app/config/agents.yaml"),
            ]
            config_file = next((p for p in search_paths if p.exists()), None)
        else:
            config_file = Path(config_path)

        if config_file is None or not config_file.exists():
            logger.warning(
                "agent_config_not_found",
                searched_paths=(
                    [str(p) for p in search_paths]
                    if config_path is None
                    else [config_path]
                ),
                using="hardcoded defaults",
            )
            return

        try:
            async with aiofiles.open(config_file, encoding="utf-8") as fh:
                content = await fh.read()
                config_data = yaml.safe_load(content)

            if not config_data or "agents" not in config_data:
                logger.error(
                    "agent_config_invalid",
                    file=str(config_file),
                    reason="missing 'agents' key",
                )
                return

            defaults = config_data.get("defaults", {})
            model_aliases = config_data.get("model_aliases", {})

            loaded: list[AgentConfig] = []
            _type_map = {
                "task": AgentType.TASK,
                "chat": AgentType.CHAT,
                "analysis": AgentType.ANALYSIS,
                "monitoring": AgentType.MONITORING,
                "support": AgentType.SUPPORT,
                "specialist": AgentType.TASK,
            }

            for agent_data in config_data["agents"]:
                try:
                    agent_type_str = agent_data.get("type", "chat").lower()
                    agent_type = _type_map.get(agent_type_str, AgentType.CHAT)

                    model_name = agent_data.get(
                        "model", defaults.get("model", "gpt-4o-mini")
                    )
                    if model_name in model_aliases:
                        model_name = model_aliases[model_name]

                    config = AgentConfig(
                        id=agent_data["id"],
                        name=agent_data.get("name", agent_data["id"]),
                        agent_type=agent_type,
                        role=agent_data.get("role", agent_data.get("name", "")),
                        goal=agent_data.get("goal", ""),
                        backstory=agent_data.get("backstory", "").strip(),
                        tools=agent_data.get("tools", []),
                        model_name=model_name,
                        memory=agent_data.get("memory", defaults.get("memory", True)),
                    )
                    loaded.append(config)
                    logger.debug(
                        "agent_config_loaded",
                        agent_id=config.id,
                        model=config.model_name,
                    )

                except KeyError as exc:
                    logger.error(
                        "agent_config_missing_field",
                        agent=agent_data.get("id", "unknown"),
                        field=str(exc),
                    )
                except Exception as exc:
                    logger.error(
                        "agent_config_parse_error",
                        agent=agent_data.get("id", "unknown"),
                        error=str(exc),
                    )

            if loaded:
                self.agent_configs = loaded
                logger.info(
                    "agents_loaded_from_config",
                    file=str(config_file),
                    count=len(loaded),
                    agents=[c.id for c in loaded],
                )
            else:
                logger.warning(
                    "no_valid_agents_loaded",
                    file=str(config_file),
                    using="hardcoded defaults",
                )

        except yaml.YAMLError as exc:
            logger.error(
                "agent_config_yaml_error",
                file=str(config_file),
                error=str(exc),
            )
        except Exception as exc:
            logger.error(
                "agent_config_load_error",
                file=str(config_file),
                error=str(exc),
            )

    # -----------------------------------------------------------------
    # Agent retrieval / creation
    # -----------------------------------------------------------------

    async def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve (or create on first access) an agent by *agent_id*.

        Uses an asyncio lock to prevent two coroutines from creating the
        same agent simultaneously.
        """
        # First check without lock (fast path for existing agents)
        if agent_id in self.agents:
            return self.agents[agent_id]

        # Acquire lock before creating agent to prevent race condition
        async with self._get_agent_lock():
            # Double-check: another coroutine may have created it while waiting
            if agent_id in self.agents:
                return self.agents[agent_id]

            # Now create agent inside the lock to ensure atomic creation
            agent = await self._create_agent(agent_id)
            if agent is not None:
                self.agents[agent_id] = agent
            return agent

    async def _create_agent(self, agent_id: str) -> Agent | None:
        """Instantiate an ``Agent`` for *agent_id* using its config."""
        try:
            agent_config = next(
                (c for c in self.agent_configs if c.id == agent_id), None
            )
            if agent_config is None:
                logger.warning("agent_config_not_found", agent_id=agent_id)
                return None

            # Ensure TWS client is available for tools
            await self._ensure_tws_client()

            # Filter tools to only those allowed by this agent's config
            allowed_tools = self._tools_for_config(agent_config)

            agent = Agent(
                model=agent_config.model_name,
                tools=list(allowed_tools.values()),
                instructions=(
                    f"You are a {agent_config.name} assistant for "
                    f"TWS operations. {agent_config.backstory}"
                ),
                name=agent_config.name,
                description=agent_config.backstory,
            )
            logger.info(
                "agent_created",
                agent_id=agent_id,
                agent_name=agent_config.name,
                model=agent_config.model_name,
                tools_enabled=list(allowed_tools.keys()),
            )
            return agent

        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "agent_creation_failed",
                agent_id=agent_id,
                error=str(exc),
            )
            return None

    def _tools_for_config(self, agent_config: AgentConfig) -> dict[str, Any]:
        """Return only the tools that *agent_config.tools* allows.

        If the config lists no tools (empty list), no tools are injected.
        If a tool name in the config doesn't match a discovered tool it
        is silently skipped (with a debug log).
        """
        if not agent_config.tools:
            return {}

        allowed: dict[str, Any] = {}
        for name in agent_config.tools:
            if name in self.tools:
                allowed[name] = self.tools[name]
            else:
                logger.debug(
                    "tool_not_found_for_agent",
                    tool=name,
                    agent_id=agent_config.id,
                )
        return allowed

    # -----------------------------------------------------------------
    # TWS client management
    # -----------------------------------------------------------------

    async def _ensure_tws_client(self) -> None:
        """Lazily initialise ``self.tws_client`` via the factory."""
        if self.tws_client is not None:
            return

        async with self._get_tws_lock():
            if self.tws_client is not None:
                return

            if self._tws_client_factory is None:
                logger.warning("tws_client_factory_not_set")
                return

            try:
                self.tws_client = self._tws_client_factory()
                logger.info(
                    "tws_client_initialized",
                    client_type=type(self.tws_client).__name__,
                )
            except Exception as exc:
                logger.warning("tws_client_init_failed", error=str(exc))

    async def get_tws_client(self) -> Any:
        """Public accessor — ensures client exists, then returns it."""
        await self._ensure_tws_client()
        return self.tws_client

    # -----------------------------------------------------------------
    # Read-only accessors
    # -----------------------------------------------------------------

    async def get_all_agents(self) -> list[AgentConfig]:
        """Return all loaded agent configurations (async)."""
        return list(self.agent_configs)

    async def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        """Retrieve the config for a single agent (async)."""
        return next((c for c in self.agent_configs if c.id == agent_id), None)

    # -----------------------------------------------------------------
    # A2A Protocol support
    # -----------------------------------------------------------------

    async def get_agent_card(
        self, agent_id: str, base_url: str = "http://localhost:8000"
    ) -> AgentCard | None:
        """Generate an A2A Agent Card for the specified agent."""
        config = await self.get_agent_config(agent_id)
        if not config:
            return None

        # Clean strings for description
        description = config.backstory.replace("\n", " ").strip()
        if not description:
            description = config.goal

        return AgentCard(
            name=config.id,
            version="6.1.2",  # Project version
            description=description,
            capabilities=AgentCapabilities(
                actions=config.tools,
                communication_modes=["json-rpc", "websocket"],
                supports_streaming=True,
                supports_events=True,
            ),
            contact=AgentContact(
                endpoint=f"{base_url}/api/v1/a2a/{agent_id}/jsonrpc",
                websocket_endpoint=f"ws://{base_url.split('://')[1]}/ws/{agent_id}",
                auth_required=True,
            ),
            metadata={"role": config.role, "model": config.model_name},
        )

    async def export_a2a_cards(
        self, base_url: str = "http://localhost:8000"
    ) -> list[AgentCard]:
        """Export A2A cards for all configured agents."""
        cards = []
        for config in self.agent_configs:
            card = await self.get_agent_card(config.id, base_url)
            if card:
                cards.append(card)
        return cards


# =============================================================================
# MODULE-LEVEL STATE  (replaces singleton pattern)
# =============================================================================

_agent_manager: AgentManager | None = None
_init_lock = threading.Lock()


def initialize_agent_manager(
    settings_module: Any = settings,
    tws_client_factory: Callable[[], Any] | None = None,
) -> AgentManager:
    """Create (or reconfigure) the module-level ``AgentManager``.

    Called by ``wiring.init_domain_singletons()`` during app lifespan.
    """
    global _agent_manager
    with _init_lock:
        _agent_manager = AgentManager(
            settings_module=settings_module,
            tws_client_factory=tws_client_factory,
        )
    return _agent_manager


def get_agent_manager() -> AgentManager:
    """Return the module-level ``AgentManager``.

    If :func:`initialize_agent_manager` has not been called yet a bare
    instance (no TWS factory) is created as a fallback — this keeps
    import-time consumers working during tests and CLI scripts.
    """
    global _agent_manager
    if _agent_manager is None:
        with _init_lock:
            if _agent_manager is None:
                logger.warning(
                    "agent_manager_fallback_init",
                    reason="accessed before initialize_agent_manager()",
                )
                _agent_manager = AgentManager()
    return _agent_manager


# =============================================================================
# UNIFIED AGENT INTERFACE
# =============================================================================


class UnifiedAgent:
    """High-level chat interface with automatic intent routing.

    History is stored **per conversation_id** to prevent data leaking
    between users/sessions in a multi-tenant server.

    Usage::

        ua = UnifiedAgent(agent_manager=get_agent_manager())
        response = await ua.chat("Quais jobs estão em ABEND?",
                                  conversation_id="sess-123")
    """

    _MAX_HISTORY: int = 100
    _TRIM_TO: int = 50

    def __init__(
        self,
        agent_manager: AgentManager | None = None,
        skill_manager: Any = None,
    ) -> None:
        from resync.core.agent_router import create_router

        self._manager = agent_manager or get_agent_manager()
        self._router = create_router(self._manager, skill_manager=skill_manager)
        # Per-conversation history: {conversation_id: [messages]}
        self._histories: dict[str, list[dict[str, str]]] = {}
        logger.info(
            "unified_agent_initialized",
            skill_manager_enabled=skill_manager is not None,
        )

    def _get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self._histories.setdefault(conversation_id, [])

    async def chat(
        self,
        message: str,
        include_history: bool = True,
        use_llm_classification: bool = False,
        conversation_id: str = "_default",
    ) -> str:
        """Send a message and receive a routed response.

        Args:
            message: User input.
            include_history: Prepend recent history as context.
            use_llm_classification: Reserved for future use.
            conversation_id: Session key for history isolation.
        """
        _ = use_llm_classification

        history = self._get_history(conversation_id)
        context: dict[str, Any] = {}
        if include_history and history:
            context["history"] = history[-10:]

        result = await self._router.route(message, context=context)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result.response})

        if len(history) > self._MAX_HISTORY:
            self._histories[conversation_id] = history[-self._TRIM_TO :]

        logger.debug(
            "unified_agent_response",
            handler=result.handler_name,
            intent=result.classification.primary_intent.value,
            confidence=result.classification.confidence,
            processing_time_ms=result.processing_time_ms,
            conversation_id=conversation_id,
        )

        return result.response

    async def chat_with_metadata(
        self,
        message: str,
        include_history: bool = True,
        tws_instance_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
        conversation_id: str = "_default",
    ) -> dict[str, Any]:
        """Send a message and receive response with full metadata.

        Returns a dict with keys: ``response``, ``intent``, ``confidence``,
        ``handler``, ``tools_used``, ``entities``, ``processing_time_ms``,
        ``tws_instance_id``.
        """
        history = self._get_history(conversation_id)
        context: dict[str, Any] = {}
        if include_history and history:
            context["history"] = history[-10:]

        if tws_instance_id:
            context["tws_instance_id"] = tws_instance_id
            logger.debug("tws_instance_context_set", instance_id=tws_instance_id)

        if extra_context:
            context.update(extra_context)

        result = await self._router.route(message, context=context)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result.response})

        return {
            "response": result.response,
            "intent": result.classification.primary_intent.value,
            "confidence": result.classification.confidence,
            "handler": result.handler_name,
            "tools_used": result.tools_used,
            "entities": result.classification.entities,
            "processing_time_ms": result.processing_time_ms,
            "tws_instance_id": tws_instance_id,
        }

    def clear_history(self, conversation_id: str = "_default") -> None:
        """Clear history for a specific conversation."""
        self._histories.pop(conversation_id, None)
        logger.debug("conversation_history_cleared", conversation_id=conversation_id)

    def get_history(self, conversation_id: str = "_default") -> list[dict[str, str]]:
        """Return a copy of history for *conversation_id*."""
        return list(self._get_history(conversation_id))

    @property
    def router(self):
        """Access the underlying router for advanced usage."""
        return self._router


# ---------------------------------------------------------------------------
# Module-level UnifiedAgent (lazy, per-session safe)
# ---------------------------------------------------------------------------

_unified_agent: UnifiedAgent | None = None


def get_unified_agent() -> UnifiedAgent:
    """Return (or create) the module-level ``UnifiedAgent``."""
    global _unified_agent
    if _unified_agent is None:
        _unified_agent = UnifiedAgent(get_agent_manager())
    return _unified_agent


# ---------------------------------------------------------------------------
# Module __getattr__ for backward-compat named imports
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Support ``from resync.core.agent_manager import agent_manager``
    and ``from resync.core.agent_manager import unified_agent`` as lazy
    accessors that always return the currently-configured instance.
    """
    if name == "agent_manager":
        return get_agent_manager()
    if name == "unified_agent":
        return get_unified_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentManager",
    "AgentsConfig",
    "AgentType",
    "MockAgent",
    "AGNO_AVAILABLE",
    "UnifiedAgent",
    "get_agent_manager",
    "get_unified_agent",
    "initialize_agent_manager",
]
