"""AI Agent management and orchestration.

Provides agent lifecycle management, configuration via YAML, execution
orchestration, and a unified chat interface.

v6.1.2 — Removed singleton anti-pattern, fixed DI, per-session history,
          tools filtered by config, structured logging throughout.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, cast

import aiofiles  # type: ignore[import-untyped]
import structlog
import yaml  # type: ignore[import-untyped]
from cachetools import LRUCache  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from resync.core.exceptions import AgentError
from resync.core.metrics import runtime_metrics
from resync.core.utils.async_bridge import run_sync
from resync.models.a2a import AgentCapabilities, AgentCard, AgentContact
from resync.models.agents import AgentConfig, AgentType
from resync.settings import settings
from resync.tools.definitions.tws import (
    tws_status_tool,
    tws_troubleshooting_tool,
)

from .global_utils import get_environment_tags, get_global_correlation_id

agent_logger = structlog.get_logger("resync.agent_manager")
logger = structlog.get_logger(__name__)

# =============================================================================
# CONSTANTS - Exception handling
# =============================================================================

# P0-2: Consolidated exception handling constants
# Runtime exceptions that should be caught and handled gracefully
RUNTIME_EXCEPTIONS = (
    OSError,
    TimeoutError,
    ConnectionError,
    ValueError,
)

# Programming errors that should be re-raised (bugs, not runtime failures)
PROGRAMMING_EXCEPTIONS = (
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    RuntimeError,
)

# All exceptions caught in handlers (for logging + fallback)
ALL_CAUGHT_EXCEPTIONS = RUNTIME_EXCEPTIONS + PROGRAMMING_EXCEPTIONS

# =============================================================================
# CONSTANTS - Magic numbers and configuration
# =============================================================================

# P2-1: Extracted magic numbers for Agent LLM configuration
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MODEL: str = "ollama/qwen2.5:3b"

# P2-1: Extracted magic numbers for UnifiedAgent history management
DEFAULT_MAX_HISTORY: int = 100
DEFAULT_TRIM_TO: int = 50
DEFAULT_HISTORY_CACHE_SIZE: int = 1000
DEFAULT_HISTORY_CONTEXT_SIZE: int = 10

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
        self.model = model or DEFAULT_MODEL
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

        # NEW v6.2: Convert tools to LiteLLM format
        self._litellm_tools = self._convert_tools_to_litellm_format()

    def _convert_tools_to_litellm_format(self) -> list[dict[str, Any]]:
        """Convert internal tools to LiteLLM tool schema.

        LiteLLM expects tools in OpenAI function calling format:
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search TWS documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
        """
        litellm_tools = []

        # P0-01 FIX: Include RAG tool by default for all agents
        # This ensures they can always search the documentation
        try:
            from resync.core.specialists.tools import RAGTool
            rag = RAGTool()
            search_func = rag.search_knowledge_base
            if not hasattr(search_func, "__name__"):
                search_func.__name__ = "search_knowledge_base"  # type: ignore[attr-defined]
            
            # Avoid duplicate registration if already in self.tools
            if not any(getattr(t, "__name__", None) == "search_knowledge_base" for t in self.tools):
                self.tools.append(search_func)
        except Exception as e:
            logger.warning("failed_to_register_rag_tool", error=str(e))

        for tool in self.tools:
            # Skip if tool doesn't have callable or metadata
            if not callable(tool):
                continue

            tool_name = getattr(tool, "__name__", str(tool))
            tool_doc = getattr(tool, "__doc__", None) or f"Execute {tool_name}"

            # Extract function signature for parameters
            sig = inspect.signature(tool)
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}",
                }

                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            litellm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_doc.split("\n")[0][
                            :200
                        ],  # First line, max 200 chars
                        "parameters": parameters,
                    },
                }
            )

        return litellm_tools

    async def arun(self, message: str, skill_context: str = "") -> str:
        """Process *message* via LiteLLM (async).

        v6.2: Enhanced with automatic tool calling.
        The LLM will analyze the query and call appropriate tools.
        """
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
                f"{full_instructions}\n\n"
            )

            # NEW v6.2: Add tool usage instructions
            if self._litellm_tools:
                system_prompt += "\nYou have access to the following tools:\n"
                for tool_config in self._litellm_tools:
                    func = tool_config["function"]
                    system_prompt += f"- {func['name']}: {func['description']}\n"

                system_prompt += (
                    "\nIMPORTANT: When the user asks a question:\n"
                    "1. Analyze if you need to call any tools to answer accurately\n"
                    "2. Call search_knowledge_base for documentation/how-to questions\n"
                    "3. Call get_tws_status for status checks\n"
                    "4. Call analyze_failures for troubleshooting\n"
                    "5. Synthesize the tool results into a helpful response\n"
                    "6. Always respond in Portuguese (Brazilian) unless user writes in English\n"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            # NEW v6.2: Call LiteLLM with tools
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=self._litellm_tools if self._litellm_tools else None,
                tool_choice="auto",  # Let LLM decide when to use tools
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )

            # P0-1: Verificar resposta vazia antes de acessar
            choices = getattr(response, "choices", None)
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                agent_logger.error("empty_response_from_llm", agent=self.name)
                return self._fallback_response(message)

            first_choice = choices[0]
            message_obj = getattr(first_choice, "message", None)
            if message_obj is None:
                agent_logger.error("empty_message_in_response", agent=self.name)
                return self._fallback_response(message)

            # NEW v6.2: Check if LLM wants to call tools
            tool_calls = getattr(message_obj, "tool_calls", None)

            if tool_calls:
                # LLM decided to call tools
                logger.info("llm_calling_tools", count=len(tool_calls))

                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    # Find and execute the tool
                    tool_result = await self._execute_tool(func_name, func_args)
                    tool_results.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": tool_result,
                        }
                    )

                # Send tool results back to LLM for synthesis
                messages.append(message_obj)
                messages.extend(tool_results)

                final_response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )

                final_choice = final_response.choices[0]
                content = final_choice.message.content
                return content if isinstance(content, str) else str(content)

            # No tools called, return direct response
            content = getattr(message_obj, "content", None)
            if content is None:
                agent_logger.error("empty_content_in_message", agent=self.name)
                return self._fallback_response(message)

            return content if isinstance(content, str) else str(content)

        except asyncio.CancelledError:
            raise
        except PROGRAMMING_EXCEPTIONS:
            # Re-raise programming errors — these are bugs, not runtime failures
            raise
        except RUNTIME_EXCEPTIONS as exc:
            agent_logger.error("agent_arun_error", error=str(exc), agent=self.name)
            return self._fallback_response(message)

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments.

        Returns the tool result as a string for LLM consumption.
        """
        try:
            # Find tool by name
            tool = next(
                (
                    t
                    for t in self.tools
                    if getattr(t, "__name__", None) == tool_name
                ),
                None,
            )

            if tool is None:
                return f"Error: Tool {tool_name} not found"

            # Execute tool (offload blocking calls to thread if needed)
            if inspect.iscoroutinefunction(tool):
                result = await tool(**arguments)
            else:
                result = await asyncio.to_thread(tool, **arguments)

            # Convert result to string for LLM
            if isinstance(result, str):
                return result
            elif isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return str(result)

        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _fallback_response(self, message: str) -> str:
        """Provide a keyword-based fallback when the LLM is unavailable."""
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "agent_llm_fallback", 
                attributes={"agent.name": self.name, "fallback_triggered": True}
            ):
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
        except RUNTIME_EXCEPTIONS as e:
            agent_logger.error("fallback_response_error", error=str(e), agent=self.name)
            return (
                "Ocorreu um erro inesperado ao tentar gerar uma resposta alternativa."
            )

    def run(self, message: str) -> str:
        """Synchronous wrapper around :meth:`arun`."""
        coro = self.arun(message)
        try:
            result = run_sync(coro)
            return result if isinstance(result, str) else str(result)
        except RuntimeError:
            coro.close()
            raise

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
class AgentsConfig(BaseModel):  # type: ignore[misc]
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
        import sys as _sys

        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("tool_discovery_failed", error=str(exc))
        return {}

class AgentManager:
    """Manages agent lifecycle: creation, caching, configuration, and tools.

    Cached module-level instance. Create via :func:`initialize_agent_manager`
    (called by the app lifespan in ``wiring.py``) and retrieve with
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
            "agent_manager_init",
            component="agent_manager",
            global_correlation=get_global_correlation_id(),
            environment=get_environment_tags(),
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
                    max_rpm=None,
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
                    max_rpm=None,
                ),
            ]

            self.tools: dict[str, Any] = _discover_tools()
            self._tws_client_factory = tws_client_factory
            self.tws_client: Any = None
            self._lock_registry_lock = threading.Lock()
            # P1-1: Use int | None to allow fallback key for contexts without running loop
            self._tws_locks: dict[int | None, asyncio.Lock] = {}
            self._agent_locks: dict[int | None, asyncio.Lock] = {}

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
        """P1-1: Get or create a lock for the current event loop.
        
        Includes fallback for contexts without running loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback for contexts without a running loop (e.g., during initialization)
            with self._lock_registry_lock:
                if None not in self._tws_locks:
                    self._tws_locks[None] = asyncio.Lock()
                return self._tws_locks[None]
        
        loop_id = id(loop)
        with self._lock_registry_lock:
            lock = self._tws_locks.get(loop_id)
            if lock is None:
                lock = asyncio.Lock()
                self._tws_locks[loop_id] = lock
            return lock

    def _get_agent_lock(self) -> asyncio.Lock:
        """P1-1: Get or create a lock for the current event loop.
        
        Includes fallback for contexts without running loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback for contexts without a running loop (e.g., during initialization)
            with self._lock_registry_lock:
                if None not in self._agent_locks:
                    self._agent_locks[None] = asyncio.Lock()
                return self._agent_locks[None]
        
        loop_id = id(loop)
        with self._lock_registry_lock:
            lock = self._agent_locks.get(loop_id)
            if lock is None:
                lock = asyncio.Lock()
                self._agent_locks[loop_id] = lock
            return lock

    # -----------------------------------------------------------------
    # YAML configuration loader
    # -----------------------------------------------------------------

    async def load_agents_from_config(self, config_path: str | None = None) -> None:
        """Load agent configurations from a YAML file.

        If the file is missing or invalid the hardcoded defaults are kept.
        """
        search_paths = [
            Path("config/agents.yaml"),
            Path(__file__).parent.parent.parent / "config" / "agents.yaml",
            Path("/app/config/agents.yaml"),
        ]

        def _find_and_validate() -> tuple[Path | None, bool]:
            if config_path is None:
                config_file = next((p for p in search_paths if p.exists()), None)
            else:
                config_file = Path(config_path)
            
            exists = config_file is not None and config_file.exists()
            return config_file, exists
        
        config_file, exists = await asyncio.to_thread(_find_and_validate)

        if not exists:
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
                config_data = await asyncio.to_thread(yaml.safe_load, content)

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
                        max_rpm=agent_data.get("max_rpm"),
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
                except ALL_CAUGHT_EXCEPTIONS as exc:
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
        except asyncio.CancelledError:
            raise
        except ALL_CAUGHT_EXCEPTIONS as exc:
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

            # NEW v6.2: Convert tools dict to list of callables
            tool_callables = []
            for tool_name, tool_func in allowed_tools.items():
                # Ensure tool is callable
                if callable(tool_func):
                    # Set __name__ for tool identification
                    if not hasattr(tool_func, "__name__"):
                        try:
                            tool_func.__name__ = tool_name
                        except (AttributeError, TypeError):
                            # Fallback if __name__ is not writable
                            pass
                    tool_callables.append(tool_func)
                else:
                    logger.warning("tool_not_callable", tool=tool_name)

            agent = Agent(
                model=agent_config.model_name,
                tools=tool_callables,  # Pass list of callables
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

        except PROGRAMMING_EXCEPTIONS:
            # Re-raise programming errors — these are bugs, not runtime failures
            raise
        except RUNTIME_EXCEPTIONS as exc:
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
                if inspect.iscoroutinefunction(self._tws_client_factory):
                    self.tws_client = await self._tws_client_factory()
                else:
                    self.tws_client = await asyncio.to_thread(self._tws_client_factory)
                logger.info(
                    "tws_client_initialized",
                    client_type=type(self.tws_client).__name__,
                )
            except RUNTIME_EXCEPTIONS as exc:
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
                supports_push_notifications=False,
                max_concurrent_tasks=10,
            ),
            contact=AgentContact(
                endpoint=f"{base_url}/api/v1/a2a/{agent_id}/jsonrpc",
                websocket_endpoint=f"ws://{base_url.split('://')[1]}/ws/{agent_id}",
                auth_required=True,
                protocol="A2A",
                event_endpoint=None,
            ),
            protocol_version="A2A-2024-11-05",
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

    _MAX_HISTORY: int = DEFAULT_MAX_HISTORY
    _TRIM_TO: int = DEFAULT_TRIM_TO
    _HISTORY_CACHE_SIZE: int = DEFAULT_HISTORY_CACHE_SIZE

    def __init__(
        self,
        agent_manager: AgentManager | None = None,
        skill_manager: Any = None,
    ) -> None:
        from resync.core.agent_router import create_router

        self._manager = agent_manager or get_agent_manager()
        self._router = create_router(self._manager, skill_manager=skill_manager)
        # Per-conversation history with an LRU cache to prevent OOM
        self._histories: LRUCache[str, list[dict[str, str]]] = LRUCache(
            maxsize=self._HISTORY_CACHE_SIZE
        )
        # P0-01 & P0-04: Two-layer locking
        # asyncio.Lock for async callers; threading.RLock for sync helpers.
        # Fixed P0-04: Added _history_locks_lock to guard dictionary access.
        self._history_locks: dict[str, asyncio.Lock] = {}   # chat / chat_with_metadata
        self._history_locks_lock = threading.Lock()         # P0-04 FIX: Guard for the dict
        self._sync_locks: dict[str, threading.RLock] = {}   # for sync _get_history (sync)
        logger.info(
            "unified_agent_initialized",
            skill_manager_enabled=skill_manager is not None,
        )

    def _get_history_lock(self, conversation_id: str) -> asyncio.Lock:
        """P0-04 FIX: Thread-safe Lock retrieval/creation."""
        with self._history_locks_lock:
            if conversation_id not in self._history_locks:
                self._history_locks[conversation_id] = asyncio.Lock()
            return self._history_locks[conversation_id]

    def _get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """P0-01 FIX: Thread-safe sync history access with threading.RLock."""
        lock = self._sync_locks.setdefault(conversation_id, threading.RLock())
        with lock:
            history = cast(list[dict[str, str]] | None, self._histories.get(conversation_id))
            if history is None:
                history = []
                self._histories[conversation_id] = history
            return history

    async def chat(
        self,
        message: str,
        include_history: bool = True,
        use_llm_classification: bool = False,
        conversation_id: str = "_default",
    ) -> str:
        """Send a message and receive a routed response.

        NEW v6.2: Always calls LLM agent, which decides which tools to use.
        The router is bypassed for better UX and simpler architecture.

        Args:
            message: User input.
            include_history: Prepend recent history as context.
            use_llm_classification: DEPRECATED (always uses LLM now).
            conversation_id: Session key for history isolation.
        """
        # P2-2: Parameter is intentionally reserved for future use
        _ = use_llm_classification

        # P0-3: Use lock to prevent race conditions on history access
        async with self._get_history_lock(conversation_id):
            history = self._get_history(conversation_id)

            # Build context with history
            context_parts = []
            if include_history and history:
                context_parts.append("## Histórico da Conversa")
                for msg in history[-DEFAULT_HISTORY_CONTEXT_SIZE:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    context_parts.append(f"**{role}**: {content}")

            # Build full prompt for LLM
            if context_parts:
                full_message = (
                    "\n".join(context_parts) + f"\n\n**Nova Pergunta**: {message}"
                )
            else:
                full_message = message

            # NEW v6.2: Always call LLM agent (no router)
            # The LLM will decide which tools to use via litellm tool calling
            agent = await self._manager.get_agent("tws-general")

            if agent is None:
                logger.error("tws_general_agent_not_found")
                response = (
                    "Desculpe, o agente não está disponível no momento. "
                    "Tente novamente em alguns instantes."
                )
            else:
                response = await agent.arun(full_message)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

            if len(history) > self._MAX_HISTORY:
                self._histories[conversation_id] = history[-self._TRIM_TO :]

        logger.debug(
            "unified_agent_response",
            has_agent=agent is not None,
        )

        return response

    async def chat_with_metadata(
        self,
        message: str,
        include_history: bool = True,
        tws_instance_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
        conversation_id: str = "_default",
    ) -> dict[str, Any]:
        """Send a message and receive response with full metadata.

        NEW v6.2: Simplified - always calls LLM, no router.

        Returns a dict with keys: ``response``, ``tools_used``,
        ``processing_time_ms``, ``tws_instance_id``.
        """
        import time

        start_time = time.time()

        # Call chat() which now always uses LLM
        response = await self.chat(
            message=message,
            include_history=include_history,
            conversation_id=conversation_id,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "response": response,
            "tools_used": [],  # TODO: Track from LLM tool_calls
            "processing_time_ms": processing_time_ms,
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
    def router(self) -> Any:
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
    # Exported constants
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_HISTORY",
    "DEFAULT_TRIM_TO",
    "DEFAULT_HISTORY_CACHE_SIZE",
    "RUNTIME_EXCEPTIONS",
    "PROGRAMMING_EXCEPTIONS",
    "ALL_CAUGHT_EXCEPTIONS",
]

# =============================================================================
# Module-level `unified_agent` singleton proxy
# =============================================================================

class _UnifiedAgentProxy:
    """Lazy proxy: delegates all attribute access to the real UnifiedAgent.

    This allows ``from resync.core.agent_manager import unified_agent``
    to work without triggering initialization at import time.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(get_unified_agent(), name)

    def __repr__(self) -> str:
        return f"<UnifiedAgentProxy wrapping {get_unified_agent()!r}>"


#: Module-level singleton used by ``resync.api.routes.core.chat``.
unified_agent: _UnifiedAgentProxy = _UnifiedAgentProxy()
