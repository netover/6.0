# ruff: noqa: E501
# pylint
"""
Prompt Manager with LangFuse Integration.

This module provides centralized prompt management with:
- YAML-based prompt definitions (local fallback)
- LangFuse cloud integration (when configured)
- Version control and A/B testing
- Runtime prompt compilation with variables
- Admin API for CRUD operations

Architecture:
1. Prompts are defined in YAML files (prompts/*.yaml)
2. LangFuse syncs and versions prompts in the cloud
3. Admin UI allows editing without code deployment
4. Fallback to local YAML if LangFuse is unavailable
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import aiofiles
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Try to import langfuse (optional dependency)
try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    logger.warning(
        "langfuse_prompt_manager_disabled reason=%s",
        type(exc).__name__,
    )

# =============================================================================
# MODELS
# =============================================================================

class PromptType(str, Enum):
    """Types of prompts in the system."""

    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    RAG = "rag"
    ROUTER = "router"
    TOOL = "tool"

class PromptConfig(BaseModel):
    """Configuration for a single prompt."""

    id: str = Field(..., description="Unique prompt identifier (e.g., 'tws-agent-v2')")
    name: str = Field(..., description="Human-readable name")
    type: PromptType = Field(default=PromptType.SYSTEM, description="Prompt type")
    version: str = Field(default="1.0.0", description="Semantic version")

    # Content
    content: str = Field(..., description="Prompt template with {{variables}}")
    description: str = Field(default="", description="Prompt description/purpose")

    # Metadata
    model_hint: str | None = Field(
        None, description="Recommended model for this prompt"
    )
    temperature_hint: float | None = Field(
        None, ge=0, le=2, description="Recommended temperature"
    )
    max_tokens_hint: int | None = Field(None, description="Recommended max tokens")

    # Variables
    variables: list[str] = Field(
        default_factory=list, description="Required variables in template"
    )
    default_values: dict[str, str] = Field(
        default_factory=dict, description="Default values for variables"
    )

    # Status
    is_active: bool = Field(default=True, description="Whether prompt is active")
    is_default: bool = Field(
        default=False, description="Whether this is the default for its type"
    )

    # Tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="system")

    # A/B Testing
    ab_test_group: str | None = Field(None, description="A/B test group identifier")
    ab_test_weight: float = Field(
        default=1.0, ge=0, le=1, description="Weight for A/B selection"
    )

    model_config = ConfigDict()

    @field_serializer("created_at", "updated_at")
    def serialize_datetimes(self, value: datetime) -> str:
        return value.isoformat()

class PromptTemplate:
    """
    Compiled prompt template ready for use.

    Usage:
        template = PromptTemplate(config)
        message = template.compile(user_name="John", context="TWS status")
    """

    def __init__(self, config: PromptConfig):
        self.config = config
        self._compiled_cache: dict[str, str] = {}

    def compile(self, **variables: str) -> str:
        """
        Compile the template with provided variables.

        Args:
            **variables: Key-value pairs to substitute in template

        Returns:
            Compiled prompt string

        Raises:
            ValueError: If required variables are missing
        """
        # Merge with defaults
        final_vars = {**self.config.default_values, **variables}

        # Check required variables
        missing = set(self.config.variables) - set(final_vars.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Substitute variables (using {{var}} syntax)
        result = self.config.content
        for var_name, var_value in final_vars.items():
            result = result.replace(f"{{{{{var_name}}}}}", str(var_value))

        return result

    def to_message(self, role: str = "system", **variables: str) -> dict[str, str]:
        """
        Create a chat message dictionary.

        Args:
            role: Message role (system, user, assistant)
            **variables: Variables for template compilation

        Returns:
            Dict with 'role' and 'content' keys
        """
        return {"role": role, "content": self.compile(**variables)}

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def model_hint(self) -> str | None:
        return self.config.model_hint

    @property
    def temperature_hint(self) -> float | None:
        return self.config.temperature_hint

# =============================================================================
# PROMPT MANAGER
# =============================================================================

class PromptManager:
    """
    Centralized prompt management with LangFuse integration.

    Features:
    - Load prompts from YAML files
    - Sync with LangFuse for cloud management
    - Version control and A/B testing
    - Runtime compilation with variables
    - Admin CRUD operations

    The manager uses a tiered approach:
    1. Check LangFuse (if available and configured)
    2. Fall back to local YAML files
    3. Fall back to hardcoded defaults
    """

    _instance: PromptManager | None = None
    _initialized: bool = False

    def __new__(cls) -> PromptManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the prompt manager."""
        if self._initialized:
            return

        self._prompts: dict[str, PromptConfig] = {}
        self._templates: dict[str, PromptTemplate] = {}
        self._langfuse_client: Langfuse | None = None
        self._lock = asyncio.Lock()

        # Paths
        self._prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self._prompts_dir.mkdir(exist_ok=True)

        self._initialized = True
        logger.info("prompt_manager_created")

    async def initialize(self) -> None:
        """
        Initialize the prompt manager.

        Loads prompts from:
        1. LangFuse (if configured)
        2. Local YAML files
        3. Default prompts
        """
        async with self._lock:
            # Initialize LangFuse client if available
            if LANGFUSE_AVAILABLE and self._should_use_langfuse():
                try:
                    self._langfuse_client = Langfuse(
                        public_key=getattr(settings, "langfuse_public_key", None),
                        secret_key=getattr(settings, "langfuse_secret_key", None),
                        host=getattr(
                            settings, "langfuse_host", "https://cloud.langfuse.com"
                        ),
                    )
                    logger.info("langfuse_client_initialized")
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    logger.warning("langfuse_init_failed", error=str(e))

            # Load prompts
            self._load_default_prompts()
            await self._load_yaml_prompts()

            if self._langfuse_client:
                self._sync_from_langfuse()

            logger.info(
                "prompt_manager_initialized",
                prompt_count=len(self._prompts),
                langfuse_enabled=self._langfuse_client is not None,
            )

    def _should_use_langfuse(self) -> bool:
        """Check if LangFuse should be used."""
        return (
            getattr(settings, "langfuse_enabled", False)
            and getattr(settings, "langfuse_public_key", None)
            and getattr(settings, "langfuse_secret_key", None)
        )

    def _load_default_prompts(self) -> None:
        """Load hardcoded default prompts."""
        defaults = [
            PromptConfig(
                id="tws-agent-system-v1",
                name="TWS Agent System Prompt",
                type=PromptType.AGENT,
                version="1.0.0",
                content="""Você é um assistente de IA especializado em HCL Workload Automation (TWS).

Sua função é ajudar operadores e administradores a:
- Monitorar status de jobs e workstations
- Diagnosticar problemas e falhas
- Fornecer recomendações baseadas em melhores práticas

Contexto atual:
{{context}}

Responda de forma clara, concisa e profissional em português brasileiro.
Se não tiver certeza sobre algo, indique claramente.""",
                description="System prompt for TWS troubleshooting agent",
                variables=["context"],
                default_values={"context": "Nenhum contexto adicional disponível."},
                model_hint="meta/llama-3.1-70b-instruct",
                temperature_hint=0.7,
                is_default=True,
            ),
            PromptConfig(
                id="tws-rag-system-v1",
                name="TWS RAG System Prompt",
                type=PromptType.RAG,
                version="1.0.0",
                content="""Você é um assistente de IA especializado em responder perguntas sobre HCL Workload Automation (TWS) baseado no contexto fornecido.

Use APENAS as informações do contexto para responder. Se a resposta não estiver no contexto, diga que não tem informação suficiente.

Contexto relevante da base de conhecimento:
{{rag_context}}

Instruções adicionais:
- Cite a fonte quando possível
- Seja preciso e técnico
- Responda em português brasileiro""",
                description="System prompt for RAG-based Q&A",
                variables=["rag_context"],
                default_values={"rag_context": "Nenhum contexto disponível."},
                model_hint="meta/llama-3.1-70b-instruct",
                temperature_hint=0.3,
                is_default=True,
            ),
            PromptConfig(
                id="intent-router-v1",
                name="Intent Router Prompt",
                type=PromptType.ROUTER,
                version="1.0.0",
                content="""Classifique a intenção do usuário em uma das seguintes categorias:

Categorias:
- STATUS: Consultas sobre status de jobs, workstations, sistema
- TROUBLESHOOT: Diagnóstico de problemas, erros, falhas
- QUERY: Perguntas sobre documentação, conceitos, procedimentos
- ACTION: Solicitações para executar ações (cancelar, reiniciar, etc.)
- GENERAL: Conversas gerais, saudações, ajuda

Mensagem do usuário: "{{user_message}}"

Responda APENAS com o nome da categoria em maiúsculas.""",
                description="Prompt for intent classification",
                variables=["user_message"],
                model_hint="meta/llama-3.1-8b-instruct",
                temperature_hint=0.1,
                max_tokens_hint=10,
                is_default=True,
            ),
            PromptConfig(
                id="tws-status-report-v1",
                name="TWS Status Report Prompt",
                type=PromptType.SYSTEM,
                version="1.0.0",
                content="""Gere um relatório de status do sistema TWS baseado nos dados fornecidos.

Dados do sistema:
{{system_data}}

Formato do relatório:
1. Resumo executivo (2-3 linhas)
2. Status das workstations
3. Jobs críticos/problemáticos
4. Recomendações (se houver problemas)

Seja conciso e priorize informações críticas.""",
                description="Prompt for generating status reports",
                variables=["system_data"],
                default_values={"system_data": "{}"},
                is_default=True,
            ),
            PromptConfig(
                id="tool-response-formatter-v1",
                name="Tool Response Formatter",
                type=PromptType.TOOL,
                version="1.0.0",
                content="""Formate a resposta da ferramenta de forma clara para o usuário.

Ferramenta executada: {{tool_name}}
Resultado bruto: {{tool_output}}

Formate de forma legível e amigável, destacando informações importantes.
Se houver erros, explique o que pode ter acontecido.""",
                description="Format tool outputs for user consumption",
                variables=["tool_name", "tool_output"],
                is_default=True,
            ),
        ]

        for prompt in defaults:
            self._prompts[prompt.id] = prompt
            self._templates[prompt.id] = PromptTemplate(prompt)

    async def _load_yaml_prompts(self) -> None:
        """Load prompts from YAML files."""
        if not self._prompts_dir.exists():
            return

        for yaml_file in self._prompts_dir.glob("*.yaml"):
            try:
                async with aiofiles.open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                # Handle single prompt or list of prompts
                prompts_data = data if isinstance(data, list) else [data]

                for prompt_data in prompts_data:
                    try:
                        config = PromptConfig(**prompt_data)
                        self._prompts[config.id] = config
                        self._templates[config.id] = PromptTemplate(config)
                        logger.debug(
                            "prompt_loaded_from_yaml",
                            prompt_id=config.id,
                            file=yaml_file.name,
                        )
                    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                        logger.warning(
                            "invalid_prompt_in_yaml", file=yaml_file.name, error=str(e)
                        )

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("yaml_load_failed", file=yaml_file.name, error=str(e))

    def _sync_from_langfuse(self) -> None:
        """Sync prompts from LangFuse."""
        if not self._langfuse_client:
            return

        try:
            # Get all prompts from LangFuse
            # Note: This is a simplified implementation - actual LangFuse API may differ
            logger.info("syncing_prompts_from_langfuse")

            # LangFuse SDK usage would go here
            # For now, we'll just log that we would sync
            logger.debug("langfuse_sync_placeholder")

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("langfuse_sync_failed", error=str(e))

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def get_prompt(
        self, prompt_id: str, version: str | None = None
    ) -> PromptTemplate | None:
        """
        Get a prompt template by ID.

        Args:
            prompt_id: The prompt identifier
            version: Optional specific version (defaults to latest)

        Returns:
            PromptTemplate if found, None otherwise
        """
        # Try exact ID first
        if prompt_id in self._templates:
            return self._templates[prompt_id]

        # Try with version suffix
        if version:
            versioned_id = f"{prompt_id}-v{version}"
            if versioned_id in self._templates:
                return self._templates[versioned_id]

        return None

    async def get_default_prompt(
        self, prompt_type: PromptType
    ) -> PromptTemplate | None:
        """
        Get the default prompt for a given type.

        Args:
            prompt_type: The type of prompt to get

        Returns:
            Default PromptTemplate for the type, or None
        """
        for config in self._prompts.values():
            if config.type == prompt_type and config.is_default and config.is_active:
                return self._templates[config.id]
        return None

    async def list_prompts(
        self, prompt_type: PromptType | None = None, active_only: bool = True
    ) -> list[PromptConfig]:
        """
        List all prompts, optionally filtered.

        Args:
            prompt_type: Filter by type
            active_only: Only return active prompts

        Returns:
            List of PromptConfig objects
        """
        result = []
        for config in self._prompts.values():
            if active_only and not config.is_active:
                continue
            if prompt_type and config.type != prompt_type:
                continue
            result.append(config)
        return result

    async def create_prompt(self, config: PromptConfig) -> PromptConfig:
        """
        Create a new prompt.

        Args:
            config: The prompt configuration

        Returns:
            The created PromptConfig

        Raises:
            ValueError: If prompt ID already exists
        """
        if config.id in self._prompts:
            raise ValueError(f"Prompt '{config.id}' already exists")

        config.created_at = datetime.now(timezone.utc)
        config.updated_at = datetime.now(timezone.utc)

        self._prompts[config.id] = config
        self._templates[config.id] = PromptTemplate(config)

        # Save to YAML
        await self._save_prompt_to_yaml(config)

        # Sync to LangFuse
        if self._langfuse_client:
            self._sync_prompt_to_langfuse(config)

        logger.info("prompt_created", prompt_id=config.id)
        return config

    async def update_prompt(
        self, prompt_id: str, updates: dict[str, Any]
    ) -> PromptConfig | None:
        """
        Update an existing prompt.

        Args:
            prompt_id: The prompt to update
            updates: Dictionary of fields to update

        Returns:
            Updated PromptConfig, or None if not found
        """
        if prompt_id not in self._prompts:
            return None

        config = self._prompts[prompt_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now(timezone.utc)

        # Rebuild template
        self._templates[prompt_id] = PromptTemplate(config)

        # Save to YAML
        await self._save_prompt_to_yaml(config)

        # Sync to LangFuse
        if self._langfuse_client:
            self._sync_prompt_to_langfuse(config)

        logger.info("prompt_updated", prompt_id=prompt_id)
        return config

    async def delete_prompt(self, prompt_id: str) -> bool:
        """
        Delete a prompt.

        Args:
            prompt_id: The prompt to delete

        Returns:
            True if deleted, False if not found
        """
        if prompt_id not in self._prompts:
            return False

        del self._prompts[prompt_id]
        del self._templates[prompt_id]

        # Remove from YAML
        yaml_file = self._prompts_dir / f"{prompt_id}.yaml"
        try:
            import aiofiles.os

            await aiofiles.os.remove(yaml_file)
        except FileNotFoundError:
            pass

        logger.info("prompt_deleted", prompt_id=prompt_id)
        return True

    async def _save_prompt_to_yaml(self, config: PromptConfig) -> None:
        """Save a prompt to its YAML file."""
        yaml_file = self._prompts_dir / f"{config.id}.yaml"

        data = config.model_dump(mode="json")

        async with aiofiles.open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, allow_unicode=True, default_flow_style=False, sort_keys=False
            )

    def _sync_prompt_to_langfuse(self, config: PromptConfig) -> None:
        """Sync a prompt to LangFuse."""
        if not self._langfuse_client:
            return

        try:
            # LangFuse SDK prompt creation would go here
            logger.debug("langfuse_prompt_sync", prompt_id=config.id)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning(
                "langfuse_prompt_sync_failed", prompt_id=config.id, error=str(e)
            )

    # =========================================================================
    # A/B TESTING
    # =========================================================================

    def get_ab_test_prompt(
        self, base_prompt_id: str, user_id: str | None = None
    ) -> PromptTemplate | None:
        """
        Get a prompt for A/B testing.

        Uses consistent hashing based on user_id to ensure
        the same user always gets the same variant.

        Args:
            base_prompt_id: Base prompt ID (e.g., 'tws-agent')
            user_id: User identifier for consistent assignment

        Returns:
            Selected PromptTemplate
        """
        # Find all variants
        variants = [
            config
            for config in self._prompts.values()
            if config.id.startswith(base_prompt_id) and config.is_active
        ]

        if not variants:
            return None

        if len(variants) == 1:
            return self._templates[variants[0].id]

        # Select based on user_id hash
        if user_id:
            hash_value = hash(user_id) % 100
            cumulative_weight = 0

            for variant in sorted(variants, key=lambda x: x.id):
                cumulative_weight += variant.ab_test_weight * 100
                if hash_value < cumulative_weight:
                    return self._templates[variant.id]

        # Fallback to first active variant
        return self._templates[variants[0].id]

# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_prompt_manager: PromptManager | None = None

@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Get or create the singleton prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager

async def initialize_prompt_manager() -> PromptManager:
    """Initialize and return the prompt manager."""
    pm = get_prompt_manager()
    await pm.initialize()
    return pm
