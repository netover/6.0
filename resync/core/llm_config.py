"""
Central LLM Configuration Provider

Provides LLM model and settings to ALL agents, specialists, and LLM consumers.
Ensures NO hardcoded models - everything uses central config.

Author: Resync Team
Version: 6.1.2
"""

from pathlib import Path
from typing import Any, Optional

# Python 3.11+ baseline: stdlib tomllib is always available.
import tomllib

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class LLMConfig:
    """
    Central LLM configuration.

    All specialists, agents, and LLM consumers MUST use this
    instead of hardcoded models.

    Usage:
        config = get_llm_config()
        model = config.get_model()  # Returns central model

        # For specialist:
        model = config.get_model_for_task("analysis")
    """

    _instance: Optional["LLMConfig"] = None
    _config: dict[str, Any] = {}

    def __init__(self):
        """Initialize LLM config."""
        self._load_config()

    def _load_config(self):
        """Load LLM configuration from TOML file using standard tomllib."""
        try:
            # Resolved dynamically relative to the project structure
            config_file = (
                Path(__file__).resolve().parent.parent.parent / "config" / "llm.toml"
            )

            if not config_file.exists():
                logger.warning(
                    "llm_config_not_found",
                    path=str(config_file),
                    fallback="using_defaults",
                )
                self._config = self._get_defaults()
                return

            with open(config_file, "rb") as f:
                self._config = tomllib.load(f)

            logger.info(
                "llm_config_loaded",
                provider=self._config.get("llm", {}).get("provider"),
                default_model=self._config.get("llm", {}).get("default_model"),
            )

        except Exception as e:
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("llm_config_load_failed", error=str(e), exc_info=True)
            self._config = self._get_defaults()

    def _get_defaults(self) -> dict[str, Any]:
        """Get default LLM configuration."""
        return {
            "llm": {
                "provider": "litellm",
                "default_model": "ollama/llama3.2",
                "fallback_model": "ollama/llama3.2",
                "litellm": {
                    "enabled": True,
                    "base_url": "http://localhost:11434",
                    "timeout": 60,
                },
            }
        }

    def get_model(self, task_type: str | None = None) -> str:
        """
        Get model name for LLM calls.

        This is the SINGLE source of truth for model selection.
        NO hardcoded models allowed!

        Args:
            task_type: Optional task type for routing
                      (analysis, dependencies, resources, knowledge, chat)

        Returns:
            Model name (e.g., "ollama/llama3.2")

        Example:
            >>> config = get_llm_config()
            >>> model = config.get_model()
            >>> # Use model for LLM call
        """
        llm_config = self._config.get("llm", {})

        # Check routing rules if enabled
        if task_type and llm_config.get("routing", {}).get("enabled"):
            routed_model = self._get_routed_model(task_type)
            if routed_model:
                return routed_model

        # Return default model
        default_model = llm_config.get("default_model", "ollama/llama3.2")

        logger.debug("llm_model_selected", model=default_model, task_type=task_type)

        return default_model

    def _get_routed_model(self, task_type: str) -> str | None:
        """Get routed model based on task type."""
        routing = self._config.get("llm", {}).get("routing", {})
        rules = routing.get("rules", {})

        for _rule_name, rule in rules.items():
            if task_type in rule.get("task_types", []):
                return rule.get("model")

        return None

    def get_temperature(self, task_type: str, default: float = 0.3) -> float:
        """
        Get temperature for task type.

        Args:
            task_type: Task type (analysis, dependencies, etc)
            default: Default if not configured

        Returns:
            Temperature value
        """
        temperatures = self._config.get("llm", {}).get("temperatures", {})
        return temperatures.get(task_type, default)

    def get_max_tokens(self, task_type: str, default: int = 2048) -> int:
        """
        Get max tokens for task type.

        Args:
            task_type: Task type
            default: Default if not configured

        Returns:
            Max tokens
        """
        token_limits = self._config.get("llm", {}).get("token_limits", {})
        return token_limits.get(task_type, default)

    def get_provider_config(self) -> dict[str, Any]:
        """
        Get provider-specific configuration.

        Returns:
            Provider config (litellm, openai, etc)
        """
        llm_config = self._config.get("llm", {})
        provider = llm_config.get("provider", "litellm")

        return llm_config.get(provider, {})

    def get_base_url(self) -> str:
        """
        Get LLM base URL.

        Returns:
            Base URL for LLM provider
        """
        provider_config = self.get_provider_config()
        return provider_config.get("base_url", "http://localhost:11434")

    def is_ollama(self) -> bool:
        """Check if using Ollama."""
        model = self.get_model()
        return model.startswith("ollama/")

    def get_cache_config(self) -> dict[str, Any]:
        """Get cache configuration."""
        return self._config.get("llm", {}).get(
            "cache", {"enabled": True, "ttl_seconds": 3600}
        )

    def get_retry_config(self) -> dict[str, Any]:
        """Get retry configuration."""
        return self._config.get("llm", {}).get(
            "retry", {"max_attempts": 3, "base_backoff": 1.0, "max_backoff": 10.0}
        )

    def reload(self):
        """Reload configuration from file (hot reload support)."""
        self._load_config()
        logger.info("llm_config_reloaded")


def get_llm_config() -> LLMConfig:
    """
    Get global LLM configuration instance.

    This is the SINGLE point of access for LLM configuration.
    All agents, specialists, and LLM consumers MUST use this.

    Returns:
        LLMConfig instance

    Example:
        >>> from resync.core.llm_config import get_llm_config
        >>>
        >>> config = get_llm_config()
        >>> model = config.get_model()
        >>>
        >>> # Use model for LLM call
        >>> response = llm_client.chat(model=model, ...)
    """
    if LLMConfig._instance is None:
        LLMConfig._instance = LLMConfig()

    return LLMConfig._instance


def get_model_for_specialist(specialist_type: str) -> str:
    """
    Get model for specialist.

    Maps specialist types to task types for routing.

    Args:
        specialist_type: Type of specialist
                        (job_analyst, dependency_specialist, etc)

    Returns:
        Model name

    Example:
        >>> model = get_model_for_specialist("job_analyst")
        >>> # Returns: "ollama/llama3.2"
    """
    # Map specialist to task type
    task_type_map = {
        "job_analyst": "analysis",
        "dependency_specialist": "dependencies",
        "resource_specialist": "resources",
        "knowledge_specialist": "knowledge",
    }

    task_type = task_type_map.get(specialist_type, "analysis")

    config = get_llm_config()
    return config.get_model(task_type)


# Auto-reload on config file change (for hot reload)
def _on_llm_config_change(event):
    """Called by UnifiedConfigManager when llm.toml changes."""
    config = get_llm_config()
    config.reload()
    logger.info("llm_config_hot_reloaded")
