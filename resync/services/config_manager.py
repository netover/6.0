"""
Unified Configuration Manager Service

Manages configuration with clear precedence:
1. Environment Variables (read-only, highest priority)
2. Database (read/write, editable via UI)
3. YAML/JSON files (defaults, lowest priority)

Supports hot-reload where possible and signals restart requirements.

Part of Admin Interface 2.0 - Resync v5.4.2
"""

import builtins
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml
import aiofiles

logger = structlog.get_logger(__name__)


class ConfigSource(Enum):
    """Configuration source types"""

    ENVIRONMENT = "environment"
    DATABASE = "database"
    FILE = "file"
    DEFAULT = "default"


class RestartRequirement(Enum):
    """Whether a config change requires restart"""

    NONE = "none"  # Hot-reload supported
    GRACEFUL = "graceful"  # Can wait for restart
    IMMEDIATE = "immediate"  # Should restart ASAP


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata"""

    key: str
    value: Any
    source: ConfigSource
    editable: bool
    restart_required: RestartRequirement
    last_modified: datetime | None = None
    modified_by: str | None = None
    description: str = ""


@dataclass
class ConfigSection:
    """A section of related configurations"""

    name: str
    description: str
    values: dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class ConfigChangeEvent:
    """Event emitted when config changes"""

    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime
    user: str | None = None
    restart_required: RestartRequirement = RestartRequirement.NONE


class ConfigManager:
    """
    Unified Configuration Manager

    Manages all application configuration with:
    - Clear precedence (env > db > file > defaults)
    - Hot-reload support where possible
    - Change tracking and auditing
    - Restart requirement signaling
    """

    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config_cache: dict[str, ConfigValue] = {}
        self._change_listeners: list[Callable[[ConfigChangeEvent], None]] = []
        self._pending_restart: set[str] = set()  # Keys that need restart
        self._db_config: dict[str, Any] = {}
        self._file_config: dict[str, Any] = {}

        # Configuration file paths
        self._config_dir = Path(__file__).parent.parent.parent / "config"
        self._admin_config_path = self._config_dir / "admin_config.json"
        self._redis_strategy_path = self._config_dir / "redis_strategy.yaml"
        self._agents_config_path = self._config_dir / "agents.yaml"

        # Define configuration schema with defaults and metadata
        self._schema = self._build_schema()

        self._initialized = True
        logger.info("ConfigManager initialized")

    def _build_schema(self) -> dict[str, dict[str, Any]]:
        """Build configuration schema with metadata"""
        return {
            # Redis Configuration
            "redis.url": {
                "default": "redis://localhost:6379",
                "env_var": "REDIS_URL",
                "restart": RestartRequirement.GRACEFUL,
                "description": "Redis connection URL",
            },
            "redis.fail_fast_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable Redis fail-fast strategy",
            },
            "redis.fail_fast_timeout": {
                "default": 5.0,
                "restart": RestartRequirement.NONE,
                "description": "Fail-fast timeout in seconds",
            },
            # LLM Configuration
            "llm.primary_model": {
                "default": "gpt-4",
                "env_var": "LLM_MODEL",
                "restart": RestartRequirement.NONE,
                "description": "Primary LLM model",
            },
            "llm.fallback_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable LLM fallback chain",
            },
            "llm.timeout": {
                "default": 30,
                "restart": RestartRequirement.NONE,
                "description": "LLM request timeout",
            },
            # TWS Configuration
            "tws.primary_instance": {
                "default": "TWS_NAZ",
                "restart": RestartRequirement.GRACEFUL,
                "description": "Primary TWS instance",
            },
            "tws.monitored_instances": {
                "default": ["TWS_NAZ", "TWS_SAZ"],
                "restart": RestartRequirement.NONE,
                "description": "List of monitored TWS instances",
            },
            "tws.circuit_breaker_threshold": {
                "default": 5,
                "restart": RestartRequirement.NONE,
                "description": "TWS circuit breaker failure threshold",
            },
            # RAG Configuration
            "rag.chunking_strategy": {
                "default": "tws_optimized",
                "restart": RestartRequirement.GRACEFUL,
                "description": "Chunking strategy: tws_optimized, hierarchical, semantic",
            },
            "rag.chunk_size": {
                "default": 512,
                "restart": RestartRequirement.GRACEFUL,
                "description": "Default chunk size in tokens",
            },
            "rag.chunk_overlap": {
                "default": 50,
                "restart": RestartRequirement.GRACEFUL,
                "description": "Chunk overlap in tokens",
            },
            "rag.reranker_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable cross-encoder reranking",
            },
            # Teams Integration
            "teams.enabled": {
                "default": False,
                "restart": RestartRequirement.NONE,
                "description": "Enable Microsoft Teams integration",
            },
            "teams.webhook_url": {
                "default": "",
                "restart": RestartRequirement.NONE,
                "description": "Teams webhook URL",
            },
            # System Settings
            "system.environment": {
                "default": "production",
                "env_var": "ENVIRONMENT",
                "restart": RestartRequirement.IMMEDIATE,
                "description": "Deployment environment",
            },
            "system.debug_mode": {
                "default": False,
                "env_var": "DEBUG",
                "restart": RestartRequirement.GRACEFUL,
                "description": "Enable debug mode",
            },
            "system.maintenance_mode": {
                "default": False,
                "restart": RestartRequirement.NONE,
                "description": "Enable maintenance mode (blocks user traffic)",
            },
            # Security Settings
            "security.csp_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable Content Security Policy",
            },
            "security.cors_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable CORS",
            },
            # Observability
            "observability.metrics_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable metrics collection",
            },
            "observability.tracing_enabled": {
                "default": True,
                "restart": RestartRequirement.NONE,
                "description": "Enable distributed tracing",
            },
        }

    async def initialize(self) -> None:
        """Initialize configuration from all sources"""
        # Load file configurations
        await self._load_file_configs()

        # DEBT: Load configurations from database (currently loaded from YAML/JSON files only)
        # await self._load_db_configs()

        # Build config cache with precedence
        self._rebuild_cache()

        logger.info(
            "ConfigManager fully initialized", config_count=len(self._config_cache)
        )

    async def _load_file_configs(self) -> None:
        """Load configurations from files"""
        # Load admin config JSON
        if self._admin_config_path.exists():
            try:
                async with aiofiles.open(self._admin_config_path) as f:
                    content = await f.read()
                    data = json.loads(content) if content.strip() else {}
                    self._flatten_config(data, self._file_config, "")
            except Exception as e:
                logger.warning("Failed to load admin config", error=str(e))

        # Load redis strategy YAML
        if self._redis_strategy_path.exists():
            try:
                async with aiofiles.open(self._redis_strategy_path) as f:
                    content = await f.read()
                    data = yaml.safe_load(content) if content.strip() else None
                    if data:
                        self._flatten_config(data, self._file_config, "redis.")
            except Exception as e:
                logger.warning("Failed to load redis strategy", error=str(e))

        # Load agents config YAML
        if self._agents_config_path.exists():
            try:
                async with aiofiles.open(self._agents_config_path) as f:
                    content = await f.read()
                    data = yaml.safe_load(content) if content.strip() else None
                    if data:
                        self._flatten_config(data, self._file_config, "agents.")
            except Exception as e:
                logger.warning("Failed to load agents config", error=str(e))

    def _flatten_config(self, data: dict, target: dict, prefix: str) -> None:
        """Flatten nested dict into dot-notation keys"""
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_config(value, target, f"{full_key}.")
            else:
                target[full_key] = value

    def _rebuild_cache(self) -> None:
        """Rebuild config cache applying precedence"""
        self._config_cache.clear()

        for key, schema in self._schema.items():
            # Check precedence: env > db > file > default
            value = schema.get("default")
            source = ConfigSource.DEFAULT

            # Check file config
            if key in self._file_config:
                value = self._file_config[key]
                source = ConfigSource.FILE

            # Check database config
            if key in self._db_config:
                value = self._db_config[key]
                source = ConfigSource.DATABASE

            # Check environment variable (highest priority)
            env_var = schema.get("env_var")
            if env_var and env_var in os.environ:
                env_value = os.environ[env_var]
                # Type coercion based on default type
                default_type = type(schema.get("default"))
                if default_type is bool:
                    value = env_value.lower() in ("true", "1", "yes")
                elif default_type is int:
                    value = int(env_value)
                elif default_type is float:
                    value = float(env_value)
                else:
                    value = env_value
                source = ConfigSource.ENVIRONMENT

            self._config_cache[key] = ConfigValue(
                key=key,
                value=value,
                source=source,
                editable=source != ConfigSource.ENVIRONMENT,
                restart_required=schema.get("restart", RestartRequirement.NONE),
                description=schema.get("description", ""),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        if key in self._config_cache:
            return self._config_cache[key].value
        return default

    def get_with_metadata(self, key: str) -> ConfigValue | None:
        """Get configuration value with full metadata"""
        return self._config_cache.get(key)

    async def set(
        self, key: str, value: Any, user: str | None = None, persist: bool = True
    ) -> ConfigChangeEvent:
        """
        Set a configuration value

        Args:
            key: Configuration key
            value: New value
            user: User making the change
            persist: Whether to persist to database

        Returns:
            ConfigChangeEvent describing the change

        Raises:
            ValueError: If key is not editable (env var)
        """
        current = self._config_cache.get(key)

        if current and current.source == ConfigSource.ENVIRONMENT:
            raise ValueError(
                f"Configuration '{key}' is set via environment variable and cannot be modified"
            )

        schema = self._schema.get(key, {})
        restart_req = schema.get("restart", RestartRequirement.NONE)

        old_value = current.value if current else None

        # Create change event
        event = ConfigChangeEvent(
            key=key,
            old_value=old_value,
            new_value=value,
            source=ConfigSource.DATABASE if persist else ConfigSource.FILE,
            timestamp=datetime.now(timezone.utc),
            user=user,
            restart_required=restart_req,
        )

        # Update cache
        self._config_cache[key] = ConfigValue(
            key=key,
            value=value,
            source=ConfigSource.DATABASE if persist else ConfigSource.FILE,
            editable=True,
            restart_required=restart_req,
            last_modified=event.timestamp,
            modified_by=user,
            description=schema.get("description", ""),
        )

        # Track if restart needed
        if restart_req != RestartRequirement.NONE:
            self._pending_restart.add(key)

        # Persist if requested
        if persist:
            await self._persist_to_file(key, value)

        # Notify listeners
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("Config change listener failed", error=str(e))

        logger.info(
            "Configuration updated",
            key=key,
            old_value=old_value,
            new_value=value,
            user=user,
            restart_required=restart_req.value,
        )

        return event

    async def _persist_to_file(self, key: str, value: Any) -> None:
        """Persist configuration change to file"""
        # Load current file config
        config = {}
        if self._admin_config_path.exists():
            try:
                async with aiofiles.open(self._admin_config_path) as f:
                    content = await f.read()
                    config = json.loads(content) if content.strip() else {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("suppressed_exception", error=str(exc))

        # Update nested structure from dot notation
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

        # Save
        self._config_dir.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._admin_config_path, "w") as f:
            await f.write(json.dumps(config, indent=2, ensure_ascii=False))

    def get_all(self) -> dict[str, ConfigValue]:
        """Get all configuration values"""
        return self._config_cache.copy()

    def get_section(self, prefix: str) -> dict[str, ConfigValue]:
        """Get all configs in a section (by prefix)"""
        return {k: v for k, v in self._config_cache.items() if k.startswith(prefix)}

    def get_pending_restarts(self) -> builtins.set[str]:
        """Get keys that have changes requiring restart"""
        return self._pending_restart.copy()

    def clear_pending_restart(self, key: str) -> None:
        """Clear pending restart flag after restart"""
        self._pending_restart.discard(key)

    def requires_restart(self) -> bool:
        """Check if any pending changes require restart"""
        return len(self._pending_restart) > 0

    def get_restart_requirement(self) -> RestartRequirement:
        """Get the most urgent restart requirement"""
        if not self._pending_restart:
            return RestartRequirement.NONE

        # Check if any require immediate restart
        for key in self._pending_restart:
            config = self._config_cache.get(key)
            if config and config.restart_required == RestartRequirement.IMMEDIATE:
                return RestartRequirement.IMMEDIATE

        return RestartRequirement.GRACEFUL

    def add_change_listener(
        self, listener: Callable[[ConfigChangeEvent], None]
    ) -> None:
        """Add a listener for configuration changes"""
        self._change_listeners.append(listener)

    def remove_change_listener(
        self, listener: Callable[[ConfigChangeEvent], None]
    ) -> None:
        """Remove a configuration change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)


# Singleton access
_config_manager: ConfigManager | None = None


async def get_config_manager() -> ConfigManager:
    """Get or create the configuration manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        await _config_manager.initialize()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Quick access to get a config value (sync)"""
    if _config_manager is None:
        # Return default if not initialized
        return default
    return _config_manager.get(key, default)
