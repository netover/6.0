# pylint: disable=all
"""
Unified Configuration Manager with Hot Reload

Manages ALL Resync configurations with:
- Single source of truth (TOML files)
- Hot reload (file watcher)
- Persistence across restarts
- Atomic updates with backups
- Validation

Author: Resync Team
Version: 5.9.9
"""

import asyncio
import inspect
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import toml

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    # Fallback for environments without watchdog (hot reload disabled)
    FileSystemEventHandler = object
    Observer = None

import aiofiles

from resync.core.config_persistence import ConfigPersistenceManager

logger = structlog.get_logger(__name__)


class ConfigChangeEvent:
    """Event for configuration changes."""

    def __init__(self, section: str, old_value: Any, new_value: Any):
        self.section = section
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = datetime.now(timezone.utc)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config changes."""

    def __init__(self, config_manager: "UnifiedConfigManager"):
        self.config_manager = config_manager
        self.last_modified = {}

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process TOML files
        if file_path.suffix != ".toml":
            return

        # Debounce - ignore rapid successive events
        now = datetime.now(timezone.utc)
        last_mod = self.last_modified.get(file_path)

        if last_mod and (now - last_mod).total_seconds() < 1.0:
            return

        self.last_modified[file_path] = now

        # Trigger reload.
        # NOTE: watchdog callbacks run on a background thread. We must schedule
        # the coroutine on the application's running event loop.
        self.config_manager._schedule_reload_from_thread(file_path)


class UnifiedConfigManager:
    """
    Central configuration manager for ALL Resync configs.

    Features:
    - Hot reload (auto-detect file changes)
    - Persistence (TOML files)
    - Validation
    - Change notifications
    - Atomic updates with backups
    """

    # Config file paths
    CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

    CONFIG_FILES = {
        "graphrag": CONFIG_DIR / "graphrag.toml",
        "ai": CONFIG_DIR / "ai.toml",
        "monitoring": CONFIG_DIR / "monitoring.toml",
        "system": CONFIG_DIR / "system.toml",
        "llm": CONFIG_DIR / "llm.toml",
    }

    def __init__(self):
        """Initialize unified config manager."""
        self.configs: dict[str, dict] = {}
        self.persistence_managers: dict[str, ConfigPersistenceManager] = {}
        self.change_callbacks: dict[str, list[Callable]] = {}
        self.observer: Observer | None = None

        # The running asyncio event loop used for scheduling hot-reload tasks.
        # Populated when hot reload is started from within the app.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Ensure config directory exists
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize persistence managers
        for name, path in self.CONFIG_FILES.items():
            if path.exists():
                self.persistence_managers[name] = ConfigPersistenceManager(
                    config_file=path, backup_dir=self.CONFIG_DIR / "backups"
                )

        logger.info("UnifiedConfigManager initialized")

    def start_hot_reload(self):
        """Start file watcher for hot reload."""
        if self.observer:
            logger.warning("Hot reload already started")
            return

        # Capture the application's running event loop so the watchdog thread
        # can schedule reload coroutines safely.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # Hot reload can still be used in non-async contexts, but file...
            self._loop = None

        if Observer is None:
            logger.warning("Watchdog not installed. Hot reload unavailable.")
            return

        event_handler = ConfigFileHandler(self)
        self.observer = Observer()

        self.observer.schedule(
            event_handler, path=str(self.CONFIG_DIR), recursive=False
        )

        self.observer.start()
        logger.info("Hot reload started - watching config directory")

    def _schedule_reload_from_thread(self, file_path: Path) -> None:
        """Schedule a config reload from the watchdog thread.

        Watchdog callbacks run in a separate thread. This helper uses
        ``asyncio.run_coroutine_threadsafe`` to submit the reload coroutine to
        the captured running loop.
        """
        if self._loop is None or not self._loop.is_running():
            logger.warning(
                "hot_reload_no_running_loop",
                file_path=file_path,
            )
            return

        future = asyncio.run_coroutine_threadsafe(
            self.reload_config_file(file_path), self._loop
        )

        def _done_callback(fut):
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover
                logger.error("hot_reload_failed", file_path=file_path, error=str(exc))

        future.add_done_callback(_done_callback)

    def stop_hot_reload(self):
        """Stop file watcher."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Hot reload stopped")

    async def load_all_configs(self):
        """Load all configuration files."""
        for name, path in self.CONFIG_FILES.items():
            if path.exists():
                await self.reload_config_file(path)
            else:
                logger.warning("Config file not found: %s", path)
                self.configs[name] = {}

    async def reload_config_file(self, file_path: Path):
        """
        Reload a specific config file and notify listeners.

        This is called automatically by file watcher (hot reload).
        """
        try:
            # Find config name
            config_name = None
            for name, path in self.CONFIG_FILES.items():
                if path == file_path:
                    config_name = name
                    break

            if not config_name:
                logger.debug("Ignoring unknown config file: %s", file_path)
                return

            # Load new config
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                new_config = toml.loads(content)
            # Get old config
            old_config = self.configs.get(config_name, {})

            # Update in-memory config
            self.configs[config_name] = new_config

            logger.info(
                f"Config reloaded (hot reload): {config_name}", file=str(file_path)
            )

            # Notify listeners
            await self._notify_change(config_name, old_config, new_config)

            # Apply changes to runtime
            await self._apply_config_changes(config_name, new_config)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "Failed to reload config file %s: %s", file_path, e, exc_info=True
            )

    async def _notify_change(self, config_name: str, old_value: Any, new_value: Any):
        """Notify registered callbacks about config change."""
        callbacks = self.change_callbacks.get(config_name, [])

        event = ConfigChangeEvent(config_name, old_value, new_value)

        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    result = await asyncio.to_thread(callback, event)
                    if inspect.isawaitable(result):
                        await result
            except Exception as e:
                logger.error("Config change callback failed: %s", e, exc_info=True)

    async def _apply_config_changes(self, config_name: str, new_config: dict):
        """Apply configuration changes to runtime objects."""
        try:
            if config_name == "graphrag":
                self._apply_graphrag_config(new_config)
            elif config_name == "ai":
                self._apply_ai_config(new_config)
            elif config_name == "monitoring":
                self._apply_monitoring_config(new_config)
            elif config_name == "system":
                self._apply_system_config(new_config)
            elif config_name == "llm":
                self._apply_llm_config(new_config)

            logger.info("Config changes applied: %s", config_name)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "Failed to apply config changes for %s: %s",
                config_name,
                e,
                exc_info=True,
            )

    def _apply_graphrag_config(self, config: dict):
        """Apply GraphRAG configuration changes."""
        from resync.core.event_driven_discovery import DiscoveryConfig
        from resync.core.smart_cache_validator import CacheValidationConfig

        graphrag = config.get("graphrag", {})

        # Budget
        budget = graphrag.get("budget", {})
        if "max_discoveries_per_day" in budget:
            DiscoveryConfig.MAX_DISCOVERIES_PER_DAY = budget["max_discoveries_per_day"]
        if "max_discoveries_per_hour" in budget:
            DiscoveryConfig.MAX_DISCOVERIES_PER_HOUR = budget[
                "max_discoveries_per_hour"
            ]

        # Cache
        cache = graphrag.get("cache", {})
        if "ttl_days" in cache:
            DiscoveryConfig.DISCOVERY_CACHE_DAYS = cache["ttl_days"]

        # Triggers
        triggers = graphrag.get("triggers", {})
        if "min_failures_to_trigger" in triggers:
            DiscoveryConfig.MIN_FAILURES_TO_TRIGGER = triggers[
                "min_failures_to_trigger"
            ]

        # Validation
        validation = graphrag.get("validation", {})
        if "validate_on_abend" in validation:
            CacheValidationConfig.VALIDATE_ON_ABEND = validation["validate_on_abend"]
        if "validate_on_failed" in validation:
            CacheValidationConfig.VALIDATE_ON_FAILED = validation["validate_on_failed"]
        if "auto_invalidate" in validation:
            CacheValidationConfig.AUTO_INVALIDATE = validation["auto_invalidate"]

        logger.info("GraphRAG config applied to runtime")

    def _apply_ai_config(self, config: dict):
        """Apply AI configuration changes."""
        # Update specialists config
        _specialists = config.get("specialists", {})  # TODO: use specialists config
        # This would update specialist manager settings
        # Implementation depends on how specialists are initialized

        logger.info("AI config applied to runtime")

    def _apply_monitoring_config(self, config: dict):
        """Apply monitoring configuration changes."""
        _monitoring = config.get("monitoring", {})  # TODO: use monitoring config
        # Update monitoring settings
        # This would call existing monitoring config update functions

        logger.info("Monitoring config applied to runtime")

    def _apply_system_config(self, config: dict):
        """Apply system configuration changes."""
        _system = config.get("system", {})  # TODO: use system config
        # Update system-wide settings
        # Rate limits, timeouts, etc.

        logger.info("System config applied to runtime")

    def _apply_llm_config(self, config: dict):
        """Apply LLM configuration changes."""
        from resync.core.llm_config import get_llm_config

        # Reload LLM config (hot reload)
        llm_config = get_llm_config()
        llm_config.reload()

        llm = config.get("llm", {})

        logger.info(
            "LLM config applied to runtime",
            provider=llm.get("provider"),
            default_model=llm.get("default_model"),
        )

    def register_change_callback(self, config_name: str, callback: Callable):
        """
        Register callback for config changes.

        Example:
            def on_graphrag_change(event):
                print(f"GraphRAG config changed: {event.section}")

            manager.register_change_callback("graphrag", on_graphrag_change)
        """
        if config_name not in self.change_callbacks:
            self.change_callbacks[config_name] = []

        self.change_callbacks[config_name].append(callback)
        logger.debug("Registered callback for %s", config_name)

    async def update_config(
        self, config_name: str, section: str, data: dict, create_backup: bool = True
    ):
        """
        Update configuration section and persist to file.

        Changes are applied immediately (hot) and saved to file.

        Args:
            config_name: Name of config (graphrag, ai, monitoring, etc)
            section: Section within config
            data: Data to update
            create_backup: Create backup before saving
        """
        if config_name not in self.CONFIG_FILES:
            raise ValueError(f"Unknown config: {config_name}")

        persistence = self.persistence_managers.get(config_name)

        if not persistence:
            raise RuntimeError(f"No persistence manager for {config_name}")

        # Get current config
        _cfg = self.configs.get(config_name, {})  # TODO: use returned config
        # Save to file
        await persistence.save_config(
            section=section, data=data, create_backup=create_backup
        )

        # Reload (triggers hot reload)
        config_file = self.CONFIG_FILES[config_name]
        await self.reload_config_file(config_file)

        logger.info(
            f"Config updated: {config_name}.{section}", fields=list(data.keys())
        )

    def get_config(self, config_name: str, section: str | None = None) -> dict:
        """
        Get configuration.

        Args:
            config_name: Name of config
            section: Specific section (optional)

        Returns:
            Configuration dict
        """
        config = self.configs.get(config_name, {})

        if section:
            return config.get(section, {})

        return config

    def get_all_configs(self) -> dict[str, dict]:
        """Get all configurations."""
        return self.configs.copy()


# Global instance
_config_manager: UnifiedConfigManager | None = None


def get_config_manager() -> UnifiedConfigManager:
    """Get global config manager instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = UnifiedConfigManager()

    return _config_manager


async def initialize_config_system():
    """
    Initialize configuration system.

    Call this on application startup.
    """
    manager = get_config_manager()

    # Load all configs
    await manager.load_all_configs()

    # Start hot reload
    manager.start_hot_reload()

    logger.info("Configuration system initialized with hot reload")


def shutdown_config_system():
    """
    Shutdown configuration system.

    Call this on application shutdown.
    """
    manager = get_config_manager()
    manager.stop_hot_reload()

    logger.info("Configuration system shutdown")
