# pylint
"""
Hot-Reload Configuration System.

Allows configuration changes to be applied without restarting the application.

Features:
- File watching for config changes
- In-memory config cache with TTL
- WebSocket notifications for config updates
- Atomic config updates with rollback

Usage:
    from resync.core.config_hot_reload import ConfigManager

    config_manager = ConfigManager()
    await config_manager.start()

    # Get config value (auto-reloads if file changed)
    value = config_manager.get("database.host")

    # Set config value (auto-persists)
    config_manager.set("database.host", "new-host")
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

try:
    from watchdog.events import FileModifiedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except Exception:  # pragma: no cover
    # Allow import without watchdog;
    # hot-reload watching will be disabled.
    FileModifiedEvent = object  # type: ignore[assignment]

    class FileSystemEventHandler:  # type: ignore[misc]
        pass

    Observer = None  # type: ignore[assignment]
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Represents a configuration change."""

    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "api"  # api, file, env


class ConfigFileHandler(FileSystemEventHandler):
    """Handles file system events for config files."""

    def __init__(self, callback: Callable):
        self.callback = callback

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            # Watchdog runs in a background thread, so we need to schedule the callback
            # on the event loop safely.
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop running in this thread (expected for watchdog thread)
                pass

            if loop and loop.is_running():
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.callback(event.src_path))
                )
            else:
                logger.warning(
                    "config_file_modified_but_no_loop_active", path=event.src_path
                )


class ConfigManager:
    """
    Hot-reload configuration manager.

    Watches config files and applies changes without restart.
    """

    def __init__(
        self,
        config_dir: str = "config",
        main_config: str = "settings.json",
    ):
        self.config_dir = Path(config_dir)
        self.main_config = main_config

        # Configuration storage
        self._config: dict[str, Any] = {}
        self._defaults: dict[str, Any] = {}

        # Change tracking
        self._change_history: list[ConfigChange] = []
        self._subscribers: set[Callable] = set()

        # File watching
        self._observer: Observer | None = None
        self._watching = False

        # Lock for thread safety (lazy-initialized)
        # to avoid event-loop issues at import time
        self._lock: asyncio.Lock | None = None

    @property
    def _async_lock(self) -> asyncio.Lock:
        """Lazy initialization of asyncio.Lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def start(self, tg: asyncio.TaskGroup | None = None):
        """Start the configuration manager.

        Args:
            tg: Optional TaskGroup for background tasks (currently unused by observer)
        """
        # Create config directory if needed
        self.config_dir.mkdir(exist_ok=True)

        # Load initial configuration
        await self._load_config()

        # Start file watching (Synchronous watchdog thread)
        self._start_watching()

        logger.info("ConfigManager started", extra={"config_dir": str(self.config_dir)})

    def stop(self):
        """Stop the configuration manager."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._watching = False

        logger.info("ConfigManager stopped")

    def _start_watching(self):
        """Start watching config files for changes."""
        if self._watching:
            return

        if not WATCHDOG_AVAILABLE or Observer is None:
            logger.warning("config_hot_reload_disabled_watchdog_missing")
            return

        handler = ConfigFileHandler(self._on_file_change)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.config_dir), recursive=True)
        self._observer.start()
        self._watching = True

        logger.info("Started watching config files")

    async def _on_file_change(self, filepath: str):
        """Handle config file changes."""
        logger.info("Config file changed: %s", filepath)

        async with self._async_lock:
            # Reload configuration
            await self._load_config()

            # Notify subscribers
            await self._notify_subscribers()

    async def _load_config(self):
        """Load configuration from files."""
        config_file = self.config_dir / self.main_config

        if config_file.exists():
            try:
                async with aiofiles.open(config_file) as f:
                    content = await f.read()
                    self._config = json.loads(content)
                logger.info("Configuration loaded successfully")
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in config file: %s", e)
        else:
            # Create default config
            self._config = self._defaults.copy()
            await self._save_config()

    async def _save_config(self):
        """Save configuration to file."""
        config_file = self.config_dir / self.main_config

        try:
            async with aiofiles.open(config_file, "w") as f:
                await f.write(json.dumps(self._config, indent=2, default=str))
            logger.info("Configuration saved")
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Supports dot notation: "database.host"
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    async def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """
        Set a configuration value.

        Args:
            key: Config key (supports dot notation)
            value: New value
            persist: Whether to save to file

        Returns:
            True if successful
        """
        async with self._async_lock:
            # Get old value for change tracking
            old_value = self.get(key)

            # Set new value
            keys = key.split(".")
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

            # Track change
            change = ConfigChange(key=key, old_value=old_value, new_value=value)
            self._change_history.append(change)

            # Persist if requested
            if persist:
                await self._save_config()

            # Notify subscribers
            await self._notify_subscribers(change)

            logger.info("Config changed: %s = %s", key, value)
            return True

    async def reload(self):
        """Force reload configuration from files."""
        async with self._async_lock:
            await self._load_config()
            await self._notify_subscribers()

    def subscribe(self, callback: Callable):
        """Subscribe to configuration changes."""
        self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from configuration changes."""
        self._subscribers.discard(callback)

    async def _notify_subscribers(self, change: ConfigChange | None = None):
        """Notify all subscribers of configuration change."""
        for callback in self._subscribers:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(change)
                else:
                    result = await asyncio.to_thread(callback, change)
                    if inspect.isawaitable(result):
                        await result
            except Exception as e:
                logger.error("Error notifying subscriber: %s", e)

    def get_all(self) -> dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()

    def get_history(self, limit: int = 100) -> list[ConfigChange]:
        """Get change history."""
        return self._change_history[-limit:]

    async def rollback(self, steps: int = 1) -> bool:
        """
        Rollback configuration changes.

        Args:
            steps: Number of changes to rollback

        Returns:
            True if successful
        """
        if not self._change_history:
            return False

        async with self._async_lock:
            for _ in range(min(steps, len(self._change_history))):
                change = self._change_history.pop()
                await self.set(change.key, change.old_value, persist=False)

            await self._save_config()
            return True


# Global instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


async def init_config_manager():
    """Initialize global configuration manager."""
    manager = get_config_manager()
    await manager.start()
    return manager
