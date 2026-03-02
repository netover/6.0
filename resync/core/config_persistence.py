# pylint
"""Configuration Persistence Manager for Resync.

This module handles persistent storage of configuration changes,
ensuring that modifications survive application restarts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import toml
except ImportError:
    import tomli as toml
    import tomli_w

logger = logging.getLogger(__name__)

class ConfigPersistenceError(Exception):
    """Base exception for configuration persistence errors."""

class ConfigPersistenceManager:
    """Manages persistent storage of application configuration.

    This class provides safe, atomic updates to configuration files
    with automatic backup and rollback capabilities.

    Attributes:
        config_file: Path to the configuration file
        backup_dir: Directory for configuration backups
        max_backups: Maximum number of backup files to retain
    """

    def __init__(
        self,
        config_file: Path | str,
        backup_dir: Path | str | None = None,
        max_backups: int = 10,
    ):
        """Initialize the configuration persistence manager.

        Args:
            config_file: Path to the TOML configuration file
            backup_dir: Directory for backups (default: same as config file)
            max_backups: Maximum number of backup files to keep
        """
        self.config_file = Path(config_file)
        self.backup_dir = (
            Path(backup_dir) if backup_dir else self.config_file.parent / "backups"
        )
        self.max_backups = max_backups

        self._validate_config_file()
        self._ensure_backup_dir()

    def _validate_config_file(self) -> None:
        """Validate that configuration file exists and is accessible.

        Raises:
            ConfigPersistenceError: If file doesn't exist or isn't writable
        """
        if not self.config_file.exists():
            raise ConfigPersistenceError(
                f"Configuration file not found: {self.config_file}"
            )

        if not os.access(self.config_file, os.W_OK):
            raise ConfigPersistenceError(
                f"Configuration file not writable: {self.config_file}"
            )

    def _ensure_backup_dir(self) -> None:
        """Ensure backup directory exists."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def load_config(self) -> dict[str, Any]:
        """Load current configuration from file.

        Returns:
            Dictionary containing the full configuration

        Raises:
            ConfigPersistenceError: If loading fails
        """
        try:
            return await asyncio.to_thread(self._load_config_sync)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to load configuration: %s", e, exc_info=True)
            raise ConfigPersistenceError(f"Failed to load config: {e}") from e

    def _load_config_sync(self) -> dict[str, Any]:
        """Synchronous implementation of load_config."""
        with open(self.config_file, encoding="utf-8") as f:
            return toml.load(f)

    async def save_config(
        self, section: str, data: dict[str, Any], create_backup: bool = True
    ) -> None:
        """Save configuration section to file with atomic write.

        Args:
            section: Configuration section name (e.g., 'teams', 'tws', 'system')
            data: Configuration data to save
            create_backup: Whether to create backup before saving

        Raises:
            ConfigPersistenceError: If save operation fails
        """
        try:
            await asyncio.to_thread(
                self._save_config_sync, section, data, create_backup
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("Failed to save configuration: %s", e, exc_info=True)
            raise ConfigPersistenceError(f"Failed to save config: {e}") from e

    def _save_config_sync(
        self, section: str, data: dict[str, Any], create_backup: bool
    ) -> None:
        """Synchronous implementation of save_config."""
        backup_file = None

        try:
            # Load current configuration
            current_config = self._load_config_sync()

            # Create backup if requested
            if create_backup:
                backup_file = self._create_backup_sync()

            # Update section
            if section not in current_config:
                current_config[section] = {}

            # Merge new data (preserve existing keys not in update)
            current_config[section].update(data)

            # Write to temporary file first (atomic write)
            temp_file = self.config_file.with_suffix(".tmp")

            # tomli_w.dump() requires a *binary* file handle, while toml.dump()
            # expects a text handle.
            if "tomli_w" in globals():
                with open(temp_file, "wb") as f:
                    tomli_w.dump(current_config, f)
            else:
                with open(temp_file, "w", encoding="utf-8") as f:
                    toml.dump(current_config, f)

            # Atomic replace
            temp_file.replace(self.config_file)

            logger.info(
                "Configuration saved successfully: "
                f"section={section}, keys={list(data.keys())}"
            )

            # Clean old backups
            self._cleanup_old_backups_sync()

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            # Restore from backup if save failed
            if backup_file and backup_file.exists():
                try:
                    shutil.copy2(backup_file, self.config_file)
                    logger.info("Configuration restored from backup after failure")
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as restore_error:
                    import sys as _sys
                    from resync.core.exception_guard import maybe_reraise_programming_error
                    _exc_type, _exc, _tb = _sys.exc_info()
                    maybe_reraise_programming_error(_exc, _tb)

                    logger.error("Failed to restore backup: %s", restore_error)
            raise

    async def _create_backup(self) -> Path:
        """Create timestamped backup of configuration file (async)."""
        return await asyncio.to_thread(self._create_backup_sync)

    def _create_backup_sync(self) -> Path:
        """Create timestamped backup of configuration file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{self.config_file.stem}_{timestamp}.toml.bak"

        shutil.copy2(self.config_file, backup_file)
        logger.debug("Created configuration backup: %s", backup_file)

        return backup_file

    async def _cleanup_old_backups(self) -> None:
        """Remove old backup files (async)."""
        await asyncio.to_thread(self._cleanup_old_backups_sync)

    def _cleanup_old_backups_sync(self) -> None:
        """Remove old backup files, keeping only the most recent ones."""
        backups = sorted(
            self.backup_dir.glob(f"{self.config_file.stem}_*.toml.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove backups beyond max_backups limit
        for old_backup in backups[self.max_backups :]:
            try:
                old_backup.unlink()
                logger.debug("Removed old backup: %s", old_backup)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning("Failed to remove old backup %s: %s", old_backup, e)

    async def get_section(self, section: str) -> dict[str, Any]:
        """Get specific configuration section.

        Args:
            section: Section name to retrieve

        Returns:
            Configuration section dictionary

        Raises:
            ConfigPersistenceError: If section doesn't exist
        """
        config = await self.load_config()

        if section not in config:
            raise ConfigPersistenceError(f"Configuration section not found: {section}")

        return config[section]

    async def update_key(self, section: str, key: str, value: Any) -> None:
        """Update a single configuration key.

        Args:
            section: Configuration section
            key: Key to update
            value: New value

        Raises:
            ConfigPersistenceError: If update fails
        """
        await self.save_config(section, {key: value})

    async def list_backups(self) -> list[Path]:
        """List all available backup files.

        Returns:
            List of backup file paths, sorted by modification time (newest first)
        """
        return await asyncio.to_thread(self._list_backups_sync)

    def _list_backups_sync(self) -> list[Path]:
        """Synchronous implementation of list_backups."""
        return sorted(
            self.backup_dir.glob(f"{self.config_file.stem}_*.toml.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    async def restore_backup(self, backup_file: Path) -> None:
        """Restore configuration from a specific backup.

        Args:
            backup_file: Path to the backup file to restore

        Raises:
            ConfigPersistenceError: If restore fails
        """
        try:
            await asyncio.to_thread(self._restore_backup_sync, backup_file)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("Failed to restore backup: %s", e, exc_info=True)
            raise ConfigPersistenceError(f"Failed to restore backup: {e}") from e

    def _restore_backup_sync(self, backup_file: Path) -> None:
        """Synchronous implementation of restore_backup."""
        if not backup_file.exists():
            raise ConfigPersistenceError(f"Backup file not found: {backup_file}")

        try:
            # Create backup of current config before restoring
            current_backup = self._create_backup_sync()

            # Restore from backup
            shutil.copy2(backup_file, self.config_file)

            logger.info(
                f"Configuration restored from backup: {backup_file.name}, "
                f"previous config backed up to: {current_backup.name}"
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            raise

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate configuration structure and values.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Basic structure validation
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors

        # Add more specific validations as needed
        # This is a basic implementation - extend based on your needs

        return len(errors) == 0, errors
