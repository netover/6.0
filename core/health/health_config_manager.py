"""
Health Check Configuration Manager

This module provides comprehensive configuration management for health checks,
including thresholds, intervals, and component-specific settings.
"""

import os
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.health.health_models import HealthCheckConfig

logger = structlog.get_logger(__name__)


class HealthCheckConfigurationManager:
    """
    Manages comprehensive health check configuration.

    This class provides functionality for:
    - Managing health check intervals and timeouts
    - Configuring component-specific thresholds
    - Handling environment-based configuration
    - Providing configuration validation and defaults
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the configuration manager.

        Args:
            config: Base health check configuration (creates default if None)
        """
        self.config = config or self._create_default_config()
        self._config_history: list[dict[str, Any]] = []
        self._max_history_size = 100

    def _create_default_config(self) -> HealthCheckConfig:
        """Create default health check configuration."""
        return HealthCheckConfig(
            check_interval_seconds=30,
            timeout_seconds=30,
            database_connection_threshold_percent=80,
            alert_enabled=True,
            enable_memory_monitoring=True,
            max_history_entries=10000,
            history_retention_days=30,
            memory_usage_threshold_mb=100,
            history_cleanup_threshold=0.8,
            history_cleanup_batch_size=100,
        )

    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration parameters to update
        """
        old_config = self.config.model_dump()

        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("unknown_config_parameter", parameter=key, value=value)

        # Store in history
        self._add_to_config_history(old_config, kwargs)

        logger.info("health_check_config_updated", updated_parameters=list(kwargs.keys()))

    def get_config(self) -> HealthCheckConfig:
        """Get current configuration."""
        return self.config

    def get_component_config(self, component_name: str) -> dict[str, Any]:
        """
        Get configuration specific to a component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with component-specific configuration
        """
        config_dict = self.config.model_dump()
        component_configs = {
            "database": {
                "connection_threshold_percent": config_dict.get(
                    "database_connection_threshold_percent", 80
                ),
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 3,
            },
            "redis": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 2,
            },
            "cache_hierarchy": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 2,
            },
            "file_system": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "disk_space_warning_percent": 85,
                "disk_space_critical_percent": 95,
            },
            "memory": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "warning_percent": 85,
                "critical_percent": 95,
            },
            "cpu": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "warning_percent": 85,
                "critical_percent": 95,
            },
            "tws_monitor": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 1,
            },
            "connection_pools": {
                "connection_threshold_percent": config_dict.get(
                    "database_connection_threshold_percent", 80
                ),
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
            },
            "websocket_pool": {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 2,
            },
        }

        return component_configs.get(
            component_name,
            {
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "retry_attempts": 2,
            },
        )

    def load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "check_interval_seconds": ("HEALTH_CHECK_INTERVAL_SECONDS", int, 30),
            "timeout_seconds": ("HEALTH_CHECK_TIMEOUT_SECONDS", int, 30),
            "database_connection_threshold_percent": (
                "HEALTH_DB_THRESHOLD_PERCENT",
                int,
                80,
            ),
            "alert_enabled": (
                "HEALTH_ALERTS_ENABLED",
                lambda x: x.lower() == "true",
                True,
            ),
            "enable_memory_monitoring": (
                "HEALTH_MEMORY_MONITORING_ENABLED",
                lambda x: x.lower() == "true",
                True,
            ),
            "max_history_entries": ("HEALTH_MAX_HISTORY_ENTRIES", int, 10000),
            "history_retention_days": ("HEALTH_HISTORY_RETENTION_DAYS", int, 30),
            "memory_usage_threshold_mb": ("HEALTH_MEMORY_THRESHOLD_MB", float, 100.0),
            "history_cleanup_threshold": ("HEALTH_CLEANUP_THRESHOLD", float, 0.8),
            "cleanup_batch_size": ("HEALTH_CLEANUP_BATCH_SIZE", int, 100),
        }

        updates = {}
        for config_key, (env_var, converter, _default) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if converter is int:
                        updates[config_key] = int(env_value)
                    elif converter is float:
                        updates[config_key] = float(env_value)
                    elif callable(converter):
                        updates[config_key] = converter(env_value)
                    else:
                        updates[config_key] = env_value
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "invalid_environment_variable_value",
                        env_var=env_var,
                        value=env_value,
                        error=str(e),
                    )

        if updates:
            self.update_config(**updates)

    def validate_config(self) -> list[str]:
        """
        Validate current configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate intervals
        if self.config.check_interval_seconds <= 0:
            errors.append("check_interval_seconds must be positive")

        if self.config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")

        if self.config.timeout_seconds > self.config.check_interval_seconds:
            errors.append("timeout_seconds cannot be greater than check_interval_seconds")

        # Validate thresholds
        if not (0 < self.config.database_connection_threshold_percent <= 100):
            errors.append("database_connection_threshold_percent must be between 0 and 100")

        if self.config.memory_usage_threshold_mb < 0:
            errors.append("memory_usage_threshold_mb cannot be negative")

        # Validate history settings
        if self.config.max_history_entries <= 0:
            errors.append("max_history_entries must be positive")

        if self.config.history_retention_days <= 0:
            errors.append("history_retention_days must be positive")

        if not (0 < self.config.history_cleanup_threshold <= 1):
            errors.append("history_cleanup_threshold must be between 0 and 1")

        return errors

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of current configuration."""
        config_dict = self.config.model_dump()
        return {
            "check_interval_seconds": config_dict.get("check_interval_seconds", 30),
            "timeout_seconds": config_dict.get("timeout_seconds", 30),
            "database_threshold_percent": config_dict.get(
                "database_connection_threshold_percent", 80
            ),
            "alert_enabled": config_dict.get("alert_enabled", True),
            "memory_monitoring_enabled": config_dict.get("enable_memory_monitoring", True),
            "max_history_entries": config_dict.get("max_history_entries", 10000),
            "history_retention_days": config_dict.get("history_retention_days", 30),
            "memory_threshold_mb": config_dict.get("memory_usage_threshold_mb", 100.0),
            "validation_errors": self.validate_config(),
            "is_valid": len(self.validate_config()) == 0,
        }

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        old_config = self.config.model_dump()
        self.config = self._create_default_config()

        self._add_to_config_history(old_config, {})

        logger.info("health_check_config_reset_to_defaults")

    def _add_to_config_history(self, old_config: dict[str, Any], changes: dict[str, Any]) -> None:
        """Add configuration change to history."""
        self._config_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old_config": old_config,
                "changes": changes,
            }
        )

        # Cleanup old entries if needed
        if len(self._config_history) > self._max_history_size:
            self._config_history = self._config_history[-self._max_history_size :]

    def get_config_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get configuration change history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of configuration change entries
        """
        if limit:
            return self._config_history[-limit:]
        return self._config_history.copy()

    def export_config(self) -> dict[str, Any]:
        """Export current configuration as dictionary."""
        config_dict = self.config.model_dump()
        return {
            "config": config_dict,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "validation_errors": self.validate_config(),
            "is_valid": len(self.validate_config()) == 0,
        }

    def import_config(self, config_data: dict[str, Any]) -> bool:
        """
        Import configuration from dictionary.

        Args:
            config_data: Configuration data to import

        Returns:
            True if import successful, False otherwise
        """
        try:
            # Extract config values
            config_values = config_data.get("config", {})

            # Update configuration
            self.update_config(**config_values)

            logger.info(
                "health_check_config_imported",
                imported_parameters=list(config_values.keys()),
            )
            return True

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("health_check_config_import_failed", error=str(e))
            return False

    def get_component_thresholds(self, component_name: str) -> dict[str, float]:
        """
        Get threshold values for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with threshold values
        """
        thresholds = {
            "database": {
                "connection_usage_warning": float(
                    self.config.database_connection_threshold_percent
                ),
                "connection_usage_critical": 95.0,
            },
            "memory": {
                "usage_warning": 85.0,
                "usage_critical": 95.0,
            },
            "cpu": {
                "usage_warning": 85.0,
                "usage_critical": 95.0,
            },
            "file_system": {
                "space_warning": 85.0,
                "space_critical": 95.0,
            },
        }

        return thresholds.get(
            component_name,
            {
                "warning": 80.0,
                "critical": 95.0,
            },
        )

    def set_component_thresholds(self, component_name: str, thresholds: dict[str, float]) -> None:
        """
        Set threshold values for a specific component.

        Args:
            component_name: Name of the component
            thresholds: Dictionary with threshold values
        """
        if component_name == "database" and "connection_usage_warning" in thresholds:
            self.update_config(
                database_connection_threshold_percent=int(thresholds["connection_usage_warning"])
            )

        logger.info(
            "component_thresholds_updated",
            component=component_name,
            thresholds=thresholds,
        )

    def get_monitoring_intervals(self) -> dict[str, int]:
        """Get monitoring intervals for different components."""
        return {
            "health_check": self.config.check_interval_seconds,
            "proactive_check": max(60, self.config.check_interval_seconds * 2),
            "performance_collection": 30,
            "cleanup": 300,  # 5 minutes
        }

    def is_component_enabled(self, component_name: str) -> bool:
        """
        Check if a component is enabled for health checking.

        Args:
            component_name: Name of the component

        Returns:
            True if component is enabled
        """
        # All core components are enabled by default
        enabled_components = {
            "database",
            "redis",
            "cache_hierarchy",
            "file_system",
            "memory",
            "cpu",
            "tws_monitor",
            "connection_pools",
            "websocket_pool",
        }

        return component_name in enabled_components

    def get_retry_configuration(self, component_name: str) -> dict[str, Any]:
        """
        Get retry configuration for a component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with retry settings
        """
        component_configs = {
            "database": {"max_retries": 3, "backoff_multiplier": 2},
            "redis": {"max_retries": 2, "backoff_multiplier": 2},
            "cache_hierarchy": {"max_retries": 2, "backoff_multiplier": 2},
            "file_system": {"max_retries": 2, "backoff_multiplier": 2},
            "memory": {"max_retries": 1, "backoff_multiplier": 1},
            "cpu": {"max_retries": 1, "backoff_multiplier": 1},
            "tws_monitor": {"max_retries": 1, "backoff_multiplier": 1},
            "connection_pools": {"max_retries": 3, "backoff_multiplier": 2},
            "websocket_pool": {"max_retries": 2, "backoff_multiplier": 2},
        }

        return component_configs.get(component_name, {"max_retries": 2, "backoff_multiplier": 2})
