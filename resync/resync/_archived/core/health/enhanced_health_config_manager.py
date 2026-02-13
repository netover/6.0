"""
Enhanced Health Configuration Manager

This module provides enhanced configuration management for health checks,
integrating with the new health checker architecture.
"""

from typing import Any

import structlog

from resync.core.health.health_models import HealthCheckConfig

from .health_checkers.health_checker_factory import HealthCheckerFactory
from .health_config_manager import HealthCheckConfigurationManager

logger = structlog.get_logger(__name__)


class EnhancedHealthConfigurationManager(HealthCheckConfigurationManager):
    """
    Enhanced configuration manager that integrates with health checker factory.

    Provides comprehensive configuration management with health checker integration.
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the enhanced configuration manager.

        Args:
            config: Base health check configuration
        """
        super().__init__(config)
        self._checker_factory: HealthCheckerFactory | None = None

    def set_health_checker_factory(self, factory: HealthCheckerFactory) -> None:
        """
        Set the health checker factory for integration.

        Args:
            factory: Health checker factory instance
        """
        self._checker_factory = factory

    def get_checker_specific_config(self, component_name: str) -> dict[str, Any]:
        """
        Get configuration specific to a health checker component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with component-specific configuration
        """
        if self._checker_factory:
            checker = self._checker_factory.get_health_checker(component_name)
            if checker:
                return checker.get_component_config()

        # Fallback to parent implementation
        return super().get_component_config(component_name)

    def validate_all_checkers_config(self) -> dict[str, list[str]]:
        """
        Validate configuration for all registered health checkers.

        Returns:
            Dictionary mapping component names to validation error lists
        """
        if not self._checker_factory:
            return {}

        return self._checker_factory.validate_all_checkers()

    def get_component_thresholds_enhanced(self, component_name: str) -> dict[str, float]:
        """
        Get enhanced threshold values for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with threshold values including checker-specific ones
        """
        # Get base thresholds
        base_thresholds = self.get_component_thresholds(component_name)

        # Get checker-specific configuration if available
        if self._checker_factory:
            checker = self._checker_factory.get_health_checker(component_name)
            if checker:
                config = checker.get_component_config()
                # Merge with base thresholds
                for key, value in config.items():
                    if (
                        ("threshold" in key.lower() or "percent" in key.lower())
                        and isinstance(value, (int, float))
                    ):
                        base_thresholds[f"checker_{key}"] = float(value)

        return base_thresholds

    def get_monitoring_intervals_enhanced(self) -> dict[str, int]:
        """
        Get enhanced monitoring intervals including checker-specific intervals.

        Returns:
            Dictionary with monitoring intervals
        """
        base_intervals = self.get_monitoring_intervals()

        # Add checker-specific intervals if factory is available
        if self._checker_factory:
            for name in self._checker_factory.get_enabled_health_checker_names():
                checker = self._checker_factory.get_health_checker(name)
                if checker:
                    config = checker.get_component_config()
                    timeout = config.get("timeout_seconds", 30)
                    base_intervals[f"{name}_timeout"] = timeout

        return base_intervals

    def export_config_enhanced(self) -> dict[str, Any]:
        """
        Export enhanced configuration including checker-specific settings.

        Returns:
            Dictionary with comprehensive configuration data
        """
        base_export = self.export_config()

        enhanced_export = {
            "config": base_export["config"],
            "exported_at": base_export["exported_at"],
            "validation_errors": base_export["validation_errors"],
            "is_valid": base_export["is_valid"],
            "checker_specific_configs": {},
            "component_thresholds": {},
            "monitoring_intervals": self.get_monitoring_intervals_enhanced(),
        }

        # Add checker-specific configurations
        if self._checker_factory:
            for name in self._checker_factory.get_enabled_health_checker_names():
                enhanced_export["checker_specific_configs"][name] = (
                    self.get_checker_specific_config(name)
                )
                enhanced_export["component_thresholds"][name] = (
                    self.get_component_thresholds_enhanced(name)
                )

        return enhanced_export

    def get_config_summary_enhanced(self) -> dict[str, Any]:
        """
        Get enhanced configuration summary.

        Returns:
            Dictionary with comprehensive configuration summary
        """
        base_summary = self.get_config_summary()

        enhanced_summary = {
            "base_config": base_summary,
            "checker_validation": self.validate_all_checkers_config(),
            "enabled_checkers": [],
            "checker_configs": {},
        }

        if self._checker_factory:
            enhanced_summary["enabled_checkers"] = (
                self._checker_factory.get_enabled_health_checker_names()
            )
            for name in enhanced_summary["enabled_checkers"]:
                enhanced_summary["checker_configs"][name] = self.get_checker_specific_config(name)

        return enhanced_summary

    def optimize_config_for_performance(self) -> dict[str, Any]:
        """
        Optimize configuration for better performance.

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "interval_optimizations": {},
            "threshold_adjustments": {},
            "resource_savings": {},
        }

        # Analyze current intervals
        base_intervals = self.get_monitoring_intervals()
        health_check_interval = base_intervals.get("health_check", 30)

        # Recommend optimizations based on component types
        if self._checker_factory:
            for name in self._checker_factory.get_enabled_health_checker_names():
                checker = self._checker_factory.get_health_checker(name)
                if checker:
                    checker.get_component_config()

                    # Recommend interval adjustments based on component type
                    if name in ["memory", "cpu"]:
                        # System resources can be checked less frequently
                        recommendations["interval_optimizations"][name] = health_check_interval * 2
                    elif name in ["database", "redis"]:
                        # Critical components should be checked more frequently
                        recommendations["interval_optimizations"][name] = max(
                            10, health_check_interval // 2
                        )
                    else:
                        # Default interval
                        recommendations["interval_optimizations"][name] = health_check_interval

        return recommendations
