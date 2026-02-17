"""
Centralized node configuration for workflow nodes.

This module provides a unified configuration system to replace
the dual-mode (verbose/optimized) pattern.
"""

from dataclasses import dataclass
from enum import Enum
import os


class LogLevel(Enum):
    """Verbosity levels for node execution."""
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


class NodeMode(Enum):
    """Legacy mode compatibility - deprecated."""
    VERBOSE = "verbose"
    OPTIMIZED = "optimized"


@dataclass
class NodeConfig:
    """
    Centralized configuration for workflow nodes.

    Replaces the old WORKFLOW_MODE environment variable pattern
    with a more flexible configuration system.
    """
    log_level: LogLevel = LogLevel.NORMAL
    enable_profiling: bool = False
    enable_intermediate_logs: bool = False
    enable_state_checkpointing: bool = True
    max_concurrent_nodes: int = 10

    # Legacy compatibility
    legacy_mode: NodeMode = NodeMode.OPTIMIZED

    @classmethod
    def from_env(cls) -> "NodeConfig":
        """Create config from environment variables."""
        # Legacy: map old WORKFLOW_MODE to new config
        workflow_mode = os.getenv("WORKFLOW_MODE", "optimized").lower()

        if workflow_mode == "verbose":
            log_level = LogLevel.VERBOSE
            enable_intermediate_logs = True
            legacy_mode = NodeMode.VERBOSE
        else:
            log_level = LogLevel.NORMAL
            enable_intermediate_logs = False
            legacy_mode = NodeMode.OPTIMIZED

        # Override with specific env vars if set
        if log_level_str := os.getenv("NODE_LOG_LEVEL"):
            try:
                log_level = LogLevel[log_level_str.upper()]
            except KeyError:
                pass

        enable_profiling = os.getenv("NODE_ENABLE_PROFILING", "").lower() in ("1", "true", "yes")

        return cls(
            log_level=log_level,
            enable_profiling=enable_profiling,
            enable_intermediate_logs=enable_intermediate_logs,
            legacy_mode=legacy_mode,
        )

    def should_log_verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return self.log_level.value >= LogLevel.VERBOSE.value

    def should_log_debug(self) -> bool:
        """Check if debug logging is enabled."""
        return self.log_level.value >= LogLevel.DEBUG.value


# Global configuration instance
_config: NodeConfig | None = None


def get_node_config() -> NodeConfig:
    """Get the global node configuration."""
    global _config
    if _config is None:
        _config = NodeConfig.from_env()
    return _config


def configure_nodes(**kwargs) -> None:
    """
    Configure node behavior at runtime.

    Usage:
        configure_nodes(log_level=LogLevel.DEBUG, enable_profiling=True)
    """
    global _config
    config = get_node_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
