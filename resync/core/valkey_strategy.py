# pylint
"""
Valkey FAIL-FAST Strategy Module

Manages Valkey dependency strategy per endpoint with documented tiers:
- READ_ONLY: Never needs Valkey (health checks, docs)
- BEST_EFFORT: Cache optional, degrades gracefully (TWS queries, RAG)
- CRITICAL: Valkey required, returns 503 if unavailable (TWS operations)

Usage:
    strategy = get_valkey_strategy()
    tier = strategy.get_tier("POST", "/tws/execute/job1")

    if not valkey_available and strategy.should_fail_fast(
        "POST", "/tws/execute/job1", False
    ):
        raise HTTPException(503)

Author: Resync Team
Version: 5.4.2
"""

import os
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class ValkeyTier(str, Enum):
    """Valkey dependency tier for endpoints."""

    READ_ONLY = "read_only"  # Never needs Valkey
    BEST_EFFORT = "best_effort"  # Cache optional, can degrade
    CRITICAL = "critical"  # Valkey required, 503 if unavailable

class DegradedBehavior(str, Enum):
    """Possible degradation behaviors for BEST_EFFORT endpoints."""

    QUERY_TWS_DIRECT = "query_tws_direct"
    CALL_LLM_DIRECT = "call_llm_direct"
    DISABLE_REALTIME = "disable_realtime_updates"
    READONLY_MODE = "readonly_mode"
    RETURN_CACHED_ONLY = "return_cached_only"

# =============================================================================
# ENDPOINT PATTERN MATCHER
# =============================================================================

class EndpointPattern:
    """
    Pattern matching for endpoints.

    Supports:
    - Exact match: "GET /health"
    - Wildcard: "GET /tws/*"
    - Method wildcard: "/health" (matches any method)
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.regex = self._compile_pattern(pattern)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Convert pattern to regex."""
        # Escape special chars except *
        escaped = re.escape(pattern).replace(r"\*", ".*")

        # If no method specified, accept any
        if pattern.split()[0].upper() not in (
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
        ):
            escaped = r".* " + escaped

        return re.compile(f"^{escaped}$", re.IGNORECASE)

    def matches(self, method: str, path: str) -> bool:
        """Check if request matches pattern."""
        request_str = f"{method.upper()} {path}"
        return bool(self.regex.match(request_str))

# =============================================================================
# CONFIGURATION
# =============================================================================

class ValkeyStrategyConfig:
    """Parsed configuration for Valkey strategy."""

    def __init__(self, config_dict: dict) -> None:
        # Startup config
        startup = config_dict.get("startup", {})
        self.startup_fail_fast: bool = startup.get("fail_fast", True)
        self.max_retries: int = startup.get("max_retries", 3)
        self.backoff_seconds: float = startup.get("backoff_seconds", 0.5)

        # Runtime config
        runtime = config_dict.get("runtime", {})
        exceptions = runtime.get("exceptions", {})

        # Parse tiers
        self.read_only_patterns = self._parse_patterns(exceptions.get("read_only", []))
        self.best_effort_configs = self._parse_best_effort(
            exceptions.get("best_effort", [])
        )
        self.critical_configs = self._parse_critical(runtime.get("critical", []))

        # Default policy
        default = runtime.get("default_policy", "fail_fast")
        self.default_tier = (
            ValkeyTier.CRITICAL if default == "fail_fast" else ValkeyTier.BEST_EFFORT
        )

        # Monitoring config
        monitoring = config_dict.get("monitoring", {})
        self.alert_on_degraded = monitoring.get("alert_on_degraded", True)
        self.alert_threshold_percent = monitoring.get("alert_threshold_percent", 5)

    def _parse_patterns(self, pattern_list: list) -> list[EndpointPattern]:
        """Parse list of patterns."""
        patterns = []
        for item in pattern_list:
            pattern = item.get("pattern", "") if isinstance(item, dict) else item
            if pattern:
                patterns.append(EndpointPattern(pattern))
        return patterns

    def _parse_best_effort(self, config_list: list) -> list[dict]:
        """Parse best effort configurations."""
        configs = []
        for item in config_list:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern", "")
            if pattern:
                configs.append(
                    {
                        "pattern": EndpointPattern(pattern),
                        "degraded_behavior": item.get("degraded_behavior"),
                        "warning_message": item.get("warning_message"),
                        "cost_impact": item.get("cost_impact"),
                        "refresh_interval_seconds": item.get(
                            "refresh_interval_seconds"
                        ),
                    }
                )
        return configs

    def _parse_critical(self, config_list: list) -> list[dict]:
        """Parse critical endpoint configurations."""
        configs = []
        for item in config_list:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern", "")
            if pattern:
                configs.append(
                    {
                        "pattern": EndpointPattern(pattern),
                        "reason": item.get("reason", "Valkey required"),
                        "http_status": item.get("http_status", 503),
                        "retry_after": item.get("retry_after", 60),
                    }
                )
        return configs

# =============================================================================
# VALKEY STRATEGY
# =============================================================================

class ValkeyStrategy:
    """
    Valkey dependency strategy manager.

    Determines the Valkey dependency tier for each endpoint and provides
    configuration for degradation behavior.

    Usage:
        strategy = ValkeyStrategy("config/valkey_strategy.yaml")
        tier = strategy.get_tier("POST", "/tws/execute/job1")
        # ValkeyTier.CRITICAL

        if not valkey_available and strategy.should_fail_fast(
        "POST", "/tws/execute/job1", False
    ):
            raise HTTPException(503)
    """

    def __init__(self, config_path: str = "config/valkey_strategy.yaml") -> None:
        """
        Initialize strategy with configuration file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._get_tier_cached = lru_cache(maxsize=1024)(self._get_tier_impl)

        logger.info(
            "valkey_strategy_initialized",
            config_path=str(self.config_path),
            fail_fast=self.config.startup_fail_fast,
            default_tier=self.config.default_tier.value,
        )

    def _load_config(self) -> ValkeyStrategyConfig:
        """Load and validate configuration."""
        if not self.config_path.exists():
            logger.warning(
                "valkey_strategy_config_not_found",
                path=str(self.config_path),
                using_defaults=True,
            )
            return ValkeyStrategyConfig({})

        try:
            with open(self.config_path) as f:
                config_dict = yaml.safe_load(f) or {}
            return ValkeyStrategyConfig(config_dict)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "valkey_strategy_config_error",
                error=str(e),
                using_defaults=True,
            )
            return ValkeyStrategyConfig({})

    def get_tier(self, method: str, path: str) -> ValkeyTier:
        """
        Determine the tier of an endpoint.

        Args:
            method: HTTP method (GET, POST, etc)
            path: Request path (/tws/execute/job1)

        Returns:
            ValkeyTier: Tier of the endpoint

        Order of precedence:
        1. READ_ONLY (never needs Valkey)
        2. CRITICAL (must have Valkey)
        3. BEST_EFFORT (can degrade)
        4. Default policy
        """
        return self._get_tier_cached(method, path)

    def _get_tier_impl(self, method: str, path: str) -> ValkeyTier:
        # Check READ_ONLY first
        for pattern in self.config.read_only_patterns:
            if pattern.matches(method, path):
                return ValkeyTier.READ_ONLY

        # Check CRITICAL (takes precedence over BEST_EFFORT)
        for config in self.config.critical_configs:
            if config["pattern"].matches(method, path):
                return ValkeyTier.CRITICAL

        # Check BEST_EFFORT
        for config in self.config.best_effort_configs:
            if config["pattern"].matches(method, path):
                return ValkeyTier.BEST_EFFORT

        # Default
        return self.config.default_tier

    def is_valkey_required(self, method: str, path: str) -> bool:
        """
        Check if Valkey is required for this endpoint.

        Returns:
            True if tier is CRITICAL, False otherwise
        """
        return self.get_tier(method, path) == ValkeyTier.CRITICAL

    def get_degraded_config(self, method: str, path: str) -> dict[str, Any] | None:
        """
        Get degradation configuration for BEST_EFFORT endpoints.

        Returns:
            Dict with configuration or None if not applicable
        """
        for config in self.config.best_effort_configs:
            if config["pattern"].matches(method, path):
                return {
                    "behavior": config.get("degraded_behavior"),
                    "warning": config.get("warning_message"),
                    "cost_impact": config.get("cost_impact"),
                    "refresh_interval": config.get("refresh_interval_seconds"),
                }
        return None

    def get_critical_config(self, method: str, path: str) -> dict[str, Any] | None:
        """
        Get configuration for CRITICAL endpoints.

        Returns:
            Dict with reason, http_status, retry_after or None
        """
        for config in self.config.critical_configs:
            if config["pattern"].matches(method, path):
                return {
                    "reason": config.get("reason"),
                    "http_status": config.get("http_status", 503),
                    "retry_after": config.get("retry_after", 60),
                }
        return None

    def should_fail_fast(self, method: str, path: str, valkey_available: bool) -> bool:
        """
        Decide if request should return 503 or be allowed.

        Args:
            method: HTTP method
            path: Request path
            valkey_available: Whether Valkey is available

        Returns:
            True if should return 503, False if can continue
        """
        # If Valkey is UP, always allow
        if valkey_available:
            return False

        # If Valkey is DOWN, depends on tier
        tier = self.get_tier(method, path)

        # READ_ONLY: never needs Valkey
        if tier == ValkeyTier.READ_ONLY:
            return False

        # BEST_EFFORT: degrades but doesn't fail
        if tier == ValkeyTier.BEST_EFFORT:
            return False

        # CRITICAL: fails without Valkey
        return True

    def get_startup_config(self) -> dict[str, Any]:
        """Get startup configuration."""
        return {
            "fail_fast": self.config.startup_fail_fast,
            "max_retries": self.config.max_retries,
            "backoff_seconds": self.config.backoff_seconds,
        }

    def get_all_tiers_summary(self) -> dict[str, int]:
        """Get summary of configured patterns per tier."""
        return {
            "read_only": len(self.config.read_only_patterns),
            "best_effort": len(self.config.best_effort_configs),
            "critical": len(self.config.critical_configs),
        }

# =============================================================================
# SINGLETON
# =============================================================================

_strategy_instance: ValkeyStrategy | None = None
_strategy_lock = __import__("threading").Lock()

def get_valkey_strategy() -> ValkeyStrategy:
    """
    Get singleton ValkeyStrategy instance.

    Thread-safe: uses double-checked locking to prevent duplicate initialization
    when multiple threads call this concurrently during startup.

    The strategy is loaded once and cached for the application lifetime.
    """
    global _strategy_instance
    if _strategy_instance is None:
        with _strategy_lock:
            # Re-check after acquiring lock
            if _strategy_instance is None:
                # Try multiple config paths
                config_paths = [
                    "config/valkey_strategy.yaml",
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "config",
                        "valkey_strategy.yaml",
                    ),
                    "/app/config/valkey_strategy.yaml",
                ]

                for path in config_paths:
                    if Path(path).exists():
                        _strategy_instance = ValkeyStrategy(path)
                        return _strategy_instance

                # Use default config
                _strategy_instance = ValkeyStrategy("config/valkey_strategy.yaml")

    return _strategy_instance

def reset_valkey_strategy() -> None:
    """Reset singleton for testing purposes."""
    global _strategy_instance
    with _strategy_lock:
        _strategy_instance = None

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_endpoint_critical(method: str, path: str) -> bool:
    """Check if endpoint is CRITICAL tier."""
    return get_valkey_strategy().get_tier(method, path) == ValkeyTier.CRITICAL

def is_endpoint_read_only(method: str, path: str) -> bool:
    """Check if endpoint is READ_ONLY tier."""
    return get_valkey_strategy().get_tier(method, path) == ValkeyTier.READ_ONLY

def get_endpoint_tier(method: str, path: str) -> str:
    """Get tier name for endpoint."""
    return get_valkey_strategy().get_tier(method, path).value

def get_valkey_strategy_status() -> dict[str, Any]:
    """
    Get current Valkey strategy status for admin UI.

    Returns dict with:
    - enabled: bool
    - mode: str (normal, degraded, fail_fast)
    - fail_fast_timeout: float
    - degraded_endpoints: list
    - healthy: bool
    """
    try:
        strategy = get_valkey_strategy()

        # Get configuration
        startup_config = strategy.get_startup_config()
        tiers_summary = strategy.get_all_tiers_summary()

        # Determine mode based on current state
        # In normal operation, mode is "normal"
        mode = "normal"
        degraded_endpoints = []


        return {
            "enabled": True,
            "mode": mode,
            "fail_fast_timeout": startup_config.get("backoff_seconds", 5.0),
            "degraded_endpoints": degraded_endpoints,
            "healthy": mode == "normal",
            "tiers": tiers_summary,
            "startup_fail_fast": startup_config.get("fail_fast", True),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("Failed to get valkey strategy status", error=str(e))
        return {
            "enabled": False,
            "mode": "unknown",
            "fail_fast_timeout": 5.0,
            "degraded_endpoints": [],
            "healthy": False,
            "error": str(e),
        }
