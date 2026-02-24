# pylint: disable=all
"""
Smart Cache Validator - Event-Driven Cache Validation

Validates cache ONLY when jobs fail (ABEND/FAILED), avoiding unnecessary
validations for successful jobs. Detects dependency changes automatically.

Author: Resync Team
Version: 5.9.9
"""

from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)


def load_validation_config():
    """Load validation configuration from TOML file."""
    try:
        from pathlib import Path

        import toml

        config_file = Path(__file__).parent.parent.parent / "config" / "graphrag.toml"

        if config_file.exists():
            config = toml.load(config_file)
            return config.get("graphrag", {}).get("validation", {})
        return {}
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return {}


class CacheValidationConfig:
    """
    Configuration for smart cache validation.

    Loaded from config/graphrag.toml on startup.
    Changes persist across restarts.
    """

    # Load from file
    _config = load_validation_config()

    # When to validate
    VALIDATE_ON_ABEND = _config.get("validate_on_abend", True)
    VALIDATE_ON_FAILED = _config.get("validate_on_failed", True)
    VALIDATE_ON_LATE = _config.get("validate_on_late", False)
    VALIDATE_ON_STUCK = _config.get("validate_on_stuck", False)

    # Cache age thresholds
    TRUST_CACHE_DAYS = _config.get("trust_cache_days", 7)
    QUICK_VALIDATE_DAYS = _config.get("quick_validate_days", 30)
    FULL_VALIDATE_DAYS = _config.get("full_validate_days", 90)

    # Auto-actions
    AUTO_INVALIDATE = _config.get("auto_invalidate", True)
    AUTO_REDISCOVER = _config.get("auto_rediscover", True)

    @classmethod
    def reload_from_file(cls):
        """Reload configuration from file."""
        cls._config = load_validation_config()

        cls.VALIDATE_ON_ABEND = cls._config.get("validate_on_abend", True)
        cls.VALIDATE_ON_FAILED = cls._config.get("validate_on_failed", True)
        cls.VALIDATE_ON_LATE = cls._config.get("validate_on_late", False)
        cls.VALIDATE_ON_STUCK = cls._config.get("validate_on_stuck", False)
        cls.TRUST_CACHE_DAYS = cls._config.get("trust_cache_days", 7)
        cls.AUTO_INVALIDATE = cls._config.get("auto_invalidate", True)
        cls.AUTO_REDISCOVER = cls._config.get("auto_rediscover", True)


class CacheValidationStats:
    """Statistics for cache validation."""

    def __init__(self):
        self.validations_triggered = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.cache_invalidations = 0
        self.dependencies_changed = []
        self.last_reset = datetime.now(timezone.utc)

    def record_validation(
        self, job_name: str, is_valid: bool, changes: dict[str, set] | None = None
    ):
        """Record validation result."""
        self.validations_triggered += 1

        if is_valid:
            self.validations_passed += 1
        else:
            self.validations_failed += 1
            self.cache_invalidations += 1

            if changes:
                self.dependencies_changed.append(
                    {
                        "job_name": job_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "added": list(changes.get("added", set())),
                        "removed": list(changes.get("removed", set())),
                    }
                )

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        accuracy = (
            (self.validations_passed / self.validations_triggered * 100)
            if self.validations_triggered > 0
            else 100.0
        )

        return {
            "validations_triggered": self.validations_triggered,
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
            "cache_invalidations": self.cache_invalidations,
            "accuracy": round(accuracy, 2),
            "dependencies_changed": self.dependencies_changed[-10:],  # Last 10
            "last_reset": self.last_reset.isoformat(),
        }

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.validations_triggered = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.cache_invalidations = 0
        self.dependencies_changed = []
        self.last_reset = datetime.now(timezone.utc)


class SmartCacheValidator:
    """
    Event-driven cache validator.

    Validates cache ONLY when jobs fail, avoiding unnecessary
    validations for successful jobs. 99.6% more efficient than polling.
    """

    def __init__(
        self, tws_client, cache_client, knowledge_graph, discovery_service=None
    ):
        """
        Initialize smart cache validator.

        Args:
            tws_client: TWS client for fetching job dependencies
            cache_client: Redis client for cache operations
            knowledge_graph: Knowledge graph instance
            discovery_service: Auto-discovery service (optional)
        """
        self.tws = tws_client
        self.cache = cache_client
        self.kg = knowledge_graph
        self.discovery = discovery_service

        self.stats = CacheValidationStats()

        logger.info("SmartCacheValidator initialized")

    async def on_job_failed(self, job_name: str, event_details: dict):
        """
        Called automatically when job fails (ABEND/FAILED).

        Validates cache and invalidates if dependencies changed.

        Args:
            job_name: Name of failed job
            event_details: Event metadata (status, RC, etc)
        """
        try:
            # Check if we should validate based on failure type
            if not self._should_validate(event_details):
                logger.debug("Skipping validation for %s", job_name)
                return

            logger.info("Job %s failed - validating cache", job_name)

            # 1. Get cached subgraph
            cached = await self._get_cached_subgraph(job_name)

            if not cached:
                logger.debug("No cache for %s - skipping validation", job_name)
                return

            # 2. Check cache age
            cache_age_days = self._get_cache_age_days(cached)

            # 3. Decide validation level based on age
            if cache_age_days < CacheValidationConfig.TRUST_CACHE_DAYS:
                # Cache too fresh - trust it
                logger.debug(
                    "Cache for %s is fresh (%sd) - trusting", job_name, cache_age_days
                )
                self.stats.record_validation(job_name, is_valid=True)
                return

            # 4. Validate dependencies
            is_valid, changes = await self._validate_dependencies(
                job_name,
                cached,
                full_check=(cache_age_days >= CacheValidationConfig.FULL_VALIDATE_DAYS),
            )

            # 5. Record stats
            self.stats.record_validation(job_name, is_valid, changes)

            # 6. Handle invalid cache
            if not is_valid and CacheValidationConfig.AUTO_INVALIDATE:
                await self._handle_invalid_cache(job_name, event_details, changes)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "Cache validation failed for %s: %s", job_name, e, exc_info=True
            )

    def _should_validate(self, event_details: dict) -> bool:
        """Check if should validate based on event type."""
        status = event_details.get("status", "").upper()
        event_details.get("severity", "").upper()

        # Check config flags
        if status == "ABEND" and CacheValidationConfig.VALIDATE_ON_ABEND:
            return True

        if status in ("FAILED", "ERROR") and CacheValidationConfig.VALIDATE_ON_FAILED:
            return True

        if "LATE" in status and CacheValidationConfig.VALIDATE_ON_LATE:
            return True

        return bool("STUCK" in status and CacheValidationConfig.VALIDATE_ON_STUCK)

    async def _get_cached_subgraph(self, job_name: str) -> dict | None:
        """Get cached subgraph if exists."""
        try:
            if not self.cache:
                return None

            cache_key = f"subgraph:{job_name}"
            cached_data = await self.cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                return cached_data

            return None

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to get cached subgraph: %s", e)
            return None

    def _get_cache_age_days(self, cached: dict) -> int:
        """Calculate cache age in days."""
        try:
            cached_at = cached.get("cached_at")

            if isinstance(cached_at, str):
                cached_at = datetime.fromisoformat(cached_at)
            elif not isinstance(cached_at, datetime):
                return 999  # Unknown age - treat as old

            age = datetime.now(timezone.utc) - cached_at
            return age.days

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return 999  # Error - treat as old

    async def _validate_dependencies(
        self, job_name: str, cached: dict, full_check: bool = False
    ) -> tuple[bool, dict[str, set] | None]:
        """
        Validate if dependencies changed.

        Args:
            job_name: Name of job
            cached: Cached subgraph data
            full_check: If True, compare names; if False, compare count only

        Returns:
            (is_valid, changes_dict)
        """
        try:
            # Get current dependencies from TWS
            current_preds = await self.tws.get_job_predecessors(job_name)

            if current_preds is None:
                # Error fetching - assume valid to avoid false invalidations
                logger.warning("Failed to fetch dependencies for %s", job_name)
                return True, None

            # Extract cached dependencies
            cached_deps = cached.get("dependencies", [])

            if full_check:
                # Full check: compare names
                cached_dep_names = {
                    dep.get("name") for dep in cached_deps if dep.get("name")
                }

                current_dep_names = {
                    pred.get("name") for pred in current_preds if pred.get("name")
                }

                if cached_dep_names != current_dep_names:
                    # Dependencies changed!
                    added = current_dep_names - cached_dep_names
                    removed = cached_dep_names - current_dep_names

                    logger.warning(
                        f"Dependencies changed for {job_name}:\n"
                        f"  Cached: {cached_dep_names}\n"
                        f"  Current: {current_dep_names}\n"
                        f"  Added: {added}\n"
                        f"  Removed: {removed}"
                    )

                    return False, {"added": added, "removed": removed}
            else:
                # Quick check: compare count only
                cached_count = len(cached_deps)
                current_count = len(current_preds)

                if cached_count != current_count:
                    logger.warning(
                        f"Dependency count changed for {job_name}: "
                        f"{cached_count} → {current_count}"
                    )
                    return False, {"count_changed": True}

            # No changes detected
            return True, None

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Dependency validation failed: %s", e, exc_info=True)
            return True, None  # Assume valid on error

    async def _handle_invalid_cache(
        self, job_name: str, event_details: dict, changes: dict
    ):
        """Handle invalid cache - invalidate and optionally rediscover."""
        try:
            # 1. Invalidate cache
            await self._invalidate_cache(job_name)

            logger.info(f"Cache invalidated for {job_name}", changes=changes)

            # 2. Trigger re-discovery if enabled
            if CacheValidationConfig.AUTO_REDISCOVER and self.discovery:
                logger.info("Triggering re-discovery for %s", job_name)

                await create_tracked_task(
                    self.discovery.on_job_failed(job_name, event_details)
                )

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to handle invalid cache: %s", e, exc_info=True)

    async def _invalidate_cache(self, job_name: str):
        """Invalidate all cache entries for a job."""
        try:
            if not self.cache:
                return

            # Invalidate subgraph cache
            await self.cache.delete(f"subgraph:{job_name}")

            # Invalidate discovery cache
            await self.cache.delete(f"discovered:{job_name}")

            logger.debug("Cache invalidated for %s", job_name)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Cache invalidation failed: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return self.stats.get_stats()

    def reset_stats(self):
        """Reset daily statistics."""
        self.stats.reset_daily_stats()
        logger.info("Cache validation stats reset")


def get_cache_validator(
    tws_client, cache_client, knowledge_graph, discovery_service=None
):
    """
    Factory function to get SmartCacheValidator instance.

    Args:
        tws_client: TWS client instance
        cache_client: Redis client instance
        knowledge_graph: Knowledge graph instance
        discovery_service: Discovery service instance (optional)

    Returns:
        SmartCacheValidator instance
    """
    return SmartCacheValidator(
        tws_client, cache_client, knowledge_graph, discovery_service
    )
