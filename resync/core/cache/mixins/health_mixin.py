"""
Cache Health Check Mixin.

Provides health check functionality for cache implementations.
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CacheHealthMixin:
    """
    Mixin providing health check capabilities for cache.

    Requires base class to have:
    - self.shards: List of cache shards  # type: ignore[attr-defined]
    - self.shard_locks: List of shard locks  # type: ignore[attr-defined]
    - self.is_running: bool  # type: ignore[attr-defined]
    - self.cleanup_task: Optional[asyncio.Task]  # type: ignore[attr-defined]
    """

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check on cache system.

        Returns:
            Dict with health status and diagnostics
        """
        from resync.core.metrics import runtime_metrics

        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "async_cache",
                "operation": "health_check",
            }
        )

        try:
            checks = await asyncio.gather(  # type: ignore[call-overload]
                self._health_check_functionality(correlation_id),
                self._health_check_integrity(correlation_id),
                self._health_check_background_tasks(correlation_id),
                self._health_check_performance(correlation_id),
                self._health_check_config(correlation_id),
                return_exceptions=True,
            )

            statuses = []
            check_results = {}

            check_names = [
                "functionality",
                "integrity",
                "background_tasks",
                "performance",
                "config",
            ]

            for name, result in zip(check_names, checks, strict=False):
                if isinstance(result, Exception):
                    check_results[name] = {
                        "status": "error",
                        "error": str(result),
                    }
                    statuses.append("error")
                else:
                    check_results[name] = result
                    statuses.append(result.get("status", "unknown"))

            # Determine overall status
            if "error" in statuses:
                overall_status = "unhealthy"
            elif "warning" in statuses:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            return {
                "status": overall_status,
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "checks": check_results,
            }

        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "status": "error",
                "correlation_id": correlation_id,
                "error": str(e),
            }

    async def _health_check_functionality(self, correlation_id: str) -> dict[str, Any]:
        """Check basic cache operations."""
        try:
            test_key = f"__health_check_{correlation_id}"
            test_value = {"test": True, "timestamp": time.time()}

            # Test set
            await self.set(test_key, test_value, ttl_seconds=10)  # type: ignore[attr-defined]

            # Test get
            retrieved = await self.get(test_key)  # type: ignore[attr-defined]

            # Test delete
            await self.delete(test_key)  # type: ignore[attr-defined]

            if retrieved != test_value:
                return {"status": "error", "message": "Value mismatch"}

            return {"status": "healthy", "operations": ["set", "get", "delete"]}

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return {"status": "error", "error": str(e)}

    async def _health_check_integrity(self, correlation_id: str) -> dict[str, Any]:
        """Check cache integrity."""
        try:
            total_entries = 0
            expired_entries = 0
            current_time = time.time()

            for i, shard in enumerate(self.shards):  # type: ignore[attr-defined]
                async with self.shard_locks[i]:  # type: ignore[attr-defined]
                    for _key, entry in shard.items():
                        total_entries += 1
                        if current_time > entry.timestamp + entry.ttl:
                            expired_entries += 1

            status = "healthy"
            if expired_entries > total_entries * 0.1:
                status = "warning"

            return {
                "status": status,
                "total_entries": total_entries,
                "expired_entries": expired_entries,
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return {"status": "error", "error": str(e)}

    def _health_check_background_tasks(self, correlation_id: str) -> dict[str, Any]:
        """Check background task status."""
        try:
            cleanup_running = (
                self.cleanup_task is not None and not self.cleanup_task.done()  # type: ignore[attr-defined]
            )

            status = "healthy" if cleanup_running or not self.is_running else "warning"  # type: ignore[attr-defined]

            return {
                "status": status,
                "cleanup_task_running": cleanup_running,
                "cache_running": self.is_running,  # type: ignore[attr-defined]
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return {"status": "error", "error": str(e)}

    def _health_check_performance(self, correlation_id: str) -> dict[str, Any]:
        """Check cache performance metrics."""
        try:
            metrics = self.get_detailed_metrics()  # type: ignore[attr-defined]

            hit_rate = metrics.get("hit_rate", 0)
            status = "healthy"

            if hit_rate < 0.5 and metrics.get("total_requests", 0) > 100:
                status = "warning"

            return {
                "status": status,
                "hit_rate": hit_rate,
                "total_entries": metrics.get("total_entries", 0),
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return {"status": "error", "error": str(e)}

    def _health_check_config(self, correlation_id: str) -> dict[str, Any]:
        """Check configuration validity."""
        try:
            issues = []

            if hasattr(self, "ttl_seconds") and self.ttl_seconds < 1:
                issues.append("TTL too low")

            if hasattr(self, "max_entries") and self.max_entries < 100:
                issues.append("Max entries too low")

            status = "healthy" if not issues else "warning"

            return {
                "status": status,
                "issues": issues,
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return {"status": "error", "error": str(e)}
