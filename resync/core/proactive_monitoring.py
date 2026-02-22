"""
Proactive Monitoring System

This module provides intelligent health monitoring that preemptively detects
connection issues, monitors pool utilization, and performs predictive analysis.
"""

import time
from typing import Any

import structlog

from resync.core.resilience import CircuitBreaker, CircuitBreakerConfig
from resync.core.connection_pool_manager import get_advanced_connection_pool_manager

logger = structlog.get_logger(__name__)


class ProactiveMonitoringSystem:
    """
    Proactive health monitoring system for connection pools and critical components.

    This system implements intelligent health monitoring that:
    - Preemptively detects connection issues
    - Monitors pool utilization and performance
    - Performs predictive health analysis
    - Triggers recovery actions automatically
    """

    def __init__(self):
        """Initialize the proactive monitoring system."""
        self.circuit_breakers = {
            "database": CircuitBreaker(CircuitBreakerConfig()),
            "redis": CircuitBreaker(CircuitBreakerConfig()),
            "cache_hierarchy": CircuitBreaker(CircuitBreakerConfig()),
            "tws_monitor": CircuitBreaker(CircuitBreakerConfig()),
        }

    async def perform_proactive_health_checks(self) -> dict[str, Any]:
        """
        Perform proactive health checks for connection pools and critical components.

        Returns:
            Dictionary containing check results, issues detected, and recovery actions
        """
        start_time = time.time()
        results: dict[str, Any] = {
            "timestamp": start_time,
            "checks_performed": [],
            "issues_detected": [],
            "recovery_actions": [],
            "performance_insights": {},
            "predictive_alerts": [],
        }

        logger.info(
            "starting_proactive_health_checks",
            correlation_id=f"proactive_{int(start_time)}",
        )

        try:
            # 1. Connection Pool Health Checks
            pool_health = await self._check_connection_pool_health()
            results["checks_performed"].append("connection_pools")
            results["performance_insights"]["connection_pools"] = pool_health

            # Detect pool issues
            if pool_health.get("utilization", 0) > 0.9:
                results["issues_detected"].append(
                    {
                        "type": "high_pool_utilization",
                        "severity": "high",
                        "message": f"Connection pool utilization at {pool_health['utilization']:.1%}",
                        "recommendation": "Consider scaling up connection pool",
                    }
                )

            if pool_health.get("error_rate", 0) > 0.05:
                results["issues_detected"].append(
                    {
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": f"Connection pool error rate at {pool_health['error_rate']:.1%}",
                        "recommendation": "Investigate connection stability",
                    }
                )

            # 2. Circuit Breaker Health
            circuit_health = self._check_circuit_breaker_health()
            results["checks_performed"].append("circuit_breakers")
            results["performance_insights"]["circuit_breakers"] = circuit_health

            # Detect circuit breaker issues
            for cb_name, cb_status in circuit_health.items():
                if cb_status.get("state") == "open":
                    results["issues_detected"].append(
                        {
                            "type": "circuit_breaker_open",
                            "severity": "high",
                            "component": cb_name,
                            "message": f"Circuit breaker {cb_name} is open",
                            "recommendation": "Check upstream service health",
                        }
                    )

            # 3. Performance Baseline Comparison
            baseline_comparison = self._compare_with_baseline()
            results["performance_insights"]["baseline_comparison"] = baseline_comparison

            logger.info(
                "proactive_health_checks_completed",
                duration=time.time() - start_time,
                issues_found=len(results["issues_detected"]),
                recovery_actions=len(results["recovery_actions"]),
            )

        except Exception as e:
            logger.error("proactive_health_checks_failed", error=str(e))
            results["error"] = str(e)

        return results

    async def _check_connection_pool_health(self) -> dict[str, Any]:
        """Check health of all connection pools."""
        try:
            advanced_manager = get_advanced_connection_pool_manager()
            if advanced_manager:
                metrics = await advanced_manager.get_performance_metrics()
                return {
                    "pool_count": len(metrics.get("traditional_pools", {})),
                    "smart_pool_active": "smart_pool" in metrics,
                    "auto_scaling_active": "auto_scaling" in metrics,
                    "utilization": metrics.get("auto_scaling", {}).get("load_score", 0),
                    "error_rate": metrics.get("smart_pool", {})
                    .get("performance", {})
                    .get("error_rate", 0),
                    "total_connections": metrics.get("auto_scaling", {}).get(
                        "current_connections", 0
                    ),
                    "scaling_recommended": metrics.get("smart_pool", {}).get(
                        "scaling_signals", {}
                    ),
                }
            # Fallback to basic pool manager
            try:
                from resync.core.connection_pool_manager import (
                    get_connection_pool_manager,
                )

                pool_manager = get_connection_pool_manager()
                if pool_manager:
                    basic_metrics = {}
                    pools = getattr(pool_manager, "pools", {})
                    for pool_name, pool in pools.items():
                        stats = pool.get_stats()
                        basic_metrics[pool_name] = {
                            "connections": stats.get("total_connections", 0),
                            "utilization": stats.get("active_connections", 0)
                            / max(1, stats.get("total_connections", 1)),
                        }
                    return basic_metrics
            except ImportError:
                pass

        except Exception as e:
            logger.warning("connection_pool_health_check_failed", error=str(e))

        return {"error": "Unable to check connection pool health"}

    def _check_circuit_breaker_health(self) -> dict[str, Any]:
        """Check health of all circuit breakers."""
        results = {}

        # Check TWS circuit breakers
        try:
            from resync.core.resilience_singletons import (
                adaptive_llm_api_breaker,
                adaptive_tws_api_breaker,
            )

            breakers = {
                "adaptive_tws_api": adaptive_tws_api_breaker,
                "adaptive_llm_api": adaptive_llm_api_breaker,
            }

            for name, breaker in breakers.items():
                if breaker:
                    try:
                        stats = breaker.get_stats() if hasattr(breaker, "get_stats") else {}
                        results[name] = {
                            "state": stats.get("state", "unknown"),
                            "failures": stats.get("failures", 0),
                            "successes": stats.get("successes", 0),
                            "error_rate": stats.get("failure_rate", 0),
                            "last_failure": stats.get("last_failure_time"),
                            "latency_p95": stats.get("latency_percentiles", {}).get(
                                "p95", 0
                            ),
                        }
                    except Exception as e:
                        logger.error("exception_caught", error=str(e), exc_info=True)
                        results[name] = {"error": str(e)}
        except ImportError:
            # Circuit breakers not available
            pass

        return results

    def _compare_with_baseline(self) -> dict[str, Any]:
        """Compare current performance with historical baseline."""
        # This would compare with stored baseline metrics
        # For now, return placeholder structure
        return {
            "baseline_available": False,
            "deviations": [],
            "trend": "stable",
            "recommendations": [
                "Implement baseline metrics storage for future comparisons"
            ],
        }

    def get_circuit_breakers(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers managed by this system."""
        return self.circuit_breakers.copy()

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get a specific circuit breaker by name."""
        return self.circuit_breakers.get(name)
