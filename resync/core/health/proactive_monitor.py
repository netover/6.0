"""
Proactive Health Monitor

This module provides proactive health monitoring functionality for
predictive analysis and automated recovery of system components.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from resync.core.connection_pool_manager import get_advanced_connection_pool_manager

logger = structlog.get_logger(__name__)


class ProactiveHealthMonitor:
    """
    Provides proactive health monitoring and predictive analysis.

    This class implements intelligent health monitoring that:
    - Preemptively detects connection issues
    - Monitors pool utilization and performance
    - Performs predictive health analysis
    - Triggers recovery actions automatically
    """

    def __init__(self):
        """Initialize the proactive health monitor."""
        self._monitoring_history: list[dict[str, Any]] = []
        self._max_history_entries = 1000

    async def perform_proactive_health_checks(self) -> dict[str, Any]:
        """
        Perform proactive health checks for connection pools and critical components.

        This method implements intelligent health monitoring that:
        - Preemptively detects connection issues
        - Monitors pool utilization and performance
        - Performs predictive health analysis
        - Triggers recovery actions automatically

        Returns:
            Dictionary containing proactive health check results
        """
        start_time = time.time()
        results = {
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
                        "message": "Connection pool utilization at {pool_health['utilization']:.1%}",
                        "recommendation": "Consider scaling up connection pool",
                    }
                )

            if pool_health.get("error_rate", 0) > 0.05:
                results["issues_detected"].append(
                    {
                        "type": "high_error_rate",
                        "severity": "critical",
                        "message": "Connection pool error rate at {pool_health['error_rate']:.1%}",
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

            # 3. Predictive Analysis
            predictions = await self._perform_predictive_analysis()
            results["checks_performed"].append("predictive_analysis")
            results["predictive_alerts"] = predictions

            # 4. Auto-Recovery Actions
            recovery_actions = await self._execute_auto_recovery()
            results["recovery_actions"] = recovery_actions

            # 5. Performance Baseline Comparison
            baseline_comparison = self._compare_with_baseline()
            results["performance_insights"]["baseline_comparison"] = baseline_comparison

            logger.info(
                "proactive_health_checks_completed",
                duration=time.time() - start_time,
                issues_found=len(results["issues_detected"]),
                recovery_actions=len(results["recovery_actions"]),
            )

            # Store results in history
            self._add_to_monitoring_history(results)

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
                    "scaling_recommended": metrics.get("smart_pool", {}).get("scaling_signals", {}),
                }
            # Fallback to basic pool manager
            # Use pool manager from pools.pool_manager (connection_manager does not define this)
            from resync.core.pools.pool_manager import get_connection_pool_manager

            pool_manager = get_connection_pool_manager()
            if pool_manager:
                basic_metrics = {}
                for pool_name, pool in pool_manager.pools.items():
                    stats = pool.get_stats()
                    basic_metrics[pool_name] = {
                        "connections": stats.get("total_connections", 0),
                        "utilization": stats.get("active_connections", 0)
                        / max(1, stats.get("total_connections", 1)),
                    }
                return basic_metrics

        except Exception as e:
            logger.warning("connection_pool_health_check_failed", error=str(e))

        return {"error": "Unable to check connection pool health"}

    def _check_circuit_breaker_health(self) -> dict[str, Any]:
        """Check health of all circuit breakers."""
        results = {}

        # Check TWS circuit breakers
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
                    stats = (
                        breaker.get_stats()
                        if hasattr(breaker, "get_stats")
                        else breaker.get_enhanced_stats()
                    )
                    results[name] = {
                        "state": stats.get("state", "unknown"),
                        "failures": stats.get("failures", 0),
                        "successes": stats.get("successes", 0),
                        "error_rate": stats.get("failure_rate", 0),
                        "last_failure": stats.get("last_failure_time"),
                        "latency_p95": stats.get("latency_percentiles", {}).get("p95", 0),
                    }
                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    results[name] = {"error": str(e)}

        return results

    async def _perform_predictive_analysis(self) -> list[dict[str, Any]]:
        """Perform predictive analysis for potential issues."""
        alerts = []

        try:
            # Analyze connection pool trends
            pool_health = await self._check_connection_pool_health()

            utilization = pool_health.get("utilization", 0)
            if utilization > 0.8:
                # Predict potential exhaustion in next hour
                alerts.append(
                    {
                        "type": "pool_exhaustion_prediction",
                        "severity": "medium",
                        "timeframe": "1_hour",
                        "confidence": 0.75,
                        "message": "Connection pool may exhaust at current utilization {utilization:.1%}",
                        "recommendation": "Monitor closely and prepare scaling",
                    }
                )

            # Analyze error rate trends
            error_rate = pool_health.get("error_rate", 0)
            if error_rate > 0.03:
                alerts.append(
                    {
                        "type": "error_rate_trend",
                        "severity": "high",
                        "timeframe": "immediate",
                        "confidence": 0.8,
                        "message": "Rising error rate detected: {error_rate:.1%}",
                        "recommendation": "Investigate root cause immediately",
                    }
                )

            # Analyze circuit breaker patterns
            circuit_health = self._check_circuit_breaker_health()
            open_breakers = sum(1 for cb in circuit_health.values() if cb.get("state") == "open")

            if open_breakers > 0:
                alerts.append(
                    {
                        "type": "multiple_circuit_breakers_open",
                        "severity": "critical",
                        "timeframe": "immediate",
                        "confidence": 1.0,
                        "message": f"{open_breakers} circuit breaker(s) are open",
                        "recommendation": "Check upstream service availability",
                    }
                )

        except Exception as e:
            logger.error("predictive_analysis_failed", error=str(e))

        return alerts

    async def _execute_auto_recovery(self) -> list[dict[str, Any]]:
        """Execute automatic recovery actions."""
        actions = []

        try:
            # Force health check on unhealthy connections
            pool_health = await self._check_connection_pool_health()
            if pool_health.get("error_rate", 0) > 0.1:
                # Trigger connection pool health check
                advanced_manager = get_advanced_connection_pool_manager()
                if advanced_manager:
                    health_results = await advanced_manager.force_health_check()
                    actions.append(
                        {
                            "action": "force_connection_health_check",
                            "timestamp": time.time(),
                            "results": health_results,
                            "reason": "High error rate detected",
                        }
                    )

            # Reset circuit breakers if appropriate
            circuit_health = self._check_circuit_breaker_health()
            for cb_name, cb_status in circuit_health.items():
                if cb_status.get("state") == "open":
                    # Check if it's been long enough to attempt reset
                    last_failure = cb_status.get("last_failure")
                    if last_failure and (time.time() - last_failure) > 300:  # 5 minutes
                        # In real implementation, this would trigger circuit breaker reset
                        actions.append(
                            {
                                "action": "circuit_breaker_reset_candidate",
                                "component": cb_name,
                                "timestamp": time.time(),
                                "reason": "Circuit breaker open for extended period",
                                "recommendation": "Manual intervention may be needed",
                            }
                        )

        except Exception as e:
            logger.error("auto_recovery_execution_failed", error=str(e))

        return actions

    def _compare_with_baseline(self) -> dict[str, Any]:
        """Compare current performance with historical baseline."""
        # This would compare with stored baseline metrics
        # For now, return placeholder structure
        return {
            "baseline_available": False,
            "deviations": [],
            "trend": "stable",
            "recommendations": ["Implement baseline metrics storage for future comparisons"],
        }

    def _add_to_monitoring_history(self, results: dict[str, Any]) -> None:
        """Add proactive monitoring results to history."""
        self._monitoring_history.append({"timestamp": datetime.now(timezone.utc), "results": results.copy()})

        # Cleanup old entries if needed
        if len(self._monitoring_history) > self._max_history_entries:
            self._monitoring_history = self._monitoring_history[-self._max_history_entries :]

    def get_monitoring_history(self, hours: int = 24, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get proactive monitoring history.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of entries to return

        Returns:
            List of monitoring history entries
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered_history = [
            entry for entry in self._monitoring_history if entry["timestamp"] >= cutoff_time
        ]

        # Sort by timestamp (most recent first)
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply limit
        if limit and len(filtered_history) > limit:
            filtered_history = filtered_history[:limit]

        return filtered_history

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get proactive monitoring statistics."""
        if not self._monitoring_history:
            return {"total_checks": 0, "avg_issues_per_check": 0.0}

        total_checks = len(self._monitoring_history)
        total_issues = sum(
            len(entry["results"].get("issues_detected", [])) for entry in self._monitoring_history
        )

        return {
            "total_checks": total_checks,
            "avg_issues_per_check": round(total_issues / total_checks, 2),
            "history_size": len(self._monitoring_history),
            "max_history_entries": self._max_history_entries,
        }
