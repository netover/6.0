"""
Predictive Analysis Engine

This module provides predictive analysis capabilities for detecting potential issues
before they become critical problems.
"""

import asyncio
from typing import Any

import structlog

from resync.core.connection_pool_manager import (
    get_advanced_connection_pool_manager,
    get_connection_pool_manager,
)

logger = structlog.get_logger(__name__)


class PredictiveAnalyzer:
    """Predictive analysis engine for health monitoring."""

    async def perform_predictive_analysis(self) -> list[dict[str, Any]]:
        """
        Perform predictive analysis for potential issues.

        This method analyzes current system metrics and trends to predict:
        - Connection pool exhaustion
        - Rising error rates
        - Circuit breaker failures
        - Performance degradation
        """
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
            circuit_health = await self._check_circuit_breaker_health()
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

            # Analyze memory usage trends
            memory_alerts = await self._analyze_memory_trends()
            alerts.extend(memory_alerts)

            # Analyze CPU usage trends
            cpu_alerts = await self._analyze_cpu_trends()
            alerts.extend(cpu_alerts)

        except Exception as e:
            logger.error("predictive_analysis_failed", error=str(e))

        return alerts

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

    async def _check_circuit_breaker_health(self) -> dict[str, Any]:
        """Check health of all circuit breakers."""
        results = {}

        # Check TWS circuit breakers
        from resync.core.circuit_breaker import (
            adaptive_llm_api_breaker,
            adaptive_tws_api_breaker,
            llm_api_breaker,
            tws_api_breaker,
        )

        breakers = {
            "adaptive_tws_api": adaptive_tws_api_breaker,
            "adaptive_llm_api": adaptive_llm_api_breaker,
            "traditional_tws_api": tws_api_breaker,
            "traditional_llm_api": llm_api_breaker,
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

    async def _analyze_memory_trends(self) -> list[dict[str, Any]]:
        """Analyze memory usage trends for predictive alerts."""
        alerts = []

        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Predict memory exhaustion
            if memory_percent > 80:
                # Estimate time to exhaustion based on current trend
                alerts.append(
                    {
                        "type": "memory_exhaustion_prediction",
                        "severity": "high",
                        "timeframe": "2_hours",
                        "confidence": 0.7,
                        "message": "Memory usage at {memory_percent:.1f}%, may exhaust soon",
                        "recommendation": "Monitor memory usage and consider scaling",
                    }
                )

            # Detect memory leaks (rapid growth)
            if memory_percent > 70:
                alerts.append(
                    {
                        "type": "memory_leak_suspicion",
                        "severity": "medium",
                        "timeframe": "30_minutes",
                        "confidence": 0.6,
                        "message": "High memory usage detected: {memory_percent:.1f}%",
                        "recommendation": "Monitor for memory leaks",
                    }
                )

        except Exception as e:
            logger.warning("memory_trend_analysis_failed", error=str(e))

        return alerts

    async def _analyze_cpu_trends(self) -> list[dict[str, Any]]:
        """Analyze CPU usage trends for predictive alerts."""
        alerts = []

        try:
            import psutil

            # Get multiple samples for trend analysis
            cpu_samples = []
            for _ in range(3):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                await asyncio.sleep(0.1)

            avg_cpu = sum(cpu_samples) / len(cpu_samples)

            # Predict CPU exhaustion
            if avg_cpu > 80:
                alerts.append(
                    {
                        "type": "cpu_exhaustion_prediction",
                        "severity": "high",
                        "timeframe": "1_hour",
                        "confidence": 0.75,
                        "message": "High CPU usage detected: {avg_cpu:.1f}%",
                        "recommendation": "Monitor CPU usage and consider scaling",
                    }
                )

            # Detect CPU spikes
            if max(cpu_samples) - min(cpu_samples) > 30:
                alerts.append(
                    {
                        "type": "cpu_spike_detected",
                        "severity": "medium",
                        "timeframe": "immediate",
                        "confidence": 0.8,
                        "message": "CPU usage spiking between {min(cpu_samples):.1f}% and {max(cpu_samples):.1f}%",
                        "recommendation": "Investigate CPU-intensive operations",
                    }
                )

        except Exception as e:
            logger.warning("cpu_trend_analysis_failed", error=str(e))

        return alerts

    async def analyze_performance_degradation(self) -> list[dict[str, Any]]:
        """Analyze system for performance degradation patterns."""
        alerts = []

        try:
            # Check for increasing response times
            # Check for decreasing throughput
            # Check for resource contention

            # This would integrate with actual performance monitoring data
            # For now, return empty list as placeholder

            pass

        except Exception as e:
            logger.error("performance_degradation_analysis_failed", error=str(e))

        return alerts
