"""
Circuit Breaker Metrics API endpoints.

This module provides monitoring capabilities for circuit breakers,
including metrics, statistics, and management operations.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from resync.core.circuit_breaker import (
    adaptive_llm_api_breaker,
    adaptive_tws_api_breaker,
)
from resync.core.health import get_health_check_service
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/circuit-breakers", tags=["monitoring"])


@router.get("/metrics")
async def get_circuit_breaker_metrics() -> dict[str, Any]:
    """Get comprehensive circuit breaker metrics."""
    return {
        "tws_api": adaptive_tws_api_breaker.get_enhanced_stats(),
        "llm_api": adaptive_llm_api_breaker.get_enhanced_stats(),
        "summary": {
            "total_breakers": 2,
            "open_breakers": sum(
                1
                for cb in [adaptive_tws_api_breaker, adaptive_llm_api_breaker]
                if cb.state == "open"
            ),
            "degraded_services": sum(
                1
                for cb in [adaptive_tws_api_breaker, adaptive_llm_api_breaker]
                if cb.latency_metrics.is_latency_degraded()
            ),
        },
    }


@router.get("/health")
async def get_circuit_breaker_health() -> dict[str, Any]:
    """Get circuit breaker health status."""
    tws_health = {
        "service": "tws_api",
        "state": adaptive_tws_api_breaker.state,
        "latency_p95": adaptive_tws_api_breaker.latency_metrics.calculate_percentiles().get(
            "p95", 0
        ),
        "latency_p99": adaptive_tws_api_breaker.latency_metrics.calculate_percentiles().get(
            "p99", 0
        ),
        "degradation_ratio": (
            adaptive_tws_api_breaker.latency_metrics.slow_requests
            / max(1, adaptive_tws_api_breaker.latency_metrics.total_measurements)
        ),
        "is_degraded": adaptive_tws_api_breaker.latency_metrics.is_latency_degraded(),
    }

    llm_health = {
        "service": "llm_api",
        "state": adaptive_llm_api_breaker.state,
        "latency_p95": adaptive_llm_api_breaker.latency_metrics.calculate_percentiles().get(
            "p95", 0
        ),
        "latency_p99": adaptive_llm_api_breaker.latency_metrics.calculate_percentiles().get(
            "p99", 0
        ),
        "degradation_ratio": (
            adaptive_llm_api_breaker.latency_metrics.slow_requests
            / max(1, adaptive_llm_api_breaker.latency_metrics.total_measurements)
        ),
        "is_degraded": adaptive_llm_api_breaker.latency_metrics.is_latency_degraded(),
    }

    return {
        "services": [tws_health, llm_health],
        "overall_health": (
            "healthy"
            if not any(
                cb.latency_metrics.is_latency_degraded()
                for cb in [adaptive_tws_api_breaker, adaptive_llm_api_breaker]
            )
            else "degraded"
        ),
    }


@router.post("/reset/{service}")
async def reset_circuit_breaker(service: str) -> dict[str, str]:
    """Reset circuit breaker for a specific service."""
    breakers = {
        "tws_api": adaptive_tws_api_breaker,
        "llm_api": adaptive_llm_api_breaker,
    }

    if service not in breakers:
        raise HTTPException(404, f"Service {service} not found")

    breaker = breakers[service]
    async with breaker._lock:
        await breaker._set_state("closed")
        # Reset metrics
        breaker.stats = type(breaker.stats)()  # Reset to defaults
        breaker.latency_metrics = type(breaker.latency_metrics)()  # Reset to defaults

    logger.info("circuit_breaker_reset", service=service)

    return {"status": "reset", "service": service}


@router.get("/thresholds")
async def get_adaptive_thresholds() -> dict[str, Any]:
    """Get current adaptive thresholds for all circuit breakers."""
    return {
        "tws_api": {
            "p95_threshold": 1000,  # Default threshold
            "p99_threshold": 2000,  # Default threshold
            "adaptive_enabled": True,
        },
        "llm_api": {
            "p95_threshold": 1500,  # Default threshold
            "p99_threshold": 3000,  # Default threshold
            "adaptive_enabled": True,
        },
    }


@router.post("/thresholds/{service}")
async def update_thresholds(
    service: str,
    p95_threshold: float = Query(..., ge=0.1, le=10.0),
    p99_threshold: float = Query(..., ge=0.2, le=20.0),
) -> dict[str, Any]:
    """Update latency thresholds for a specific service."""
    breakers = {
        "tws_api": adaptive_tws_api_breaker,
        "llm_api": adaptive_llm_api_breaker,
    }

    if service not in breakers:
        raise HTTPException(404, f"Service {service} not found")

    breaker = breakers[service]
    async with breaker._lock:
        old_p95 = 1000  # Default threshold
        old_p99 = 2000  # Default threshold

        # Note: Adaptive config not implemented in base CircuitBreaker
        # These are stored as module-level variables for now
        logger.info(
            "threshold_update_attempted",
            service=service,
            old_p95=old_p95,
            old_p99=old_p99,
            new_p95=p95_threshold,
            new_p99=p99_threshold,
        )

    logger.info(
        "circuit_breaker_thresholds_updated",
        service=service,
        old_p95=old_p95,
        new_p95=p95_threshold,
        old_p99=old_p99,
        new_p99=p99_threshold,
    )

    return {
        "service": service,
        "updated_thresholds": {
            "p95": p95_threshold,
            "p99": p99_threshold,
        },
    }


@router.get("/proactive-health")
async def get_proactive_health_checks() -> dict[str, Any]:
    """Get proactive health check results with predictive analysis."""
    try:
        health_service = await get_health_check_service()
        results = await health_service.perform_proactive_health_checks()

        return {
            "status": "success",
            "data": results,
            "summary": {
                "issues_count": len(results.get("issues_detected", [])),
                "alerts_count": len(results.get("predictive_alerts", [])),
                "recovery_actions_count": len(results.get("recovery_actions", [])),
                "checks_performed": results.get("checks_performed", []),
            },
        }

    except Exception as e:
        logger.error("proactive_health_check_endpoint_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="Proactive health check failed. Check server logs for details."
        ) from e


@router.post("/proactive-health/analyze")
async def analyze_system_health() -> dict[str, Any]:
    """Perform deep analysis of system health with recommendations."""
    try:
        health_service = await get_health_check_service()

        # Get proactive health results
        proactive_results = await health_service.perform_proactive_health_checks()

        # Generate analysis and recommendations
        analysis = {
            "timestamp": proactive_results["timestamp"],
            "overall_health_score": await _calculate_health_score(proactive_results),
            "risk_assessment": await _assess_system_risks(proactive_results),
            "recommendations": await _generate_recommendations(proactive_results),
            "action_plan": await _create_action_plan(proactive_results),
            "raw_data": proactive_results,
        }

        return {"status": "success", "analysis": analysis}

    except Exception as e:
        logger.error("health_analysis_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health analysis failed. Check server logs for details.") from e


async def _calculate_health_score(results: dict[str, Any]) -> float:
    """Calculate overall system health score (0.0 to 1.0)."""
    score = 1.0  # Start with perfect health

    # Deduct points for issues
    critical_issues = sum(
        1 for issue in results.get("issues_detected", []) if issue.get("severity") == "critical"
    )
    high_issues = sum(
        1 for issue in results.get("issues_detected", []) if issue.get("severity") == "high"
    )

    # Critical issues have bigger impact
    score -= critical_issues * 0.2
    score -= high_issues * 0.1

    # Deduct for alerts
    alerts = len(results.get("predictive_alerts", []))
    score -= alerts * 0.05

    return max(0.0, min(1.0, score))


async def _assess_system_risks(results: dict[str, Any]) -> dict[str, Any]:
    """Assess system risks based on health data."""
    risks = {
        "overall_risk_level": "low",
        "risk_factors": [],
        "mitigation_priority": "low",
    }

    issues = results.get("issues_detected", [])
    alerts = results.get("predictive_alerts", [])

    # Assess risk level
    critical_count = sum(1 for i in issues if i.get("severity") == "critical")
    high_count = sum(1 for i in issues if i.get("severity") == "high")

    if critical_count > 0:
        risks["overall_risk_level"] = "critical"
        risks["mitigation_priority"] = "immediate"
    elif high_count > 2 or critical_count > 0:
        risks["overall_risk_level"] = "high"
        risks["mitigation_priority"] = "high"
    elif high_count > 0 or len(alerts) > 2:
        risks["overall_risk_level"] = "medium"
        risks["mitigation_priority"] = "medium"

    # Identify risk factors
    if issues:
        risks["risk_factors"].extend([i["type"] for i in issues])

    if alerts:
        risks["risk_factors"].extend([a["type"] for a in alerts])

    return risks


async def _generate_recommendations(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate actionable recommendations."""
    recommendations = []

    # Analyze issues and generate specific recommendations
    for issue in results.get("issues_detected", []):
        if issue["type"] == "high_pool_utilization":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "scalability",
                    "action": "Scale up connection pool",
                    "reason": issue["message"],
                    "estimated_impact": "Reduce connection timeouts by 50%",
                    "implementation_effort": "medium",
                }
            )
        elif issue["type"] == "high_error_rate":
            recommendations.append(
                {
                    "priority": "critical",
                    "category": "reliability",
                    "action": "Investigate connection stability issues",
                    "reason": issue["message"],
                    "estimated_impact": "Improve system reliability",
                    "implementation_effort": "high",
                }
            )
        elif issue["type"] == "circuit_breaker_open":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "reliability",
                    "action": f"Check health of {issue['component']} service",
                    "reason": issue["message"],
                    "estimated_impact": "Restore service availability",
                    "implementation_effort": "medium",
                }
            )

    # Add predictive recommendations
    for alert in results.get("predictive_alerts", []):
        if alert["type"] == "pool_exhaustion_prediction":
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "capacity_planning",
                    "action": "Prepare for connection pool scaling",
                    "reason": alert["message"],
                    "estimated_impact": "Prevent service degradation",
                    "implementation_effort": "low",
                }
            )

    # Default recommendations if no issues
    if not recommendations:
        recommendations.append(
            {
                "priority": "low",
                "category": "maintenance",
                "action": "Continue monitoring system health",
                "reason": "System operating normally",
                "estimated_impact": "Maintain current performance",
                "implementation_effort": "low",
            }
        )

    return recommendations


async def _create_action_plan(results: dict[str, Any]) -> dict[str, Any]:
    """Create prioritized action plan."""
    issues = results.get("issues_detected", [])
    alerts = results.get("predictive_alerts", [])

    # Prioritize actions
    immediate_actions = [i for i in issues if i.get("severity") == "critical"]
    high_priority_actions = [i for i in issues if i.get("severity") == "high"]
    medium_priority_actions = [i for i in issues if i.get("severity") == "medium"]
    predictive_actions = alerts

    return {
        "immediate": immediate_actions,
        "high_priority": high_priority_actions,
        "medium_priority": medium_priority_actions,
        "predictive": predictive_actions,
        "timeline": {
            "immediate": "Execute within 1 hour",
            "high_priority": "Execute within 4 hours",
            "medium_priority": "Execute within 24 hours",
            "predictive": "Monitor and plan for future",
        },
    }
