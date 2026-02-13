"""
Service Level Objectives and Key Performance Indicators configuration
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ServiceLevelObjectives:
    """Service Level Objectives for Resync system"""

    # API Performance SLOs
    API_RESPONSE_TIME_P95: float = 500.0  # milliseconds
    API_AVAILABILITY: float = 0.999  # 99.9%
    ERROR_RATE_THRESHOLD: float = 0.001  # 0.1%

    # Business-specific SLOs
    TWS_CONNECTION_AVAILABILITY: float = 0.995  # 99.5%
    AI_AGENT_RESPONSE_TIME: float = 2000.0  # milliseconds
    CHAT_MESSAGE_DELIVERY_TIME: float = 200.0  # milliseconds

    # Capacity SLOs
    MAX_CONCURRENT_USERS: int = 1000
    MAX_REQUESTS_PER_MINUTE: int = 10000
    CACHE_HIT_RATIO_THRESHOLD: float = 0.8  # 80%


# KPI definitions
KPI_DEFINITIONS = {
    "api_response_time": {
        "description": "95th percentile of API response time",
        "target": 500,  # ms
        "warning_threshold": 400,  # ms
        "critical_threshold": 800,  # ms
    },
    "api_error_rate": {
        "description": "Percentage of failed API requests",
        "target": 0.001,  # 0.1%
        "warning_threshold": 0.0005,  # 0.05%
        "critical_threshold": 0.005,  # 0.5%
    },
    "availability": {
        "description": "Percentage of time the system is available",
        "target": 0.999,  # 99.9%
        "warning_threshold": 0.997,  # 99.7%
        "critical_threshold": 0.990,  # 99.0%
    },
    "tws_connection_success_rate": {
        "description": "Percentage of successful TWS connections",
        "target": 0.995,  # 99.5%
        "warning_threshold": 0.990,  # 99.0%
        "critical_threshold": 0.980,  # 98.0%
    },
    "ai_agent_response_time": {
        "description": "Average time for AI agent to respond",
        "target": 1000,  # ms
        "warning_threshold": 800,  # ms
        "critical_threshold": 2000,  # ms
    },
    "cache_hit_ratio": {
        "description": "Percentage of cache hits vs misses",
        "target": 0.85,  # 85%
        "warning_threshold": 0.80,  # 80%
        "critical_threshold": 0.70,  # 70%
    },
}


def validate_slo_compliance(metrics: dict[str, Any]) -> dict[str, bool]:
    """
    Validate if current metrics are compliant with SLOs

    Args:
        metrics: Dictionary containing current system metrics

    Returns:
        Dictionary with compliance status for each metric
    """
    compliance_report = {}

    for kpi_name, kpi_config in KPI_DEFINITIONS.items():
        if kpi_name in metrics:
            current_value = metrics[kpi_name]
            target = kpi_config["target"]

            # For percentages/ratios, higher is better
            if kpi_name in [
                "availability",
                "cache_hit_ratio",
                "tws_connection_success_rate",
            ]:
                compliance_report[kpi_name] = current_value >= target
            # For times and error rates, lower is better
            else:
                compliance_report[kpi_name] = current_value <= target
        else:
            compliance_report[kpi_name] = False  # Missing metric is non-compliant

    return compliance_report
