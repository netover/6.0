"""
AI Monitoring Module.

Provides drift detection and quality monitoring for the Resync AI system
using Evidently AI framework.

Features:
- Data drift detection (query patterns)
- Prediction drift detection (response quality)
- Target drift detection (user feedback)
- Scheduled monitoring with resource limits
- Admin configuration interface

Author: Resync Team
Version: 5.2.3.29
"""

from resync.core.monitoring.evidently_monitor import (
    DEFAULT_MONITORING_CONFIG,
    EVIDENTLY_AVAILABLE,
    AIMonitoringService,
    AlertSeverity,
    DriftAlert,
    DriftDetector,
    DriftType,
    MonitoringConfig,
    MonitoringDataCollector,
    MonitoringSchedule,
    ResourceLimits,
    get_monitoring_service,
    init_monitoring_service,
)

__all__ = [
    "EVIDENTLY_AVAILABLE",
    "DriftType",
    "MonitoringSchedule",
    "AlertSeverity",
    "DriftAlert",
    "ResourceLimits",
    "MonitoringConfig",
    "DEFAULT_MONITORING_CONFIG",
    "MonitoringDataCollector",
    "DriftDetector",
    "AIMonitoringService",
    "get_monitoring_service",
    "init_monitoring_service",
]
