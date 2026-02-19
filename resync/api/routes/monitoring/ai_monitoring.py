"""
Admin API Routes for AI Monitoring and Specialist Agents.

Provides endpoints for:
- Configuring specialist agents (enable/disable, model selection)
- Configuring AI monitoring (drift detection, schedules, resource limits)
- Viewing monitoring status and alerts
- Running manual monitoring

Endpoints are mounted under /api/v1/admin/ai

Author: Resync Team
Version: 5.2.3.29
"""

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ai", tags=["Admin - AI Configuration"])

# ============================================================================
# CONFIGURATION FILE PATH
# ============================================================================

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "config" / "ai_config.json"
)


# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================


class ResourceLimitsConfig(BaseModel):
    """Resource limits configuration."""

    max_cpu_percent: float = Field(
        default=25.0, ge=5.0, le=100.0, description="Maximum CPU usage percentage"
    )
    max_memory_mb: int = Field(
        default=512, ge=128, le=4096, description="Maximum memory usage in MB"
    )
    max_execution_time_seconds: int = Field(
        default=300, ge=30, le=3600, description="Maximum execution time"
    )
    nice_level: int = Field(
        default=10, ge=0, le=19, description="Process nice level (higher = lower priority)"
    )


class SpecialistAgentConfig(BaseModel):
    """Configuration for a single specialist agent."""

    enabled: bool = Field(default=True, description="Whether the specialist is enabled")
    model_name: str = Field(default="gpt-4o", description="LLM model to use")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2048, ge=100, le=8192, description="Maximum response tokens")
    timeout_seconds: int = Field(default=30, ge=5, le=120, description="Request timeout")
    retry_attempts: int = Field(default=3, ge=0, le=5, description="Retry attempts on failure")
    custom_instructions: str | None = Field(
        default=None, description="Additional custom instructions"
    )


class SpecialistsConfig(BaseModel):
    """Configuration for all specialist agents."""

    enabled: bool = Field(default=True, description="Enable specialist team")
    execution_mode: str = Field(
        default="coordinate",
        pattern="^(coordinate|collaborate|route|parallel)$",
        description="Team execution mode",
    )
    parallel_execution: bool = Field(default=True, description="Run specialists in parallel")
    max_parallel_specialists: int = Field(
        default=4, ge=1, le=10, description="Max parallel specialists"
    )
    timeout_seconds: int = Field(default=45, ge=10, le=180, description="Overall team timeout")
    orchestrator_model: str = Field(default="gpt-4o", description="Orchestrator model")
    synthesizer_model: str = Field(default="gpt-4o", description="Synthesizer model")
    fallback_to_general: bool = Field(
        default=True, description="Fallback to general assistant on failure"
    )

    job_analyst: SpecialistAgentConfig = Field(
        default_factory=lambda: SpecialistAgentConfig(temperature=0.2)
    )
    dependency_specialist: SpecialistAgentConfig = Field(
        default_factory=lambda: SpecialistAgentConfig(temperature=0.1)
    )
    resource_specialist: SpecialistAgentConfig = Field(
        default_factory=lambda: SpecialistAgentConfig(temperature=0.2, max_tokens=1536)
    )
    knowledge_specialist: SpecialistAgentConfig = Field(
        default_factory=lambda: SpecialistAgentConfig(temperature=0.4, max_tokens=3072)
    )


class MonitoringScheduleConfig(BaseModel):
    """Monitoring schedule configuration."""

    type: str = Field(
        default="daily",
        pattern="^(hourly|every_4_hours|daily|weekly|manual)$",
        description="Schedule type",
    )
    time: str = Field(
        default="03:00",
        pattern=r"^\d{2}:\d{2}$",
        description="Time to run (HH:MM) for daily/weekly",
    )
    day_of_week: int | None = Field(
        default=None, ge=0, le=6, description="Day of week for weekly (0=Monday, 6=Sunday)"
    )


class DriftDetectionConfig(BaseModel):
    """Drift detection configuration."""

    data_drift_enabled: bool = Field(default=True, description="Monitor data/query drift")
    prediction_drift_enabled: bool = Field(default=True, description="Monitor prediction drift")
    target_drift_enabled: bool = Field(default=True, description="Monitor target/feedback drift")
    drift_threshold: float = Field(
        default=0.15, ge=0.01, le=0.5, description="Drift detection threshold"
    )
    alert_threshold: float = Field(
        default=0.25, ge=0.05, le=0.5, description="Alert trigger threshold"
    )
    reference_window_days: int = Field(default=7, ge=1, le=90, description="Reference data window")
    current_window_hours: int = Field(default=24, ge=1, le=168, description="Current data window")


class AIMonitoringConfig(BaseModel):
    """Complete AI monitoring configuration."""

    enabled: bool = Field(default=True, description="Enable AI monitoring")
    schedule: MonitoringScheduleConfig = Field(default_factory=MonitoringScheduleConfig)
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    resource_limits: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    reports_path: str = Field(default="data/evidently_reports", description="Path to store reports")
    max_reports_stored: int = Field(default=30, ge=1, le=365, description="Max reports to retain")


class AIConfigResponse(BaseModel):
    """Complete AI configuration response."""

    specialists: SpecialistsConfig = Field(default_factory=SpecialistsConfig)
    monitoring: AIMonitoringConfig = Field(default_factory=AIMonitoringConfig)
    last_updated: str | None = Field(default=None, description="Last update timestamp")
    updated_by: str | None = Field(default=None, description="Who made the last update")


class MonitoringStatusResponse(BaseModel):
    """Monitoring status response."""

    enabled: bool
    running: bool
    schedule: str
    last_run: str | None = None
    total_alerts: int
    recent_alerts: int
    evidently_available: bool


class DriftAlertResponse(BaseModel):
    """Drift alert response."""

    alert_id: str
    drift_type: str
    severity: str
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: str
    details: dict[str, Any] = Field(default_factory=dict)


class MonitoringRunResponse(BaseModel):
    """Monitoring run result response."""

    timestamp: str
    data_drift: dict[str, Any] | None = None
    prediction_drift: dict[str, Any] | None = None
    target_drift: dict[str, Any] | None = None
    alerts: list[dict[str, Any]]
    duration_seconds: float
    error: str | None = None


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_AI_CONFIG: dict[str, Any] = {
    "specialists": {
        "enabled": True,
        "execution_mode": "coordinate",
        "parallel_execution": True,
        "max_parallel_specialists": 4,
        "timeout_seconds": 45,
        "orchestrator_model": "gpt-4o",
        "synthesizer_model": "gpt-4o",
        "fallback_to_general": True,
        "job_analyst": {
            "enabled": True,
            "model_name": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 2048,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "custom_instructions": None,
        },
        "dependency_specialist": {
            "enabled": True,
            "model_name": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "custom_instructions": None,
        },
        "resource_specialist": {
            "enabled": True,
            "model_name": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 1536,
            "timeout_seconds": 25,
            "retry_attempts": 3,
            "custom_instructions": None,
        },
        "knowledge_specialist": {
            "enabled": True,
            "model_name": "gpt-4o",
            "temperature": 0.4,
            "max_tokens": 3072,
            "timeout_seconds": 35,
            "retry_attempts": 3,
            "custom_instructions": None,
        },
    },
    "monitoring": {
        "enabled": True,
        "schedule": {
            "type": "daily",
            "time": "03:00",
            "day_of_week": None,
        },
        "drift_detection": {
            "data_drift_enabled": True,
            "prediction_drift_enabled": True,
            "target_drift_enabled": True,
            "drift_threshold": 0.15,
            "alert_threshold": 0.25,
            "reference_window_days": 7,
            "current_window_hours": 24,
        },
        "resource_limits": {
            "max_cpu_percent": 25.0,
            "max_memory_mb": 512,
            "max_execution_time_seconds": 300,
            "nice_level": 10,
        },
        "reports_path": "data/evidently_reports",
        "max_reports_stored": 30,
    },
    "last_updated": None,
    "updated_by": None,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _load_config() -> dict[str, Any]:
    """Load configuration from file."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return _deep_merge(DEFAULT_AI_CONFIG.copy(), config)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("config_load_error", error=str(e))
    return DEFAULT_AI_CONFIG.copy()


def _save_config(config: dict[str, Any], updated_by: str = "admin") -> None:
    """Save configuration to file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    config["last_updated"] = datetime.now(timezone.utc).isoformat()
    config["updated_by"] = updated_by

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("config_saved", path=str(CONFIG_PATH), updated_by=updated_by)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_monitoring_service():
    """Get the monitoring service, lazy import to avoid circular dependency."""
    try:
        from resync.core.monitoring import get_monitoring_service

        return get_monitoring_service()
    except ImportError:
        return None


# ============================================================================
# ENDPOINTS - CONFIGURATION
# ============================================================================


@router.get(
    "/config",
    response_model=AIConfigResponse,
    summary="Get AI Configuration",
    description="Retrieve current configuration for specialist agents and AI monitoring.",
)
async def get_ai_config() -> dict[str, Any]:
    """Get current AI configuration."""
    return _load_config()


@router.put(
    "/config",
    response_model=AIConfigResponse,
    summary="Update AI Configuration",
    description="Update configuration for specialist agents and AI monitoring.",
)
async def update_ai_config(
    config: AIConfigResponse,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Update AI configuration."""
    try:
        config_dict = config.model_dump()
        _save_config(config_dict)

        # Reload services in background
        background_tasks.add_task(_reload_services, config_dict)

        return _load_config()

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("config_update_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update configuration. Check server logs for details.",
        ) from e


@router.patch(
    "/config/specialists",
    response_model=SpecialistsConfig,
    summary="Update Specialists Configuration",
    description="Partially update specialist agents configuration.",
)
async def update_specialists_config(
    specialists: SpecialistsConfig,
) -> dict[str, Any]:
    """Update specialists configuration."""
    config = _load_config()
    config["specialists"] = specialists.model_dump()
    _save_config(config)
    return config["specialists"]


@router.patch(
    "/config/monitoring",
    response_model=AIMonitoringConfig,
    summary="Update Monitoring Configuration",
    description="Partially update AI monitoring configuration.",
)
async def update_monitoring_config(
    monitoring: AIMonitoringConfig,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Update monitoring configuration."""
    config = _load_config()
    config["monitoring"] = monitoring.model_dump()
    _save_config(config)

    # Restart monitoring service with new config
    background_tasks.add_task(_restart_monitoring, config["monitoring"])

    return config["monitoring"]


# ============================================================================
# ENDPOINTS - SPECIALISTS
# ============================================================================


@router.get(
    "/specialists/status",
    summary="Get Specialists Status",
    description="Get current status of all specialist agents.",
)
async def get_specialists_status() -> dict[str, Any]:
    """Get status of specialist agents."""
    config = _load_config()
    specialists_config = config.get("specialists", {})

    specialists = [
        "job_analyst",
        "dependency_specialist",
        "resource_specialist",
        "knowledge_specialist",
    ]

    status_list = []
    for spec_name in specialists:
        spec_config = specialists_config.get(spec_name, {})
        status_list.append(
            {
                "name": spec_name.replace("_", " ").title(),
                "type": spec_name,
                "enabled": spec_config.get("enabled", True),
                "model": spec_config.get("model_name", "gpt-4o"),
                "temperature": spec_config.get("temperature", 0.3),
                "max_tokens": spec_config.get("max_tokens", 2048),
            }
        )

    return {
        "team_enabled": specialists_config.get("enabled", True),
        "execution_mode": specialists_config.get("execution_mode", "coordinate"),
        "parallel_execution": specialists_config.get("parallel_execution", True),
        "specialists": status_list,
    }


@router.post(
    "/specialists/{specialist_type}/toggle",
    summary="Toggle Specialist",
    description="Enable or disable a specific specialist agent.",
)
async def toggle_specialist(
    specialist_type: str,
    enabled: bool,
) -> dict[str, Any]:
    """Toggle a specialist agent on/off."""
    valid_types = [
        "job_analyst",
        "dependency_specialist",
        "resource_specialist",
        "knowledge_specialist",
    ]

    if specialist_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid specialist type. Must be one of: {valid_types}",
        )

    config = _load_config()
    if specialist_type in config["specialists"]:
        config["specialists"][specialist_type]["enabled"] = enabled
    else:
        config["specialists"][specialist_type] = {"enabled": enabled}

    _save_config(config)

    return {
        "specialist": specialist_type,
        "enabled": enabled,
        "message": f"Specialist {specialist_type} {'enabled' if enabled else 'disabled'}",
    }


# ============================================================================
# ENDPOINTS - MONITORING
# ============================================================================


@router.get(
    "/monitoring/status",
    response_model=MonitoringStatusResponse,
    summary="Get Monitoring Status",
    description="Get current status of the AI monitoring service.",
)
async def get_monitoring_status() -> dict[str, Any]:
    """Get monitoring service status."""
    service = _get_monitoring_service()

    if service:
        return service.get_status()

    # Return default status if service not initialized
    config = _load_config()
    return {
        "enabled": config.get("monitoring", {}).get("enabled", False),
        "running": False,
        "schedule": config.get("monitoring", {}).get("schedule", {}).get("type", "manual"),
        "last_run": None,
        "total_alerts": 0,
        "recent_alerts": 0,
        "evidently_available": False,
    }


@router.post(
    "/monitoring/toggle",
    summary="Toggle Monitoring",
    description="Enable or disable AI monitoring.",
)
async def toggle_monitoring(
    enabled: bool,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Toggle monitoring service on/off."""
    config = _load_config()
    config["monitoring"]["enabled"] = enabled
    _save_config(config)

    # Start/stop service
    background_tasks.add_task(_toggle_monitoring_service, enabled)

    return {
        "enabled": enabled,
        "message": f"Monitoring {'enabled' if enabled else 'disabled'}",
    }


@router.post(
    "/monitoring/run",
    response_model=MonitoringRunResponse,
    summary="Run Monitoring Now",
    description="Trigger a manual monitoring run.",
)
async def run_monitoring_now() -> dict[str, Any]:
    """Run monitoring manually."""
    service = _get_monitoring_service()

    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring service not initialized",
        )

    try:
        return await service.run_monitoring()
    except Exception as e:
        logger.error("manual_monitoring_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Monitoring run failed. Check server logs for details.",
        ) from e


@router.get(
    "/monitoring/alerts",
    response_model=list[DriftAlertResponse],
    summary="Get Monitoring Alerts",
    description="Get recent drift detection alerts.",
)
async def get_monitoring_alerts(
    drift_type: str | None = None,
    severity: str | None = None,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get recent monitoring alerts."""
    service = _get_monitoring_service()

    if not service:
        return []

    from datetime import timedelta

    from resync.core.monitoring import AlertSeverity, DriftType

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Convert string filters to enums
    drift_filter = None
    if drift_type:
        with contextlib.suppress(ValueError):
            drift_filter = DriftType(drift_type)

    severity_filter = None
    if severity:
        with contextlib.suppress(ValueError):
            severity_filter = AlertSeverity(severity)

    alerts = service.get_alerts(
        since=since,
        drift_type=drift_filter,
        severity=severity_filter,
    )

    return [a.to_dict() for a in alerts]


@router.get(
    "/monitoring/reports",
    summary="List Monitoring Reports",
    description="List available drift detection reports.",
)
async def list_monitoring_reports(
    limit: int = 10,
) -> dict[str, Any]:
    """List available monitoring reports."""
    config = _load_config()
    reports_path = Path(config.get("monitoring", {}).get("reports_path", "data/evidently_reports"))

    reports = []
    if reports_path.exists():
        for report_file in sorted(reports_path.glob("*.html"), reverse=True)[:limit]:
            reports.append(
                {
                    "name": report_file.name,
                    "path": str(report_file),
                    "size_kb": report_file.stat().st_size / 1024,
                    "created": datetime.fromtimestamp(report_file.stat().st_ctime).isoformat(),
                }
            )

    return {
        "reports_path": str(reports_path),
        "total_reports": len(list(reports_path.glob("*.html"))) if reports_path.exists() else 0,
        "reports": reports,
    }


# ============================================================================
# BACKGROUND TASKS
# ============================================================================


async def _reload_services(config: dict[str, Any]) -> None:
    """Reload services with new configuration."""
    logger.info("reloading_services")

    # Reload monitoring if config changed
    if "monitoring" in config:
        await _restart_monitoring(config["monitoring"])

    # Reload specialists if config changed
    if "specialists" in config:
        await _reload_specialists(config["specialists"])


async def _restart_monitoring(monitoring_config: dict[str, Any]) -> None:
    """Restart monitoring service with new configuration."""
    try:
        from resync.core.monitoring import (
            MonitoringConfig,
            get_monitoring_service,
            init_monitoring_service,
        )

        # Stop existing service
        service = get_monitoring_service()
        if service:
            await service.stop()

        if monitoring_config.get("enabled", False):
            # Create new config and start
            config = MonitoringConfig(**monitoring_config)
            await init_monitoring_service(config)
            logger.info("monitoring_service_restarted")

    except ImportError:
        logger.warning("monitoring_module_not_available")
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("monitoring_restart_error", error=str(e))


async def _reload_specialists(specialists_config: dict[str, Any]) -> None:
    """Reload specialist agents with new configuration."""
    try:
        from resync.core.specialists import create_specialist_team
        from resync.core.specialists.models import SpecialistConfig, SpecialistType, TeamConfig

        # Build team config from dict
        team_config = TeamConfig(
            enabled=specialists_config.get("enabled", True),
            execution_mode=specialists_config.get("execution_mode", "coordinate"),
            parallel_execution=specialists_config.get("parallel_execution", True),
            max_parallel_specialists=specialists_config.get("max_parallel_specialists", 4),
            timeout_seconds=specialists_config.get("timeout_seconds", 45),
            orchestrator_model=specialists_config.get("orchestrator_model", "gpt-4o"),
            synthesizer_model=specialists_config.get("synthesizer_model", "gpt-4o"),
            fallback_to_general=specialists_config.get("fallback_to_general", True),
        )

        # Build specialist configs
        for spec_type in SpecialistType:
            spec_key = spec_type.value
            if spec_key in specialists_config:
                team_config.specialists[spec_type] = SpecialistConfig(
                    specialist_type=spec_type, **specialists_config[spec_key]
                )

        await create_specialist_team(config=team_config)
        logger.info("specialists_reloaded")

    except ImportError:
        logger.warning("specialists_module_not_available")
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("specialists_reload_error", error=str(e))


async def _toggle_monitoring_service(enabled: bool) -> None:
    """Start or stop monitoring service."""
    try:
        from resync.core.monitoring import get_monitoring_service, init_monitoring_service

        service = get_monitoring_service()

        if enabled and not service:
            config = _load_config()
            from resync.core.monitoring import MonitoringConfig

            monitoring_config = MonitoringConfig(**config.get("monitoring", {}))
            await init_monitoring_service(monitoring_config)
        elif not enabled and service:
            await service.stop()

    except ImportError:
        logger.warning("monitoring_module_not_available")
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("toggle_monitoring_error", error=str(e))
