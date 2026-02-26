"""
Admin Observability Routes.

Provides endpoints for monitoring and observability management:
- LangFuse status and configuration
- Evidently drift monitoring and reports
- Combined observability status

Endpoints:
    GET  /api/v1/admin/observability/status     - Get observability status
    POST /api/v1/admin/observability/setup      - Initialize observability

    GET  /api/v1/admin/observability/langfuse/stats    - LangFuse statistics

    GET  /api/v1/admin/observability/evidently/stats   - Evidently statistics
    POST /api/v1/admin/observability/evidently/check   - Run drift check
    GET  /api/v1/admin/observability/evidently/reports - List drift reports
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from resync.core.observability import (
    get_evidently_monitor,
    get_langfuse_client,
    get_observability_config,
    get_observability_status,
    setup_observability,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/admin/observability", tags=["Admin - Observability"])

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class LangFuseStatusResponse(BaseModel):
    """LangFuse status response."""

    enabled: bool
    configured: bool
    connected: bool
    host: str | None = None

class EvidentlyStatusResponse(BaseModel):
    """Evidently status response."""

    enabled: bool
    active: bool
    reference_data_size: int = 0
    current_data_size: int = 0
    last_check: str | None = None
    reports_count: int = 0

class ObservabilityStatusResponse(BaseModel):
    """Combined observability status."""

    langfuse: LangFuseStatusResponse
    evidently: EvidentlyStatusResponse
    environment: str
    service_name: str
    service_version: str

class SetupResponse(BaseModel):
    """Response from observability setup."""

    langfuse: bool
    evidently: bool
    message: str

class DriftCheckResponse(BaseModel):
    """Response from drift check."""

    timestamp: str
    drift_detected: bool
    drift_columns: list[str] = []
    reference_size: int
    current_size: int
    error: str | None = None

class DriftReportSummary(BaseModel):
    """Summary of a drift report."""

    filename: str
    timestamp: str
    drift_detected: bool

class DriftReportsResponse(BaseModel):
    """Response with list of drift reports."""

    reports: list[DriftReportSummary]
    total: int

class LangFuseStatsResponse(BaseModel):
    """LangFuse statistics response."""

    enabled: bool
    connected: bool
    total_traces: int = 0
    success_rate: float = 1.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0

class EvidentlyStatsResponse(BaseModel):
    """Evidently statistics response."""

    enabled: bool
    reference_data_size: int
    current_data_size: int
    last_check: str | None = None
    reports_count: int
    config: dict[str, Any]

# =============================================================================
# STATUS ENDPOINTS
# =============================================================================

@router.get("/status", response_model=ObservabilityStatusResponse)
async def get_status():
    """
    Get current observability status.

    Returns status of LangFuse and Evidently integrations.
    """
    status = get_observability_status()

    # Build response
    langfuse_status = status.get("langfuse", {})
    evidently_status = status.get("evidently", {})
    evidently_stats = evidently_status.get("statistics", {}) or {}

    return ObservabilityStatusResponse(
        langfuse=LangFuseStatusResponse(
            enabled=langfuse_status.get("enabled", False),
            configured=langfuse_status.get("configured", False),
            connected=langfuse_status.get("connected", False),
            host=langfuse_status.get("host"),
        ),
        evidently=EvidentlyStatusResponse(
            enabled=evidently_status.get("enabled", False),
            active=evidently_status.get("active", False),
            reference_data_size=evidently_stats.get("reference_data_size", 0),
            current_data_size=evidently_stats.get("current_data_size", 0),
            last_check=evidently_stats.get("last_check"),
            reports_count=evidently_stats.get("reports_count", 0),
        ),
        environment=status.get("environment", "unknown"),
        service_name=status.get("service", {}).get("name", "resync"),
        service_version=status.get("service", {}).get("version", "unknown"),
    )

@router.post("/setup", response_model=SetupResponse)
async def initialize_observability():
    """
    Initialize or reinitialize observability components.

    Sets up LangFuse and Evidently based on configuration.
    """
    try:
        results = await setup_observability()

        return SetupResponse(
            langfuse=results.get("langfuse", False),
            evidently=results.get("evidently", False),
            message="Observability setup completed",
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("observability_setup_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

# =============================================================================
# LANGFUSE ENDPOINTS
# =============================================================================

@router.get("/langfuse/stats", response_model=LangFuseStatsResponse)
async def get_langfuse_stats():
    """
    Get LangFuse tracing statistics.
    """
    config = get_observability_config().langfuse
    client = get_langfuse_client()

    # Get tracer stats if available
    stats = {
        "enabled": config.enabled,
        "connected": client is not None,
    }

    try:
        from resync.core.langfuse import get_tracer

        tracer = get_tracer()
        tracer_stats = tracer.get_statistics()

        stats.update(
            {
                "total_traces": tracer_stats.get("total_traces", 0),
                "success_rate": tracer_stats.get("success_rate", 1.0),
                "total_tokens": tracer_stats.get("total_tokens", 0),
                "total_cost_usd": tracer_stats.get("total_cost_usd", 0.0),
                "avg_latency_ms": tracer_stats.get("avg_duration_ms", 0.0),
            }
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.debug("langfuse_stats_error", error=str(e))

    return LangFuseStatsResponse(**stats)

# =============================================================================
# EVIDENTLY ENDPOINTS
# =============================================================================

@router.get("/evidently/stats", response_model=EvidentlyStatsResponse)
async def get_evidently_stats():
    """
    Get Evidently monitoring statistics.
    """
    monitor = get_evidently_monitor()

    if not monitor:
        return EvidentlyStatsResponse(
            enabled=False,
            reference_data_size=0,
            current_data_size=0,
            last_check=None,
            reports_count=0,
            config={},
        )

    stats = monitor.get_statistics()

    return EvidentlyStatsResponse(
        enabled=stats.get("enabled", False),
        reference_data_size=stats.get("reference_data_size", 0),
        current_data_size=stats.get("current_data_size", 0),
        last_check=stats.get("last_check"),
        reports_count=stats.get("reports_count", 0),
        config=stats.get("config", {}),
    )

@router.post("/evidently/check", response_model=DriftCheckResponse)
async def run_drift_check():
    """
    Manually run a drift detection check.

    Compares current data against reference data to detect drift.
    """
    monitor = get_evidently_monitor()

    if not monitor:
        raise HTTPException(
            status_code=503,
            detail="Evidently monitor not initialized. Enable it in configuration.",
        )

    result = await monitor.check_drift()

    if "error" in result:
        return DriftCheckResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_detected=False,
            drift_columns=[],
            reference_size=0,
            current_size=0,
            error=result["error"],
        )

    return DriftCheckResponse(
        timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        drift_detected=result.get("drift_detected", False),
        drift_columns=result.get("drift_columns", []),
        reference_size=result.get("reference_size", 0),
        current_size=result.get("current_size", 0),
    )

@router.get("/evidently/reports", response_model=DriftReportsResponse)
async def list_drift_reports(
    limit: int = Query(50, ge=1, le=200),
):
    """
    List drift detection reports.
    """
    monitor = get_evidently_monitor()

    if not monitor:
        return DriftReportsResponse(reports=[], total=0)

    reports = monitor._reports[-limit:]
    reports.reverse()  # Most recent first

    return DriftReportsResponse(
        reports=[
            DriftReportSummary(
                filename=r["filename"],
                timestamp=r["timestamp"],
                drift_detected=r["drift_detected"],
            )
            for r in reports
        ],
        total=len(reports),
    )

# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@router.get("/config")
async def get_config():
    """
    Get current observability configuration.

    Note: Sensitive values (API keys) are masked.
    """
    config = get_observability_config()

    return {
        "langfuse": {
            "enabled": config.langfuse.enabled,
            "host": config.langfuse.host,
            "public_key_set": bool(config.langfuse.public_key),
            "secret_key_set": bool(config.langfuse.secret_key),
            "sample_rate": config.langfuse.sample_rate,
            "flush_interval_seconds": config.langfuse.flush_interval_seconds,
        },
        "evidently": {
            "enabled": config.evidently.enabled,
            "reference_window_days": config.evidently.reference_window_days,
            "current_window_hours": config.evidently.current_window_hours,
            "feature_drift_threshold": config.evidently.feature_drift_threshold,
            "prediction_drift_threshold": config.evidently.prediction_drift_threshold,
            "check_interval_minutes": config.evidently.check_interval_minutes,
            "report_retention_days": config.evidently.report_retention_days,
            "reports_dir": config.evidently.reports_dir,
        },
        "environment": config.environment,
        "service_name": config.service_name,
        "service_version": config.service_version,
    }
