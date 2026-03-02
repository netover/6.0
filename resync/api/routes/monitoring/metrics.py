"""
Metrics Dashboard API - REST endpoints for the learning metrics dashboard.

Provides:
- Dashboard data for visualization
- Time series metrics
- Real-time gauges
- System health metrics
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics Dashboard"])

# Templates directory
templates_dir = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""

    timestamp: str
    value: float

class ChartData(BaseModel):
    """Data for a chart visualization."""

    label: str
    data: list[TimeSeriesPoint]
    color: str | None = None

class GaugeData(BaseModel):
    """Current value gauge."""

    name: str
    value: float
    unit: str | None = None
    status: str = "ok"  # ok, warning, critical

class DashboardResponse(BaseModel):
    """Complete dashboard data response."""

    summary: dict[str, Any]
    charts: dict[str, list[TimeSeriesPoint]]
    gauges: list[GaugeData]
    system: dict[str, Any]
    generated_at: str

class MetricSummary(BaseModel):
    """Summary for a specific metric."""

    name: str
    current: float
    avg_1h: float
    avg_24h: float
    min_24h: float
    max_24h: float
    trend: str  # up, down, stable

# =============================================================================
# DASHBOARD PAGE
# =============================================================================

@router.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard(request: Request):
    """Serve the metrics dashboard HTML page."""
    return templates.TemplateResponse(
        request,
        "metrics_dashboard.html",
        {"title": "Continual Learning Metrics"},
    )

# =============================================================================
# DATA ENDPOINTS
# =============================================================================

@router.get("/data", response_model=DashboardResponse)
async def get_dashboard_data(
    hours: int = Query(24, ge=1, le=168, description="Hours of data to return"),
):
    """
    Get complete dashboard data.

    Returns summary, charts, gauges, and system metrics.
    """
    try:
        from resync.core.metrics import MetricNames, get_metrics_store

        store = get_metrics_store()
        await store.initialize()

        # Get summary from store
        summary = await store.get_summary()

        # Get chart data
        charts = {}

        # Query volume chart
        query_data = await store.get_aggregated(
            MetricNames.QUERY_TOTAL,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )
        charts["queries"] = [
            TimeSeriesPoint(timestamp=m.period_start.isoformat(), value=m.sum_value)
            for m in query_data
        ]

        # Response time chart
        response_time_data = await store.get_aggregated(
            MetricNames.QUERY_DURATION_MS,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )
        charts["response_time"] = [
            TimeSeriesPoint(timestamp=m.period_start.isoformat(), value=m.avg_value)
            for m in response_time_data
        ]

        # Feedback chart
        feedback_data = await store.get_aggregated(
            MetricNames.FEEDBACK_TOTAL,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )
        charts["feedback"] = [
            TimeSeriesPoint(timestamp=m.period_start.isoformat(), value=m.sum_value)
            for m in feedback_data
        ]

        # Enrichment chart
        enrichment_data = await store.get_aggregated(
            MetricNames.ENRICHMENT_TOTAL,
            period="hour" if hours > 6 else "minute",
            hours=hours,
        )
        charts["enrichment"] = [
            TimeSeriesPoint(timestamp=m.period_start.isoformat(), value=m.sum_value)
            for m in enrichment_data
        ]

        # Build gauges
        gauges = []

        # Review queue gauge
        queue_size = store.get_gauge(MetricNames.REVIEW_QUEUE_SIZE) or 0
        gauges.append(
            GaugeData(
                name="Review Queue",
                value=queue_size,
                unit="items",
                status="critical"
                if queue_size > 100
                else "warning"
                if queue_size > 50
                else "ok",
            )
        )

        # Feedback rate gauge
        total_feedback = store.get_counter(MetricNames.FEEDBACK_TOTAL)
        positive_feedback = store.get_counter(MetricNames.FEEDBACK_POSITIVE)
        positive_rate = (
            (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        )
        gauges.append(
            GaugeData(
                name="Positive Feedback",
                value=round(positive_rate, 1),
                unit="%",
                status="ok"
                if positive_rate >= 80
                else "warning"
                if positive_rate >= 60
                else "critical",
            )
        )

        # Enrichment rate gauge
        total_queries = store.get_counter(MetricNames.QUERY_TOTAL)
        enriched_queries = store.get_counter(MetricNames.QUERY_WITH_ENRICHMENT)
        enrichment_rate = (
            (enriched_queries / total_queries * 100) if total_queries > 0 else 0
        )
        gauges.append(
            GaugeData(
                name="Enrichment Rate",
                value=round(enrichment_rate, 1),
                unit="%",
                status="ok"
                if enrichment_rate >= 30
                else "warning"
                if enrichment_rate >= 10
                else "critical",
            )
        )

        # System metrics
        process = psutil.Process()
        system = {
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "db_records": summary.get("storage", {}).get("raw_records", 0)
            + summary.get("storage", {}).get("aggregated_records", 0),
        }

        # DB size placeholder for PostgreSQL deployments.
        # Future improvement: expose pg_database_size() via repository/metrics service.
        system["db_size_mb"] = 0

        return DashboardResponse(
            summary=summary,
            charts=charts,
            gauges=gauges,
            system=system,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("dashboard_data_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/series/{metric_name}")
async def get_metric_series(
    metric_name: str,
    hours: int = Query(24, ge=1, le=168),
    period: str = Query("hour", pattern="^(minute|hour|day)$"),
):
    """Get time series data for a specific metric."""
    try:
        from resync.core.metrics import AggregationPeriod, get_metrics_store

        store = get_metrics_store()
        await store.initialize()

        data = await store.get_aggregated(
            metric_name,
            period=AggregationPeriod(period),
            hours=hours,
        )

        return {
            "metric": metric_name,
            "period": period,
            "hours": hours,
            "data": [
                {
                    "timestamp": m.period_start.isoformat(),
                    "count": m.count,
                    "sum": m.sum_value,
                    "avg": m.avg_value,
                    "min": m.min_value,
                    "max": m.max_value,
                }
                for m in data
            ],
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("metric_series_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/summary")
async def get_metrics_summary():
    """Get summary statistics for all metrics."""
    try:
        from resync.core.metrics import get_metrics_store

        store = get_metrics_store()
        await store.initialize()

        summary = await store.get_summary()
        metric_names = await store.get_metric_names()

        return {
            "summary": summary,
            "available_metrics": metric_names,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("metrics_summary_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/gauges")
async def get_current_gauges():
    """Get current values for all gauge metrics."""
    try:
        from resync.core.metrics import MetricNames, get_metrics_store

        store = get_metrics_store()
        await store.initialize()

        gauges = {}

        # Get all gauge values
        gauge_names = [
            MetricNames.REVIEW_QUEUE_SIZE,
            MetricNames.FEEDBACK_RATING_AVG,
            MetricNames.SYSTEM_MEMORY_MB,
            MetricNames.SYSTEM_CPU_PERCENT,
            MetricNames.SYSTEM_DB_SIZE_MB,
        ]

        for name in gauge_names:
            value = store.get_gauge(name)
            if value is not None:
                gauges[name] = value

        # Get counter totals
        counter_names = [
            MetricNames.QUERY_TOTAL,
            MetricNames.FEEDBACK_TOTAL,
            MetricNames.FEEDBACK_POSITIVE,
            MetricNames.FEEDBACK_NEGATIVE,
            MetricNames.REVIEW_ADDED,
            MetricNames.REVIEW_COMPLETED,
            MetricNames.ENRICHMENT_TOTAL,
            MetricNames.AUDIT_PROCESSED,
        ]

        counters = {}
        for name in counter_names:
            counters[name] = store.get_counter(name)

        return {
            "gauges": gauges,
            "counters": counters,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("gauges_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/health")
async def metrics_health():
    """Check health of the metrics system."""
    try:
        from resync.core.metrics import get_metrics_store

        store = get_metrics_store()
        await store.initialize()

        # Get basic stats
        summary = await store.get_summary()

        status = "healthy"
        issues = []

        # Check if we have recent data
        raw_count = summary.get("storage", {}).get("raw_records", 0)
        if raw_count == 0:
            issues.append("No raw metrics recorded")
            status = "degraded"

        return {
            "status": status,
            "issues": issues,
            "storage": summary.get("storage", {}),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

# =============================================================================
# CONTINUAL LEARNING SPECIFIC ENDPOINTS
# =============================================================================

@router.get("/continual-learning")
async def get_cl_dashboard_data(
    hours: int = Query(24, ge=1, le=168),
):
    """Get continual learning specific dashboard data."""
    try:
        from resync.core.metrics import get_cl_metrics

        cl_metrics = get_cl_metrics()
        return await cl_metrics.get_dashboard_data(hours=hours)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("cl_dashboard_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/feedback-analysis")
async def get_feedback_analysis(
    days: int = Query(7, ge=1, le=30),
):
    """Get detailed feedback analysis."""
    try:
        from resync.core.metrics import (
            AggregationPeriod,
            MetricNames,
            get_metrics_store,
        )

        store = get_metrics_store()
        await store.initialize()

        # Get daily feedback data
        total_data = await store.get_aggregated(
            MetricNames.FEEDBACK_TOTAL,
            period=AggregationPeriod.DAY,
            hours=days * 24,
        )

        positive_data = await store.get_aggregated(
            MetricNames.FEEDBACK_POSITIVE,
            period=AggregationPeriod.DAY,
            hours=days * 24,
        )

        negative_data = await store.get_aggregated(
            MetricNames.FEEDBACK_NEGATIVE,
            period=AggregationPeriod.DAY,
            hours=days * 24,
        )

        # Calculate daily rates
        daily_stats = []
        positive_by_date = {m.period_start.date(): m.sum_value for m in positive_data}
        negative_by_date = {m.period_start.date(): m.sum_value for m in negative_data}

        for m in total_data:
            date = m.period_start.date()
            pos = positive_by_date.get(date, 0)
            neg = negative_by_date.get(date, 0)
            rate = (pos / m.sum_value * 100) if m.sum_value > 0 else 0

            daily_stats.append(
                {
                    "date": date.isoformat(),
                    "total": m.sum_value,
                    "positive": pos,
                    "negative": neg,
                    "positive_rate": round(rate, 1),
                }
            )

        return {
            "daily_stats": daily_stats,
            "period_days": days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("feedback_analysis_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e
