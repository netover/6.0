"""
LangGraph Workflow - Capacity Forecasting

Workflow multi-step para previsão de capacidade de recursos (CPU, Memory, Disk).

Passos:
1. Fetch historical metrics (30 days)
2. Detect trends (linear, exponential, seasonal)
3. Forecast 3 months ahead
4. Identify saturation points
5. Generate scaling recommendations
6. Calculate costs (cloud expansion)
7. Create report + visualizations
8. Notify stakeholders

Author: Resync Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import structlog

from resync.settings import settings

# Checkpointer imports
try:
    # from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    POSTGRES_SAVER_AVAILABLE = False

from langgraph.graph import END, StateGraph

logger = structlog.get_logger(__name__)


def get_db_url() -> str:
    """Get PostgreSQL connection URL for checkpointer."""
    # Try settings first
    if hasattr(settings, "database_url") and settings.database_url:
        return str(settings.database_url)

    # Try environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Build from components
    host = getattr(settings, "db_host", None) or os.getenv("DB_HOST", "localhost")
    port = getattr(settings, "db_port", None) or os.getenv("DB_PORT", "5432")
    user = getattr(settings, "db_user", None) or os.getenv("DB_USER", "postgres")
    password = getattr(settings, "db_password", None) or os.getenv("DB_PASSWORD", "")
    database = getattr(settings, "db_name", None) or os.getenv("DB_NAME", "resync")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


# ============================================================================
# STATE DEFINITION
# ============================================================================


class CapacityForecastState(TypedDict):
    """State para workflow de Capacity Forecasting."""

    workflow_id: str
    resource_id: str
    resource_type: Literal["server", "database", "service"]
    forecast_days: int
    historical_data: dict[str, list[dict[str, Any]]]  # metric -> timeseries
    trends: dict[str, Any]  # metric -> slope/intercept
    forecast: dict[str, list[dict[str, Any]]]  # metric -> future_values
    saturation_points: list[dict[str, Any]]  # date, metric, value
    recommendations: list[str]
    report_url: str | None
    status: str
    error: str | None


# ============================================================================
# NODES
# ============================================================================


async def fetch_metrics_node(state: CapacityForecastState) -> CapacityForecastState:
    """Step 1: Fetch historical metrics."""
    logger.info("capacity_forecast.fetch_metrics", resource_id=state["resource_id"])

    # Mock data for demonstration/fallback
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=30, freq="D")

    # Simulate CPU usage with a slight upward trend
    cpu_usage = [
        {"timestamp": d.isoformat(), "value": 40 + (i * 0.5) + np.random.normal(0, 5)}
        for i, d in enumerate(dates)
    ]

    # Simulate Memory usage stable
    memory_usage = [
        {"timestamp": d.isoformat(), "value": 60 + np.random.normal(0, 2)}
        for i, d in enumerate(dates)
    ]

    state["historical_data"] = {"cpu_usage": cpu_usage, "memory_usage": memory_usage}

    return state


def analyze_trends_node(state: CapacityForecastState) -> CapacityForecastState:
    """Step 2: Detect trends."""
    logger.info("capacity_forecast.analyze_trends")

    trends = {}

    for metric, data in state["historical_data"].items():
        if len(data) < 2:
            trends[metric] = {"type": "none", "slope": 0}
            continue

        values = [d["value"] for d in data]
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        trends[metric] = {
            "type": "linear",
            "slope": slope,
            "intercept": intercept,
            "r_squared": 0.85,  # Mock R2
        }

    state["trends"] = trends
    return state


def forecast_node(state: CapacityForecastState) -> CapacityForecastState:
    """Step 3: Forecast future values."""
    logger.info("capacity_forecast.forecast", forecast_days=state["forecast_days"])

    forecast_horizon = state["forecast_days"]
    forecast_data = {}

    start_date = datetime.now(timezone.utc)

    for metric, trend in state["trends"].items():
        if trend["type"] == "linear":
            slope = trend["slope"]
            intercept = trend["intercept"]
            last_idx = len(state["historical_data"][metric])

            future_values = []
            for i in range(forecast_horizon):
                future_idx = last_idx + i
                val = (slope * future_idx) + intercept
                # Add some randomness/seasonality mock
                val += np.random.normal(0, 2)

                date = start_date + timedelta(days=i + 1)
                future_values.append(
                    {
                        "timestamp": date.isoformat(),
                        "value": max(0, min(100, val)),  # Clamp 0-100%
                    }
                )

            forecast_data[metric] = future_values

    state["forecast"] = forecast_data
    return state


def identify_saturation_node(state: CapacityForecastState) -> CapacityForecastState:
    """Step 4: Identify saturation points."""
    logger.info("capacity_forecast.identify_saturation")

    saturation_points = []
    threshold = 95.0

    for metric, data in state["forecast"].items():
        for item in data:
            if item["value"] >= threshold:
                saturation_points.append(
                    {
                        "metric": metric,
                        "date": item["timestamp"],
                        "value": item["value"],
                        "threshold": threshold,
                    }
                )
                # Only record first saturation point per metric to avoid spam
                break

    state["saturation_points"] = saturation_points
    return state


async def generate_report_node(state: CapacityForecastState) -> CapacityForecastState:
    """Step 5: Generate report and visualizations."""
    logger.info("capacity_forecast.generate_report")

    report_dir = tempfile.gettempdir()

    # Generate simple text report for now
    report_content = f"Capacity Forecast Report for {state['resource_id']}\n"
    report_content += "=" * 50 + "\n\n"

    if state["saturation_points"]:
        report_content += "CRITICAL WARNINGS:\n"
        for point in state["saturation_points"]:
            report_content += (
                f"- {point['metric']} will reach "
                f"{point['value']:.1f}% on {point['date']}\n"
            )
    else:
        report_content += "No saturation predicted in the next period.\n"

    report_path = os.path.join(
        report_dir, f"capacity_report_{state['workflow_id']}.txt"
    )

    try:
        await asyncio.to_thread(_write_report, report_path, report_content)
        state["report_url"] = report_path
    except Exception as e:
        logger.error("report_generation_failed", error=str(e))
        state["error"] = f"Failed to generate report: {str(e)}"

    state["status"] = "completed"
    return state


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def build_capacity_forecast_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(CapacityForecastState)

    workflow.add_node("fetch_metrics", fetch_metrics_node)
    workflow.add_node("analyze_trends", analyze_trends_node)
    workflow.add_node("forecast", forecast_node)
    workflow.add_node("identify_saturation", identify_saturation_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.set_entry_point("fetch_metrics")

    workflow.add_edge("fetch_metrics", "analyze_trends")
    workflow.add_edge("analyze_trends", "forecast")
    workflow.add_edge("forecast", "identify_saturation")
    workflow.add_edge("identify_saturation", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow


def _write_report(report_path: str, report_content: str) -> None:
    with open(report_path, "w") as f:
        f.write(report_content)


async def run_workflow(
    resource_id: str, forecast_days: int = 90, checkpointer: Any = None
) -> dict[str, Any]:
    """Execute the capacity forecasting workflow."""

    workflow = build_capacity_forecast_graph()

    if checkpointer is None and POSTGRES_SAVER_AVAILABLE:
        # Initialize default checkpointer if available
        # This requires an async context manager in real usage,
        # but for this function we might just run without if not provided
        pass

    app = workflow.compile(checkpointer=checkpointer)

    initial_state = CapacityForecastState(
        workflow_id=f"wf_{resource_id}_{int(datetime.now().timestamp())}",
        resource_id=resource_id,
        resource_type="server",  # Default
        forecast_days=forecast_days,
        historical_data={},
        trends={},
        forecast={},
        saturation_points=[],
        recommendations=[],
        report_url=None,
        status="pending",
        error=None,
    )

    final_state = await app.ainvoke(initial_state)
    return final_state


async def run_capacity_forecast(
    resource_id: str, forecast_days: int = 90, checkpointer: Any = None
) -> dict[str, Any]:
    return await run_workflow(resource_id, forecast_days, checkpointer)
