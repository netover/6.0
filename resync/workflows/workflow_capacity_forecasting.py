# pylint: skip-file
# mypy: ignore-errors
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

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import structlog
from resync.core.utils.llm_factories import LLMFactory
from sqlalchemy.ext.asyncio import AsyncSession
from resync.core.database import get_async_session

POSTGRES_SAVER_AVAILABLE = False
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:

    class HumanMessage:
        pass

    class SystemMessage:
        pass


try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = "END"

    class StateGraph:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    try:
        from langgraph_checkpoint_postgres import AsyncPostgresSaver

        POSTGRES_SAVER_AVAILABLE = True
    except ImportError:
        AsyncPostgresSaver = None  # type: ignore
        POSTGRES_SAVER_AVAILABLE = False

logger = structlog.get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_checkpointer_db_url() -> str:
    """Get PostgreSQL connection URL for checkpointer."""
    import os
    from resync.settings import settings

    # Try settings first
    if hasattr(settings, "database_url") and settings.database_url:
        return settings.database_url

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

    # Input
    workstation: str | None  # None = all workstations
    lookback_days: int
    forecast_days: int

    # Fetched data
    metrics_history: list[dict[str, Any]]
    job_history: list[dict[str, Any]]

    # Trend analysis
    cpu_trend: dict[str, Any]
    memory_trend: dict[str, Any]
    disk_trend: dict[str, Any]
    workload_trend: dict[str, Any]  # job counts

    # Forecasts
    cpu_forecast: list[dict[str, Any]]
    memory_forecast: list[dict[str, Any]]
    disk_forecast: list[dict[str, Any]]
    workload_forecast: list[dict[str, Any]]

    # Saturation analysis
    cpu_saturation_date: datetime | None
    memory_saturation_date: datetime | None
    disk_saturation_date: datetime | None
    saturation_confidence: float

    # Recommendations
    recommendations: list[dict[str, Any]]
    scaling_options: list[dict[str, Any]]
    estimated_costs: dict[str, float]

    # Report
    report_path: str | None
    visualizations: list[str]

    # Metadata
    workflow_id: str
    started_at: datetime
    completed_at: datetime | None
    status: Literal["running", "completed", "failed"]
    error: str | None


# ============================================================================
# NODES
# ============================================================================


async def fetch_metrics_node(
    state: CapacityForecastState, db: AsyncSession
) -> CapacityForecastState:
    """
    Step 1: Fetch historical metrics.

    Busca:
    - Workstation metrics (CPU, memory, disk) - 30 dias
    - Job execution history (workload) - 30 dias
    """
    logger.info(
        "capacity_forecast.fetch_metrics",
        workstation=state["workstation"],
        lookback_days=state["lookback_days"],
    )

    try:
        from resync.workflows.nodes import (
            fetch_job_execution_history,
            fetch_workstation_metrics_history,
        )

        # Fetch metrics
        metrics_history = await fetch_workstation_metrics_history(
            db=db, workstation=state["workstation"], days=state["lookback_days"]
        )

        # Fetch job history
        job_history = await fetch_job_execution_history(
            db=db, workstation=state["workstation"], days=state["lookback_days"]
        )

        logger.info(
            "capacity_forecast.data_fetched",
            metrics_count=len(metrics_history),
            jobs_count=len(job_history),
        )

        return {**state, "metrics_history": metrics_history, "job_history": job_history}

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("capacity_forecast.fetch_failed", error=str(e))
        return {**state, "status": "failed", "error": f"Failed to fetch data: {str(e)}"}


async def analyze_trends_node(
    state: CapacityForecastState,
    llm: Any,  # Typed as Any to support flexible backend (ChatLiteLLM)
) -> CapacityForecastState:
    """
    Step 2: Detect trends.

    Analisa tendências usando:
    - Linear regression
    - Exponential smoothing
    - Seasonal decomposition
    - LLM para pattern recognition
    """
    logger.info("capacity_forecast.analyze_trends")

    if not state["metrics_history"]:
        return {
            **state,
            "cpu_trend": {"type": "none", "slope": 0},
            "memory_trend": {"type": "none", "slope": 0},
            "disk_trend": {"type": "none", "slope": 0},
            "workload_trend": {"type": "none", "slope": 0},
        }

    try:
        # Convert to DataFrame
        df = pd.DataFrame(state["metrics_history"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Analyze CPU trend
        cpu_trend = analyze_metric_trend(df, "cpu_percent")

        # Analyze Memory trend
        memory_trend = analyze_metric_trend(df, "memory_percent")

        # Analyze Disk trend
        disk_trend = analyze_metric_trend(df, "disk_percent")

        # Analyze Workload trend (job counts per day)
        workload_df = pd.DataFrame(state["job_history"])
        if not workload_df.empty:
            workload_df["date"] = pd.to_datetime(workload_df["start_time"]).dt.date
            daily_jobs = (
                workload_df.groupby("date").size().reset_index(name="job_count")
            )
            workload_trend = analyze_workload_trend(daily_jobs)
        else:
            workload_trend = {"type": "none", "slope": 0}

        # Use LLM to enrich analysis
        llm_analysis = await llm.ainvoke(
            [
                SystemMessage(content="You are a capacity planning expert."),
                HumanMessage(
                    content=f"""
                Analyze these trends:

                CPU: {cpu_trend["type"]} trend, slope {cpu_trend["slope"]:.2f}%/day
                Memory: {memory_trend["type"]} trend, slope {memory_trend["slope"]:.2f}%/day
                Disk: {disk_trend["type"]} trend, slope {disk_trend["slope"]:.2f}%/day
                Workload: {workload_trend["type"]} trend, slope {workload_trend["slope"]:.2f} jobs/day

                Provide insights on:
                1. Most concerning trend
                2. Root causes
                3. Business impact

                Respond in JSON format.
            """
                ),
            ]
        )

        # Parse LLM response (simplified)
        insights = {
            "most_concerning": "cpu"
            if cpu_trend["slope"] > max(memory_trend["slope"], disk_trend["slope"])
            else "memory",
            "analysis": llm_analysis.content,
        }

        return {
            **state,
            "cpu_trend": {**cpu_trend, "insights": insights},
            "memory_trend": memory_trend,
            "disk_trend": disk_trend,
            "workload_trend": workload_trend,
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("capacity_forecast.trend_analysis_failed", error=str(e))
        return state


def analyze_metric_trend(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    """
    Analyze trend for a specific metric.

    Returns:
        dict with type, slope, r_squared, etc.
    """
    # Linear regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric].values

    # Simple linear fit
    coeffs = np.polyfit(X.flatten(), y, 1)
    slope_per_sample = coeffs[0]

    # Convert to per-day slope (assuming 5min intervals)
    samples_per_day = 24 * 60 / 5  # 288 samples/day
    slope_per_day = slope_per_sample * samples_per_day

    # R-squared
    y_pred = np.polyval(coeffs, X.flatten())
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Determine trend type
    if abs(slope_per_day) < 0.1:
        trend_type = "stable"
    elif slope_per_day > 0.5:
        trend_type = "increasing"
    elif slope_per_day < -0.5:
        trend_type = "decreasing"
    else:
        trend_type = "slight"

    return {
        "type": trend_type,
        "slope": slope_per_day,
        "r_squared": r_squared,
        "current_value": float(df[metric].iloc[-1]),
        "mean_value": float(df[metric].mean()),
    }


def analyze_workload_trend(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze workload (job count) trend."""
    if len(df) < 2:
        return {"type": "none", "slope": 0}

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["job_count"].values

    coeffs = np.polyfit(X.flatten(), y, 1)
    slope_per_day = coeffs[0]

    trend_type = "stable"
    if slope_per_day > 1:
        trend_type = "increasing"
    elif slope_per_day < -1:
        trend_type = "decreasing"

    return {
        "type": trend_type,
        "slope": slope_per_day,
        "current_value": int(df["job_count"].iloc[-1]),
        "mean_value": float(df["job_count"].mean()),
    }


def forecast_node(state: CapacityForecastState, llm: Any) -> CapacityForecastState:
    """
    Step 3: Forecast 3 months ahead.

    Usa:
    - Linear extrapolation
    - Exponential smoothing
    - LLM para ajustes (seasonal, events)
    """
    logger.info("capacity_forecast.forecast", forecast_days=state["forecast_days"])

    try:
        # Generate forecasts
        cpu_forecast = generate_forecast(state["cpu_trend"], state["forecast_days"])

        memory_forecast = generate_forecast(
            state["memory_trend"], state["forecast_days"]
        )

        disk_forecast = generate_forecast(state["disk_trend"], state["forecast_days"])

        workload_forecast = generate_forecast(
            state["workload_trend"], state["forecast_days"]
        )

        return {
            **state,
            "cpu_forecast": cpu_forecast,
            "memory_forecast": memory_forecast,
            "disk_forecast": disk_forecast,
            "workload_forecast": workload_forecast,
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("capacity_forecast.forecast_failed", error=str(e))
        return state


def generate_forecast(trend: dict[str, Any], days: int) -> list[dict[str, Any]]:
    """Generate forecast for next N days."""
    current_value = trend.get("current_value", 0)
    slope = trend.get("slope", 0)

    forecast = []
    for day in range(1, days + 1):
        predicted_value = current_value + (slope * day)

        # Clamp to 0-100 for percentages
        if "percent" in str(trend):
            predicted_value = max(0, min(100, predicted_value))

        forecast.append(
            {
                "day": day,
                "date": (datetime.now(timezone.utc) + timedelta(days=day))
                .date()
                .isoformat(),
                "value": round(predicted_value, 2),
            }
        )

    return forecast


def analyze_saturation_node(state: CapacityForecastState) -> CapacityForecastState:
    """
    Step 4: Identify saturation points.

    Determina quando recursos atingirão 95% (saturation).
    """
    logger.info("capacity_forecast.analyze_saturation")

    # CPU saturation
    cpu_saturation = find_saturation_date(state["cpu_forecast"], threshold=95)

    # Memory saturation
    memory_saturation = find_saturation_date(state["memory_forecast"], threshold=95)

    # Disk saturation
    disk_saturation = find_saturation_date(
        state["disk_forecast"],
        threshold=90,  # Lower threshold for disk
    )

    # Calculate confidence based on R²
    confidence = min(
        state["cpu_trend"].get("r_squared", 0),
        state["memory_trend"].get("r_squared", 0),
        state["disk_trend"].get("r_squared", 0),
    )

    return {
        **state,
        "cpu_saturation_date": cpu_saturation,
        "memory_saturation_date": memory_saturation,
        "disk_saturation_date": disk_saturation,
        "saturation_confidence": confidence,
    }


def find_saturation_date(
    forecast: list[dict[str, Any]], threshold: float
) -> datetime | None:
    """Find when metric will exceed threshold."""
    for item in forecast:
        if item["value"] >= threshold:
            return datetime.fromisoformat(item["date"])

    return None


def recommend_node(state: CapacityForecastState, llm: Any) -> CapacityForecastState:
    """
    Step 5: Generate recommendations.

    Recommendations:
    - Scale up (CPU, memory, disk)
    - Optimize workload
    - Archive data
    - Redistribute jobs
    """
    logger.info("capacity_forecast.recommend")

    recommendations = []
    scaling_options = []
    estimated_costs = {}

    # CPU recommendations
    if state["cpu_saturation_date"]:
        days_until = (state["cpu_saturation_date"] - datetime.now(timezone.utc)).days

        recommendations.append(
            {
                "priority": "high" if days_until < 30 else "medium",
                "resource": "cpu",
                "action": "scale_up",
                "reason": f"CPU will saturate in {days_until} days",
                "options": [
                    "Upgrade to 16 cores (from 8)",
                    "Optimize job scheduling",
                    "Redistribute heavy jobs",
                ],
            }
        )

        scaling_options.append(
            {
                "resource": "cpu",
                "current": "8 cores",
                "recommended": "16 cores",
                "cost_monthly": 500,
                "timeline": f"{days_until} days",
            }
        )

        estimated_costs["cpu_upgrade"] = 500

    # Memory recommendations
    if state["memory_saturation_date"]:
        days_until = (state["memory_saturation_date"] - datetime.now(timezone.utc)).days

        recommendations.append(
            {
                "priority": "high" if days_until < 30 else "medium",
                "resource": "memory",
                "action": "scale_up",
                "reason": f"Memory will saturate in {days_until} days",
                "options": [
                    "Increase to 64GB (from 32GB)",
                    "Implement memory limits per job",
                    "Review memory leaks",
                ],
            }
        )

        scaling_options.append(
            {
                "resource": "memory",
                "current": "32GB",
                "recommended": "64GB",
                "cost_monthly": 300,
                "timeline": f"{days_until} days",
            }
        )

        estimated_costs["memory_upgrade"] = 300

    # Disk recommendations
    if state["disk_saturation_date"]:
        days_until = (state["disk_saturation_date"] - datetime.now(timezone.utc)).days

        recommendations.append(
            {
                "priority": "critical" if days_until < 14 else "high",
                "resource": "disk",
                "action": "expand_storage",
                "reason": f"Disk will saturate in {days_until} days",
                "options": [
                    "Add 500GB storage",
                    "Archive logs older than 90 days",
                    "Implement data lifecycle policy",
                ],
            }
        )

        scaling_options.append(
            {
                "resource": "disk",
                "current": "500GB",
                "recommended": "1TB",
                "cost_monthly": 200,
                "timeline": f"{days_until} days",
            }
        )

        estimated_costs["disk_expansion"] = 200

    return {
        **state,
        "recommendations": recommendations,
        "scaling_options": scaling_options,
        "estimated_costs": estimated_costs,
    }


def generate_report_node(state: CapacityForecastState) -> CapacityForecastState:
    """
    Step 6: Generate report + visualizations.

    Creates:
    - PDF report
    - Charts (CPU, memory, disk trends)
    - Executive summary
    """
    logger.info("capacity_forecast.generate_report")

    # DEBT: Implement PDF/chart report generation (requires matplotlib/reportlab)

    report_path = f"/tmp/capacity_forecast_{state['workflow_id']}.pdf"
    visualizations = [
        f"/tmp/cpu_forecast_{state['workflow_id']}.png",
        f"/tmp/memory_forecast_{state['workflow_id']}.png",
        f"/tmp/disk_forecast_{state['workflow_id']}.png",
    ]

    return {
        **state,
        "report_path": report_path,
        "visualizations": visualizations,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc),
    }


# ============================================================================
# WORKFLOW GRAPH
# ============================================================================


def create_capacity_forecast_workflow(
    llm: Any, db: AsyncSession, checkpointer: Any
) -> StateGraph:
    """Create Capacity Forecasting workflow graph."""

    workflow = StateGraph(CapacityForecastState)

    # Add nodes
    workflow.add_node("fetch", lambda s: fetch_metrics_node(s, db))
    workflow.add_node("analyze", lambda s: analyze_trends_node(s, llm))
    workflow.add_node("forecast", lambda s: forecast_node(s, llm))
    workflow.add_node("saturation", analyze_saturation_node)
    workflow.add_node("recommend", lambda s: recommend_node(s, llm))
    workflow.add_node("report", generate_report_node)

    # Define edges
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "analyze")
    workflow.add_edge("analyze", "forecast")
    workflow.add_edge("forecast", "saturation")
    workflow.add_edge("saturation", "recommend")
    workflow.add_edge("recommend", "report")
    workflow.add_edge("report", END)

    return workflow.compile(checkpointer=checkpointer)


# ============================================================================
# RUNNER
# ============================================================================


async def run_capacity_forecast(
    workstation: str | None = None, lookback_days: int = 30, forecast_days: int = 90
) -> dict[str, Any]:
    """Run Capacity Forecasting workflow."""

    from resync.settings import settings

    model_name = getattr(settings, "agent_model_name", None) or getattr(
        settings, "llm_model", "gpt-4o"
    )
    llm = LLMFactory.get_langchain_llm(model=model_name)

    # Create checkpointer first (outside the db session)
    if POSTGRES_SAVER_AVAILABLE and AsyncPostgresSaver is not None:
        try:
            db_url = _get_checkpointer_db_url()
            checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
            await checkpointer.setup()
        except Exception as e:
            logger.warning("postgres_checkpointer_failed_using_memory", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    # Then use db for the workflow
    async with get_async_session() as db:
        workflow = create_capacity_forecast_workflow(
            llm=llm, db=db, checkpointer=checkpointer
        )

        workflow_id = (
            f"cf_{workstation or 'all'}_{datetime.now(timezone.utc).timestamp()}"
        )

        initial_state: CapacityForecastState = {
            "workstation": workstation,
            "lookback_days": lookback_days,
            "forecast_days": forecast_days,
            "metrics_history": [],
            "job_history": [],
            "cpu_trend": {},
            "memory_trend": {},
            "disk_trend": {},
            "workload_trend": {},
            "cpu_forecast": [],
            "memory_forecast": [],
            "disk_forecast": [],
            "workload_forecast": [],
            "cpu_saturation_date": None,
            "memory_saturation_date": None,
            "disk_saturation_date": None,
            "saturation_confidence": 0.0,
            "recommendations": [],
            "scaling_options": [],
            "estimated_costs": {},
            "report_path": None,
            "visualizations": [],
            "workflow_id": workflow_id,
            "started_at": datetime.now(timezone.utc),
            "completed_at": None,
            "status": "running",
            "error": None,
        }

        config = {"configurable": {"thread_id": workflow_id}}

        result = await workflow.ainvoke(initial_state, config=config)

        logger.info(
            "capacity_forecast.completed",
            workflow_id=workflow_id,
            status=result["status"],
        )

        return result
