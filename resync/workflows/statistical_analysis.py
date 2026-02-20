"""Statistical analysis utilities for predictive workflows.

This module provides functions for correlating metrics, detecting degradation,
and generating predictions. Extracted from nodes_optimized.py to reduce complexity.
"""

from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_NUMPY_AVAILABLE = False
try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore

_PANDAS_AVAILABLE = False
try:
    import pandas as _pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _pd = None  # type: ignore

_SCIPY_AVAILABLE = False
try:
    import scipy.stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None  # type: ignore


def build_hour_bucket(ts: Any) -> datetime | None:
    """Convert timestamp to hour bucket (rounds down to hour)."""
    if isinstance(ts, datetime):
        return ts.replace(minute=0, second=0, microsecond=0)
    return None


def extract_job_rows(job_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract and normalize job data into rows with hour buckets."""
    rows: list[dict[str, Any]] = []

    for record in job_history:
        ts = record.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        duration = _safe_float(record.get("duration_seconds"))
        if duration is None:
            start = record.get("start_time")
            end = record.get("end_time")
            if start and end:
                duration = _safe_float(_duration_seconds(start, end))

        if duration is None:
            continue

        status = (record.get("status") or "").upper()
        rows.append(
            {
                "t": build_hour_bucket(ts),
                "duration": duration,
                "failed": 1.0
                if status in {"ABEND", "ERROR", "FAILED", "FAIL"}
                else 0.0,
            }
        )

    return rows


def extract_workstation_rows(
    workstation_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract and normalize workstation metrics into rows with hour buckets."""
    rows: list[dict[str, Any]] = []

    for metric in workstation_metrics:
        ts = metric.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        rows.append(
            {
                "t": build_hour_bucket(ts),
                "cpu": _safe_float(metric.get("cpu_usage")),
                "mem": _safe_float(metric.get("memory_usage")),
                "active_jobs": _safe_float(metric.get("active_jobs")),
            }
        )

    return rows


def aggregate_by_hour(rows: list[dict[str, Any]], key: str) -> dict[datetime, float]:
    """Aggregate rows by hour, computing mean for the given key."""
    buckets: dict[datetime, list[float]] = {}

    for row in rows:
        t = row.get("t")
        value = row.get(key)
        if t is None or value is None:
            continue
        buckets.setdefault(t, []).append(float(value))

    return {t: sum(values) / len(values) for t, values in buckets.items() if values}


def calculate_correlations(
    duration: list[float],
    cpu: list[float],
    mem: list[float],
    active_jobs: list[float],
    failures: list[float],
) -> dict[str, Any]:
    """Calculate Pearson correlations between duration and other metrics."""
    if _PANDAS_AVAILABLE:
        df = _pd.DataFrame(
            {
                "duration": duration,
                "cpu": cpu,
                "mem": mem,
                "active_jobs": active_jobs,
                "failures": failures,
            }
        )
        corr = df.corr(method="pearson")
        return {
            "method": "pandas",
            "corr_matrix": corr.to_dict(),
            "corr_duration": corr["duration"].to_dict(),
        }

    return {
        "method": "manual",
        "corr_duration": {
            "cpu": pearson_corr(duration, cpu),
            "mem": pearson_corr(duration, mem),
            "active_jobs": pearson_corr(duration, active_jobs),
            "failures": pearson_corr(duration, failures),
        },
    }


def select_best_factor(
    corr_dur: dict[str, Any], threshold: float = 0.45
) -> tuple[str | None, float | None, bool]:
    """Select the best correlating factor based on absolute correlation value.

    Returns:
        Tuple of (factor_name, correlation_value, is_significant)
    """
    scored = [(k, v) for k, v in corr_dur.items() if isinstance(v, (int, float))]
    scored.sort(key=lambda kv: abs(kv[1]), reverse=True)

    if not scored:
        return None, None, False

    best_key, best_val = scored[0]
    is_significant = abs(best_val) >= threshold
    return best_key, best_val, is_significant


def interpret_factor(factor_key: str | None) -> tuple[str | None, list[str]]:
    """Interpret the best factor into root cause and contributing factors."""
    if factor_key is None:
        return None, []

    factor_map = {
        "cpu": ("CPU saturation correlates with increased job runtime", ["cpu"]),
        "mem": ("Memory pressure correlates with increased job runtime", ["memory"]),
        "active_jobs": (
            "High concurrency correlates with increased job runtime",
            ["active_jobs"],
        ),
        "failures": (
            "Failures correlate with increased runtime (possible retries/resource issues)",
            ["failures"],
        ),
    }

    return factor_map.get(factor_key, (None, []))


def pearson_corr(x: list[float], y: list[float]) -> float | None:
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 3:
        return None

    if _NUMPY_AVAILABLE:
        try:
            return float(_np.corrcoef(_np.array(x), _np.array(y))[0, 1])
        except Exception as e:
            logger.warning(
                "NumPy correlation failed, falling back to manual", error=str(e)
            )
            # Continue to manual calculation below

    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5

    if denx == 0 or deny == 0:
        logger.warning("Zero standard deviation in correlation calculation")
        return None

    return numerator / (denx * deny)


def _safe_float(v: Any) -> float | None:
    """Safely convert value to float."""
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _duration_seconds(start: Any, end: Any) -> float | None:
    """Calculate duration in seconds between two timestamps."""
    if not start or not end:
        return None
    try:
        return max(0.0, (end - start).total_seconds())
    except Exception as e:
        logger.warning("Failed to calculate duration between timestamps", error=str(e))
        return None


def fetch_job_history_from_db(
    db: Any,
    job_name: str,
    since: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch job history from database.

    Note: This function creates its own event loop. For async contexts,
    use fetch_job_history_from_db_async instead.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
        # Loop is already running - need to use a different approach
        # Return empty list with warning - caller should use async version
        logger.warning(
            "fetch_job_history_from_db called from async context, "
            "consider using async version",
            job_name=job_name,
        )
        return []
    except RuntimeError:
        # No event loop exists, create one
        pass

    return asyncio.run(_fetch_job_history_sync(db, job_name, since, limit))


async def _fetch_job_history_sync(
    db: Any,
    job_name: str,
    since: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Internal async implementation for fetching job history."""
    from sqlalchemy import select

    try:
        from resync.db.models.tws_job_status import TWSJobStatus

        stmt = (
            select(TWSJobStatus)
            .where(TWSJobStatus.job_name == job_name)
            .where(TWSJobStatus.timestamp >= since)
            .order_by(TWSJobStatus.timestamp.desc())
            .limit(limit)
        )
        rows = (await db.execute(stmt)).scalars().all()
        if not rows:
            return []

        out = []
        for r in rows:
            out.append(
                {
                    "job_name": r.job_name,
                    "job_stream": r.job_stream,
                    "workstation": r.workstation,
                    "status": r.status,
                    "run_number": r.run_number,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration_seconds": _duration_seconds(r.start_time, r.end_time),
                    "return_code": r.return_code,
                    "timestamp": r.timestamp,
                    "metadata": getattr(r, "metadata_", None) or {},
                    "source": "db",
                }
            )
        return list(reversed(out))
    except Exception as e:
        logger.exception(
            "Failed to fetch job history from DB", error=str(e), job_name=job_name
        )
        return []


def extract_job_items_from_tws_response(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract job items from TWS API response."""
    items = plan.get("jobs") if isinstance(plan, dict) else None

    if items is None and isinstance(plan, dict):
        for key in ("items", "results", "data"):
            if isinstance(plan.get(key), list):
                items = plan[key]
                break

    if not isinstance(items, list):
        return []

    return items


def map_tws_job_item(item: dict[str, Any], job_name: str) -> dict[str, Any] | None:
    """Map TWS API job item to standard format."""
    if not isinstance(item, dict):
        return None

    status = item.get("status") or item.get("jobStatus")
    workstation = (
        item.get("workstation")
        or item.get("workStation")
        or item.get("workstationName")
    )
    timestamp = item.get("timestamp") or item.get("startTime") or item.get("lastUpdate")

    return {
        "job_name": item.get("name") or item.get("jobName") or job_name,
        "job_stream": item.get("jobStream") or item.get("jobstream"),
        "workstation": workstation,
        "status": status,
        "run_number": item.get("runNumber") or 1,
        "start_time": item.get("startTime"),
        "end_time": item.get("endTime"),
        "duration_seconds": _safe_float(item.get("duration")),
        "return_code": item.get("returnCode"),
        "timestamp": timestamp,
        "metadata": item,
        "source": "tws",
    }


async def fetch_job_history_from_tws(
    tws_client: Any,
    job_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch job history from TWS API as fallback."""
    try:
        plan = await tws_client.query_current_plan_jobs(
            q=job_name, limit=min(200, limit)
        )
        items = extract_job_items_from_tws_response(plan)

        out = []
        for item in items:
            mapped = map_tws_job_item(item, job_name)
            if mapped:
                out.append(mapped)
        return out
    except Exception as e:
        logger.exception(
            "Failed to fetch job history from TWS", error=str(e), job_name=job_name
        )
        return []


def extract_runtime_series(
    job_history: list[dict[str, Any]],
) -> list[tuple[datetime, float]]:
    """Extract and sort runtime series from job history."""
    series: list[tuple[datetime, float]] = []

    for record in job_history:
        ts = record.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        duration = _safe_float(record.get("duration_seconds"))
        if duration is None:
            duration = _safe_float(
                _duration_seconds(record.get("start_time"), record.get("end_time"))
            )

        if duration is None:
            continue

        series.append((ts, duration))

    series.sort(key=lambda x: x[0])
    return series


def calculate_danger_threshold(values: list[float]) -> tuple[float, float, float]:
    """Calculate danger threshold using Median Absolute Deviation (MAD).

    Returns:
        Tuple of (median, mad, danger_threshold)
    """
    if not values:
        return 0.0, 0.0, 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Correct median calculation for both odd and even lengths
    if n % 2 == 0:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median = sorted_values[n // 2]

    # Calculate MAD: median of absolute deviations from median
    mad_values = sorted(abs(v - median) for v in values)

    # Correct MAD median calculation for both odd and even lengths
    if n % 2 == 0:
        mad = (mad_values[n // 2 - 1] + mad_values[n // 2]) / 2 or 1.0
    else:
        mad = mad_values[n // 2] or 1.0

    danger = median + 3.0 * mad

    return median, mad, danger


def linear_regression(
    x_values: list[float], y_values: list[float]
) -> tuple[float, float, float]:
    """Perform simple linear regression: y = a + b*x

    Returns:
        Tuple of (intercept_a, slope_b, residual_std)
    """
    n = len(x_values)
    if n < 2:
        return 0.0, 0.0, 0.0

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    sxx = sum((x - x_mean) ** 2 for x in x_values)
    if sxx == 0:
        return y_mean, 0.0, 0.0

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sxx
    intercept = y_mean - slope * x_mean

    residuals = [y - (intercept + slope * x) for x, y in zip(x_values, y_values)]
    sse = sum(r**2 for r in residuals)
    dof = max(1, n - 2)
    residual_std = (sse / dof) ** 0.5

    return intercept, slope, residual_std


def calculate_confidence_interval(
    slope: float,
    _intercept: float,  # Kept for API compatibility, not used in calculation
    _x_values: list[float],  # Kept for API compatibility, not used in calculation
    _y_values: list[float],  # Kept for API compatibility, not used in calculation
    residual_std: float,
    x_mean: float,
    sxx: float,
    n: int,
    predicted_x: float | None = None,
) -> dict[str, Any]:
    """Calculate confidence interval for predictions.

    Note: intercept, x_values, and y_values are kept for API compatibility
    but are not used in the current calculation.

    Returns:
        Dict with confidence interval details
    """
    if not _SCIPY_AVAILABLE:
        tcrit = 1.96
    else:
        dof = max(1, n - 2)
        tcrit = float(_scipy_stats.t.ppf(0.975, dof))

    ci = {}

    if predicted_x is not None and slope != 0 and sxx > 0:
        se_predicted = (
            residual_std * (1.0 + 1.0 / n + ((predicted_x - x_mean) ** 2) / sxx) ** 0.5
        )
        dt_days = abs(tcrit * se_predicted / slope)
        ci = {
            "slope": slope,
            "tcrit": tcrit,
            "std_error": se_predicted,
            "delta_days": dt_days,
        }
    elif sxx <= 0:
        logger.warning("Cannot calculate confidence interval: sxx is zero or negative")

    return ci


def calculate_failure_probability(
    degradation_severity: float,
    estimated_days_to_failure: float | None,
    horizon_days: int,
    _confidence: float,  # Kept for API compatibility, not used in calculation
    _y_mean: float,  # Kept for API compatibility, not used in calculation
    _residual_std: float,  # Kept for API compatibility, not used in calculation
) -> float:
    """Calculate failure probability based on timeline and degradation.

    Note: confidence, y_mean, and residual_std are kept for API compatibility
    but are not used in the current calculation.
    """
    base_prob = min(1.0, 0.25 + 0.55 * degradation_severity)

    if estimated_days_to_failure is not None:
        if estimated_days_to_failure <= horizon_days:
            base_prob = min(1.0, max(base_prob, 0.75))
        elif estimated_days_to_failure <= horizon_days * 2:
            base_prob = min(1.0, max(base_prob, 0.55))

    return base_prob


def calculate_confidence_score(
    degradation_severity: float,
    residual_std: float,
    y_mean: float,
    has_estimate: bool,
) -> float:
    """Calculate confidence score for the prediction."""
    confidence = min(
        1.0,
        max(
            0.1,
            0.35 + 0.5 * degradation_severity - 0.1 * (residual_std / max(1.0, y_mean)),
        ),
    )

    if not has_estimate:
        confidence *= 0.7

    return confidence


def extract_runtimes_and_failures(
    job_history: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Extract runtimes and failures from job history."""
    runtimes: list[float] = []
    failures: list[float] = []

    for record in job_history:
        duration = record.get("duration_seconds")
        if duration is None:
            duration = _duration_seconds(
                record.get("start_time"), record.get("end_time")
            )
        dur_f = _safe_float(duration)
        if dur_f is None:
            continue
        runtimes.append(dur_f)
        status = (record.get("status") or "").upper()
        failures.append(1.0 if status in {"ABEND", "ERROR", "FAILED", "FAIL"} else 0.0)

    return runtimes, failures


def calculate_runtime_growth(
    runtimes: list[float],
    window: int,
) -> tuple[float, float, float]:
    """Calculate runtime growth between recent and prior windows.

    Returns:
        Tuple of (mu_recent, mu_prior, growth_ratio)
    """
    if len(runtimes) < max(8, window * 2):
        return 0.0, 0.0, 0.0

    recent = runtimes[-window:]
    prior = runtimes[-2 * window : -window]
    mu_recent = sum(recent) / len(recent)
    mu_prior = sum(prior) / len(prior)
    growth = (mu_recent - mu_prior) / max(1e-9, mu_prior)

    return mu_recent, mu_prior, growth


def calculate_failure_metrics(
    failures: list[float],
    window: int,
) -> tuple[float, float, float, list[float]]:
    """Calculate failure metrics including z-score and growth.

    Returns:
        Tuple of (fail_recent, fail_prior, fail_growth, z_scores)
    """
    if len(failures) < max(8, window * 2):
        return 0.0, 0.0, 0.0, []

    z_scores = rolling_zscore(failures, window=window)
    fail_recent = sum(failures[-window:]) / window
    fail_prior = sum(failures[-2 * window : -window]) / window
    fail_growth = fail_recent - fail_prior

    return fail_recent, fail_prior, fail_growth, z_scores


def rolling_zscore(data: list[float], window: int = 10) -> list[float]:
    """Calculate rolling z-score for data."""
    if len(data) < window:
        return [0.0] * len(data)

    result: list[float] = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(0.0)
        else:
            chunk = data[i - window + 1 : i + 1]
            mean = sum(chunk) / window
            variance = sum((x - mean) ** 2 for x in chunk) / window
            std = variance**0.5
            if std == 0:
                result.append(0.0)
            else:
                result.append((data[i] - mean) / std)

    return result


def generate_recommendation(
    title: str,
    priority: str,
    rationale: str,
    job: str,
    confidence: float,
    action_type: str | None = None,
    action_target: str | None = None,
) -> dict[str, Any]:
    """Generate a single recommendation with optional action."""
    rec = {
        "title": title,
        "priority": priority,
        "rationale": rationale,
        "job": job,
        "confidence": confidence,
    }

    if action_type:
        action = {"type": action_type, "job": job}
        if action_target:
            action["target"] = action_target
        return {"recommendation": rec, "action": action}

    return {"recommendation": rec, "action": None}


def build_recommendations(
    job_name: str,
    degradation_type: str | None,
    degradation_severity: float,
    correlation: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build recommendations based on analysis results.

    Returns:
        Tuple of (recommendations, actions)
    """
    recommendations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    root = (correlation or {}).get("root_cause")
    factors = (correlation or {}).get("factors") or []
    prob = (prediction or {}).get("failure_probability") or 0.0

    if degradation_type == "runtime_growth":
        recommendations.append(
            {
                "title": "Investigate runtime growth trend",
                "priority": "high" if degradation_severity >= 0.6 else "medium",
                "rationale": "Job runtime increased materially vs baseline",
                "job": job_name,
                "confidence": min(0.95, 0.5 + degradation_severity / 2),
            }
        )

    if "cpu" in factors:
        recommendations.append(
            {
                "title": "Check CPU saturation on workstation",
                "priority": "high" if prob >= 0.7 else "medium",
                "rationale": root or "CPU correlates with runtime degradation",
                "job": job_name,
                "confidence": 0.8,
            }
        )
        actions.append(
            {"type": "workstation_capacity_review", "target": "cpu", "job": job_name}
        )

    if "memory" in factors:
        recommendations.append(
            {
                "title": "Check memory pressure / swap",
                "priority": "high" if prob >= 0.7 else "medium",
                "rationale": root or "Memory correlates with runtime degradation",
                "job": job_name,
                "confidence": 0.75,
            }
        )
        actions.append(
            {"type": "workstation_capacity_review", "target": "memory", "job": job_name}
        )

    if prob >= 0.75:
        recommendations.append(
            {
                "title": "Schedule proactive validation run / controlled restart",
                "priority": "high",
                "rationale": "High probability of failure within horizon",
                "job": job_name,
                "confidence": 0.7,
            }
        )
        actions.append({"type": "proactive_validation", "job": job_name})

    return recommendations, actions
