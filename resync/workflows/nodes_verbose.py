import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import text

logger = structlog.get_logger(__name__)

_JOB_EXECUTION_HISTORY_LIMIT = 2000
_METRICS_HISTORY_LIMIT = 5000

def _cutoff_from_days(days: int) -> datetime:
    safe_days = max(1, int(days or 1))
    return datetime.now(timezone.utc) - timedelta(days=safe_days)

async def fetch_job_history(
    *,
    db: Any,
    job_name: str,
    days: int = 30,
    limit: int = _JOB_EXECUTION_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Verbose-mode fetch compatible with optimized node interface."""
    if not job_name:
        return []

    cutoff_date = _cutoff_from_days(days)
    query = text(
        """
        SELECT * FROM jobs
        WHERE job_name = :job_name
          AND timestamp >= :cutoff_date
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )

    result = await db.execute(
        query,
        {
            "job_name": job_name,
            "cutoff_date": cutoff_date,
            "limit": max(1, int(limit or 1)),
        },
    )
    rows = result.mappings().fetchall()

    return [
        {
            "timestamp": row["timestamp"].isoformat() if row.get("timestamp") else None,
            "job_name": row.get("job_name"),
            "workstation": row.get("workstation") or "UNKNOWN",
            "status": row.get("status") or "UNKNOWN",
            "return_code": row.get("return_code") or 0,
            "runtime_seconds": row.get("runtime_seconds") or 0,
            "scheduled_time": row["scheduled_time"].isoformat()
            if row.get("scheduled_time")
            else None,
            "actual_start_time": row["actual_start_time"].isoformat()
            if row.get("actual_start_time")
            else None,
            "completed_time": row["completed_time"].isoformat()
            if row.get("completed_time")
            else None,
            "source": "verbose_db",
        }
        for row in rows
    ]

async def fetch_workstation_metrics(
    *,
    db: Any,
    workstation: str | None,
    days: int = 30,
    limit: int = _METRICS_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Verbose-mode fetch compatible with optimized node interface."""
    if not workstation:
        return []

    cutoff_date = _cutoff_from_days(days)
    query = text(
        """
        SELECT * FROM metrics
        WHERE workstation = :workstation
          AND timestamp >= :cutoff_date
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )

    result = await db.execute(
        query,
        {
            "workstation": workstation,
            "cutoff_date": cutoff_date,
            "limit": max(1, int(limit or 1)),
        },
    )
    rows = result.mappings().fetchall()

    return [
        {
            "timestamp": row["timestamp"].isoformat() if row.get("timestamp") else None,
            "workstation": row.get("workstation"),
            "cpu_percent": float(row["cpu_percent"])
            if row.get("cpu_percent") is not None
            else 0.0,
            "memory_percent": float(row["memory_percent"])
            if row.get("memory_percent") is not None
            else 0.0,
            "disk_percent": float(row["disk_percent"])
            if row.get("disk_percent") is not None
            else 0.0,
            "load_avg_1min": float(row["load_avg_1min"])
            if row.get("load_avg_1min") is not None
            else 0.0,
            "cpu_count": int(row["cpu_count"])
            if row.get("cpu_count") is not None
            else 0,
            "total_memory_gb": float(row["total_memory_gb"])
            if row.get("total_memory_gb") is not None
            else 0.0,
            "total_disk_gb": float(row["total_disk_gb"])
            if row.get("total_disk_gb") is not None
            else 0.0,
            "source": "verbose_db",
        }
        for row in rows
    ]

async def fetch_job_execution_history(
    db: Any,
    job_name: str | None = None,
    workstation: str | None = None,
) -> list[dict[str, Any]]:
    query_str = "SELECT * FROM jobs WHERE 1=1"
    params: dict[str, Any] = {}
    if job_name:
        query_str += " AND job_name = :job_name"
        params["job_name"] = job_name
    if workstation:
        query_str += " AND workstation = :workstation"
        params["workstation"] = workstation

    query_str += " ORDER BY timestamp DESC LIMIT 2000"

    result = await db.execute(text(query_str), params)
    rows = result.mappings().fetchall()

    job_history = [
        {
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "job_name": row["job_name"],
            "workstation": row["workstation"] or "UNKNOWN",
            "status": row["status"] or "UNKNOWN",
            "return_code": row["return_code"] or 0,
            "runtime_seconds": row["runtime_seconds"] or 0,
            "scheduled_time": row["scheduled_time"].isoformat()
            if row["scheduled_time"]
            else None,
            "actual_start_time": row["actual_start_time"].isoformat()
            if row["actual_start_time"]
            else None,
            "completed_time": row["completed_time"].isoformat()
            if row["completed_time"]
            else None,
        }
        for row in rows
    ]
    return job_history

async def fetch_workstation_metrics_history(
    db: Any, workstation: str | None = None
) -> list[dict[str, Any]]:
    query_str = "SELECT * FROM metrics WHERE 1=1"
    params: dict[str, Any] = {}
    if workstation:
        query_str += " AND workstation = :workstation"
        params["workstation"] = workstation

    query_str += " ORDER BY timestamp DESC LIMIT 5000"

    result = await db.execute(text(query_str), params)
    rows = result.mappings().fetchall()

    metrics_history = [
        {
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "workstation": row["workstation"],
            "cpu_percent": float(row["cpu_percent"])
            if row["cpu_percent"] is not None
            else 0.0,
            "memory_percent": float(row["memory_percent"])
            if row["memory_percent"] is not None
            else 0.0,
            "disk_percent": float(row["disk_percent"])
            if row["disk_percent"] is not None
            else 0.0,
            "load_avg_1min": float(row["load_avg_1min"])
            if row["load_avg_1min"] is not None
            else 0.0,
            "cpu_count": int(row["cpu_count"]) if row["cpu_count"] is not None else 0,
            "total_memory_gb": float(row["total_memory_gb"])
            if row["total_memory_gb"] is not None
            else 0.0,
            "total_disk_gb": float(row["total_disk_gb"])
            if row["total_disk_gb"] is not None
            else 0.0,
        }
        for row in rows
    ]
    return metrics_history

async def detect_degradation(
    metrics_history: list[dict[str, Any]] | None = None,
    *,
    cpu_threshold: float = 85.0,
    memory_threshold: float = 90.0,
    disk_threshold: float = 90.0,
    min_points: int = 10,
) -> dict[str, Any]:
    """Detect simple degradation signals from workstation metrics.

    This is a lightweight, deterministic implementation intended for verbose mode.
    It avoids LLM calls and provides stable signals for downstream steps.
    """
    await asyncio.sleep(0)

    history = metrics_history or []
    if len(history) < min_points:
        return {"degradation_detected": False, "reason": "insufficient_data"}

    # Metrics are ordered DESC by timestamp in fetch_workstation_metrics; reverse for trend
    ordered = list(reversed(history))

    def _last_n_avg(key: str, n: int = 10) -> float:
        vals = [float(x.get(key, 0.0) or 0.0) for x in ordered[-n:]]
        return sum(vals) / max(1, len(vals))

    avg_cpu = _last_n_avg("cpu_percent")
    avg_mem = _last_n_avg("memory_percent")
    avg_disk = _last_n_avg("disk_percent")

    degraded = (avg_cpu >= cpu_threshold) or (avg_mem >= memory_threshold) or (
        avg_disk >= disk_threshold
    )

    return {
        "degradation_detected": degraded,
        "signals": {
            "avg_cpu_percent_last10": avg_cpu,
            "avg_memory_percent_last10": avg_mem,
            "avg_disk_percent_last10": avg_disk,
        },
        "thresholds": {
            "cpu": cpu_threshold,
            "memory": memory_threshold,
            "disk": disk_threshold,
        },
    }
async def correlate_metrics(
    metrics_history: list[dict[str, Any]] | None = None,
    *,
    window: int = 50,
) -> dict[str, Any]:
    """Compute simple correlations between resource metrics.

    Returns Pearson correlations for cpu/memory/disk across the latest window.
    """
    await asyncio.sleep(0)

    history = metrics_history or []
    if len(history) < 3:
        return {"correlation": {}, "reason": "insufficient_data"}

    ordered = list(reversed(history))[-window:]

    def series(key: str) -> list[float]:
        return [float(x.get(key, 0.0) or 0.0) for x in ordered]

    def pearson(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a = a[:n]
        b = b[:n]
        ma = sum(a) / n
        mb = sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        if da == 0.0 or db == 0.0:
            return 0.0
        return num / (da * db)

    cpu = series("cpu_percent")
    mem = series("memory_percent")
    disk = series("disk_percent")

    return {
        "correlation": {
            "cpu_mem": pearson(cpu, mem),
            "cpu_disk": pearson(cpu, disk),
            "mem_disk": pearson(mem, disk),
        },
        "window_points": len(ordered),
    }
async def predict_timeline(
    degradation: dict[str, Any] | None = None,
    correlations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a coarse failure timeline.

    This is a heuristic (non-ML) placeholder that provides a stable contract:
    - If degradation is detected and correlations are strong, estimate sooner.
    """
    await asyncio.sleep(0)

    degradation = degradation or {}
    correlations = correlations or {}
    degraded = bool(degradation.get("degradation_detected", False))
    corr = correlations.get("correlation", {}) if isinstance(correlations, dict) else {}
    max_corr = max((abs(float(v)) for v in corr.values()), default=0.0)

    if not degraded:
        return {"predicted_timeline_days": None, "confidence": 0.3}

    # Strong correlation suggests systemic saturation; assume faster risk
    if max_corr >= 0.8:
        return {"predicted_timeline_days": 7, "confidence": 0.75}
    if max_corr >= 0.5:
        return {"predicted_timeline_days": 14, "confidence": 0.6}
    return {"predicted_timeline_days": 28, "confidence": 0.5}
async def generate_recommendations(
    degradation: dict[str, Any] | None = None,
    correlations: dict[str, Any] | None = None,
    timeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate deterministic recommendations for verbose mode."""
    await asyncio.sleep(0)

    degradation = degradation or {}
    signals = degradation.get("signals", {}) if isinstance(degradation, dict) else {}
    timeline = timeline or {}

    recs: list[str] = []
    if degradation.get("degradation_detected"):
        recs.append("Validate workstation capacity (CPU/memory/disk) and check for contention.")
        recs.append("Inspect top resource consumers and consider rescheduling heavy jobs off-peak.")
        recs.append("Review recent changes (deploys/config) that correlate with the start of saturation.")
    else:
        recs.append("No strong degradation signals found; continue monitoring.")

    days = timeline.get("predicted_timeline_days")
    if isinstance(days, int):
        recs.append(f"Suggested action window: within ~{days} days.")

    return {"recommendations": recs, "signals": signals}
async def notify_operators(*args: Any, **kwargs: Any) -> None:
    """No-op notification placeholder for verbose mode."""
    await asyncio.sleep(0)
    logger.info(
        "notify_operators_verbose_noop",
        args_count=len(args),
        kwargs_keys=sorted(kwargs.keys()),
    )
