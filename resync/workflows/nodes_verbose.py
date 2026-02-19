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
            "scheduled_time": row["scheduled_time"].isoformat() if row.get("scheduled_time") else None,
            "actual_start_time": row["actual_start_time"].isoformat() if row.get("actual_start_time") else None,
            "completed_time": row["completed_time"].isoformat() if row.get("completed_time") else None,
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
            "cpu_percent": float(row["cpu_percent"]) if row.get("cpu_percent") is not None else 0.0,
            "memory_percent": float(row["memory_percent"]) if row.get("memory_percent") is not None else 0.0,
            "disk_percent": float(row["disk_percent"]) if row.get("disk_percent") is not None else 0.0,
            "load_avg_1min": float(row["load_avg_1min"]) if row.get("load_avg_1min") is not None else 0.0,
            "cpu_count": int(row["cpu_count"]) if row.get("cpu_count") is not None else 0,
            "total_memory_gb": float(row["total_memory_gb"]) if row.get("total_memory_gb") is not None else 0.0,
            "total_disk_gb": float(row["total_disk_gb"]) if row.get("total_disk_gb") is not None else 0.0,
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
            "scheduled_time": row["scheduled_time"].isoformat() if row["scheduled_time"] else None,
            "actual_start_time": row["actual_start_time"].isoformat() if row["actual_start_time"] else None,
            "completed_time": row["completed_time"].isoformat() if row["completed_time"] else None,
        }
        for row in rows
    ]
    return job_history


async def fetch_workstation_metrics_history(db: Any, workstation: str | None = None) -> list[dict[str, Any]]:
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
            "cpu_percent": float(row["cpu_percent"]) if row["cpu_percent"] is not None else 0.0,
            "memory_percent": float(row["memory_percent"]) if row["memory_percent"] is not None else 0.0,
            "disk_percent": float(row["disk_percent"]) if row["disk_percent"] is not None else 0.0,
            "load_avg_1min": float(row["load_avg_1min"]) if row["load_avg_1min"] is not None else 0.0,
            "cpu_count": int(row["cpu_count"]) if row["cpu_count"] is not None else 0,
            "total_memory_gb": float(row["total_memory_gb"]) if row["total_memory_gb"] is not None else 0.0,
            "total_disk_gb": float(row["total_disk_gb"]) if row["total_disk_gb"] is not None else 0.0,
        }
        for row in rows
    ]
    return metrics_history


async def detect_degradation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """TODO: Verbose-mode placeholder until full analytical implementation is restored."""
    await asyncio.sleep(0)
    _ = (args, kwargs)
    return {}


async def correlate_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """TODO: Verbose-mode placeholder until full analytical implementation is restored."""
    await asyncio.sleep(0)
    _ = (args, kwargs)
    return {}


async def predict_timeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """TODO: Verbose-mode placeholder until full analytical implementation is restored."""
    await asyncio.sleep(0)
    _ = (args, kwargs)
    return {}


async def generate_recommendations(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """TODO: Verbose-mode placeholder until full recommendation implementation is restored."""
    await asyncio.sleep(0)
    _ = (args, kwargs)
    return {}


async def notify_operators(*args: Any, **kwargs: Any) -> None:
    """No-op notification placeholder for verbose mode."""
    await asyncio.sleep(0)
    logger.info("notify_operators_verbose_noop", args_count=len(args), kwargs_keys=sorted(kwargs.keys()))
