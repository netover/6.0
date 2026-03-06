# pylint
"""
System status routes for FastAPI
"""

import platform
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from resync.api.dependencies_v2 import get_logger
from resync.api.models.responses_v2 import SystemStatusResponse
from resync.core.async_utils import with_timeout, classify_exception
from resync.settings import get_settings

router = APIRouter()


class LivenessResponse(BaseModel):
    status: str
    timestamp: str


class DependencyCheck(BaseModel):
    healthy: bool
    error: str | None = None
    critical: bool | None = None


class ReadinessResponse(BaseModel):
    status: str
    timestamp: str
    checks: dict[str, object]


class DetailedHealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    checks: dict[str, object]


class WorkstationRegistrationResponse(BaseModel):
    message: str
    workstation: dict[str, str]


# In-memory status store (replace with Valkey/DB in production)
_status_store = {
    "workstations": [],
    "jobs": [],
}

def get_system_metrics() -> dict:
    """Get basic system metrics."""
    try:
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except ImportError:
        return {
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_percent": 0,
        }

async def check_database_health() -> tuple[bool, str | None]:
    """Check database connectivity."""
    try:
        from sqlalchemy import text

        from resync.core.database import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return False, str(e)

async def check_valkey_health() -> tuple[bool, str | None]:
    """Check Valkey connectivity."""
    try:
        from resync.core.valkey_init import get_valkey_client
        valkey = get_valkey_client()
        if valkey:
            settings = get_settings()
            await with_timeout(valkey.ping(), getattr(settings, 'valkey_health_timeout', 2.0), op='valkey.ping')
            return True, None
        return True, "Valkey not configured (optional)"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return False, str(e)

@router.get("/liveness", response_model=LivenessResponse)
async def liveness_probe() -> LivenessResponse:
    """
    Kubernetes Liveness Probe.
    Returns 200 if the application is running.
    Use for: livenessProbe in k8s deployment.
    """
    return LivenessResponse(status="alive", timestamp=datetime.now(timezone.utc).isoformat())

@router.get("/readiness", response_model=ReadinessResponse)
async def readiness_probe(response: Response, logger_instance=Depends(get_logger)) -> ReadinessResponse:
    """
    Kubernetes Readiness Probe.
    Returns 200 only if ALL critical dependencies are healthy.
    Use for: readinessProbe in k8s deployment.

    Checks:
    - Database connectivity (critical)
    - Valkey connectivity (optional, degrades gracefully)
    """
    checks = {}
    is_ready = True

    # Check database (critical)
    db_healthy, db_error = await check_database_health()
    checks["database"] = {"healthy": db_healthy, "error": db_error}
    if not db_healthy:
        is_ready = False
        logger_instance.error(
            "readiness_check_failed", component="database", error=db_error
        )

    # Check Valkey (optional - degrades gracefully)
    valkey_healthy, valkey_error = await check_valkey_health()
    checks["valkey"] = {
        "healthy": valkey_healthy,
        "error": valkey_error,
        "critical": False,
    }
    if (
        not valkey_healthy
        and valkey_error
        and "not configured" not in valkey_error.lower()
    ):
        logger_instance.warning(
            "readiness_check_degraded", component="valkey", error=valkey_error
        )

    # Add system metrics
    checks["system"] = get_system_metrics()

    result = {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }

    if not is_ready:
        response.status_code = 503  # Service Unavailable

    return ReadinessResponse(**result)

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    response: Response, logger_instance=Depends(get_logger)
) -> DetailedHealthResponse:
    """
    Detailed health check for monitoring dashboards.
    Returns comprehensive status of all components.
    """
    checks = {}
    overall_healthy = True
    degraded = False

    # Database check
    db_healthy, db_error = await check_database_health()
    checks["database"] = {
        "status": "healthy" if db_healthy else "unhealthy",
        "latency_ms": None,  # Could add latency measurement
        "error": db_error,
    }
    if not db_healthy:
        overall_healthy = False

    # Valkey check
    valkey_healthy, valkey_error = await check_valkey_health()
    checks["valkey"] = {
        "status": "healthy"
        if valkey_healthy
        else (
            "degraded" if "not configured" in str(valkey_error or "") else "unhealthy"
        ),
        "error": valkey_error,
    }
    if (
        not valkey_healthy
        and valkey_error
        and "not configured" not in valkey_error.lower()
    ):
        degraded = True

    # System metrics
    metrics = get_system_metrics()
    checks["system"] = {
        "status": "healthy"
        if metrics["cpu_percent"] < 90 and metrics["memory_percent"] < 90
        else "warning",
        "metrics": metrics,
    }

    # Determine overall status
    if not overall_healthy:
        status = "unhealthy"
        response.status_code = 503
    elif degraded:
        status = "degraded"
    else:
        status = "healthy"

    return DetailedHealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="5.3.19",
        checks=checks,
    )

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(logger_instance=Depends(get_logger)):
    """Get system status including workstations and jobs"""
    try:
        # Get status from store (production: use Valkey/database)
        workstations = _status_store.get("workstations", [])
        jobs = _status_store.get("jobs", [])

        # Add system info
        {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }

        logger_instance.info(
            "system_status_retrieved",
            user_id="system",
            workstation_count=len(workstations),
            job_count=len(jobs),
        )

        return SystemStatusResponse(
            workstations=workstations,
            jobs=jobs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("system_status_retrieval_error", error=str(e))
        return SystemStatusResponse(
            workstations=[], jobs=[], timestamp=datetime.now(timezone.utc).isoformat()
        )

@router.post("/status/workstation", response_model=WorkstationRegistrationResponse)
async def register_workstation(
    name: str, status: str = "online", logger_instance=Depends(get_logger)
):
    """Register or update a workstation status."""
    workstation = {
        "name": name,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Update or add workstation
    existing = next(
        (w for w in _status_store["workstations"] if w["name"] == name), None
    )
    if existing:
        existing.update(workstation)
    else:
        _status_store["workstations"].append(workstation)

    return {"message": "Workstation registered", "workstation": workstation}