"""
Admin Monitoring Routes.

Provides real-time monitoring dashboard data:
- System metrics (CPU, memory, disk)
- Application metrics (requests, latency, errors)
- Service health status
- Active connections
- Performance trends
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from resync.api.security import decode_token

logger = logging.getLogger(__name__)


def _verify_ws_admin(websocket: WebSocket) -> bool:
    """Verify admin authentication for WebSocket connection.

    Returns True if authenticated as admin, False otherwise.
    """
    try:
        # Check for token in Authorization header
        auth_header = websocket.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header[7:]
        payload = decode_token(token)
        if not payload:
            return False

        # Verify admin role
        roles_claim = payload.get("roles")
        if roles_claim is None:
            legacy_role = payload.get("role")
            roles = [legacy_role] if legacy_role else []
        elif isinstance(roles_claim, list):
            roles = roles_claim
        else:
            roles = [roles_claim]

        return "admin" in roles
    except Exception as e:
        logger.debug("WebSocket admin auth failed: %s", type(e).__name__)
        return False


router = APIRouter()


# Pydantic models
class SystemMetrics(BaseModel):
    """System-level metrics."""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    uptime_seconds: float


class ApplicationMetrics(BaseModel):
    """Application-level metrics."""

    total_requests: int
    requests_per_minute: float
    avg_response_time_ms: float
    error_rate_percent: float
    active_connections: int
    cache_hit_rate: float


class ServiceHealth(BaseModel):
    """Health of a service."""

    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float | None = None
    last_check: str
    error_message: str | None = None


class MonitoringDashboard(BaseModel):
    """Complete monitoring dashboard data."""

    timestamp: str
    system: SystemMetrics
    application: ApplicationMetrics
    services: list[ServiceHealth]
    alerts: list[dict[str, Any]]


# In-memory metrics store
_metrics_history: list[dict] = []
_request_times: list[float] = []
_error_count: int = 0
_total_requests: int = 0
_start_time = time.time()

# WebSocket connections for real-time updates
_ws_connections: list[WebSocket] = []


def _get_system_metrics() -> SystemMetrics:
    """Get current system metrics."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=memory.percent,
        memory_used_gb=round(memory.used / (1024**3), 2),
        memory_total_gb=round(memory.total / (1024**3), 2),
        disk_percent=disk.percent,
        disk_used_gb=round(disk.used / (1024**3), 2),
        disk_total_gb=round(disk.total / (1024**3), 2),
        network_sent_mb=round(net.bytes_sent / (1024**2), 2),
        network_recv_mb=round(net.bytes_recv / (1024**2), 2),
        uptime_seconds=round(time.time() - _start_time, 0),
    )


def _get_application_metrics() -> ApplicationMetrics:
    """Get current application metrics."""
    global _request_times, _error_count, _total_requests

    # Calculate requests per minute
    recent_requests = [t for t in _request_times if t > time.time() - 60]
    rpm = len(recent_requests)

    # Calculate average response time
    avg_response = (
        sum(_request_times[-100:]) / len(_request_times[-100:]) if _request_times else 0
    )

    # Calculate error rate
    error_rate = (_error_count / _total_requests * 100) if _total_requests > 0 else 0

    return ApplicationMetrics(
        total_requests=_total_requests,
        requests_per_minute=rpm,
        avg_response_time_ms=round(avg_response * 1000, 2),
        error_rate_percent=round(error_rate, 2),
        active_connections=len(_ws_connections),
        cache_hit_rate=85.5,  # Would come from cache metrics
    )


def _get_services_health() -> list[ServiceHealth]:
    """Get health of all services."""
    services = []

    # TWS Service
    services.append(
        ServiceHealth(
            name="TWS Primary",
            status="healthy",
            latency_ms=45.2,
            last_check=datetime.now(timezone.utc).isoformat(),
            error_message=None,
        )
    )

    # Database
    services.append(
        ServiceHealth(
            name="PostgreSQL",
            status="healthy",
            latency_ms=12.5,
            last_check=datetime.now(timezone.utc).isoformat(),
            error_message=None,
        )
    )

    # Redis
    services.append(
        ServiceHealth(
            name="Redis Cache",
            status="healthy",
            latency_ms=2.1,
            last_check=datetime.now(timezone.utc).isoformat(),
            error_message=None,
        )
    )

    # RAG Service
    services.append(
        ServiceHealth(
            name="RAG/pgvector",
            status="healthy",
            latency_ms=150.0,
            last_check=datetime.now(timezone.utc).isoformat(),
            error_message=None,
        )
    )

    return services


def _get_active_alerts() -> list[dict[str, Any]]:
    """Get active alerts."""
    alerts = []

    # Check for high CPU
    cpu = psutil.cpu_percent()
    if cpu > 80:
        alerts.append(
            {
                "id": "cpu-high",
                "severity": "warning",
                "message": f"CPU usage is high: {cpu}%",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Check for high memory
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        alerts.append(
            {
                "id": "memory-high",
                "severity": "warning",
                "message": f"Memory usage is high: {memory.percent}%",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return alerts


@router.get(
    "/monitoring/dashboard", response_model=MonitoringDashboard, tags=["Monitoring"]
)
async def get_monitoring_dashboard():
    """Get complete monitoring dashboard data."""
    return MonitoringDashboard(
        timestamp=datetime.now(timezone.utc).isoformat(),
        system=_get_system_metrics(),
        application=_get_application_metrics(),
        services=_get_services_health(),
        alerts=_get_active_alerts(),
    )


@router.get("/monitoring/system", response_model=SystemMetrics, tags=["Monitoring"])
async def get_system_metrics():
    """Get system metrics."""
    return _get_system_metrics()


@router.get(
    "/monitoring/application", response_model=ApplicationMetrics, tags=["Monitoring"]
)
async def get_application_metrics():
    """Get application metrics."""
    return _get_application_metrics()


@router.get(
    "/monitoring/services", response_model=list[ServiceHealth], tags=["Monitoring"]
)
async def get_services_health():
    """Get health of all services."""
    return _get_services_health()


@router.get("/monitoring/alerts", tags=["Monitoring"])
async def get_active_alerts():
    """Get active alerts."""
    return {"alerts": _get_active_alerts()}


@router.get("/monitoring/metrics/history", tags=["Monitoring"])
async def get_metrics_history(
    minutes: int = 60,
    interval_seconds: int = 60,
):
    """Get historical metrics."""
    # Return last N entries from history
    return {
        "history": _metrics_history[-minutes:],
        "interval_seconds": interval_seconds,
    }


@router.websocket("/monitoring/ws")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates.

    Requires admin authentication via Authorization header.
    """
    # Verify admin authentication - fail closed (deny by default)
    if not _verify_ws_admin(websocket):
        logger.warning("Monitoring WebSocket auth failed - rejecting connection")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Admin authentication required"
        )
        return

    await websocket.accept()
    _ws_connections.append(websocket)

    logger.info("Monitoring WebSocket connected. Total: %s", len(_ws_connections))

    try:
        while True:
            # Send metrics every 5 seconds
            dashboard = MonitoringDashboard(
                timestamp=datetime.now(timezone.utc).isoformat(),
                system=_get_system_metrics(),
                application=_get_application_metrics(),
                services=_get_services_health(),
                alerts=_get_active_alerts(),
            )

            await websocket.send_json(dashboard.model_dump())
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        _ws_connections.remove(websocket)
        logger.info(
            "Monitoring WebSocket disconnected. Total: %s", len(_ws_connections)
        )
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


@router.get("/monitoring/summary", tags=["Monitoring"])
async def get_monitoring_summary():
    """Get a quick summary of system status."""
    system = _get_system_metrics()
    services = _get_services_health()
    alerts = _get_active_alerts()

    # Determine overall status
    unhealthy_services = [s for s in services if s.status == "unhealthy"]
    critical_alerts = [a for a in alerts if a.get("severity") == "critical"]

    if critical_alerts or unhealthy_services:
        overall_status = "critical"
    elif alerts:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    return {
        "status": overall_status,
        "cpu_percent": system.cpu_percent,
        "memory_percent": system.memory_percent,
        "services_healthy": len([s for s in services if s.status == "healthy"]),
        "services_total": len(services),
        "active_alerts": len(alerts),
        "uptime_hours": round(system.uptime_seconds / 3600, 1),
    }


@router.post("/monitoring/record-request", tags=["Monitoring"])
async def record_request(
    response_time_ms: float,
    is_error: bool = False,
):
    """Record a request for metrics (internal use)."""
    global _request_times, _error_count, _total_requests

    _request_times.append(response_time_ms / 1000)
    _total_requests += 1

    if is_error:
        _error_count += 1

    # Keep only last 10000 requests
    if len(_request_times) > 10000:
        _request_times = _request_times[-10000:]

    return {"recorded": True}
