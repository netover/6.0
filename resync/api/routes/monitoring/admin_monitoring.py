"""Admin monitoring routes with async-safe websocket broadcasting.

This module exposes REST and WebSocket endpoints for operational monitoring.
It hardens concurrency, validates runtime websocket parameters, and avoids
blocking the event loop with synchronous system probes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import orjson
import psutil
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, ValidationError

from resync.api.security import decode_token

logger = logging.getLogger(__name__)
router = APIRouter()


class WebSocketRuntimeConfig(BaseModel):
    """Runtime validated websocket config."""

    interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    max_connections: int = Field(default=100, ge=1, le=5000)


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
    requests_per_minute: int
    avg_response_time_ms: float
    error_rate_percent: float
    active_connections: int
    cache_hit_rate: float


class ServiceHealth(BaseModel):
    """Health of a service."""

    name: str
    status: str
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


class RequestStats(BaseModel):
    """Request counters maintained in memory."""

    request_times_seconds: list[float] = Field(default_factory=list)
    error_count: int = 0
    total_requests: int = 0


class WebSocketHub:
    """Concurrency-safe registry of websocket clients."""

    def __init__(self, max_connections: int) -> None:
        self._lock = asyncio.Lock()
        self._clients: set[WebSocket] = set()
        self._max_connections = max_connections

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept websocket if connection cap allows it."""
        async with self._lock:
            if len(self._clients) >= self._max_connections:
                return False
            await websocket.accept()
            self._clients.add(websocket)
            return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove websocket if present."""
        async with self._lock:
            self._clients.discard(websocket)

    async def count(self) -> int:
        """Current active client count."""
        async with self._lock:
            return len(self._clients)


_metrics_history: list[dict[str, Any]] = []
_start_time = time.time()
_stats = RequestStats()
_stats_lock = asyncio.Lock()


def _load_ws_runtime_config() -> WebSocketRuntimeConfig:
    """Load websocket runtime parameters from environment."""
    try:
        return WebSocketRuntimeConfig(
            interval_seconds=float(os.getenv("MONITORING_WS_INTERVAL_SECONDS", "5")),
            max_connections=int(os.getenv("MONITORING_WS_MAX_CONNECTIONS", "100")),
        )
    except (ValidationError, ValueError) as exc:
        logger.warning("invalid_admin_monitoring_ws_config: %s", exc)
        return WebSocketRuntimeConfig()


_WS_CONFIG = _load_ws_runtime_config()
_WS_HUB = WebSocketHub(max_connections=_WS_CONFIG.max_connections)


def _verify_ws_admin(websocket: WebSocket) -> bool:
    """Return True when websocket bearer token has admin role."""
    try:
        auth_header = websocket.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        payload = decode_token(auth_header[7:])
        if not payload:
            return False
        roles_claim = payload.get("roles")
        if roles_claim is None:
            legacy_role = payload.get("role")
            roles: list[str] = [legacy_role] if isinstance(legacy_role, str) else []
        elif isinstance(roles_claim, list):
            roles = [str(role) for role in roles_claim]
        else:
            roles = [str(roles_claim)]
        return "admin" in roles
    except Exception as exc:
        logger.debug("websocket_auth_failed: %s", type(exc).__name__)
        return False


async def _collect_system_metrics() -> SystemMetrics:
    """Collect system metrics in worker threads to avoid loop blocking."""
    memory = await asyncio.to_thread(psutil.virtual_memory)
    disk = await asyncio.to_thread(psutil.disk_usage, "/")
    net = await asyncio.to_thread(psutil.net_io_counters)
    cpu_percent = await asyncio.to_thread(psutil.cpu_percent, 0.1)
    return SystemMetrics(
        cpu_percent=float(cpu_percent),
        memory_percent=float(memory.percent),
        memory_used_gb=round(memory.used / (1024**3), 2),
        memory_total_gb=round(memory.total / (1024**3), 2),
        disk_percent=float(disk.percent),
        disk_used_gb=round(disk.used / (1024**3), 2),
        disk_total_gb=round(disk.total / (1024**3), 2),
        network_sent_mb=round(net.bytes_sent / (1024**2), 2),
        network_recv_mb=round(net.bytes_recv / (1024**2), 2),
        uptime_seconds=round(time.time() - _start_time, 0),
    )


async def _collect_active_alerts() -> list[dict[str, Any]]:
    """Collect active alerts using non-blocking thread offload."""
    alerts: list[dict[str, Any]] = []
    cpu = float(await asyncio.to_thread(psutil.cpu_percent))
    if cpu > 80:
        alerts.append(
            {
                "id": "cpu-high",
                "severity": "warning",
                "message": f"CPU usage is high: {cpu}%",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    memory = await asyncio.to_thread(psutil.virtual_memory)
    if float(memory.percent) > 85:
        alerts.append(
            {
                "id": "memory-high",
                "severity": "warning",
                "message": f"Memory usage is high: {memory.percent}%",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return alerts


async def _collect_application_metrics() -> ApplicationMetrics:
    """Collect application metrics from in-memory counters."""
    async with _stats_lock:
        now = time.time()
        recent_requests = [t for t in _stats.request_times_seconds if t > now - 60]
        rpm = len(recent_requests)
        avg_response = (
            sum(_stats.request_times_seconds[-100:])
            / len(_stats.request_times_seconds[-100:])
            if _stats.request_times_seconds
            else 0.0
        )
        error_rate = (
            (_stats.error_count / _stats.total_requests) * 100
            if _stats.total_requests > 0
            else 0.0
        )
        total_requests = _stats.total_requests

    return ApplicationMetrics(
        total_requests=total_requests,
        requests_per_minute=rpm,
        avg_response_time_ms=round(avg_response * 1000, 2),
        error_rate_percent=round(error_rate, 2),
        active_connections=await _WS_HUB.count(),
        cache_hit_rate=85.5,
    )


def _collect_services_health() -> list[ServiceHealth]:
    """Collect static service health placeholders."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return [
        ServiceHealth(
            name="TWS Primary", status="healthy", latency_ms=45.2, last_check=now_iso
        ),
        ServiceHealth(
            name="PostgreSQL", status="healthy", latency_ms=12.5, last_check=now_iso
        ),
        ServiceHealth(
            name="Redis Cache", status="healthy", latency_ms=2.1, last_check=now_iso
        ),
        ServiceHealth(
            name="RAG/pgvector", status="healthy", latency_ms=150.0, last_check=now_iso
        ),
    ]


async def _build_dashboard() -> MonitoringDashboard:
    """Build dashboard payload shared by REST and websocket endpoints."""
    return MonitoringDashboard(
        timestamp=datetime.now(timezone.utc).isoformat(),
        system=await _collect_system_metrics(),
        application=await _collect_application_metrics(),
        services=_collect_services_health(),
        alerts=await _collect_active_alerts(),
    )


@router.get(
    "/monitoring/dashboard", response_model=MonitoringDashboard, tags=["Monitoring"]
)
async def get_monitoring_dashboard() -> MonitoringDashboard:
    """Return complete monitoring dashboard data."""
    return await _build_dashboard()


@router.get("/monitoring/system", response_model=SystemMetrics, tags=["Monitoring"])
async def get_system_metrics() -> SystemMetrics:
    """Return current system metrics."""
    return await _collect_system_metrics()


@router.get(
    "/monitoring/application", response_model=ApplicationMetrics, tags=["Monitoring"]
)
async def get_application_metrics() -> ApplicationMetrics:
    """Return application metrics."""
    return await _collect_application_metrics()


@router.get(
    "/monitoring/services", response_model=list[ServiceHealth], tags=["Monitoring"]
)
async def get_services_health() -> list[ServiceHealth]:
    """Return service health payload."""
    return _collect_services_health()


@router.get("/monitoring/alerts", tags=["Monitoring"])
async def get_active_alerts() -> dict[str, list[dict[str, Any]]]:
    """Return active alerts."""
    return {"alerts": await _collect_active_alerts()}


@router.get("/monitoring/metrics/history", tags=["Monitoring"])
async def get_metrics_history(
    minutes: int = Query(default=60, ge=1, le=240),
    interval_seconds: int = Query(default=60, ge=1, le=3600),
) -> dict[str, Any]:
    """Return historical metrics within bounded period."""
    return {
        "history": _metrics_history[-minutes:],
        "interval_seconds": interval_seconds,
    }


@router.websocket("/monitoring/ws")
async def monitoring_websocket(websocket: WebSocket) -> None:
    """Push monitoring updates to authenticated admins."""
    if not _verify_ws_admin(websocket):
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Admin authentication required",
        )
        return

    if not await _WS_HUB.connect(websocket):
        await websocket.close(
            code=status.WS_1013_TRY_AGAIN_LATER,
            reason="Connection limit reached",
        )
        return

    logger.info("monitoring websocket connected")
    try:
        while True:
            dashboard = await _build_dashboard()
            await websocket.send_bytes(orjson.dumps(dashboard.model_dump(mode="json")))
            await asyncio.sleep(_WS_CONFIG.interval_seconds)
    except WebSocketDisconnect:
        logger.info("monitoring websocket disconnected")
    except Exception as exc:
        logger.exception("monitoring websocket error", extra={"error": str(exc)})
    finally:
        await _WS_HUB.disconnect(websocket)


@router.get("/monitoring/summary", tags=["Monitoring"])
async def get_monitoring_summary() -> dict[str, Any]:
    """Return quick operational summary."""
    system = await _collect_system_metrics()
    services = _collect_services_health()
    alerts = await _collect_active_alerts()

    unhealthy_services = [
        service for service in services if service.status == "unhealthy"
    ]
    critical_alerts = [alert for alert in alerts if alert.get("severity") == "critical"]

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
        "services_healthy": len(
            [service for service in services if service.status == "healthy"]
        ),
        "services_total": len(services),
        "active_alerts": len(alerts),
        "uptime_hours": round(system.uptime_seconds / 3600, 1),
    }


@router.post("/monitoring/record-request", tags=["Monitoring"])
async def record_request(
    response_time_ms: float = Query(..., ge=0.0, le=120000.0),
    is_error: bool = Query(default=False),
) -> dict[str, bool]:
    """Record synthetic request telemetry for dashboard counters."""
    async with _stats_lock:
        _stats.request_times_seconds.append(response_time_ms / 1000)
        _stats.total_requests += 1
        if is_error:
            _stats.error_count += 1
        if len(_stats.request_times_seconds) > 10000:
            _stats.request_times_seconds = _stats.request_times_seconds[-10000:]
    return {"recorded": True}
