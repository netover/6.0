# monitoring_dashboard.py — API endpoint para dashboard interno de monitoramento
# Substitui necessidade de Prometheus/Grafana com solução leve e integrada
#
# Características:
#   - Rolling window de 2 horas em memória (~1.4 MB)
#   - Atualização a cada 5 segundos
#   - Zero dependências externas
#   - Integrado com métricas existentes do Resync


import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Configurações do rolling buffer
HISTORY_WINDOW_SECONDS = 2 * 60 * 60  # 2 horas
SAMPLE_INTERVAL_SECONDS = 5  # Amostragem a cada 5s
MAX_SAMPLES = HISTORY_WINDOW_SECONDS // SAMPLE_INTERVAL_SECONDS  # 1440 amostras


@dataclass
class MetricSample:
    """Uma amostra de métricas em um ponto no tempo."""

    timestamp: float
    datetime_str: str

    # API Metrics
    requests_total: int = 0
    requests_per_sec: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_avg: float = 0.0

    # Cache Metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    cache_size: int = 0
    cache_evictions: int = 0

    # Agent Metrics
    agents_active: int = 0
    agents_created: int = 0
    agents_failed: int = 0

    # LLM Metrics
    llm_requests: int = 0
    llm_tokens_used: int = 0
    llm_errors: int = 0

    # TWS Metrics
    tws_connected: bool = False
    tws_latency_ms: float = 0.0
    tws_errors: int = 0
    tws_requests_success: int = 0
    tws_requests_failed: int = 0

    # System Metrics
    system_uptime: float = 0.0
    system_availability: float = 100.0
    async_operations_active: int = 0
    correlation_ids_active: int = 0


@dataclass
class DashboardMetricsStore:
    """Store para métricas do dashboard com rolling window."""

    samples: deque[MetricSample] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    start_time: float = field(default_factory=time.time)
    last_sample_time: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Contadores acumulados para calcular deltas
    _prev_requests: int = 0
    _prev_cache_hits: int = 0
    _prev_cache_misses: int = 0
    _prev_agents_created: int = 0
    _prev_llm_requests: int = 0

    # Alertas ativos
    alerts: list[dict[str, Any]] = field(default_factory=list)

    async def add_sample(self, sample: MetricSample) -> None:
        """Adiciona uma nova amostra ao histórico."""
        async with self._lock:
            self.samples.append(sample)
            self.last_sample_time = sample.timestamp

            # Verificar alertas
            self._check_alerts(sample)

    def _check_alerts(self, sample: MetricSample) -> None:
        """Verifica condições de alerta."""
        new_alerts = []

        # Alerta: Error rate > 5%
        if sample.error_rate > 5.0:
            new_alerts.append(
                {
                    "type": "error_rate",
                    "severity": "warning" if sample.error_rate < 10 else "critical",
                    "message": "Error rate elevado: {sample.error_rate:.1f}%",
                    "timestamp": sample.datetime_str,
                }
            )

        # Alerta: Cache hit ratio < 80%
        if sample.cache_hit_ratio < 80.0 and (sample.cache_hits + sample.cache_misses) > 100:
            new_alerts.append(
                {
                    "type": "cache_ratio",
                    "severity": "warning",
                    "message": "Cache hit ratio baixo: {sample.cache_hit_ratio:.1f}%",
                    "timestamp": sample.datetime_str,
                }
            )

        # Alerta: Response time > 500ms
        if sample.response_time_p95 > 500:
            new_alerts.append(
                {
                    "type": "latency",
                    "severity": "warning" if sample.response_time_p95 < 1000 else "critical",
                    "message": "Latência P95 elevada: {sample.response_time_p95:.0f}ms",
                    "timestamp": sample.datetime_str,
                }
            )

        # Alerta: TWS desconectado
        if not sample.tws_connected:
            new_alerts.append(
                {
                    "type": "tws_connection",
                    "severity": "critical",
                    "message": "TWS desconectado",
                    "timestamp": sample.datetime_str,
                }
            )

        # Manter últimos 20 alertas
        self.alerts = (new_alerts + self.alerts)[:20]

    async def get_current_metrics(self) -> dict[str, Any]:
        """Retorna métricas atuais para o dashboard."""
        async with self._lock:
            if not self.samples:
                return self._empty_metrics()

            current = self.samples[-1]

            # Calcular tendências (comparando com 5 minutos atrás)
            trend_sample = None
            if len(self.samples) > 60:  # 60 amostras = 5 minutos
                trend_sample = self.samples[-61]

            return {
                "status": "ok" if current.error_rate < 5 else "degraded",
                "uptime": self._format_uptime(current.system_uptime),
                "uptime_seconds": current.system_uptime,
                "version": "5.1.0",
                "last_update": current.datetime_str,
                "timestamp": current.timestamp,
                "api": {
                    "requests_per_sec": round(current.requests_per_sec, 1),
                    "requests_total": current.requests_total,
                    "error_rate": round(current.error_rate, 2),
                    "error_count": current.error_count,
                    "response_time_p50": round(current.response_time_p50, 1),
                    "response_time_p95": round(current.response_time_p95, 1),
                    "response_time_avg": round(current.response_time_avg, 1),
                    "trend": self._calc_trend(
                        current.requests_per_sec,
                        trend_sample.requests_per_sec if trend_sample else 0,
                    ),
                },
                "cache": {
                    "hit_ratio": round(current.cache_hit_ratio, 1),
                    "hits": current.cache_hits,
                    "misses": current.cache_misses,
                    "size": current.cache_size,
                    "evictions": current.cache_evictions,
                    "trend": self._calc_trend(
                        current.cache_hit_ratio, trend_sample.cache_hit_ratio if trend_sample else 0
                    ),
                },
                "agents": {
                    "active": current.agents_active,
                    "created_total": current.agents_created,
                    "failed_total": current.agents_failed,
                    "trend": self._calc_trend(
                        current.agents_active, trend_sample.agents_active if trend_sample else 0
                    ),
                },
                "llm": {
                    "requests": current.llm_requests,
                    "tokens_used": current.llm_tokens_used,
                    "errors": current.llm_errors,
                },
                "tws": {
                    "connected": current.tws_connected,
                    "latency_ms": round(current.tws_latency_ms, 1),
                    "requests_success": current.tws_requests_success,
                    "requests_failed": current.tws_requests_failed,
                    "status": "online" if current.tws_connected else "offline",
                },
                "system": {
                    "availability": round(current.system_availability, 2),
                    "async_operations": current.async_operations_active,
                    "correlation_ids": current.correlation_ids_active,
                },
                "alerts": self.alerts[:5],  # Últimos 5 alertas
            }

    async def get_history(self, minutes: int = 120) -> dict[str, Any]:
        """Retorna histórico de métricas para gráficos."""
        async with self._lock:
            # Calcular quantas amostras pegar
            samples_needed = min(len(self.samples), (minutes * 60) // SAMPLE_INTERVAL_SECONDS)

            if samples_needed == 0:
                return self._empty_history()

            # Pegar últimas N amostras
            recent_samples = list(self.samples)[-samples_needed:]

            # Formatar para gráficos
            return {
                "timestamps": [s.datetime_str for s in recent_samples],
                "api": {
                    "requests_per_sec": [round(s.requests_per_sec, 1) for s in recent_samples],
                    "error_rate": [round(s.error_rate, 2) for s in recent_samples],
                    "response_time_p50": [round(s.response_time_p50, 1) for s in recent_samples],
                    "response_time_p95": [round(s.response_time_p95, 1) for s in recent_samples],
                },
                "cache": {
                    "hit_ratio": [round(s.cache_hit_ratio, 1) for s in recent_samples],
                    "operations": [s.cache_hits + s.cache_misses for s in recent_samples],
                },
                "agents": {
                    "active": [s.agents_active for s in recent_samples],
                    "created": [s.agents_created for s in recent_samples],
                },
                "tws": {
                    "latency": [round(s.tws_latency_ms, 1) for s in recent_samples],
                    "connected": [1 if s.tws_connected else 0 for s in recent_samples],
                },
                "sample_count": len(recent_samples),
                "interval_seconds": SAMPLE_INTERVAL_SECONDS,
            }

    def _empty_metrics(self) -> dict[str, Any]:
        """Retorna estrutura vazia de métricas."""
        return {
            "status": "initializing",
            "uptime": "0s",
            "uptime_seconds": 0,
            "version": "5.1.0",
            "last_update": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "api": {"requests_per_sec": 0, "error_rate": 0, "response_time_p95": 0, "trend": 0},
            "cache": {"hit_ratio": 0, "hits": 0, "misses": 0, "trend": 0},
            "agents": {"active": 0, "created_total": 0, "failed_total": 0, "trend": 0},
            "llm": {"requests": 0, "tokens_used": 0, "errors": 0},
            "tws": {"connected": False, "latency_ms": 0, "status": "unknown"},
            "system": {"availability": 0, "async_operations": 0},
            "alerts": [],
        }

    def _empty_history(self) -> dict[str, Any]:
        """Retorna estrutura vazia de histórico."""
        return {
            "timestamps": [],
            "api": {
                "requests_per_sec": [],
                "error_rate": [],
                "response_time_p50": [],
                "response_time_p95": [],
            },
            "cache": {"hit_ratio": [], "operations": []},
            "agents": {"active": [], "created": []},
            "tws": {"latency": [], "connected": []},
            "sample_count": 0,
            "interval_seconds": SAMPLE_INTERVAL_SECONDS,
        }

    def _format_uptime(self, seconds: float) -> str:
        """Formata uptime para exibição."""
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        if seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"

    def _calc_trend(self, current: float, previous: float) -> float:
        """Calcula tendência em porcentagem."""
        if previous == 0:
            return 0
        return round(((current - previous) / previous) * 100, 1)


# Singleton do store
_metrics_store: DashboardMetricsStore | None = None
_collector_task: asyncio.Task | None = None


def get_metrics_store() -> DashboardMetricsStore:
    """Obtém o store singleton."""
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = DashboardMetricsStore()
    return _metrics_store


async def collect_metrics_sample() -> MetricSample:
    """Coleta uma amostra das métricas atuais do sistema."""
    from resync.core.metrics import get_runtime_metrics

    try:
        metrics = get_runtime_metrics()
        snapshot = metrics.get_snapshot()

        now = time.time()
        store = get_metrics_store()

        # Calcular requests/sec baseado no delta
        requests_total = (
            snapshot.get("agent", {}).get("initializations", 0)
            + snapshot.get("tws", {}).get("success", 0)
            if "tws" in snapshot
            else snapshot.get("audit", {}).get("records_created", 0)
        )

        # Estimar requests/sec (simplificado)
        time_delta = (
            now - store.last_sample_time if store.last_sample_time > 0 else SAMPLE_INTERVAL_SECONDS
        )
        requests_delta = requests_total - store._prev_requests
        requests_per_sec = requests_delta / time_delta if time_delta > 0 else 0
        store._prev_requests = requests_total

        # Cache metrics
        cache = snapshot.get("cache", {})
        cache_hits = cache.get("hits", 0)
        cache_misses = cache.get("misses", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_ratio = (cache_hits / cache_total * 100) if cache_total > 0 else 100

        # SLO metrics
        slo = snapshot.get("slo", {})

        return MetricSample(
            timestamp=now,
            datetime_str=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            # API
            requests_total=requests_total,
            requests_per_sec=max(0, requests_per_sec),
            error_count=snapshot.get("agent", {}).get("creation_failures", 0),
            error_rate=slo.get("api_error_rate", 0) * 100,
            response_time_p50=slo.get("api_response_time_p50", 0) * 1000,  # ms
            response_time_p95=slo.get("api_response_time_p95", 0) * 1000,  # ms
            response_time_avg=(
                slo.get("api_response_time_p50", 0) + slo.get("api_response_time_p95", 0)
            )
            / 2
            * 1000,
            # Cache
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_ratio=cache_hit_ratio,
            cache_size=int(cache.get("size", 0)),
            cache_evictions=cache.get("evictions", 0),
            # Agents
            agents_active=snapshot.get("agent", {}).get("active_count", 0),
            agents_created=snapshot.get("agent", {}).get("initializations", 0),
            agents_failed=snapshot.get("agent", {}).get("creation_failures", 0),
            # LLM metrics from snapshot
            llm_requests=snapshot.get("llm", {}).get("requests", 0),
            llm_tokens_used=snapshot.get("llm", {}).get("tokens_used", 0),
            llm_errors=snapshot.get("llm", {}).get("errors", 0),
            # TWS
            tws_connected=slo.get("tws_connection_success_rate", 0) > 0.5,
            tws_latency_ms=slo.get("api_response_time_p50", 0) * 1000,
            tws_errors=0,
            tws_requests_success=0,
            tws_requests_failed=0,
            # System
            system_uptime=now - store.start_time,
            system_availability=slo.get("availability", 1.0) * 100,
            async_operations_active=snapshot.get("system", {}).get("async_operations_active", 0),
            correlation_ids_active=snapshot.get("system", {}).get("correlation_ids_active", 0),
        )

    except Exception as e:
        logger.error("Erro ao coletar métricas: %s", e)
        return MetricSample(timestamp=time.time(), datetime_str=datetime.now(timezone.utc).strftime("%H:%M:%S"))


async def metrics_collector_loop():
    """Loop de coleta de métricas em background."""
    store = get_metrics_store()

    while True:
        try:
            sample = await collect_metrics_sample()
            await store.add_sample(sample)
        except Exception as e:
            logger.error("Erro no collector loop: %s", e)

        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)


def start_metrics_collector():
    """Inicia o coletor de métricas em background."""
    global _collector_task

    if _collector_task is None or _collector_task.done():
        try:
            loop = asyncio.get_event_loop()
            _collector_task = loop.create_task(metrics_collector_loop())
            logger.info("Dashboard metrics collector iniciado")
        except RuntimeError:
            # Não há event loop rodando ainda
            logger.warning("Event loop não disponível, collector será iniciado depois")


def stop_metrics_collector():
    """Para o coletor de métricas."""
    global _collector_task

    if _collector_task and not _collector_task.done():
        _collector_task.cancel()
        logger.info("Dashboard metrics collector parado")


# =====================================
# FastAPI Router
# =====================================

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# Nota: @router.on_event("startup") foi removido (deprecado no FastAPI 0.111+)
# Lógica de inicialização do metrics collector movida para app_factory.py lifespan manager
# O collector é iniciado automaticamente no startup da aplicação


@router.get("/current")
async def get_current_metrics():
    """
    Retorna métricas atuais do sistema.

    Substitui necessidade de Prometheus + Grafana com solução integrada.
    Consumo: ~50MB RAM vs ~1.2GB do stack externo.
    """
    store = get_metrics_store()
    return JSONResponse(content=await store.get_current_metrics())


@router.get("/history")
async def get_metrics_history(minutes: int = 120):
    """
    Retorna histórico de métricas para gráficos.

    Args:
        minutes: Minutos de histórico (máx 120 = 2 horas)

    Returns:
        Dados formatados para Chart.js
    """
    minutes = min(max(1, minutes), 120)  # Limitar entre 1-120 minutos
    store = get_metrics_store()
    return JSONResponse(content=await store.get_history(minutes))


@router.get("/alerts")
async def get_active_alerts():
    """Retorna alertas ativos do sistema."""
    store = get_metrics_store()
    async with store._lock:
        return JSONResponse(content={"alerts": store.alerts, "count": len(store.alerts)})


@router.get("/health")
async def monitoring_health():
    """Health check do sistema de monitoramento."""
    store = get_metrics_store()
    return JSONResponse(
        content={
            "status": "healthy",
            "samples_collected": len(store.samples),
            "max_samples": MAX_SAMPLES,
            "history_window": f"{HISTORY_WINDOW_SECONDS // 3600}h",
            "sample_interval": f"{SAMPLE_INTERVAL_SECONDS}s",
            "memory_estimate_mb": round(len(store.samples) * 0.001, 2),  # ~1KB por amostra
        }
    )


# WebSocket para atualizações em tempo real
connected_clients: list[WebSocket] = []


@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket para métricas em tempo real.

    Envia atualizações a cada 5 segundos automaticamente.
    """
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        store = get_metrics_store()

        while True:
            # Enviar métricas atuais
            metrics = await store.get_current_metrics()
            await websocket.send_json(metrics)

            # Aguardar próximo intervalo
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        if websocket in connected_clients:
            connected_clients.remove(websocket)
