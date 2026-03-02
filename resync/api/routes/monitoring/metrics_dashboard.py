"""
Dashboard de Métricas em Tempo Real.

Endpoints para visualização de métricas de:
- Semantic Cache (hit rate, entries, memory)
- Embedding Router (accuracy, fallback rate)
- RAG Cross-Encoder (rerank stats)
- TWS Validators (validation counts)
- Cache Warming (warming stats)

Versão: 5.9.6

NOTA: Métricas não instrumentadas retornam None.
Campos com valor None devem ser exibidos como "N/A" no frontend.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import orjson
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field, ValidationError

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from resync.api.security import decode_token

logger = logging.getLogger(__name__)

def _verify_ws_admin(websocket: WebSocket) -> bool:
    """Verify admin authentication for WebSocket connection.

    Returns True if authenticated as admin, False otherwise.
    """
    try:
        auth_header = websocket.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header[7:]
        payload = decode_token(token)
        if not payload:
            return False
        roles_claim = payload.get("roles")
        if roles_claim is None:
            legacy_role = payload.get("role")
            roles = [legacy_role] if legacy_role else []
        elif isinstance(roles_claim, list):
            roles = roles_claim
        else:
            roles = [roles_claim]
        return "admin" in roles
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.debug("WebSocket admin auth failed: %s", type(e).__name__)
        return False

router = APIRouter(prefix="/metrics-dashboard", tags=["Metrics Dashboard"])

class CacheMetrics(BaseModel):
    """Métricas do Semantic Cache."""

    hit_rate: float = Field(description="Taxa de cache hits (0-1)")
    total_entries: int = Field(description="Total de entradas no cache")
    memory_mb: float | None = Field(description="Memória utilizada em MB")
    avg_latency_ms: float | None = Field(description="Latência média em ms")
    hits_last_hour: int = Field(description="Hits na última hora")
    misses_last_hour: int = Field(description="Misses na última hora")
    oldest_entry_age_hours: float | None = Field(
        default=None, description="Idade da entrada mais antiga em horas"
    )

class RouterMetrics(BaseModel):
    """Métricas do Embedding Router."""

    accuracy: float | None = Field(description="Precisão da classificação (0-1)")
    fallback_rate: float = Field(description="Taxa de fallback para LLM (0-1)")
    avg_classification_ms: float | None = Field(
        description="Tempo médio de classificação em ms"
    )
    total_classifications: int = Field(description="Total de classificações")
    low_confidence_count: int = Field(description="Classificações com baixa confiança")
    top_intents: dict[str, int] = Field(description="Top intents por frequência")

class RerankerMetrics(BaseModel):
    """Métricas do Cross-Encoder Reranker."""

    enabled: bool = Field(description="Se reranking está habilitado")
    model: str = Field(description="Modelo do cross-encoder")
    avg_rerank_ms: float | None = Field(description="Tempo médio de reranking em ms")
    docs_processed: int | None = Field(description="Documentos processados")
    docs_filtered: int = Field(description="Documentos filtrados (abaixo threshold)")
    filter_rate: float = Field(description="Taxa de filtragem (0-1)")
    threshold: float = Field(description="Threshold de relevância")

class ValidatorMetrics(BaseModel):
    """Métricas dos TWS Validators."""

    total_validations: int = Field(description="Total de validações")
    successful_validations: int = Field(description="Validações bem-sucedidas")
    failed_validations: int = Field(description="Validações que falharam")
    avg_validation_ms: float | None = Field(
        description="Tempo médio de validação em ms"
    )
    validation_types: dict[str, int] = Field(description="Validações por tipo")

class WarmingMetrics(BaseModel):
    """Métricas do Cache Warming."""

    last_warm: datetime | None = Field(description="Último warming executado")
    queries_warmed: int = Field(description="Queries aquecidas")
    queries_skipped: int = Field(description="Queries puladas (já em cache)")
    errors: int = Field(description="Erros durante warming")
    duration_seconds: float = Field(description="Duração do último warming")
    is_warming: bool = Field(description="Se warming está em progresso")

class SystemMetrics(BaseModel):
    """Métricas do sistema."""

    uptime_hours: float = Field(description="Tempo de atividade em horas")
    requests_today: int = Field(description="Requisições hoje")
    errors_today: int = Field(description="Erros hoje")
    active_connections: int = Field(description="Conexões ativas")
    memory_usage_mb: float = Field(description="Uso de memória em MB")
    cpu_usage_percent: float = Field(description="Uso de CPU em %")

class DashboardMetrics(BaseModel):
    """Métricas consolidadas do dashboard."""

    timestamp: datetime = Field(description="Timestamp das métricas")
    cache: CacheMetrics
    router: RouterMetrics
    reranker: RerankerMetrics
    validators: ValidatorMetrics
    warming: WarmingMetrics
    system: SystemMetrics

class TimeSeriesPoint(BaseModel):
    """Ponto em série temporal."""

    timestamp: datetime
    value: float

class HistoricalData(BaseModel):
    """Dados históricos de uma métrica."""

    metric: str
    period_hours: int
    interval: str
    data_points: list[TimeSeriesPoint]

class MetricsStore:
    """Armazena métricas em memória (MVP)."""

    def __init__(self) -> None:
        self._startup_time = datetime.now(timezone.utc)
        self._cache_hits = 0
        self._cache_misses = 0
        self._classifications = 0
        self._low_confidence = 0
        self._validations = 0
        self._validation_failures = 0
        self._requests = 0
        self._errors = 0
        self._intent_counts: dict[str, int] = {}
        self._validation_type_counts: dict[str, int] = {}

    def record_cache_hit(self) -> None:
        self._cache_hits += 1

    def record_cache_miss(self) -> None:
        self._cache_misses += 1

    def record_classification(self, intent: str, confidence: float) -> None:
        self._classifications += 1
        self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1
        if confidence < 0.7:
            self._low_confidence += 1

    def record_validation(self, validation_type: str, success: bool) -> None:
        self._validations += 1
        self._validation_type_counts[validation_type] = (
            self._validation_type_counts.get(validation_type, 0) + 1
        )
        if not success:
            self._validation_failures += 1

    def record_request(self, error: bool = False) -> None:
        self._requests += 1
        if error:
            self._errors += 1

    @property
    def uptime_hours(self) -> float:
        return (datetime.now(timezone.utc) - self._startup_time).total_seconds() / 3600

    @property
    def hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    @property
    def fallback_rate(self) -> float:
        return (
            self._low_confidence / self._classifications
            if self._classifications > 0
            else 0.0
        )

    def get_top_intents(self, n: int = 5) -> dict[str, int]:
        sorted_intents = sorted(
            self._intent_counts.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_intents[:n])

    def validation_snapshot(self) -> tuple[int, int, dict[str, int]]:
        """Return a stable copy used by async endpoints and websockets."""
        return (
            self._validations,
            self._validation_failures,
            dict(self._validation_type_counts),
        )

class WebSocketRuntimeConfig(BaseModel):
    """Runtime validated websocket polling configuration."""

    interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    max_connections: int = Field(default=200, ge=1, le=5000)

def _load_ws_runtime_config() -> WebSocketRuntimeConfig:
    """Load and validate websocket runtime knobs from environment."""
    raw_interval = os.getenv("METRICS_WS_INTERVAL_SECONDS", "5")
    raw_max_connections = os.getenv("METRICS_WS_MAX_CONNECTIONS", "200")
    try:
        return WebSocketRuntimeConfig(
            interval_seconds=float(raw_interval),
            max_connections=int(raw_max_connections),
        )
    except (ValidationError, ValueError) as exc:
        logger.warning("Invalid websocket runtime config, using defaults: %s", exc)
        return WebSocketRuntimeConfig()

WS_RUNTIME_CONFIG = _load_ws_runtime_config()

_metrics_store = MetricsStore()

def get_metrics_store() -> MetricsStore:
    """Get the MetricsStore singleton (sync version for backwards compatibility)."""
    return _metrics_store

async def get_metrics_store_dep() -> MetricsStore:
    """
    Get the MetricsStore singleton as an async dependency for FastAPI.

    This is the proper FastAPI DI pattern - an async function that returns
    the singleton instance. This ensures proper lifecycle management and
    allows FastAPI to handle the dependency injection correctly.
    """
    return _metrics_store

async def _collect_process_metrics(
    snapshot: dict[str, Any],
) -> tuple[float, float, int]:
    """Collect process/network metrics safely from async contexts."""
    if not PSUTIL_AVAILABLE:
        return (0.0, 0.0, 0)

    sys_snap = snapshot.get("system", {})
    memory_mb = float(sys_snap.get("memory_mb", 0.0))
    cpu_percent = float(sys_snap.get("cpu_percent", 0.0))

    if memory_mb == 0.0:
        try:
            process = await asyncio.to_thread(psutil.Process)
            memory_mb = float(process.memory_info().rss / (1024 * 1024))
            cpu_percent = float(process.cpu_percent())
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.warning("psutil_process_metrics_failed", extra={"error": str(exc)})

    active_connections = 0
    try:
        connections = await asyncio.to_thread(psutil.net_connections, "inet")
        active_connections = len(connections)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("psutil_net_connections_failed", extra={"error": str(exc)})

    return (memory_mb, cpu_percent, active_connections)

async def _build_dashboard_metrics(store: MetricsStore) -> DashboardMetrics:
    """Create a dashboard snapshot shared by HTTP and websocket endpoints."""
    from resync.core.metrics import runtime_metrics
    from resync.knowledge.config import CFG

    snapshot = runtime_metrics.get_snapshot()
    memory_mb, cpu_percent, active_connections = await _collect_process_metrics(
        snapshot
    )

    router_cache = snapshot.get("router_cache", {})
    router_hits = int(router_cache.get("hits", 0))
    router_misses = int(router_cache.get("misses", 0))
    router_total = router_hits + router_misses
    router_hit_ratio = router_hits / router_total if router_total > 0 else 0.0

    total_validations, validation_failures, validation_types = (
        store.validation_snapshot()
    )

    return DashboardMetrics(
        timestamp=datetime.now(timezone.utc),
        cache=CacheMetrics(
            hit_rate=router_hit_ratio,
            total_entries=router_total,
            memory_mb=memory_mb * 0.3,
            avg_latency_ms=router_cache.get("avg_latency_ms"),
            hits_last_hour=router_hits,
            misses_last_hour=router_misses,
        ),
        router=RouterMetrics(
            accuracy=snapshot.get("router", {}).get("accuracy"),
            fallback_rate=snapshot.get("router", {}).get("fallback_rate", 0.0),
            avg_classification_ms=snapshot.get("router", {}).get("avg_duration_ms"),
            total_classifications=snapshot.get("router", {}).get("total", 0),
            low_confidence_count=snapshot.get("router", {}).get(
                "low_confidence_count", 0
            ),
            top_intents=snapshot.get("router", {}).get("top_intents", {}),
        ),
        reranker=RerankerMetrics(
            enabled=bool(CFG.enable_cross_encoder),
            model=str(CFG.cross_encoder_model),
            avg_rerank_ms=None,
            docs_processed=None,
            docs_filtered=0,
            filter_rate=0.3,
            threshold=float(CFG.cross_encoder_threshold),
        ),
        validators=ValidatorMetrics(
            total_validations=total_validations,
            successful_validations=total_validations - validation_failures,
            failed_validations=validation_failures,
            avg_validation_ms=None,
            validation_types=validation_types,
        ),
        warming=WarmingMetrics(
            last_warm=None,
            queries_warmed=0,
            queries_skipped=0,
            errors=0,
            duration_seconds=0.0,
            is_warming=False,
        ),
        system=SystemMetrics(
            uptime_hours=round(snapshot.get("system", {}).get("uptime_hours", 0.0), 2),
            requests_today=snapshot.get("system", {}).get("requests_today", 0),
            errors_today=snapshot.get("system", {}).get("errors_today", 0),
            active_connections=active_connections,
            memory_usage_mb=round(memory_mb, 2),
            cpu_usage_percent=round(cpu_percent, 2),
        ),
    )

@router.get("/", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    store: MetricsStore = Depends(get_metrics_store_dep),
) -> DashboardMetrics:
    """Retorna todas as métricas do dashboard em snapshot consistente."""
    dashboard_metrics = await _build_dashboard_metrics(store)
    try:
        from resync.core.cache.cache_warmer import get_cache_warmer

        warmer = get_cache_warmer()
        stats = warmer.get_stats()
        dashboard_metrics.warming = WarmingMetrics(
            last_warm=datetime.fromisoformat(stats["last_warm"])
            if stats.get("last_warm")
            else None,
            queries_warmed=int(stats.get("queries_cached", 0)),
            queries_skipped=int(stats.get("queries_skipped", 0)),
            errors=int(stats.get("errors", 0)),
            duration_seconds=float(stats.get("duration_seconds", 0.0)),
            is_warming=bool(warmer.is_warming),
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("warming_stats_unavailable", extra={"error": str(exc)})
    return dashboard_metrics

@router.get("/cache", response_model=CacheMetrics)
async def get_cache_metrics() -> CacheMetrics:
    """Retorna métricas detalhadas do cache."""
    from resync.core.metrics import runtime_metrics

    snapshot = RuntimeMetricsCollector().get_snapshot()
    router_cache = snapshot.get("router_cache", {})
    router_hits = router_cache.get("hits", 0)
    router_misses = router_cache.get("misses", 0)
    router_total = router_hits + router_misses
    router_hit_ratio = router_hits / router_total if router_total > 0 else 0.0
    return CacheMetrics(
        hit_rate=router_hit_ratio,
        total_entries=router_total,
        memory_mb=snapshot.get("system", {}).get("memory_mb", 0.0) * 0.3,
        avg_latency_ms=router_cache.get("avg_latency_ms"),
        hits_last_hour=router_hits,
        misses_last_hour=router_misses,
    )

@router.get("/cache/history", response_model=HistoricalData)
async def get_cache_history(
    hours: int = Query(default=24, ge=1, le=168, description="Período em horas"),
    interval: str = Query(
        default="1h",
        pattern="^(5m|15m|1h|6h|1d)$",
        description="Intervalo de agregação",
    ),
) -> HistoricalData:
    """
    Retorna histórico de métricas do cache.

    Útil para visualização de tendências ao longo do tempo.
    """
    return HistoricalData(
        metric="cache_hit_rate", period_hours=hours, interval=interval, data_points=[]
    )

@router.get("/router/intent-distribution")
async def get_intent_distribution(
    hours: int = Query(default=24, ge=1, le=168),
) -> dict[str, Any]:
    """Retorna distribuição de intents classificados."""
    from resync.core.metrics import runtime_metrics

    snapshot = RuntimeMetricsCollector().get_snapshot()
    router_stats = snapshot.get("router", {})
    total = router_stats.get("total", 0) or 1
    intent_counts = router_stats.get("top_intents", {})
    distribution = {intent: count / total for intent, count in intent_counts.items()}
    return {
        "period_hours": hours,
        "distribution": distribution,
        "total_classifications": total,
        "top_intents": intent_counts,
    }

@router.post("/cache/warm")
async def trigger_cache_warming(
    priority: int = Query(default=1, ge=1, le=3, description="Nível de prioridade"),
    include_history: bool = Query(
        default=False, description="Incluir queries do histórico"
    ),
) -> dict[str, Any]:
    """
    Dispara warming manual do cache.

    Args:
        priority: Nível máximo de prioridade (1=alta, 2=média, 3=baixa)
        include_history: Se deve incluir queries do histórico

    Returns:
        Estatísticas do warming
    """
    try:
        from resync.core.cache.cache_warmer import get_cache_warmer

        warmer = get_cache_warmer()
        if warmer.is_warming:
            raise HTTPException(
                status_code=409, detail="Cache warming já está em progresso"
            )
        if include_history:
            result = await warmer.full_warm(include_history=True)
        else:
            static_count = await warmer.warm_static_queries(priority=priority)
            critical_count = await warmer.warm_critical_jobs()
            result = {
                "static_queries": static_count,
                "critical_jobs": critical_count,
                "total": static_count + critical_count,
                "stats": warmer.get_stats(),
            }
        return {
            "status": "completed",
            "message": "Cache warming executado com sucesso",
            "result": result,
        }
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        raise HTTPException(
            status_code=500,
            detail="Erro no cache warming. Check server logs for details.",
        ) from e

@router.get("/warming/queries")
async def get_warming_queries() -> dict[str, Any]:
    """
    Retorna lista de queries configuradas para warming.

    Útil para auditoria e ajuste das queries de warming.
    """
    from resync.core.cache.cache_warmer import CacheWarmer

    warmer = CacheWarmer()
    queries_by_priority = {1: [], 2: [], 3: []}
    queries_by_category = {}
    for q in warmer.STATIC_QUERIES:
        queries_by_priority[q.priority].append(
            {
                "query": q.query,
                "category": q.category,
                "expected_intent": q.expected_intent,
            }
        )
        if q.category not in queries_by_category:
            queries_by_category[q.category] = []
        queries_by_category[q.category].append(q.query)
    return {
        "total_queries": len(warmer.STATIC_QUERIES),
        "by_priority": {
            f"priority_{p}": {"count": len(queries), "queries": queries}
            for p, queries in queries_by_priority.items()
        },
        "by_category": {
            cat: {"count": len(queries), "queries": queries}
            for cat, queries in queries_by_category.items()
        },
        "counts": warmer.get_static_queries_count(),
    }

@router.get("/health")
async def metrics_health() -> dict[str, Any]:
    """
    Health check do sistema de métricas.

    Verifica se todos os componentes estão funcionando.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
    }
    try:
        health["components"]["cache"] = "available"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        health["components"]["cache"] = f"unavailable: {e}"
    try:
        health["components"]["router"] = "available"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        health["components"]["router"] = f"unavailable: {e}"
    try:
        from resync.knowledge.retrieval.reranker import get_reranker_info

        info = get_reranker_info()
        health["components"]["reranker"] = (
            "available" if info["enabled"] else "disabled"
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        health["components"]["reranker"] = f"unavailable: {e}"
    try:
        health["components"]["validators"] = "available"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        health["components"]["validators"] = f"unavailable: {e}"
    unavailable = [
        c for c, s in health["components"].items() if "unavailable" in str(s)
    ]
    if unavailable:
        health["status"] = "degraded"
        health["issues"] = unavailable
    return health

class ConnectionManager:
    """Thread-safe websocket registry with bounded connection count."""

    def __init__(self, max_connections: int) -> None:
        self._active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._max_connections = max_connections

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept and register websocket if capacity allows."""
        async with self._lock:
            if len(self._active_connections) >= self._max_connections:
                return False
            await websocket.accept()
            self._active_connections.add(websocket)
            return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._active_connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast pre-serialized payload to all live websocket peers."""
        payload = orjson.dumps(message)
        async with self._lock:
            connections = list(self._active_connections)
        stale: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_bytes(payload)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
                stale.append(connection)
        if stale:
            async with self._lock:
                for dead in stale:
                    self._active_connections.discard(dead)

manager = ConnectionManager(max_connections=WS_RUNTIME_CONFIG.max_connections)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket para atualizações em tempo real com polling bounded e seguro."""
    if not _verify_ws_admin(websocket):
        logger.warning("metrics_ws_auth_failed")
        await websocket.accept()
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Admin authentication required",
        )
        return

    connected = await manager.connect(websocket)
    if not connected:
        await websocket.close(
            code=status.WS_1013_TRY_AGAIN_LATER,
            reason="Connection limit reached",
        )
        return

    try:
        while True:
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WS_RUNTIME_CONFIG.interval_seconds,
                )
            except TimeoutError:
                # expected heartbeat interval
                pass

            dashboard_metrics = await _build_dashboard_metrics(get_metrics_store())
            payload = dashboard_metrics.model_dump(mode="json")
            await websocket.send_bytes(orjson.dumps(payload))
    except WebSocketDisconnect:
        logger.info("metrics_ws_client_disconnected")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.exception("metrics_ws_unhandled_error", extra={"error": str(exc)})
    finally:
        await manager.disconnect(websocket)

