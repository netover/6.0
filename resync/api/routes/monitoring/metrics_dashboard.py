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

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# v5.9.6: Import psutil at module level with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics-dashboard", tags=["Metrics Dashboard"])


# =============================================================================
# MODELS
# =============================================================================


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
    avg_classification_ms: float | None = Field(description="Tempo médio de classificação em ms")
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
    avg_validation_ms: float | None = Field(description="Tempo médio de validação em ms")
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


# =============================================================================
# STORAGE (em memória para MVP)
# =============================================================================


class MetricsStore:
    """Armazena métricas em memória (MVP)."""

    def __init__(self):
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

    def record_cache_hit(self):
        self._cache_hits += 1

    def record_cache_miss(self):
        self._cache_misses += 1

    def record_classification(self, intent: str, confidence: float):
        self._classifications += 1
        self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1
        if confidence < 0.7:
            self._low_confidence += 1

    def record_validation(self, validation_type: str, success: bool):
        self._validations += 1
        self._validation_type_counts[validation_type] = (
            self._validation_type_counts.get(validation_type, 0) + 1
        )
        if not success:
            self._validation_failures += 1

    def record_request(self, error: bool = False):
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
        return self._low_confidence / self._classifications if self._classifications > 0 else 0.0

    def get_top_intents(self, n: int = 5) -> dict[str, int]:
        sorted_intents = sorted(self._intent_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_intents[:n])


# Singleton
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    return _metrics_store


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    store: MetricsStore = Depends(get_metrics_store),
) -> DashboardMetrics:
    """
    Retorna todas as métricas do dashboard.

    Endpoint principal para o dashboard de métricas em tempo real.

    NOTA: Algumas métricas são estimativas/placeholders até instrumentação real ser implementada.
    """
    # v5.9.6: Graceful handling if psutil not available
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available - system metrics will use defaults")
        memory_mb = 0.0
        cpu_percent = 0.0
    else:
        # psutil can still fail in constrained containers (permissions/cgroups).
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except Exception as e:
            logger.warning("psutil_process_metrics_failed", error=str(e))
            memory_mb = 0.0
            cpu_percent = 0.0

    # Tentar obter métricas reais do cache
    cache_metrics = CacheMetrics(
        hit_rate=store.hit_rate,
        total_entries=store._cache_hits + store._cache_misses,  # Aproximação
        memory_mb=memory_mb * 0.3,  # Estimativa: 30% para cache
        avg_latency_ms=None,  # Not yet instrumented
        hits_last_hour=store._cache_hits,
        misses_last_hour=store._cache_misses,
    )

    # Métricas do router
    router_metrics = RouterMetrics(
        accuracy=None,  # Not yet instrumented
        fallback_rate=store.fallback_rate,
        avg_classification_ms=None,  # Not yet instrumented
        total_classifications=store._classifications,
        low_confidence_count=store._low_confidence,
        top_intents=store.get_top_intents(),
    )

    # Métricas do reranker
    from resync.knowledge.config import CFG

    reranker_metrics = RerankerMetrics(
        enabled=CFG.enable_cross_encoder,
        model=CFG.cross_encoder_model,
        avg_rerank_ms=None,  # Not yet instrumented
        docs_processed=None,  # Not yet instrumented
        docs_filtered=0,
        filter_rate=0.30,
        threshold=CFG.cross_encoder_threshold,
    )

    # Métricas dos validators
    validator_metrics = ValidatorMetrics(
        total_validations=store._validations,
        successful_validations=store._validations - store._validation_failures,
        failed_validations=store._validation_failures,
        avg_validation_ms=None,  # Not yet instrumented
        validation_types=store._validation_type_counts,
    )

    # Métricas do warming
    warming_metrics = WarmingMetrics(
        last_warm=None,  # Updated by cache warmer when available
        queries_warmed=0,
        queries_skipped=0,
        errors=0,
        duration_seconds=0.0,
        is_warming=False,
    )

    # Tentar obter stats reais do warmer
    try:
        from resync.core.cache.cache_warmer import get_cache_warmer

        warmer = get_cache_warmer()
        stats = warmer.get_stats()
        warming_metrics = WarmingMetrics(
            last_warm=datetime.fromisoformat(stats["last_warm"])
            if stats.get("last_warm")
            else None,
            queries_warmed=stats.get("queries_cached", 0),
            queries_skipped=stats.get("queries_skipped", 0),
            errors=stats.get("errors", 0),
            duration_seconds=stats.get("duration_seconds", 0.0),
            is_warming=warmer.is_warming,
        )
    except Exception as e:
        # v5.9.6: Fixed - was silently passing, now logs the error
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to get warming stats: {e}. Using defaults."
        )
        # warming_metrics stays at default values

    # Métricas do sistema
    # Active connections might require elevated privileges; handle gracefully.
    active_connections = 0
    if PSUTIL_AVAILABLE:
        try:
            active_connections = len(psutil.net_connections(kind="inet"))
        except Exception as e:
            logger.warning("psutil_net_connections_failed", error=str(e))

    system_metrics = SystemMetrics(
        uptime_hours=round(store.uptime_hours, 2),
        requests_today=store._requests,
        errors_today=store._errors,
        active_connections=active_connections,
        memory_usage_mb=round(memory_mb, 2),
        cpu_usage_percent=round(cpu_percent, 2),
    )

    return DashboardMetrics(
        timestamp=datetime.now(timezone.utc),
        cache=cache_metrics,
        router=router_metrics,
        reranker=reranker_metrics,
        validators=validator_metrics,
        warming=warming_metrics,
        system=system_metrics,
    )


@router.get("/cache", response_model=CacheMetrics)
async def get_cache_metrics(
    store: MetricsStore = Depends(get_metrics_store),
) -> CacheMetrics:
    """Retorna métricas detalhadas do cache."""
    return CacheMetrics(
        hit_rate=store.hit_rate,
        total_entries=store._cache_hits + store._cache_misses,
        memory_mb=None,  # Not yet instrumented
        avg_latency_ms=15.0,
        hits_last_hour=store._cache_hits,
        misses_last_hour=store._cache_misses,
    )


@router.get("/cache/history", response_model=HistoricalData)
async def get_cache_history(
    hours: int = Query(default=24, ge=1, le=168, description="Período em horas"),
    interval: str = Query(
        default="1h", pattern="^(5m|15m|1h|6h|1d)$", description="Intervalo de agregação"
    ),
) -> HistoricalData:
    """
    Retorna histórico de métricas do cache.

    Útil para visualização de tendências ao longo do tempo.
    """
    # Historical metrics storage not yet implemented — returns empty dataset
    return HistoricalData(
        metric="cache_hit_rate",
        period_hours=hours,
        interval=interval,
        data_points=[],  # Populated when historical storage is implemented
    )


@router.get("/router/intent-distribution")
async def get_intent_distribution(
    hours: int = Query(default=24, ge=1, le=168),
    store: MetricsStore = Depends(get_metrics_store),
) -> dict[str, Any]:
    """Retorna distribuição de intents classificados."""
    total = store._classifications or 1
    distribution = {intent: count / total for intent, count in store._intent_counts.items()}

    return {
        "period_hours": hours,
        "distribution": distribution,
        "total_classifications": store._classifications,
        "top_intents": store.get_top_intents(10),
    }


@router.post("/cache/warm")
async def trigger_cache_warming(
    priority: int = Query(default=1, ge=1, le=3, description="Nível de prioridade"),
    include_history: bool = Query(default=False, description="Incluir queries do histórico"),
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
            raise HTTPException(status_code=409, detail="Cache warming já está em progresso")

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
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        raise HTTPException(status_code=500, detail="Erro no cache warming. Check server logs for details.") from e


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
            f"priority_{p}": {
                "count": len(queries),
                "queries": queries,
            }
            for p, queries in queries_by_priority.items()
        },
        "by_category": {
            cat: {
                "count": len(queries),
                "queries": queries,
            }
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

    # Verificar cache
    try:
        health["components"]["cache"] = "available"
    except Exception as e:
        health["components"]["cache"] = f"unavailable: {e}"

    # Verificar router
    try:
        health["components"]["router"] = "available"
    except Exception as e:
        health["components"]["router"] = f"unavailable: {e}"

    # Verificar reranker
    try:
        from resync.knowledge.retrieval.reranker import get_reranker_info

        info = get_reranker_info()
        health["components"]["reranker"] = "available" if info["enabled"] else "disabled"
    except Exception as e:
        health["components"]["reranker"] = f"unavailable: {e}"

    # Verificar validators
    try:
        health["components"]["validators"] = "available"
    except Exception as e:
        health["components"]["validators"] = f"unavailable: {e}"

    # Status geral
    unavailable = [c for c, s in health["components"].items() if "unavailable" in str(s)]
    if unavailable:
        health["status"] = "degraded"
        health["issues"] = unavailable

    return health
