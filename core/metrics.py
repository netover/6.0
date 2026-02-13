# metrics.py — Runtime telemetry/metrics robusto com Prometheus e correlação real "context-aware"
# Melhorias principais:
#  - Singleton unificado (sem "instâncias duplas")
#  - contextvars para correlation_id por execução (async/thread)
#  - MetricCounter/Gauge thread-safe
#  - MetricHistogram completo (buckets + _bucket/_sum/_count + amostras)
#  - Export Prometheus correto (HELP/TYPE e tipos adequados)
#  - Uso de time.perf_counter() para durações de alta precisão
#
# Referências de boas práticas:
#  - Prometheus tipos/semântica e formato de exposição: prom docs (metric types + exposition + histogram)
#  - Buckets e "le" label: prom docs e otel compat
#  - Temporização: perf_counter() (monotônico, alta resolução)
#  - Contexto por execução: contextvars
#
# (Ver documentação: https://prometheus.io/docs/... e Python docs para contextvars e perf_counter)


import logging
import threading
import time
import uuid
from bisect import bisect_right
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------
# Utilidades
# -----------------------------
def _now_wall() -> float:
    """Wall-clock (para timestamps e logs)."""
    return time.time()


def _now_perf() -> float:
    """Monotônico, alta resolução (para medir durações)."""
    return time.perf_counter()  # recomendado para benchmarking/medição de duração


def _sanitize_metric_name(name: str) -> str:
    """Sanitiza nomes para Prometheus (básico: substitui chars inválidos por '_')."""
    out = []
    for ch in name:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in [":", "_"]:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    # Métrica deve começar por [a-zA-Z_:]
    if not s or (not (s[0].isalpha() or s[0] in (":", "_"))):
        s = "m_" + s
    return s


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


# -----------------------------
# Métricas básicas
# -----------------------------
@dataclass
class MetricCounter:
    """Counter (só cresce ou zera)."""

    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self.value += amount

    def get_and_reset(self) -> int:
        with self._lock:
            v = self.value
            self.value = 0
            return v

    def set(self, value: int) -> None:
        with self._lock:
            # Mantemos por compatibilidade, mas counters não deveriam ter 'set' arbitrário.
            self.value = value


@dataclass
class MetricGauge:
    """Gauge (sobe e desce)."""

    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        with self._lock:
            self.value = float(value)

    def get(self) -> float:
        with self._lock:
            return self.value


class MetricHistogram:
    """
    Histogram com buckets cumulativos, _sum e _count.
    - boundaries: limites de bucket (ex.: durações/latência)
    - counts[i]: contagem até boundaries[i] (cumulativa ao exportar)
    - counts[-1]: +Inf
    """

    __slots__ = (
        "boundaries",  # sorted list[float]
        "counts",  # list[int] len = len(boundaries)+1
        "samples",  # sliding window p/ quantis/diagnóstico
        "max_samples",
        "_sum",
        "_count",
        "_min",
        "_max",
        "_lock",
        "help_text",
    )

    def __init__(
        self,
        boundaries: Iterable[float] | None = None,
        max_samples: int = 1024,
        help_text: str = "",
    ) -> None:
        # Defaults bons p/ latências (segundos) — próximos aos defaults do Prom-client
        if boundaries is None:
            boundaries = [
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ]  # pode ajustar conforme o domínio
        b = sorted(set(float(x) for x in boundaries))
        self.boundaries: list[float] = b
        self.counts: list[int] = [0] * (len(b) + 1)  # +Inf
        self.samples: list[float] = []
        self.max_samples = int(max_samples)
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float | None = None
        self._max: float | None = None
        self._lock = threading.Lock()
        self.help_text = help_text

    def observe(self, value: float) -> None:
        with self._lock:
            # atualiza somatório/contagem
            self._sum += float(value)
            self._count += 1
            # min/max
            if self._min is None or value < self._min:
                self._min = value
            if self._max is None or value > self._max:
                self._max = value
            # bucket index
            idx = bisect_right(self.boundaries, value)
            self.counts[idx] += 1
            # amostra p/ depuração/quantis simples
            self.samples.append(value)
            if len(self.samples) > self.max_samples:
                self.samples.pop(0)

    # Acesso rápido
    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def min(self) -> float | None:
        return self._min

    @property
    def max(self) -> float | None:
        return self._max

    def _quantile(self, q: float) -> float | None:
        """Quantil empírico simples a partir do buffer de amostras."""
        if not self.samples:
            return None
        s = sorted(self.samples)
        idx = int(q * (len(s) - 1))
        idx = max(0, min(len(s) - 1, idx))
        return s[idx]

    def render_prometheus(self, metric_name: str) -> list[str]:
        """
        Gera linhas de exposição para Prometheus:
          # HELP ...
          # TYPE ... histogram
          <name>_bucket{le="..."} <cumul>
          <name>_count <count>
          <name>_sum <sum>
        """
        name = _sanitize_metric_name(metric_name)
        lines: list[str] = []
        help_txt = self.help_text or f"Histogram for {name}"
        lines.append(f"# HELP {name} {_escape_help(help_txt)}")
        lines.append(f"# TYPE {name} histogram")
        with self._lock:
            # cumulativos
            cumul = 0
            for b, c in zip(self.boundaries, self.counts[:-1], strict=False):
                cumul += c
                lines.append(f'{name}_bucket{{le="{b}"}} {cumul}')
            # +Inf
            cumul += self.counts[-1]
            lines.append(f'{name}_bucket{{le="+Inf"}} {cumul}')
            # sum/count
            lines.append(f"{name}_sum {self._sum}")
            lines.append(f"{name}_count {self._count}")
        return lines


# -----------------------------
# Correlation por contexto (contextvars)
# -----------------------------
_current_correlation_id: ContextVar[str | None] = ContextVar(
    "_current_correlation_id", default=None
)


class RuntimeMetrics:
    """
    Runtime telemetry — thread-safe e async-friendly.
    Mantém counters/gauges/histograms, correlação por contexto,
    health checks e export Prometheus correto.
    """

    # -------------------------
    # Construção
    # -------------------------
    def __init__(self) -> None:
        # Agent
        self.agent_initializations = MetricCounter()
        self.agent_creation_failures = MetricCounter()
        self.agent_mock_fallbacks = MetricCounter()
        self.agent_active_count = MetricGauge()
        self.agent_orchestration_time = MetricHistogram(
            help_text="Agent orchestration duration seconds"
        )

        # Cache
        self.cache_hits = MetricCounter()
        self.cache_misses = MetricCounter()
        self.cache_evictions = MetricCounter()
        self.cache_sets = MetricCounter()
        self.cache_size = MetricGauge()
        self.cache_cleanup_cycles = MetricCounter()

        # Audit
        self.audit_records_created = MetricCounter()
        self.audit_records_approved = MetricCounter()
        self.audit_records_rejected = MetricCounter()
        self.audit_batch_operations = MetricCounter()
        self.audit_pending_timeout = MetricCounter()
        self.audit_rollback_operations = MetricCounter()

        # System
        self.correlation_ids_active = MetricGauge()
        self.async_operations_active = MetricGauge()

        # LLM
        self.llm_requests = MetricCounter()
        self.llm_errors = MetricCounter()
        self.llm_duration = MetricHistogram(help_text="LLM operation duration seconds")
        self.llm_tokens = MetricCounter()

        # "error_rate" era usado para tempos de erro — manter nome por compatibilidade
        self.error_rate = MetricHistogram(help_text="Observed error processing duration seconds")

        # Error tracking
        self.error_counts: dict[str, MetricCounter] = {}
        self._error_lock = threading.Lock()

        # TWS
        self.tws_status_requests_success = MetricCounter()
        self.tws_status_requests_failed = MetricCounter()
        self.tws_workstations_total = MetricGauge()
        self.tws_jobs_total = MetricGauge()

        # Connection validation
        self.connection_validations_total = MetricCounter()
        self.connection_validation_success = MetricCounter()
        self.connection_validation_failure = MetricCounter()
        self.health_check_with_auto_enable = MetricCounter()

        # SLO
        self.api_response_time = MetricHistogram(help_text="API response time seconds")
        self.api_error_rate = MetricGauge()  # percentual agregado externamente
        self.system_availability = MetricGauge()  # percentual agregado externamente
        self.tws_connection_success_rate = MetricGauge()
        self.ai_agent_response_time = MetricHistogram(help_text="AI agent response time seconds")

        # Correlation tracking
        self._correlation_context: dict[str, dict[str, Any]] = {}
        self._correlation_lock = threading.Lock()

        # Health monitoring
        self._health_checks: dict[str, dict[str, Any]] = {}
        self._health_lock = threading.Lock()

        logger.info("RuntimeMetrics initialized")

    # -------------------------
    # Correlação
    # -------------------------
    def create_correlation_id(self, context: dict[str, Any] | None = None) -> str:
        """Cria novo correlation_id e registra contexto; não altera o contextvar atual."""
        cid = str(uuid.uuid4())
        with self._correlation_lock:
            self._correlation_context[cid] = {
                "created_at": _now_wall(),
                "context": context or {},
                "operations": [],
            }
            self.correlation_ids_active.set(len(self._correlation_context))
        logger.debug("Created correlation ID: %s", cid)
        return cid

    def set_current_correlation_id(self, correlation_id: str | None) -> None:
        """Define o correlation_id atual (contexto da tarefa/thread)."""
        _current_correlation_id.set(correlation_id)

    def get_current_correlation_id(self) -> str | None:
        return _current_correlation_id.get()

    def add_correlation_event(
        self, correlation_id: str, event: str, data: dict[str, Any] | None = None
    ) -> None:
        """Adiciona evento a um contexto de correlação existente."""
        with self._correlation_lock:
            ctx = self._correlation_context.get(correlation_id)
            if ctx is not None:
                ctx["operations"].append(
                    {"timestamp": _now_wall(), "event": event, "data": data or {}}
                )

    def close_correlation_id(self, correlation_id: str) -> None:
        """Fecha um contexto de correlação e loga resumo."""
        with self._correlation_lock:
            ctx = self._correlation_context.pop(correlation_id, None)
            self.correlation_ids_active.set(len(self._correlation_context))
        if ctx is not None:
            duration = _now_wall() - ctx["created_at"]
            op_count = len(ctx["operations"])
            logger.info(
                "Correlation %s completed: duration=%.4fs, operations=%d",
                correlation_id,
                duration,
                op_count,
            )

    # Context manager p/ spans
    class _Span:
        """_ span."""

        def __init__(self, rm: "RuntimeMetrics", name: str, context: dict[str, Any] | None = None):
            self.rm = rm
            self.name = name
            self.context = context or {}
            self.correlation_id: str | None = None
            self._t0 = 0.0

        def __enter__(self):
            self.correlation_id = self.rm.create_correlation_id({"span": self.name, **self.context})
            self.rm.set_current_correlation_id(self.correlation_id)
            self._t0 = _now_perf()
            return self.correlation_id

        def __exit__(self, exc_type, exc_val, exc_tb):
            dur = _now_perf() - self._t0
            # registrar duração no histogram genérico (opcionalmente reutilize um hist dedicado)
            self.rm.agent_orchestration_time.observe(dur)
            cid = self.correlation_id
            if cid:
                self.rm.add_correlation_event(cid, "span_end", {"duration_seconds": dur})
                self.rm.close_correlation_id(cid)
            # limpar contextvar
            self.rm.set_current_correlation_id(None)

    def span(self, name: str, context: dict[str, Any] | None = None) -> "RuntimeMetrics._Span":
        """Cria um span de correlação (context manager)."""
        return RuntimeMetrics._Span(self, name, context)

    # -------------------------
    # Health
    # -------------------------
    def record_health_check(
        self, component: str, status: str, details: dict[str, Any] | None = None
    ) -> None:
        with self._health_lock:
            self._health_checks[component] = {
                "status": status,
                "timestamp": _now_wall(),
                "details": details or {},
            }
            logger.debug("Health check for %s: %s", component, status)

    def get_health_status(self) -> dict[str, Any]:
        with self._health_lock:
            return dict(self._health_checks)

    # -------------------------
    # Erros
    # -------------------------
    def record_error(self, error_type: str, processing_time_seconds: float) -> None:
        with self._error_lock:
            ctr = self.error_counts.get(error_type)
            if ctr is None:
                ctr = self.error_counts[error_type] = MetricCounter()
            ctr.increment()
        self.error_rate.observe(float(processing_time_seconds))

    # -------------------------
    # Snapshot (para dashboards/diagnóstico rápido)
    # -------------------------
    def _calculate_error_rate_ratio(self) -> float:
        total_requests = (
            self.agent_initializations.value
            + self.tws_status_requests_success.value
            + self.tws_status_requests_failed.value
        )
        if total_requests > 0:
            return (
                self.agent_creation_failures.value + self.tws_status_requests_failed.value
            ) / total_requests
        return 0.0

    def _calculate_cache_hit_ratio(self) -> float:
        total_cache_ops = self.cache_hits.value + self.cache_misses.value
        if total_cache_ops > 0:
            return self.cache_hits.value / total_cache_ops
        return 0.0

    def get_snapshot(self) -> dict[str, Any]:
        # Erros por tipo
        error_metrics = {}
        with self._error_lock:
            for et, counter in self.error_counts.items():
                error_metrics[et] = counter.value

        return {
            "agent": {
                "initializations": self.agent_initializations.value,
                "creation_failures": self.agent_creation_failures.value,
                "mock_fallbacks": self.agent_mock_fallbacks.value,
                "active_count": self.agent_active_count.get(),
            },
            "cache": {
                "hits": self.cache_hits.value,
                "misses": self.cache_misses.value,
                "evictions": self.cache_evictions.value,
                "sets": self.cache_sets.value,
                "size": self.cache_size.get(),
                "cleanup_cycles": self.cache_cleanup_cycles.value,
                "hit_rate": self._calculate_cache_hit_ratio(),
            },
            "audit": {
                "records_created": self.audit_records_created.value,
                "records_approved": self.audit_records_approved.value,
                "records_rejected": self.audit_records_rejected.value,
                "batch_operations": self.audit_batch_operations.value,
                "pending_timeout": self.audit_pending_timeout.value,
                "rollback_operations": self.audit_rollback_operations.value,
            },
            "system": {
                "correlation_ids_active": self.correlation_ids_active.get(),
                "async_operations_active": self.async_operations_active.get(),
            },
            "slo": {
                "api_error_rate": self._calculate_error_rate_ratio(),
                "api_response_time_p50": self.api_response_time._quantile(0.50),
                "api_response_time_p95": self.api_response_time._quantile(0.95),
                "availability": self.system_availability.get(),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
                "tws_connection_success_rate": self.tws_connection_success_rate.get(),
            },
            "errors": error_metrics,
            "health": self.get_health_status(),
        }

    # -------------------------
    # SLO (atualização externa)
    # -------------------------
    def update_slo_metrics(
        self,
        availability: float | None = None,
        tws_connection_success_rate: float | None = None,
    ) -> None:
        if availability is not None:
            self.system_availability.set(float(availability))
        if tws_connection_success_rate is not None:
            self.tws_connection_success_rate.set(float(tws_connection_success_rate))

    # -------------------------
    # Export Prometheus (formato correto)
    # -------------------------
    def _iter_metrics(self) -> Iterable[tuple[str, Any]]:
        """Itera sobre atributos que são métricas."""
        for name, obj in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(obj, (MetricCounter, MetricGauge, MetricHistogram)):
                yield name, obj

        # error_counts: expor como métricas rotuladas
        with self._error_lock:
            for etype, ctr in self.error_counts.items():
                yield f"error_counts__{etype}", ctr  # nome composto → vira label no render

    def generate_prometheus_metrics(self) -> str:
        """
        Formato de exposição de texto com HELP/TYPE corretos:
        - Counter → TYPE counter
        - Gauge → TYPE gauge
        - Histogram → TYPE histogram (bucket/sum/count)
        - error_counts → counter com label type
        """
        lines: list[str] = []

        # principais métricas (counters/gauges/histograms)
        for raw_name, metric in sorted(self._iter_metrics(), key=lambda x: x[0]):
            if raw_name.startswith("error_counts__"):
                # render como uma série rotulada
                _, etype = raw_name.split("__", 1)
                mname = _sanitize_metric_name("resync_errors_total")
                if f"# TYPE {mname} counter" not in lines:
                    lines.append(f"# HELP {mname} Total errors by type")
                    lines.append(f"# TYPE {mname} counter")
                # valor
                val = metric.value if isinstance(metric, MetricCounter) else 0
                lines.append(f'{mname}{{type="{etype}"}} {val}')
                continue

            mname = _sanitize_metric_name(f"resync_{raw_name}")
            if isinstance(metric, MetricCounter):
                lines.append(f"# HELP {mname} Counter {raw_name}")
                lines.append(f"# TYPE {mname} counter")
                lines.append(f"{mname} {metric.value}")
            elif isinstance(metric, MetricGauge):
                lines.append(f"# HELP {mname} Gauge {raw_name}")
                lines.append(f"# TYPE {mname} gauge")
                lines.append(f"{mname} {metric.get()}")
            elif isinstance(metric, MetricHistogram):
                # delega render p/ histogram
                lines.extend(metric.render_prometheus(mname))

        return "\n".join(lines)


# -----------------------------
# Singleton global (unificado)
# -----------------------------
_runtime_metrics: RuntimeMetrics | None = None
_runtime_lock = threading.Lock()


def _get_runtime_metrics() -> RuntimeMetrics:
    global _runtime_metrics
    if _runtime_metrics is None:
        with _runtime_lock:
            if _runtime_metrics is None:
                _runtime_metrics = RuntimeMetrics()
    return _runtime_metrics


class _RuntimeMetricsProxy:
    """Proxy que delega dinamicamente ao singleton verdadeiro (sem instância própria)."""

    def __getattr__(self, name):
        return getattr(_get_runtime_metrics(), name)


runtime_metrics = _RuntimeMetricsProxy()


# -----------------------------
# Funções utilitárias públicas
# -----------------------------
def get_correlation_context() -> str:
    """
    Retorna o correlation_id atual do contexto, ou cria um novo (e define como atual).
    Usa a MESMA instância do singleton utilizado por `runtime_metrics`.
    """
    rm = _get_runtime_metrics()
    cid = rm.get_current_correlation_id()
    if cid:
        return cid
    cid = rm.create_correlation_id()
    rm.set_current_correlation_id(cid)
    return cid


def track_llm_metrics(func):
    """
    Decorador para instrumentar chamadas LLM (sync/async):
      - conta requisições/erros
      - mede duração (perf_counter)
      - coleta tokens quando disponível (result.usage)
    """
    import functools
    import inspect

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            t0 = _now_perf()
            runtime_metrics.llm_requests.increment()
            try:
                result = await func(*args, **kwargs)
                dt = _now_perf() - t0
                runtime_metrics.llm_duration.observe(dt)
                # tokens (padrões comuns: dict ou objeto com attrs)
                usage = getattr(result, "usage", None)
                if usage:
                    if isinstance(usage, dict):
                        itoks = int(usage.get("prompt_tokens", 0) or 0)
                        otoks = int(usage.get("completion_tokens", 0) or 0)
                    else:
                        itoks = int(getattr(usage, "prompt_tokens", 0) or 0)
                        otoks = int(getattr(usage, "completion_tokens", 0) or 0)
                    runtime_metrics.llm_tokens.increment(itoks + otoks)
                return result
            except Exception as _e:
                runtime_metrics.llm_errors.increment()
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        t0 = _now_perf()
        runtime_metrics.llm_requests.increment()
        try:
            result = func(*args, **kwargs)
            dt = _now_perf() - t0
            runtime_metrics.llm_duration.observe(dt)
            usage = getattr(result, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    itoks = int(usage.get("prompt_tokens", 0) or 0)
                    otoks = int(usage.get("completion_tokens", 0) or 0)
                else:
                    itoks = int(getattr(usage, "prompt_tokens", 0) or 0)
                    otoks = int(getattr(usage, "completion_tokens", 0) or 0)
                runtime_metrics.llm_tokens.increment(itoks + otoks)
            return result
        except Exception as _e:
            runtime_metrics.llm_errors.increment()
            raise

    return sync_wrapper


def log_with_correlation(
    level: int, message: str, correlation_id: str | None = None, **kwargs: Any
) -> None:
    """
    Loga com contexto de correlação.
    - Se correlation_id não for fornecido: tenta usar o do contexto atual (contextvars).
    - Sempre registra o evento no contexto (se existir) e emite log padrão do Python.
    """
    rm = _get_runtime_metrics()
    cid = correlation_id or rm.get_current_correlation_id()
    if cid:
        rm.add_correlation_event(cid, f"log_{level}", {"message": message, **kwargs})
    logger.log(level, f"[{cid}] {message}", **kwargs)
