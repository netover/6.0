"""
Métricas Prometheus para o sistema RAG.

Usa prometheus_client diretamente conforme stack especificada.
Criação lazy com try/except para evitar quebra de import em testes.

CORRIGIDO (P2): Define NoOp stubs no except para evitar NameError
em código que importa as métricas quando Prometheus não está disponível.
"""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# Buckets otimizados para operações RAG (ms → s)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

class _NoOpMetric:
    """
    NoOp stub para métricas Prometheus quando registry não está disponível.
    Evita NameError e AttributeError em testes e ambientes sem Prometheus.
    """

    def labels(self, **_kwargs: object) -> "_NoOpMetric":
        return self

    def observe(self, *_args: object, **_kwargs: object) -> None:
        pass

    def inc(self, *_args: object, **_kwargs: object) -> None:
        pass

    def set(self, *_args: object, **_kwargs: object) -> None:
        pass

try:
    embed_seconds: Histogram | _NoOpMetric = Histogram(
        "rag_embed_seconds",
        "Latency for embedding batches",
        buckets=_LATENCY_BUCKETS,
    )
    upsert_seconds: Histogram | _NoOpMetric = Histogram(
        "rag_upsert_seconds",
        "Latency for vector upserts",
        buckets=_LATENCY_BUCKETS,
    )
    query_seconds: Histogram | _NoOpMetric = Histogram(
        "rag_query_seconds",
        "Latency for vector queries",
        buckets=_LATENCY_BUCKETS,
    )
    rerank_seconds: Histogram | _NoOpMetric = Histogram(
        "rag_rerank_seconds",
        "Latency for cross-encoder reranking",
        buckets=_LATENCY_BUCKETS,
    )
    jobs_total: Counter | _NoOpMetric = Counter(
        "rag_jobs_total",
        "RAG jobs processed",
        ["status"],  # "success" | "error" | "skipped"
    )
    collection_vectors: Gauge | _NoOpMetric = Gauge(
        "rag_collection_vectors",
        "Vectors in current read collection",
    )
    cache_hits_total: Counter | _NoOpMetric = Counter(
        "rag_cache_hits_total",
        "Cache hits for embedding lookups",
        ["layer"],  # "redis" | "local"
    )
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as _exc:  # noqa: BLE001
    # Em testes sem Prometheus registry, não quebrar o import.
    # CORRIGIDO: define stubs para evitar NameError em importadores.
    logger.warning("prometheus_metrics_init_failed", error=str(_exc))
    embed_seconds = _NoOpMetric()
    upsert_seconds = _NoOpMetric()
    query_seconds = _NoOpMetric()
    rerank_seconds = _NoOpMetric()
    jobs_total = _NoOpMetric()
    collection_vectors = _NoOpMetric()
    cache_hits_total = _NoOpMetric()
