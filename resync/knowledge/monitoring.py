"""
Métricas Prometheus para o sistema RAG.

Usa prometheus_client diretamente conforme stack especificada.
Criação lazy com try/except para evitar quebra de import em testes.
"""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Gauge, Histogram  # ← prometheus_client direto

logger = structlog.get_logger(__name__)

# Buckets otimizados para operações RAG (ms → s)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

try:
    embed_seconds = Histogram(
        "rag_embed_seconds",
        "Latency for embedding batches",
        buckets=_LATENCY_BUCKETS,
    )
    upsert_seconds = Histogram(
        "rag_upsert_seconds",
        "Latency for vector upserts",
        buckets=_LATENCY_BUCKETS,
    )
    query_seconds = Histogram(
        "rag_query_seconds",
        "Latency for vector queries",
        buckets=_LATENCY_BUCKETS,
    )
    rerank_seconds = Histogram(  # ← novo: latência de reranking
        "rag_rerank_seconds",
        "Latency for cross-encoder reranking",
        buckets=_LATENCY_BUCKETS,
    )
    jobs_total = Counter(
        "rag_jobs_total",
        "RAG jobs processed",
        ["status"],  # valores: "success" | "error" | "skipped"
    )
    collection_vectors = Gauge(
        "rag_collection_vectors",
        "Vectors in current read collection",
    )
    cache_hits_total = Counter(  # ← novo: cache hits Redis
        "rag_cache_hits_total",
        "Cache hits for embedding lookups",
        ["layer"],  # "redis" | "local"
    )
except Exception as _exc:  # noqa: BLE001
    # Em testes sem Prometheus registry, não quebrar o import
    logger.warning("prometheus_metrics_init_failed", error=str(_exc))
