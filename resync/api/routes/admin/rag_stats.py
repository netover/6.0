"""
RAG Statistics Endpoint - Monitoramento e Observabilidade.

Fornece métricas para:
- Status do índice BM25 (tamanho, lock, integridade)
- Performance do cache (hit rate, miss rate)
- Memória de chat (turns armazenados, expirações)

Autor: Resync Team
Versão: 1.0.0
"""

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from resync.core.structured_logger import get_logger
from resync.knowledge.retrieval.hybrid_retriever import INDEX_STORAGE_PATH

logger = get_logger(__name__)

router = APIRouter(prefix="/admin/rag", tags=["RAG Admin"])

class IndexStatus(BaseModel):
    """Status do índice BM25."""

    path: str
    exists: bool
    size_bytes: int | None = None
    last_modified: str | None = None
    is_locked: bool
    is_loaded: bool
    num_documents: int = 0
    num_terms: int = 0

class CachePerformance(BaseModel):
    """Performance do cache de queries."""

    hit_rate: float
    miss_rate: float
    total_queries: int
    cache_size: int
    cache_max_size: int

class ChatMemoryStatus(BaseModel):
    """Status da memória de chat."""

    enabled: bool
    ttl_days: int
    total_turns: int = 0
    active_sessions: int = 0
    expired_turns_cleaned: int = 0

class RAGStatsResponse(BaseModel):
    """Resposta completa de estatísticas RAG."""

    timestamp: str
    index_status: IndexStatus
    cache_performance: CachePerformance
    chat_memory: ChatMemoryStatus

@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats() -> RAGStatsResponse:
    """
    Retorna estatísticas completas do sistema RAG.
    """
    from resync.core.metrics import get_runtime_metrics

    index_path = INDEX_STORAGE_PATH
    lock_path = f"{index_path}.lock"

    index_exists = os.path.exists(index_path)
    index_size_bytes = os.path.getsize(index_path) if index_exists else None

    index_modified = None
    if index_exists:
        mod_time = os.path.getmtime(index_path)
        index_modified = datetime.fromtimestamp(mod_time).isoformat()

    metrics = get_runtime_metrics()
    rag_stats = metrics.get_stats()

    cache_hit_rate = 0.0
    total_queries = 0
    cache_size = 0

    if "rag" in rag_stats:
        rag = rag_stats["rag"]
        bm25_stats = rag.get("bm25", {})
        cache_rag = rag.get("cache", {})

        cache_hits = cache_rag.get("hits", 0)
        cache_misses = cache_rag.get("misses", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0.0
        total_queries = bm25_stats.get("queries_total", 0)
        cache_size = int(cache_rag.get("size", 0))

    chat_memory_enabled = (
        os.environ.get("CHAT_MEMORY_ENABLED", "true").lower() == "true"
    )
    chat_ttl = int(os.environ.get("CHAT_MEMORY_TTL_DAYS", "30"))

    return RAGStatsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        index_status=IndexStatus(
            path=index_path,
            exists=index_exists,
            size_bytes=index_size_bytes,
            last_modified=index_modified,
            is_locked=os.path.exists(lock_path),
            is_loaded=index_exists,
            num_documents=int(
                rag_stats.get("rag", {}).get("bm25", {}).get("index_documents", 0)
            ),
            num_terms=int(
                rag_stats.get("rag", {}).get("bm25", {}).get("index_terms", 0)
            ),
        ),
        cache_performance=CachePerformance(
            hit_rate=cache_hit_rate,
            miss_rate=100.0 - cache_hit_rate,
            total_queries=total_queries,
            cache_size=cache_size,
            cache_max_size=1000,
        ),
        chat_memory=ChatMemoryStatus(
            enabled=chat_memory_enabled,
            ttl_days=chat_ttl,
            total_turns=int(
                rag_stats.get("rag", {}).get("chat", {}).get("turns_stored", 0)
            ),
            active_sessions=0,
            expired_turns_cleaned=0,
        ),
    )

@router.get("/health")
async def rag_health_check() -> dict[str, Any]:
    """
    Health check simplificado para RAG.
    """
    index_path = INDEX_STORAGE_PATH

    index_ok = os.path.exists(index_path)

    return {
        "status": "healthy" if index_ok else "degraded",
        "index_ready": index_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

__all__ = ["router", "get_rag_stats", "rag_health_check"]
