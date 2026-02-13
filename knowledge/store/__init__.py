"""
Knowledge Store Module.

v5.9.0: Persistence layer with optimized Binary+Halfvec vector search.

Components:
- pgvector_store.py: Main vector store with two-phase search
- pgvector.py: Backward-compatible service interface

Vector Search Architecture:
- Storage: halfvec (float16) for 50% reduction
- Search: Binary HNSW → Halfvec rescoring
- Auto-fill: Trigger populates embedding_half automatically

Performance:
- Storage: ~75% reduction vs float32
- Speed: ~70% faster search
- Quality: ~99% with halfvec rescoring
"""

from resync.knowledge.store.pgvector import (
    CollectionStats,
    DistanceMetric,
    PgVectorService,
    SearchResult,
    VectorDocument,
    get_vector_service,
)
from resync.knowledge.store.pgvector_store import (
    PgVectorStore,
    get_vector_store,
)

__all__ = [
    # New API (recommended)
    "PgVectorStore",
    "get_vector_store",
    # Legacy API (backward compatible)
    "PgVectorService",
    "VectorDocument",
    "SearchResult",
    "CollectionStats",
    "DistanceMetric",
    "get_vector_service",
]
