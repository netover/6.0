from __future__ import annotations

import math
from typing import Any

from resync.knowledge.config import CFG
from resync.knowledge.interfaces import Embedder, Retriever, VectorStore
from resync.knowledge.monitoring import query_seconds


class RagRetriever(Retriever):
    """Rag retriever."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        top_k = min(top_k, CFG.max_top_k)
        vec = await self.embedder.embed(query)
        ef = CFG.ef_search_base + int(math.log2(max(10, top_k)) * 8)
        ef = min(ef, CFG.ef_search_max)
        with query_seconds.time():
            hits = await self.store.query(
                vector=vec,
                top_k=top_k,
                collection=CFG.collection_read,
                filters=filters,
                ef_search=ef,
                with_vectors=bool(CFG.enable_rerank),
            )
        if not CFG.enable_rerank:
            return hits

        # Lightweight re-rank (cosine with vector from pgvector, if returned)
        if hits and "vector" in hits[0]:
            query_norm = math.sqrt(sum(x * x for x in vec))
            if query_norm == 0:
                return hits

            def cosine_similarity_score(doc_vector: list[float] | None) -> float:
                if not doc_vector:
                    return 0.0

                # Dot product
                dot = sum(q * d for q, d in zip(vec, doc_vector, strict=False))

                # Doc norm
                doc_norm = math.sqrt(sum(d * d for d in doc_vector))

                if doc_norm == 0:
                    return 0.0

                return dot / (query_norm * doc_norm)

            hits.sort(key=lambda h: cosine_similarity_score(h.get("vector")), reverse=True)
        return hits
