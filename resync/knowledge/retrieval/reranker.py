"""
Cross-Encoder Reranker for RAG Document Retrieval.

v5.3.17 - Uses BAAI/bge-reranker-small for document reranking.

The cross-encoder evaluates query-document pairs together,
providing more accurate relevance scores than cosine similarity alone.

Architecture:
    1. Vector search returns top-N candidates (fast, recall-oriented)
    2. Cross-encoder reranks candidates (slower, precision-oriented)
    3. Return top-K most relevant documents

This two-stage approach balances speed and accuracy:
- Stage 1: pgvector HNSW search (~10ms for 100K docs)
- Stage 2: Cross-encoder rerank (~50ms for 20 docs)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from resync.knowledge.config import CFG

logger = logging.getLogger(__name__)

# Global model instance (singleton, lazy loaded)
_cross_encoder_model: Any = None
_cross_encoder_available: bool | None = None


@dataclass
class RerankResult:
    """Result of cross-encoder reranking."""

    documents: list[dict[str, Any]]
    rerank_time_ms: float
    model_used: str
    original_count: int
    filtered_count: int


def is_cross_encoder_available() -> bool:
    """Check if cross-encoder model is available."""
    global _cross_encoder_available

    if _cross_encoder_available is not None:
        return _cross_encoder_available

    try:
        from sentence_transformers import CrossEncoder as _  # noqa: F401

        _cross_encoder_available = True
        logger.info("Cross-encoder is available for RAG reranking")
    except ImportError:
        _cross_encoder_available = False
        logger.warning(
            "Cross-encoder not available for RAG. Install with: pip install sentence-transformers"
        )

    return _cross_encoder_available


def get_cross_encoder() -> Any:
    """
    Get singleton cross-encoder model instance.

    Lazy loads the model on first use to avoid startup overhead.
    """
    global _cross_encoder_model, _cross_encoder_available

    if _cross_encoder_model is not None:
        return _cross_encoder_model

    if not is_cross_encoder_available():
        return None

    try:
        from sentence_transformers import CrossEncoder

        model_name = CFG.cross_encoder_model
        logger.info("Loading cross-encoder model for RAG: %s", model_name)
        start = time.perf_counter()

        _cross_encoder_model = CrossEncoder(
            model_name,
            max_length=512,  # Limit input length for speed
        )

        load_time = (time.perf_counter() - start) * 1000
        logger.info("RAG cross-encoder loaded in %.0fms", load_time)

        return _cross_encoder_model

    except Exception as e:
        logger.error("Failed to load RAG cross-encoder: %s", e)
        _cross_encoder_available = False
        return None


def preload_cross_encoder() -> bool:
    """
    Preload cross-encoder model into memory.

    Call this at startup to avoid cold-start latency on first query.

    Returns:
        True if model loaded successfully
    """
    model = get_cross_encoder()
    if model is None:
        return False

    # Warm up with a dummy prediction
    try:
        _ = model.predict([("test query", "test document")])
        logger.info("RAG cross-encoder model warmed up")
        return True
    except Exception as e:
        logger.warning("Failed to warm up RAG cross-encoder: %s", e)
        return False


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int | None = None,
    threshold: float | None = None,
) -> RerankResult:
    """
    Rerank documents using cross-encoder.

    Args:
        query: User's search query
        documents: List of documents from vector search
            Each document should have 'text' or 'content' field
        top_k: Number of top documents to return (default from config)
        threshold: Minimum score to keep document (default from config)

    Returns:
        RerankResult with reranked documents and metadata
    """
    import math

    start = time.perf_counter()

    top_k = top_k or CFG.cross_encoder_top_k
    threshold = threshold or CFG.cross_encoder_threshold
    original_count = len(documents)

    model = get_cross_encoder()

    if model is None or not documents:
        # Fallback: return documents as-is
        return RerankResult(
            documents=documents[:top_k],
            rerank_time_ms=0.0,
            model_used="fallback",
            original_count=original_count,
            filtered_count=min(len(documents), top_k),
        )

    try:
        # Extract text from documents
        pairs = []
        for doc in documents:
            text = doc.get("text") or doc.get("content") or doc.get("chunk_text", "")
            if text:
                pairs.append((query, text[:512]))  # Limit text length
            else:
                pairs.append((query, ""))

        # Get cross-encoder scores
        scores = model.predict(pairs)

        # Normalize scores to 0-1 using sigmoid
        normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]

        # Attach scores to documents
        scored_docs = list(zip(documents, normalized_scores, strict=False))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and take top_k
        filtered_docs = []
        for doc, score in scored_docs:
            if score >= threshold:
                # Add rerank score to document
                doc_with_score = dict(doc)
                doc_with_score["rerank_score"] = round(score, 4)
                doc_with_score["original_rank"] = documents.index(doc) + 1
                filtered_docs.append(doc_with_score)

                if len(filtered_docs) >= top_k:
                    break

        rerank_time = (time.perf_counter() - start) * 1000

        logger.debug(
            "Reranked %s docs → %s docs in %.1fms",
            original_count,
            len(filtered_docs),
            rerank_time,
        )

        return RerankResult(
            documents=filtered_docs,
            rerank_time_ms=round(rerank_time, 2),
            model_used=CFG.cross_encoder_model,
            original_count=original_count,
            filtered_count=len(filtered_docs),
        )

    except Exception as e:
        logger.error("Cross-encoder reranking failed: %s", e)
        # Fallback: return original documents
        return RerankResult(
            documents=documents[:top_k],
            rerank_time_ms=(time.perf_counter() - start) * 1000,
            model_used="error_fallback",
            original_count=original_count,
            filtered_count=min(len(documents), top_k),
        )


def get_reranker_info() -> dict[str, Any]:
    """Get information about the RAG reranker."""
    return {
        "available": is_cross_encoder_available(),
        "enabled": CFG.enable_cross_encoder,
        "model": CFG.cross_encoder_model if _cross_encoder_model else None,
        "loaded": _cross_encoder_model is not None,
        "top_k": CFG.cross_encoder_top_k,
        "threshold": CFG.cross_encoder_threshold,
    }


__all__ = [
    "RerankResult",
    "is_cross_encoder_available",
    "get_cross_encoder",
    "preload_cross_encoder",
    "rerank_documents",
    "get_reranker_info",
]
