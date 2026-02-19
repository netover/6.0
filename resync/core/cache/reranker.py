"""
Cross-Encoder Reranker for Semantic Cache.

v5.3.17 - Conditional reranking to reduce false positives.

Uses BAAI/bge-reranker-small:
- ~33M parameters
- ~150MB RAM
- Good quality with small footprint
- ~20-50ms per query-document pair

The reranker is only applied in the "gray zone" (uncertain matches)
to avoid adding latency to clear hits/misses.

Architecture decision:
- Distance < 0.20 → Clear HIT, skip reranking
- Distance > 0.35 → Clear MISS, skip reranking
- Distance 0.20-0.35 → Gray zone, apply reranker for confirmation
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Reranker configuration
RERANKER_MODEL = "BAAI/bge-reranker-small"
RERANKER_THRESHOLD = 0.5  # Cross-encoder score threshold (0-1, higher = more similar)
GRAY_ZONE_MIN = 0.20  # Below this distance = clear hit
GRAY_ZONE_MAX = 0.35  # Above this distance = clear miss

# Global model instance (singleton, lazy loaded)
_reranker_model: Any = None
_reranker_available: bool | None = None


@dataclass
class RerankerResult:
    """Result of cross-encoder reranking."""

    score: float  # Cross-encoder similarity score (0-1)
    is_similar: bool  # Whether queries are semantically similar
    latency_ms: float  # Time taken for reranking
    model_used: str  # Model name used


def is_reranker_available() -> bool:
    """Check if cross-encoder model is available."""
    global _reranker_available

    if _reranker_available is not None:
        return _reranker_available

    try:
        from sentence_transformers import CrossEncoder  # noqa: F401

        _reranker_available = True
        logger.info("Cross-encoder reranker is available")
    except ImportError:
        _reranker_available = False
        logger.warning(
            "Cross-encoder not available. Install with: pip install sentence-transformers"
        )

    return _reranker_available


def get_reranker_model() -> Any:
    """
    Get singleton cross-encoder model instance.

    Lazy loads the model on first use to avoid startup overhead.
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    if not is_reranker_available():
        return None

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", RERANKER_MODEL)
        start = time.perf_counter()

        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,  # Limit input length for speed
        )

        (time.perf_counter() - start) * 1000
        logger.info("Cross-encoder loaded in {load_time:.0f}ms")

        return _reranker_model

    except Exception as e:
        logger.error("Failed to load cross-encoder model: %s", e)
        _reranker_available = False
        return None


def preload_reranker() -> bool:
    """
    Preload reranker model into memory.

    Call this at startup to avoid cold-start latency on first query.

    Returns:
        True if model loaded successfully
    """
    model = get_reranker_model()
    if model is None:
        return False

    # Warm up with a dummy prediction
    try:
        _ = model.predict([("test query", "test document")])
        logger.info("Cross-encoder model warmed up")
        return True
    except Exception as e:
        logger.warning("Failed to warm up cross-encoder: %s", e)
        return False


def rerank_pair(query: str, cached_query: str) -> RerankerResult:
    """
    Compute cross-encoder similarity between two queries.

    Args:
        query: New user query
        cached_query: Query stored in cache

    Returns:
        RerankerResult with similarity score and decision
    """
    start = time.perf_counter()

    model = get_reranker_model()

    if model is None:
        # Fallback: assume similar if we can't rerank
        return RerankerResult(
            score=1.0,
            is_similar=True,
            latency_ms=0.0,
            model_used="fallback",
        )

    try:
        # Cross-encoder takes (query, document) pairs
        # For cache, both are queries, so order doesn't matter much
        score = model.predict([(query, cached_query)])[0]

        # Normalize score to 0-1 range (BGE outputs raw scores)
        # BGE reranker outputs logits, apply sigmoid for probability
        import math

        normalized_score = 1 / (1 + math.exp(-score))

        latency_ms = (time.perf_counter() - start) * 1000

        is_similar = normalized_score >= RERANKER_THRESHOLD

        logger.debug(
            f"Rerank: '{query[:30]}...' vs '{cached_query[:30]}...' "
            "→ score={normalized_score:.3f}, similar={is_similar}, "
            "latency={latency_ms:.1f}ms"
        )

        return RerankerResult(
            score=normalized_score,
            is_similar=is_similar,
            latency_ms=latency_ms,
            model_used=RERANKER_MODEL,
        )

    except Exception as e:
        logger.error("Reranking failed: %s", e)
        # On error, assume similar (don't block cache hits)
        return RerankerResult(
            score=1.0,
            is_similar=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_used="error_fallback",
        )


def should_rerank(distance: float) -> bool:
    """
    Determine if reranking should be applied based on embedding distance.

    The "gray zone" strategy:
    - Clear hits (distance < 0.20): Skip reranking, fast path
    - Clear misses (distance > 0.35): Skip reranking, fast path
    - Uncertain (0.20-0.35): Apply reranking for confirmation

    Args:
        distance: Cosine distance from embedding search

    Returns:
        True if reranking should be applied
    """
    return GRAY_ZONE_MIN <= distance <= GRAY_ZONE_MAX


def get_reranker_info() -> dict[str, Any]:
    """Get information about the reranker model."""
    return {
        "available": is_reranker_available(),
        "model": RERANKER_MODEL if _reranker_model else None,
        "loaded": _reranker_model is not None,
        "threshold": RERANKER_THRESHOLD,
        "gray_zone_min": GRAY_ZONE_MIN,
        "gray_zone_max": GRAY_ZONE_MAX,
    }


def update_reranker_config(
    threshold: float | None = None,
    gray_zone_min: float | None = None,
    gray_zone_max: float | None = None,
) -> dict[str, float]:
    """
    Update reranker configuration at runtime.

    Args:
        threshold: New cross-encoder threshold (0-1)
        gray_zone_min: New minimum distance for gray zone
        gray_zone_max: New maximum distance for gray zone

    Returns:
        Updated configuration values
    """
    global RERANKER_THRESHOLD, GRAY_ZONE_MIN, GRAY_ZONE_MAX

    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        RERANKER_THRESHOLD = threshold

    if gray_zone_min is not None:
        if not 0 <= gray_zone_min <= 1:
            raise ValueError("gray_zone_min must be between 0 and 1")
        GRAY_ZONE_MIN = gray_zone_min

    if gray_zone_max is not None:
        if not 0 <= gray_zone_max <= 1:
            raise ValueError("gray_zone_max must be between 0 and 1")
        GRAY_ZONE_MAX = gray_zone_max

    logger.info(
        f"Reranker config updated: threshold={RERANKER_THRESHOLD}, "
        f"gray_zone=[{GRAY_ZONE_MIN}, {GRAY_ZONE_MAX}]"
    )

    return {
        "threshold": RERANKER_THRESHOLD,
        "gray_zone_min": GRAY_ZONE_MIN,
        "gray_zone_max": GRAY_ZONE_MAX,
    }


__all__ = [
    "RerankerResult",
    "is_reranker_available",
    "get_reranker_model",
    "preload_reranker",
    "rerank_pair",
    "should_rerank",
    "get_reranker_info",
    "update_reranker_config",
    "RERANKER_MODEL",
    "RERANKER_THRESHOLD",
    "GRAY_ZONE_MIN",
    "GRAY_ZONE_MAX",
]
