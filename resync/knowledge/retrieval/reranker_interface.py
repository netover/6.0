"""
Reranker Interface Module - v5.9.9

Implements the Interface + NoOp + Gating pattern for CPU-optimized reranking.

Architecture:
    1. IReranker Protocol - stable contract for all rerankers
    2. NoOpReranker - Null Object pattern for disabled rerank
    3. CrossEncoderReranker - wraps the existing cross-encoder
    4. RerankGatingPolicy - decides when to activate rerank based on scores

Benefits:
    - No if/else scattered across codebase
    - Feature flag controls which reranker is registered
    - Gating reduces CPU cost by only reranking when confidence is low
    - Easy to add new rerankers (LLM-based, etc.)

References:
    - Null Object Pattern: https://en.wikipedia.org/wiki/Null_object_pattern
    - Two-stage retrieval: retrieve fast (vector/BM25) → rerank precise (cross-encoder)
"""

from __future__ import annotations

import logging
import math
import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

def safe_sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)



# =============================================================================
# CONFIGURATION
# =============================================================================


def _bool(env: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    v = os.getenv(env)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


@dataclass
class RerankGatingConfig:
    """
    Configuration for rerank gating policy.

    Controls when reranking is activated based on retrieval scores.

    Environment Variables:
        RERANK_GATING_ENABLED: Enable/disable gating (default: True)
        RERANK_SCORE_LOW_THRESHOLD: Activate if top1 score < threshold (default: 0.35)
        RERANK_MARGIN_THRESHOLD: Activate if top1-top2 < margin (default: 0.05)
        RERANK_MAX_CANDIDATES: Max docs to rerank (default: 10)
        RERANK_ENTROPY_THRESHOLD: Activate if many scores are similar (default: 0.8)
    """

    # Master switch for gating (if False, always rerank when enabled)
    enabled: bool = _bool("RERANK_GATING_ENABLED", True)

    # Rule A: Top1 score below threshold → retrieval uncertain
    score_low_threshold: float = float(os.getenv("RERANK_SCORE_LOW_THRESHOLD", "0.35"))

    # Rule B: Small margin between top1 and top2 → ambiguity
    margin_threshold: float = float(os.getenv("RERANK_MARGIN_THRESHOLD", "0.05"))

    # Rule C: High entropy (many similar scores) → retrieval "lost"
    entropy_threshold: float = float(os.getenv("RERANK_ENTROPY_THRESHOLD", "0.8"))
    enable_entropy_check: bool = _bool("RERANK_ENTROPY_CHECK_ENABLED", False)

    # Max candidates to pass to reranker (limits CPU cost)
    max_candidates: int = int(os.getenv("RERANK_MAX_CANDIDATES", "10"))

    # Target: activate rerank in ~10-30% of queries for CPU-only
    # Adjust thresholds based on your score distribution

    @classmethod
    def from_env(cls) -> RerankGatingConfig:
        """Create config from environment variables."""
        return cls()


# =============================================================================
# GATING POLICY
# =============================================================================


@dataclass
class RerankGatingPolicy:
    """
    Policy that decides whether to activate reranking.

    Uses score-based heuristics to avoid expensive reranking when
    retrieval confidence is already high.

    Typical usage:
        policy = RerankGatingPolicy()
        if policy.should_rerank(scores):
            candidates = candidates[:policy.config.max_candidates]
            reranked = await reranker.rerank(query, candidates)
    """

    config: RerankGatingConfig = field(default_factory=RerankGatingConfig)

    # Statistics for monitoring
    _total_decisions: int = field(default=0, repr=False)
    _rerank_activated: int = field(default=0, repr=False)
    _reasons: dict[str, int] = field(
        default_factory=lambda: {
            "low_score": 0,
            "small_margin": 0,
            "high_entropy": 0,
            "skipped": 0,
        },
        repr=False,
    )

    def should_rerank(
        self, scores: list[float], normalize: bool = True
    ) -> tuple[bool, str]:
        """
        Determine if reranking should be activated.

        Args:
            scores: List of retrieval scores (descending order expected)
            normalize: If True, normalize scores to 0-1 range

        Returns:
            Tuple of (should_rerank, reason)
        """
        self._total_decisions += 1

        # Gating disabled → always rerank
        if not self.config.enabled:
            self._rerank_activated += 1
            return True, "gating_disabled"

        # No scores → nothing to evaluate
        if not scores:
            self._reasons["skipped"] += 1
            return False, "no_scores"

        # Ensure descending order
        scores = sorted(scores, reverse=True)

        # Normalize scores if needed (pgvector similarity can be 0-1 or distance-based)
        if normalize and scores:
            max_score = max(scores)
            min_score = min(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                self._rerank_activated += 1
                self._reasons["high_entropy"] += 1
                return True, "uniform_scores"

        s1 = scores[0]
        s2 = scores[1] if len(scores) > 1 else 0.0

        # Rule A: Top1 score is low → retrieval uncertain
        if s1 < self.config.score_low_threshold:
            self._rerank_activated += 1
            self._reasons["low_score"] += 1
            logger.debug(
                "rerank_activated",
                reason="low_score",
                top1_score=round(s1, 3),
                threshold=self.config.score_low_threshold,
            )
            return True, "low_score"

        # Rule B: Small margin between top1 and top2 → ambiguity
        if len(scores) > 1:
            margin = s1 - s2
            if margin < self.config.margin_threshold:
                self._rerank_activated += 1
                self._reasons["small_margin"] += 1
                logger.debug(
                    "rerank_activated",
                    reason="small_margin",
                    margin=round(margin, 3),
                    threshold=self.config.margin_threshold,
                )
                return True, "small_margin"

        # Rule C: High entropy (optional, more expensive to compute)
        if self.config.enable_entropy_check and len(scores) >= 3:
            entropy = self._compute_normalized_entropy(scores)
            if entropy > self.config.entropy_threshold:
                self._rerank_activated += 1
                self._reasons["high_entropy"] += 1
                logger.debug(
                    "rerank_activated",
                    reason="high_entropy",
                    entropy=round(entropy, 3),
                    threshold=self.config.entropy_threshold,
                )
                return True, "high_entropy"

        # All checks passed → retrieval confident, skip rerank
        self._reasons["skipped"] += 1
        logger.debug(
            "rerank_skipped",
            reason="confident_retrieval",
            top1_score=round(s1, 3),
            margin=round(s1 - s2, 3) if len(scores) > 1 else 0.0,
        )
        return False, "confident"

    def _compute_normalized_entropy(self, scores: list[float]) -> float:
        """
        Compute normalized entropy of score distribution.

        Returns value in [0, 1] where:
            - 0 = one dominant score (low entropy)
            - 1 = uniform distribution (high entropy)
        """
        if not scores or len(scores) < 2:
            return 0.0

        # Normalize to probabilities
        total = sum(scores)
        if total == 0:
            return 1.0

        probs = [s / total for s in scores]

        # Compute entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)

        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log(len(scores))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get gating statistics for monitoring."""
        total = max(1, self._total_decisions)
        return {
            "total_decisions": self._total_decisions,
            "rerank_activated": self._rerank_activated,
            "rerank_rate": self._rerank_activated / total,
            "reasons": dict(self._reasons),
            "config": {
                "enabled": self.config.enabled,
                "score_low_threshold": self.config.score_low_threshold,
                "margin_threshold": self.config.margin_threshold,
                "max_candidates": self.config.max_candidates,
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_decisions = 0
        self._rerank_activated = 0
        self._reasons = {
            "low_score": 0,
            "small_margin": 0,
            "high_entropy": 0,
            "skipped": 0,
        }


# =============================================================================
# RERANKER INTERFACE (Protocol)
# =============================================================================


@runtime_checkable
class IReranker(Protocol):
    """
    Protocol defining the reranker interface.

    All rerankers (NoOp, CrossEncoder, LLM-based) implement this contract.
    The rest of the system only knows about IReranker.rerank().

    This enables:
        - Clean dependency injection
        - Easy testing with mocks
        - Runtime switching via feature flag
    """

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate documents for the given query.

        Args:
            query: Search query
            candidates: List of documents with 'content'/'text' and 'score' fields
            top_k: Max results to return (None = return all)

        Returns:
            Reranked list of documents with 'rerank_score' added
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """Get reranker information and status."""
        ...


# =============================================================================
# NOOP RERANKER (Null Object Pattern)
# =============================================================================


class NoOpReranker:
    """
    Null Object pattern implementation of IReranker.

    Returns candidates unchanged. Use when:
        - Reranking is disabled
        - Cross-encoder not available
        - Development/testing environments

    Benefits:
        - No None checks needed in calling code
        - Clean fallback behavior
        - Zero latency overhead
    """

    def __init__(self) -> None:
        self._call_count = 0

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates unchanged (pass-through)."""
        self._call_count += 1

        if top_k is not None:
            return candidates[:top_k]
        return candidates

    def get_info(self) -> dict[str, Any]:
        """Get NoOp reranker info."""
        return {
            "type": "noop",
            "enabled": False,
            "available": True,
            "model": None,
            "call_count": self._call_count,
            "description": "Pass-through reranker (disabled)",
        }


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================


class CrossEncoderReranker:
    """
    Cross-encoder based reranker using sentence-transformers.

    Wraps the existing reranker.py functionality with the IReranker interface.

    The cross-encoder evaluates (query, document) pairs together,
    providing more accurate relevance scores than cosine similarity.

    Default model: BAAI/bge-reranker-small
        - Good accuracy/speed balance
        - ~50ms for 10 documents on CPU
        - ~33M parameters
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.3,
        max_length: int = 512,
    ) -> None:
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name (default from config)
            threshold: Minimum score to keep document
            max_length: Max input length for model
        """
        from resync.knowledge.config import CFG

        self.model_name = model_name or CFG.cross_encoder_model
        self.threshold = threshold
        self.max_length = max_length

        self._model = None
        self._available: bool | None = None
        self._call_count = 0
        self._total_latency_ms = 0.0

    def _ensure_model(self) -> Any:
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return self._model

        if self._available is False:
            return None

        try:
            from sentence_transformers import CrossEncoder

            logger.info("loading_cross_encoder", model=self.model_name)
            start = time.perf_counter()

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )

            load_time = (time.perf_counter() - start) * 1000
            logger.info("cross_encoder_loaded", load_time_ms=round(load_time, 1))
            self._available = True

            return self._model

        except ImportError:
            logger.warning("sentence_transformers_not_installed")
            self._available = False
            return None

        except Exception as e:
            logger.error("failed_to_load_cross_encoder", error=str(e))
            self._available = False
            return None

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.

        Returns documents sorted by cross-encoder score with
        'rerank_score' and 'original_rank' fields added.
        """
        start = time.perf_counter()
        self._call_count += 1

        model = self._ensure_model()

        if model is None or not candidates:
            # Fallback: return as-is
            if top_k is not None:
                return candidates[:top_k]
            return candidates

        try:
            # Prepare (query, document) pairs
            pairs = []
            for doc in candidates:
                text = (
                    doc.get("text") or doc.get("content") or doc.get("chunk_text", "")
                )
                if isinstance(text, dict):
                    text = text.get("text", str(text))
                pairs.append((query, str(text)[: self.max_length]))

            # Get scores from cross-encoder
            scores = await asyncio.wait_for(
                asyncio.to_thread(model.predict, pairs), timeout=10.0
            )

            # Normalize scores to [0, 1] using safe_sigmoid
            normalized = [safe_sigmoid(float(s)) for s in scores]

            # Attach scores and original rank
            scored_docs = []
            for idx, (doc, score) in enumerate(
                zip(candidates, normalized, strict=False)
            ):
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = round(score, 4)
                doc_copy["original_rank"] = idx + 1
                scored_docs.append((doc_copy, score))

            # Sort by rerank score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold
            result = [doc for doc, score in scored_docs if score >= self.threshold]

            # If all filtered out, return top results anyway
            if not result:
                result = [doc for doc, _ in scored_docs]

            # Apply top_k limit
            if top_k is not None:
                result = result[:top_k]

            latency = (time.perf_counter() - start) * 1000
            self._total_latency_ms += latency

            logger.debug(
                "cross_encoder_reranked",
                candidates=len(candidates),
                results=len(result),
                latency_ms=round(latency, 1),
            )

            return result

        except Exception as e:
            logger.error("cross_encoder_reranking_failed", error=str(e))
            if top_k is not None:
                return candidates[:top_k]
            return candidates

    def preload(self) -> bool:
        """
        Preload model to avoid cold-start latency.

        Call at application startup.

        Returns:
            True if model loaded successfully
        """
        model = self._ensure_model()
        if model is None:
            return False

        try:
            # Warm up with dummy prediction
            _ = model.predict([("test", "test")])
            logger.info("cross_encoder_warmed_up")
            return True
        except Exception as e:
            logger.warning("cross_encoder_warmup_failed", error=str(e))
            return False

    def get_info(self) -> dict[str, Any]:
        """Get cross-encoder reranker info."""
        avg_latency = (
            self._total_latency_ms / self._call_count if self._call_count > 0 else 0.0
        )

        return {
            "type": "cross_encoder",
            "enabled": True,
            "available": self._available in (True, None),
            "loaded": self._model is not None,
            "model": self.model_name,
            "threshold": self.threshold,
            "max_length": self.max_length,
            "call_count": self._call_count,
            "avg_latency_ms": round(avg_latency, 2),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_reranker(
    enabled: bool | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
) -> IReranker:
    """
    Factory function to create appropriate reranker based on configuration.

    Uses feature flag to determine which implementation to return.

    Args:
        enabled: Override config enable flag (None = use env/config)
        model_name: Override model name
        threshold: Override score threshold

    Returns:
        IReranker implementation (NoOp or CrossEncoder)

    Example:
        # At startup
        reranker = create_reranker()

        # In retrieval
        results = reranker.rerank(query, candidates)
    """
    from resync.knowledge.config import CFG

    # Determine if reranking is enabled
    if enabled is None:
        enabled = CFG.enable_cross_encoder

    if not enabled:
        logger.info("reranker_disabled_using_noop")
        return NoOpReranker()

    # Create cross-encoder reranker
    threshold = threshold or CFG.cross_encoder_threshold

    reranker = CrossEncoderReranker(
        model_name=model_name,
        threshold=threshold,
    )

    logger.info("created_cross_encoder_reranker", model=reranker.model_name)
    return reranker


def create_gated_reranker(
    reranker: IReranker | None = None,
    gating_config: RerankGatingConfig | None = None,
) -> tuple[IReranker, RerankGatingPolicy]:
    """
    Create a reranker with gating policy.

    This is the recommended way to set up reranking for CPU-only environments.

    Args:
        reranker: Optional reranker (creates default if None)
        gating_config: Optional gating config (uses env defaults if None)

    Returns:
        Tuple of (reranker, gating_policy)

    Example:
        reranker, gating = create_gated_reranker()

        should_rerank, reason = gating.should_rerank(scores)
        if should_rerank:
            pool = candidates[:gating.config.max_candidates]
            results = reranker.rerank(query, pool)
    """
    if reranker is None:
        reranker = create_reranker()

    gating_config = gating_config or RerankGatingConfig.from_env()
    policy = RerankGatingPolicy(config=gating_config)

    logger.info(
        "created_gated_reranker",
        gating=policy.config.enabled,
        threshold=policy.config.score_low_threshold,
        margin=policy.config.margin_threshold,
        max_candidates=policy.config.max_candidates,
    )

    return reranker, policy


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Protocols
    "IReranker",
    # Implementations
    "NoOpReranker",
    "CrossEncoderReranker",
    # Gating
    "RerankGatingPolicy",
    "RerankGatingConfig",
    # Factories
    "create_reranker",
    "create_gated_reranker",
]
