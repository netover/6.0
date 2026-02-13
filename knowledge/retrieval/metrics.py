"""
Retrieval Quality Metrics for RAG Evaluation.

PR3: Implements standard IR metrics for retrieval evaluation.

Metrics:
- Recall@k: Proportion of relevant docs retrieved in top-k
- MRR (Mean Reciprocal Rank): Inverse position of first relevant result
- nDCG (Normalized Discounted Cumulative Gain): Position-weighted relevance

Based on:
- RAGAS framework
- TruLens RAG Triad
- MTEB Leaderboard methodology

Author: Resync Team
Version: 5.7.0
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# =============================================================================
# METRICS DATACLASSES
# =============================================================================


@dataclass
class RetrievalMetrics:
    """Standard IR metrics for retrieval evaluation."""

    # Recall metrics (proportion of relevant docs retrieved)
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0

    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0  # Normalized DCG
    ndcg_at_10: float = 0.0

    # Hit rate (any relevant doc in top-k)
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0

    # Latency metrics
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Evaluation metadata
    sample_count: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "recall_at_20": round(self.recall_at_20, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "hit_rate_at_1": round(self.hit_rate_at_1, 4),
            "hit_rate_at_3": round(self.hit_rate_at_3, 4),
            "hit_rate_at_5": round(self.hit_rate_at_5, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "sample_count": self.sample_count,
            "timestamp": self.timestamp,
        }


@dataclass
class EvalSample:
    """A single evaluation sample with expected results."""

    query: str
    expected_doc_ids: list[str]  # Relevant document IDs
    relevance_scores: dict[str, float] | None = None  # doc_id -> relevance (0-1)

    # Optional metadata
    sample_id: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""

    sample_id: str
    query: str

    # Retrieved results
    retrieved_ids: list[str]
    retrieved_scores: list[float]

    # Expected
    expected_ids: list[str]

    # Individual metrics
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_10: float = 0.0

    # Latency
    latency_ms: float = 0.0

    # Status
    success: bool = True
    error: str | None = None


# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """
    Calculate recall@k.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Args:
        relevant: Set of relevant document IDs
        retrieved: List of retrieved document IDs (in rank order)
        k: Cutoff position
    """
    if not relevant:
        return 1.0  # No relevant docs = perfect recall (vacuous truth)

    retrieved_k = set(retrieved[:k])
    return len(relevant & retrieved_k) / len(relevant)


def hit_rate_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """
    Calculate hit rate@k (any relevant doc in top-k).

    Returns 1.0 if any relevant doc is in top-k, 0.0 otherwise.
    """
    if not relevant:
        return 1.0

    retrieved_k = set(retrieved[:k])
    return 1.0 if (relevant & retrieved_k) else 0.0


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    """
    Calculate reciprocal rank (1/position of first relevant).

    MRR = mean of reciprocal ranks across queries.
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(
    relevance_scores: dict[str, float],
    retrieved: list[str],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    DCG@k = Σ (rel_i / log2(i + 1)) for i in 1..k
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    relevance_scores: dict[str, float],
    retrieved: list[str],
    k: int,
) -> float:
    """
    Calculate Normalized DCG at k.

    nDCG@k = DCG@k / IDCG@k

    Where IDCG is the ideal DCG (perfect ranking).
    """
    if not relevance_scores:
        return 0.0

    # Calculate actual DCG
    dcg = dcg_at_k(relevance_scores, retrieved, k)

    # Calculate ideal DCG (documents sorted by relevance)
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(relevance_scores, ideal_order, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def ndcg_at_k_binary(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    """
    Calculate nDCG@k with binary relevance (1 if relevant, 0 otherwise).

    Simplified version when we don't have graded relevance.
    """
    # Convert to relevance scores dict
    relevance_scores = {doc_id: 1.0 for doc_id in relevant}
    return ndcg_at_k(relevance_scores, retrieved, k)


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (p / 100)
    lower = int(index)
    upper = lower + 1

    if upper >= len(sorted_values):
        return sorted_values[-1]

    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================


def calculate_retrieval_metrics(
    results: list[EvalResult],
) -> RetrievalMetrics:
    """
    Calculate aggregate retrieval metrics from evaluation results.

    Args:
        results: List of evaluation results for individual samples

    Returns:
        RetrievalMetrics with aggregated metrics
    """
    if not results:
        return RetrievalMetrics()

    # Collect per-sample metrics
    recalls_1, recalls_3, recalls_5, recalls_10, recalls_20 = [], [], [], [], []
    hit_rates_1, hit_rates_3, hit_rates_5 = [], [], []
    reciprocal_ranks = []
    ndcg_5, ndcg_10 = [], []
    latencies = []

    for r in results:
        if not r.success:
            continue

        relevant = set(r.expected_ids)
        retrieved = r.retrieved_ids

        # Recall@k
        recalls_1.append(recall_at_k(relevant, retrieved, 1))
        recalls_3.append(recall_at_k(relevant, retrieved, 3))
        recalls_5.append(recall_at_k(relevant, retrieved, 5))
        recalls_10.append(recall_at_k(relevant, retrieved, 10))
        recalls_20.append(recall_at_k(relevant, retrieved, 20))

        # Hit rate@k
        hit_rates_1.append(hit_rate_at_k(relevant, retrieved, 1))
        hit_rates_3.append(hit_rate_at_k(relevant, retrieved, 3))
        hit_rates_5.append(hit_rate_at_k(relevant, retrieved, 5))

        # MRR
        reciprocal_ranks.append(reciprocal_rank(relevant, retrieved))

        # nDCG (binary relevance)
        ndcg_5.append(ndcg_at_k_binary(relevant, retrieved, 5))
        ndcg_10.append(ndcg_at_k_binary(relevant, retrieved, 10))

        # Latency
        if r.latency_ms > 0:
            latencies.append(r.latency_ms)

    # Helper to safely calculate mean
    def safe_mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    return RetrievalMetrics(
        recall_at_1=safe_mean(recalls_1),
        recall_at_3=safe_mean(recalls_3),
        recall_at_5=safe_mean(recalls_5),
        recall_at_10=safe_mean(recalls_10),
        recall_at_20=safe_mean(recalls_20),
        mrr=safe_mean(reciprocal_ranks),
        ndcg_at_5=safe_mean(ndcg_5),
        ndcg_at_10=safe_mean(ndcg_10),
        hit_rate_at_1=safe_mean(hit_rates_1),
        hit_rate_at_3=safe_mean(hit_rates_3),
        hit_rate_at_5=safe_mean(hit_rates_5),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        avg_latency_ms=safe_mean(latencies),
        sample_count=len([r for r in results if r.success]),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# REGRESSION GATE
# =============================================================================


@dataclass
class RegressionThresholds:
    """Thresholds for regression detection."""

    # Maximum allowed regression (as fraction, e.g., 0.02 = 2%)
    recall_at_5: float = 0.02
    recall_at_10: float = 0.02
    mrr: float = 0.03
    ndcg_at_10: float = 0.02

    # Maximum allowed latency increase (as fraction)
    p95_latency: float = 0.10  # 10% increase allowed


class RegressionGate:
    """
    CI/CD gate that fails builds if metrics regress beyond threshold.

    Usage:
        gate = RegressionGate(baseline_metrics)
        passed, failures = gate.check(current_metrics)
        if not passed:
            raise Exception(f"Regression detected: {failures}")
    """

    def __init__(
        self,
        baseline: RetrievalMetrics,
        thresholds: RegressionThresholds | None = None,
    ):
        self.baseline = baseline
        self.thresholds = thresholds or RegressionThresholds()

    def check(self, current: RetrievalMetrics) -> tuple[bool, list[str]]:
        """
        Check if current metrics pass the regression gate.

        Returns:
            (passed: bool, failures: list[str])
        """
        failures = []

        # Check quality metrics (higher is better)
        quality_checks = [
            ("recall_at_5", self.thresholds.recall_at_5),
            ("recall_at_10", self.thresholds.recall_at_10),
            ("mrr", self.thresholds.mrr),
            ("ndcg_at_10", self.thresholds.ndcg_at_10),
        ]

        for metric_name, threshold in quality_checks:
            baseline_val = getattr(self.baseline, metric_name, 0)
            current_val = getattr(current, metric_name, 0)

            if baseline_val > 0:
                regression = (baseline_val - current_val) / baseline_val
                if regression > threshold:
                    failures.append(
                        "{metric_name}: {current_val:.4f} vs baseline {baseline_val:.4f} "
                        "(-{regression*100:.1f}%, threshold: -{threshold*100:.0f}%)"
                    )

        # Check latency (lower is better)
        baseline_latency = self.baseline.p95_latency_ms
        current_latency = current.p95_latency_ms

        if baseline_latency > 0:
            latency_increase = (current_latency - baseline_latency) / baseline_latency
            if latency_increase > self.thresholds.p95_latency:
                failures.append(
                    "p95_latency_ms: {current_latency:.1f}ms vs baseline {baseline_latency:.1f}ms "
                    "(+{latency_increase*100:.1f}%, threshold: +{self.thresholds.p95_latency*100:.0f}%)"
                )

        return len(failures) == 0, failures

    def generate_report(self, current: RetrievalMetrics) -> str:
        """Generate a human-readable regression report."""
        passed, failures = self.check(current)

        lines = [
            "=" * 60,
            "RETRIEVAL METRICS REGRESSION REPORT",
            "=" * 60,
            "",
            f"Status: {'✅ PASSED' if passed else '❌ FAILED'}",
            "",
            "Metrics Comparison:",
            "-" * 40,
        ]

        metrics = [
            ("Recall@5", "recall_at_5"),
            ("Recall@10", "recall_at_10"),
            ("MRR", "mrr"),
            ("nDCG@10", "ndcg_at_10"),
            ("p95 Latency", "p95_latency_ms"),
        ]

        for display_name, attr in metrics:
            baseline_val = getattr(self.baseline, attr, 0)
            current_val = getattr(current, attr, 0)

            if "latency" in attr:
                change = current_val - baseline_val
                change_str = "+{change:.1f}ms" if change >= 0 else "{change:.1f}ms"
                lines.append("  {display_name}: {current_val:.1f}ms (baseline: {baseline_val:.1f}ms, {change_str})")
            else:
                change = (current_val - baseline_val) / baseline_val if baseline_val > 0 else 0
                change_str = "+{change*100:.1f}%" if change >= 0 else "{change*100:.1f}%"
                lines.append("  {display_name}: {current_val:.4f} (baseline: {baseline_val:.4f}, {change_str})")

        if failures:
            lines.extend([
                "",
                "Failures:",
                "-" * 40,
            ])
            for failure in failures:
                lines.append(f"  ❌ {failure}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# HNSW PARAMETER TUNING
# =============================================================================


@dataclass
class HNSWConfig:
    """HNSW index configuration for pgvector."""

    m: int = 16  # Connections per node (12-32 recommended)
    ef_construction: int = 256  # Build-time search depth (100-400)
    ef_search: int = 200  # Query-time search depth (100-500)

    @classmethod
    def high_recall(cls) -> HNSWConfig:
        """Configuration optimized for maximum recall (>99%)."""
        return cls(m=32, ef_construction=400, ef_search=500)

    @classmethod
    def balanced(cls) -> HNSWConfig:
        """Balanced configuration (95-98% recall)."""
        return cls(m=16, ef_construction=256, ef_search=200)

    @classmethod
    def low_latency(cls) -> HNSWConfig:
        """Configuration optimized for low latency (~85% recall)."""
        return cls(m=12, ef_construction=128, ef_search=50)

    def to_sql_create_index(self, table: str, column: str, index_name: str) -> str:
        """Generate SQL for creating HNSW index."""
        return f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table}
            USING hnsw ({column} vector_cosine_ops)
            WITH (m = {self.m}, ef_construction = {self.ef_construction})
        """

    def to_sql_set_ef_search(self) -> str:
        """Generate SQL to set ef_search for current session."""
        return f"SET hnsw.ef_search = {self.ef_search}"
