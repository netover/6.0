"""
Freshness-based scoring for RAG retrieval.

PR4: Implements temporal relevance signals for document ranking.

Based on Google's QDF (Query Deserves Freshness) and Elasticsearch decay functions.

Key concepts:
- Exponential decay: freshness_score = exp(-λ * age_days)
- Half-life calibration: score = 0.5 at half_life_days
- Deprecated document penalty
- Version preference for same document family

Author: Resync Team
Version: 5.7.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class FreshnessConfig:
    """Configuration for freshness scoring."""

    # Decay parameters
    half_life_days: int = 180  # Score = 0.5 after 6 months
    min_score: float = 0.1  # Floor to avoid zero scores

    # Bonus for recent updates
    recent_boost_days: int = 30  # "Hot" period
    recent_boost_factor: float = 1.2

    # Penalty for deprecated docs
    deprecated_penalty: float = 0.3  # Multiply score by this

    # Version preference
    prefer_latest_version: bool = True
    version_penalty_per_old: float = 0.1  # Penalty per version behind

    # Query-based freshness (QDF)
    enable_qdf: bool = True
    qdf_keywords: list[str] | None = None  # Keywords that trigger freshness boost

    def __post_init__(self) -> None:
        if self.qdf_keywords is None:
            self.qdf_keywords = [
                "latest",
                "recent",
                "new",
                "current",
                "today",
                "now",
                "update",
                "change",
                "breaking",
            ]


class FreshnessScorer:
    """
    Calculates freshness scores for documents.

    Uses exponential decay with configurable half-life:
        score = exp(-λ * age_days) where λ = ln(2) / half_life_days

    Features:
    - Age-based decay
    - Recent document boost
    - Deprecated document penalty
    - Version comparison within document families
    """

    def __init__(self, config: FreshnessConfig | None = None):
        self.config = config or FreshnessConfig()
        # Calculate decay constant: λ = ln(2) / half_life
        self._lambda = math.log(2) / self.config.half_life_days

    def calculate_age_score(
        self,
        last_updated: datetime | str | None,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Calculate age-based freshness score.

        Returns score in [min_score, 1.0+boost].

        Args:
            last_updated: Document's last update timestamp
            reference_time: Reference time for age calculation (default: now)
        """
        if last_updated is None:
            return self.config.min_score

        # Parse string timestamp
        if isinstance(last_updated, str):
            try:
                # Handle various ISO formats
                if last_updated.endswith("Z"):
                    last_updated = last_updated[:-1] + "+00:00"
                last_updated = datetime.fromisoformat(last_updated)
            except ValueError:
                return self.config.min_score

        # Ensure timezone awareness
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        reference = reference_time or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)

        age_days = (reference - last_updated).days

        if age_days < 0:
            age_days = 0  # Future dates treated as now

        # Base exponential decay
        score = math.exp(-self._lambda * age_days)

        # Recent boost
        if age_days <= self.config.recent_boost_days:
            score *= self.config.recent_boost_factor

        # Apply floor
        return max(score, self.config.min_score)

    def calculate_version_score(
        self,
        doc_version: int,
        latest_version: int,
    ) -> float:
        """
        Calculate version-based score.

        Penalizes older versions of the same document.

        Args:
            doc_version: Version of this document
            latest_version: Latest known version in the family
        """
        if not self.config.prefer_latest_version:
            return 1.0

        versions_behind = latest_version - doc_version
        if versions_behind <= 0:
            return 1.0

        return max(
            1.0 - (versions_behind * self.config.version_penalty_per_old),
            self.config.min_score,
        )

    def calculate_deprecated_penalty(self, is_deprecated: bool) -> float:
        """Calculate penalty for deprecated documents."""
        if is_deprecated:
            return self.config.deprecated_penalty
        return 1.0

    def query_deserves_freshness(self, query: str) -> bool:
        """
        Check if query deserves freshness boost (QDF).

        Returns True if query contains freshness-indicating keywords.
        """
        if not self.config.enable_qdf or not self.config.qdf_keywords:
            return False

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.config.qdf_keywords)

    def calculate_combined_score(
        self,
        doc: dict[str, Any],
        latest_versions: dict[str, int] | None = None,
        query: str | None = None,
    ) -> float:
        """
        Calculate combined freshness score from metadata.

        Combines:
        - Age decay
        - Deprecated penalty
        - Version preference

        Args:
            doc: Document with metadata
            latest_versions: Map of doc_id -> latest_version
            query: Optional query for QDF boost
        """
        metadata = doc.get("metadata", {})

        # Start with base score of 1.0
        score = 1.0

        # Age decay
        last_updated = metadata.get("last_updated")
        age_score = self.calculate_age_score(last_updated)
        score *= age_score

        # Deprecated penalty
        is_deprecated = metadata.get("is_deprecated", False)
        score *= self.calculate_deprecated_penalty(is_deprecated)

        # Version penalty
        if latest_versions:
            doc_id = doc.get("document_id", "").split("#")[0]  # Remove chunk suffix
            doc_version = metadata.get("doc_version", 1)
            latest = latest_versions.get(doc_id, doc_version)
            score *= self.calculate_version_score(doc_version, latest)

        # QDF boost
        if query and self.query_deserves_freshness(query):
            score *= self.config.recent_boost_factor

        return score


def apply_freshness_rerank(
    documents: list[dict[str, Any]],
    freshness_weight: float = 0.2,
    config: FreshnessConfig | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank documents incorporating freshness scores.

    Final score = (1 - freshness_weight) * relevance_score + freshness_weight * freshness_score

    Args:
        documents: Documents with 'score' and 'metadata' fields
        freshness_weight: Weight of freshness in final score (0-1)
        config: Freshness configuration
        query: Optional query for QDF analysis
    """
    if not documents:
        return documents

    scorer = FreshnessScorer(config)

    # Get latest versions for each document family
    latest_versions: dict[str, int] = {}
    for doc in documents:
        doc_id = doc.get("document_id", "").split("#")[0]
        version = doc.get("metadata", {}).get("doc_version", 1)
        if doc_id:
            latest_versions[doc_id] = max(latest_versions.get(doc_id, 0), version)

    # Calculate combined scores
    for doc in documents:
        relevance = doc.get("score", 0.5)
        freshness = scorer.calculate_combined_score(doc, latest_versions, query)

        # Store component scores for debugging
        doc["_freshness_score"] = freshness
        doc["_relevance_score"] = relevance

        # Combined score
        doc["score"] = (1 - freshness_weight) * relevance + freshness_weight * freshness

    # Re-sort by combined score
    documents.sort(key=lambda x: x.get("score", 0), reverse=True)

    return documents


def get_document_age_days(doc: dict[str, Any]) -> int:
    """
    Get the age of a document in days.

    Utility function for reporting and debugging.
    """
    metadata = doc.get("metadata", {})
    last_updated_raw = metadata.get("last_updated")

    if not last_updated_raw:
        return -1  # Unknown age

    try:
        last_updated: datetime
        if isinstance(last_updated_raw, str):
            ts = last_updated_raw
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            last_updated = datetime.fromisoformat(ts)
        elif isinstance(last_updated_raw, datetime):
            last_updated = last_updated_raw
        else:
            return -1

        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days: int = (now - last_updated).days
        return age_days

    except (ValueError, TypeError):
        return -1


def is_document_stale(
    doc: dict[str, Any],
    stale_threshold_days: int = 365,
) -> bool:
    """
    Check if a document is considered stale.

    Args:
        doc: Document with metadata
        stale_threshold_days: Age in days to consider stale
    """
    age = get_document_age_days(doc)
    if age < 0:
        return True  # Unknown age = assume stale
    return age > stale_threshold_days
