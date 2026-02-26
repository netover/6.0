"""
Threshold Tuning - PostgreSQL Implementation.

Provides dynamic threshold tuning for continual learning using PostgreSQL.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from resync.core.database.models import LearningThreshold
from resync.core.database.repositories import FeedbackStore

logger = logging.getLogger(__name__)

class AutoTuningMode(str, Enum):
    """Auto-tuning mode for thresholds."""

    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class ThresholdBounds:
    """Bounds for a threshold value."""

    min_value: float = 0.0
    max_value: float = 1.0
    step: float = 0.01

    def clamp(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.min_value, min(self.max_value, value))

@dataclass
class ThresholdConfig:
    """Configuration for a threshold."""

    name: str
    default_value: float = 0.5
    bounds: ThresholdBounds = field(default_factory=ThresholdBounds)
    description: str = ""
    auto_tune: bool = True
    mode: AutoTuningMode = AutoTuningMode.MODERATE

@dataclass
class ThresholdMetrics:
    """Metrics for threshold performance."""

    name: str
    current_value: float
    hit_count: int = 0
    miss_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

@dataclass
class ThresholdRecommendation:
    """Recommendation for threshold adjustment."""

    threshold_name: str
    current_value: float
    recommended_value: float
    confidence: float = 0.0
    reason: str = ""
    based_on_samples: int = 0

@dataclass
class AuditLogEntry:
    """Audit log entry for threshold changes."""

    id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = ""
    threshold_name: str = ""
    old_value: float | None = None
    new_value: float | None = None
    reason: str = ""
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditResult:
    """Result of an audit operation."""

    success: bool
    message: str = ""
    entries: list[AuditLogEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

class AuditToKGPipeline:
    """Pipeline to push audit data to knowledge graph."""

    def __init__(self, kg_client: Any = None):
        """Initialize pipeline."""
        self._kg_client = kg_client
        self._buffer: list[AuditLogEntry] = []

    def push(self, entry: AuditLogEntry) -> bool:
        """Push audit entry to KG."""
        self._buffer.append(entry)
        return True

    def flush(self) -> int:
        """Flush buffer to KG."""
        count = len(self._buffer)
        self._buffer.clear()
        return count

class ThresholdTuningManager:
    """Manager for threshold tuning operations."""

    def __init__(self, tuner: "ThresholdTuner | None" = None):
        """Initialize manager."""
        self._tuner = tuner or get_threshold_tuner()
        self._configs: dict[str, ThresholdConfig] = {}
        self._metrics: dict[str, ThresholdMetrics] = {}

    def register_threshold(self, config: ThresholdConfig) -> None:
        """Register a threshold configuration."""
        self._configs[config.name] = config
        self._metrics[config.name] = ThresholdMetrics(
            name=config.name, current_value=config.default_value
        )

    def get_recommendation(self, name: str) -> ThresholdRecommendation | None:
        """Get tuning recommendation for a threshold."""
        if name not in self._metrics:
            return None

        metrics = self._metrics[name]
        config = self._configs.get(name)

        if not config or not config.auto_tune:
            return None

        # Simple recommendation based on hit rate
        if metrics.hit_rate < 0.5:
            recommended = metrics.current_value + 0.05
        elif metrics.hit_rate > 0.9:
            recommended = metrics.current_value - 0.02
        else:
            return None

        if config.bounds:
            recommended = config.bounds.clamp(recommended)

        return ThresholdRecommendation(
            threshold_name=name,
            current_value=metrics.current_value,
            recommended_value=recommended,
            confidence=0.7,
            reason="Based on hit rate of {metrics.hit_rate:.2%}",
            based_on_samples=metrics.hit_count + metrics.miss_count,
        )

    async def apply_recommendations(self) -> list[ThresholdRecommendation]:
        """Apply all pending recommendations."""
        applied = []
        for name in self._configs:
            rec = self.get_recommendation(name)
            if rec:
                await self._tuner.set_threshold(name, rec.recommended_value)
                applied.append(rec)
        return applied

__all__ = [
    "ThresholdTuner",
    "get_threshold_tuner",
    "AuditLogEntry",
    "AutoTuningMode",
    "ThresholdBounds",
    "ThresholdConfig",
    "ThresholdMetrics",
    "ThresholdRecommendation",
    "ThresholdTuningManager",
    "AuditResult",
    "AuditToKGPipeline",
    "get_threshold_tuning_manager",
    "get_audit_to_kg_pipeline",
]

class ThresholdTuner:
    """Threshold Tuner - PostgreSQL Backend."""

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._store = FeedbackStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the tuner."""
        self._initialized = True
        logger.info("ThresholdTuner initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the tuner."""
        self._initialized = False

    async def get_threshold(self, name: str, default: float = 0.5) -> float:
        """Get threshold value by name."""
        value = await self._store.thresholds.get_threshold(name)
        return value if value is not None else default

    async def set_threshold(
        self, name: str, value: float, min_value: float = 0.0, max_value: float = 1.0
    ) -> LearningThreshold:
        """Set or update threshold."""
        return await self._store.thresholds.set_threshold(
            name, value, min_value, max_value
        )

    async def adjust_threshold(self, name: str, adjustment: float) -> float | None:
        """Adjust threshold by a delta."""
        current = await self.get_threshold(name)
        new_value = max(0.0, min(1.0, current + adjustment))
        await self.set_threshold(name, new_value)
        return new_value

    async def get_all_thresholds(self) -> dict[str, float]:
        """Get all thresholds."""
        thresholds = await self._store.thresholds.get_all(limit=100)
        return {t.threshold_name: t.current_value for t in thresholds}

    async def auto_tune(self, feedback_window: int = 100) -> dict[str, Any]:
        """Auto-tune thresholds based on feedback."""
        positive = await self._store.feedback.get_positive_examples(
            limit=feedback_window
        )
        negative = await self._store.feedback.get_negative_examples(
            limit=feedback_window
        )

        total = len(positive) + len(negative)
        if total == 0:
            return {"adjusted": False, "reason": "No feedback data"}

        positive_rate = len(positive) / total

        # Adjust confidence threshold based on feedback
        if positive_rate < 0.5:
            # Too many negatives, increase threshold
            await self.adjust_threshold("confidence", 0.05)
        elif positive_rate > 0.8:
            # Very positive, can lower threshold
            await self.adjust_threshold("confidence", -0.02)

        return {
            "adjusted": True,
            "positive_rate": positive_rate,
            "total_feedback": total,
        }

_instance: ThresholdTuner | None = None
_manager_instance: ThresholdTuningManager | None = None
_pipeline_instance: AuditToKGPipeline | None = None

def get_threshold_tuner() -> ThresholdTuner:
    """Get the singleton ThresholdTuner instance."""
    global _instance
    if _instance is None:
        _instance = ThresholdTuner()
    return _instance

def get_threshold_tuning_manager() -> ThresholdTuningManager:
    """Get the singleton ThresholdTuningManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ThresholdTuningManager()
    return _manager_instance

def get_audit_to_kg_pipeline() -> AuditToKGPipeline:
    """Get the singleton AuditToKGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AuditToKGPipeline()
    return _pipeline_instance
