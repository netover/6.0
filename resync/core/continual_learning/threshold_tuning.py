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

    def __init__(self, kg_client: Any = None) -> None:
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

    # Mode parameters for auto-tuning
    MODE_PARAMS = {
        AutoTuningMode.DISABLED: {"adjustment_pct": 0, "interval_hours": 0},
        AutoTuningMode.CONSERVATIVE: {"adjustment_pct": 2, "interval_hours": 48},
        AutoTuningMode.MODERATE: {"adjustment_pct": 5, "interval_hours": 24},
        AutoTuningMode.AGGRESSIVE: {"adjustment_pct": 10, "interval_hours": 12},
        AutoTuningMode.ADAPTIVE: {"adjustment_pct": 5, "interval_hours": 24},
    }

    CIRCUIT_BREAKER_COOLDOWN_HOURS = 24

    def __init__(self, tuner: "ThresholdTuner | None" = None) -> None:
        """Initialize manager."""
        self._tuner = tuner or get_threshold_tuner()
        self._configs: dict[str, ThresholdConfig] = {}
        self._metrics: dict[str, ThresholdMetrics] = {}
        self._mode: AutoTuningMode = AutoTuningMode.DISABLED
        self._circuit_breaker_active = False
        self._circuit_breaker_activated_at: datetime | None = None
        self._baseline_f1: float | None = None
        self._pending_recommendations: list[ThresholdRecommendation] = []
        self._audit_log: list[AuditLogEntry] = []

    def register_threshold(self, config: ThresholdConfig) -> None:
        """Register a threshold configuration."""
        self._configs[config.name] = config
        self._metrics[config.name] = ThresholdMetrics(
            name=config.name, current_value=config.default_value
        )

    # =============================================================================
    # Status & Mode
    # =============================================================================

    async def get_full_status(self) -> dict[str, Any]:
        """Get full threshold tuning status."""
        return {
            "mode": self._mode.value,
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_activated_at": (
                self._circuit_breaker_activated_at.isoformat()
                if self._circuit_breaker_activated_at else None
            ),
            "baseline_f1": self._baseline_f1,
            "thresholds": await self.get_thresholds(),
            "pending_recommendations": await self.get_pending_recommendations(),
            "recent_audit": await self.get_audit_log(limit=10),
        }

    async def get_mode(self) -> AutoTuningMode:
        """Get current auto-tuning mode."""
        return self._mode

    async def set_mode(self, mode: AutoTuningMode, admin_user: str = "admin") -> dict[str, Any]:
        """Set auto-tuning mode."""
        old_mode = self._mode
        self._mode = mode
        
        # Log the change
        self._audit_log.append(AuditLogEntry(
            action="mode_change",
            threshold_name="",
            old_value=float(AutoTuningMode(old_mode).value == "disabled") if old_mode else 0,
            new_value=float(mode.value == "disabled") if mode else 1,
            user_id=admin_user,
            reason=f"Mode changed from {old_mode.value} to {mode.value}",
        ))
        
        return {"status": "success", "mode": mode.value}

    # =============================================================================
    # Thresholds
    # =============================================================================

    async def get_thresholds(self) -> dict[str, Any]:
        """Get all threshold configurations."""
        thresholds = {}
        for name, config in self._configs.items():
            metrics = self._metrics.get(name)
            thresholds[name] = {
                "value": metrics.current_value if metrics else config.default_value,
                "default": config.default_value,
                "bounds": {
                    "min": config.bounds.min_value,
                    "max": config.bounds.max_value,
                    "step": config.bounds.step,
                },
                "auto_tune": config.auto_tune,
                "description": config.description,
            }
        return thresholds

    async def get_threshold(self, name: str) -> ThresholdConfig | None:
        """Get a specific threshold configuration."""
        return self._configs.get(name)

    async def set_threshold(
        self, name: str, value: float, admin_user: str = "admin", reason: str = ""
    ) -> dict[str, Any]:
        """Set a threshold value."""
        config = self._configs.get(name)
        if not config:
            return {"status": "error", "message": f"Threshold not found: {name}"}
        
        # Clamp to bounds
        clamped_value = config.bounds.clamp(value)
        old_value = config.default_value
        
        config.default_value = clamped_value
        
        # Update metrics
        if name in self._metrics:
            self._metrics[name].current_value = clamped_value
        
        # Log the change
        self._audit_log.append(AuditLogEntry(
            action="threshold_set",
            threshold_name=name,
            old_value=old_value,
            new_value=clamped_value,
            user_id=admin_user,
            reason=reason,
        ))
        
        return {"status": "success", "threshold": name, "value": clamped_value}

    async def reset_to_defaults(self, admin_user: str = "admin") -> dict[str, Any]:
        """Reset all thresholds to default values."""
        for name, config in self._configs.items():
            config.default_value = 0.5
            if name in self._metrics:
                self._metrics[name].current_value = 0.5
        
        self._audit_log.append(AuditLogEntry(
            action="reset_defaults",
            user_id=admin_user,
            reason="All thresholds reset to defaults",
        ))
        
        return {"status": "success", "message": "All thresholds reset to defaults"}

    async def rollback_to_last_good(self, admin_user: str = "admin") -> dict[str, Any]:
        """Rollback to last known good thresholds."""
        # Find last good from audit log
        last_good: dict[str, float] = {}
        for entry in reversed(self._audit_log):
            if entry.action == "threshold_set" and entry.old_value is not None:
                last_good[entry.threshold_name] = entry.old_value
        
        if not last_good:
            return {"status": "error", "message": "No rollback point found"}
        
        for name, value in last_good.items():
            if name in self._configs:
                self._configs[name].default_value = value
                if name in self._metrics:
                    self._metrics[name].current_value = value
        
        return {"status": "success", "thresholds": last_good}

    # =============================================================================
    # Metrics
    # =============================================================================

    async def get_metrics_summary(self, days: int = 30) -> ThresholdMetrics:
        """Get metrics summary for the specified period."""
        # Return aggregated metrics
        total_hits = sum(m.hit_count for m in self._metrics.values())
        total_misses = sum(m.miss_count for m in self._metrics.values())
        
        return ThresholdMetrics(
            name="aggregated",
            current_value=0.5,
            hit_count=total_hits,
            miss_count=total_misses,
        )

    async def get_daily_metrics(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily metrics for charting."""
        # Return empty for now - would need historical data storage
        return []

    # =============================================================================
    # Recommendations
    # =============================================================================

    async def get_pending_recommendations(self) -> list[dict[str, Any]]:
        """Get all pending recommendations."""
        return [rec.__dict__ for rec in self._pending_recommendations]

    async def generate_recommendations(self) -> list[ThresholdRecommendation]:
        """Generate new threshold recommendations."""
        self._pending_recommendations = []
        
        for name in self._configs:
            rec = self.get_recommendation(name)
            if rec:
                self._pending_recommendations.append(rec)
        
        return self._pending_recommendations

    async def approve_recommendation(
        self, recommendation_id: int, admin_user: str = "admin"
    ) -> dict[str, Any]:
        """Approve and apply a pending recommendation."""
        if recommendation_id >= len(self._pending_recommendations):
            return {"status": "error", "message": "Recommendation not found"}
        
        rec = self._pending_recommendations[recommendation_id]
        
        # Apply the recommendation
        result = await self.set_threshold(
            rec.threshold_name,
            rec.recommended_value,
            admin_user,
            f"Approved recommendation: {rec.reason}",
        )
        
        # Remove from pending
        self._pending_recommendations.pop(recommendation_id)
        
        return result

    async def reject_recommendation(
        self, recommendation_id: int, admin_user: str = "admin", reason: str = ""
    ) -> dict[str, Any]:
        """Reject a pending recommendation."""
        if recommendation_id >= len(self._pending_recommendations):
            return {"status": "error", "message": "Recommendation not found"}
        
        rec = self._pending_recommendations[recommendation_id]
        
        self._audit_log.append(AuditLogEntry(
            action="recommendation_rejected",
            threshold_name=rec.threshold_name,
            old_value=rec.current_value,
            new_value=rec.recommended_value,
            user_id=admin_user,
            reason=reason or f"Rejected: {rec.reason}",
        ))
        
        # Remove from pending
        self._pending_recommendations.pop(recommendation_id)
        
        return {"status": "success", "message": "Recommendation rejected"}

    # =============================================================================
    # Auto-Adjustment
    # =============================================================================

    async def run_auto_adjustment_cycle(self) -> dict[str, Any]:
        """Manually trigger an auto-adjustment cycle."""
        if self._mode == AutoTuningMode.DISABLED:
            return {"status": "error", "message": "Auto-tuning is disabled"}
        
        if self._circuit_breaker_active:
            return {"status": "error", "message": "Circuit breaker is active"}
        
        # Generate and apply recommendations
        recs = await self.generate_recommendations()
        
        applied = []
        for rec in recs:
            result = await self.set_threshold(
                rec.threshold_name,
                rec.recommended_value,
                "auto_tuner",
                f"Auto-adjustment: {rec.reason}",
            )
            if result["status"] == "success":
                applied.append(rec.threshold_name)
        
        return {
            "status": "success",
            "generated": len(recs),
            "applied": len(applied),
            "thresholds": applied,
        }

    async def reset_circuit_breaker(self, admin_user: str = "admin") -> dict[str, Any]:
        """Reset the circuit breaker."""
        self._circuit_breaker_active = False
        self._circuit_breaker_activated_at = None
        
        self._audit_log.append(AuditLogEntry(
            action="circuit_breaker_reset",
            user_id=admin_user,
            reason="Circuit breaker manually reset",
        ))
        
        return {"status": "success", "message": "Circuit breaker reset"}

    # =============================================================================
    # Audit Log
    # =============================================================================

    async def get_audit_log(
        self, limit: int = 50, threshold: str | None = None
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        logs = self._audit_log[-limit:]
        
        if threshold:
            logs = [e for e in logs if e.threshold_name == threshold]
        
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "action": e.action,
                "threshold_name": e.threshold_name,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "reason": e.reason,
                "user_id": e.user_id,
            }
            for e in logs
        ]

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
            reason=f"Based on hit rate of {metrics.hit_rate:.2%}",
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

    def __init__(self) -> None:
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
