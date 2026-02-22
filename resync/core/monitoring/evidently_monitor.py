# pylint: skip-file
# mypy: ignore-errors
"""
Evidently AI Monitoring Module.

Provides drift detection and AI quality monitoring for the Resync system.

Features:
- Data drift detection (query patterns changing)
- Prediction drift detection (response quality changing)
- Target drift detection (user feedback degrading)
- Scheduled monitoring with configurable intervals
- Resource limits (CPU/memory) to not impact production

Author: Resync Team
Version: 5.2.3.29
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import json
import os

try:
    import resource
except ImportError:
    resource = None

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, ConfigDict

logger = structlog.get_logger(__name__)

# ============================================================================
# EVIDENTLY AVAILABILITY CHECK
# ============================================================================

try:
    import pandas as pd
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
    )
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
    logger.info("evidently_available", version="0.4+")
except ImportError:
    EVIDENTLY_AVAILABLE = False
    pd = None
    logger.warning(
        "evidently_not_available", message="Install with: pip install evidently"
    )


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================


class DriftType(str, Enum):
    """Types of drift that can be monitored."""

    DATA = "data"  # Input query patterns
    PREDICTION = "prediction"  # Model output patterns
    TARGET = "target"  # User feedback/outcomes
    CONCEPT = "concept"  # Input-output relationship


class MonitoringSchedule(str, Enum):
    """Monitoring schedule options."""

    HOURLY = "hourly"
    EVERY_4_HOURS = "every_4_hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"  # Only on-demand


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Represents a drift detection alert."""

    alert_id: str
    drift_type: DriftType
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class ResourceLimits(BaseModel):
    """Resource limits for monitoring jobs."""

    max_cpu_percent: float = Field(
        default=25.0, ge=5.0, le=100.0, description="Maximum CPU usage percentage"
    )
    max_memory_mb: int = Field(
        default=512, ge=128, le=4096, description="Maximum memory usage in MB"
    )
    max_execution_time_seconds: int = Field(
        default=300, ge=30, le=3600, description="Maximum execution time"
    )
    nice_level: int = Field(
        default=10,
        ge=0,
        le=19,
        description="Process nice level (higher = lower priority)",
    )


class MonitoringConfig(BaseModel):
    """Configuration for AI monitoring."""

    enabled: bool = Field(default=True, description="Enable/disable monitoring")

    # Drift detection settings
    data_drift_enabled: bool = Field(
        default=True, description="Monitor data/query drift"
    )
    prediction_drift_enabled: bool = Field(
        default=True, description="Monitor prediction/response drift"
    )
    target_drift_enabled: bool = Field(
        default=True, description="Monitor target/feedback drift"
    )

    # Thresholds
    drift_threshold: float = Field(
        default=0.15, ge=0.01, le=0.5, description="Drift detection threshold (0-1)"
    )
    alert_threshold: float = Field(
        default=0.25, ge=0.05, le=0.5, description="Alert trigger threshold"
    )

    # Scheduling
    schedule: MonitoringSchedule = Field(
        default=MonitoringSchedule.DAILY, description="Monitoring schedule"
    )
    schedule_time: str = Field(
        default="03:00",
        pattern=r"^\d{2}:\d{2}$",
        description="Time to run scheduled monitoring (HH:MM)",
    )

    # Resource limits
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits, description="Resource usage limits"
    )

    # Data retention
    reference_window_days: int = Field(
        default=7, ge=1, le=90, description="Days of reference data to use"
    )
    current_window_hours: int = Field(
        default=24, ge=1, le=168, description="Hours of current data to compare"
    )

    # Storage
    reports_path: str = Field(
        default="data/evidently_reports", description="Path to store reports"
    )
    max_reports_stored: int = Field(
        default=30, ge=1, le=365, description="Maximum reports to retain"
    )

    model_config = ConfigDict(use_enum_values=True)


# Default configuration
DEFAULT_MONITORING_CONFIG = MonitoringConfig()


# ============================================================================
# DATA COLLECTOR
# ============================================================================


class MonitoringDataCollector:
    """
    Collects and stores data for drift monitoring.

    Stores:
    - Query embeddings (for semantic drift)
    - Query metadata (length, intent, entities)
    - Response metadata (length, confidence, latency)
    - User feedback (ratings, actions taken)
    """

    def __init__(
        self,
        storage_path: str = "data/monitoring_data",
        max_records: int = 100000,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records

        # In-memory buffer
        self._query_buffer: list[dict[str, Any]] = []
        self._response_buffer: list[dict[str, Any]] = []
        self._feedback_buffer: list[dict[str, Any]] = []

        self._lock = threading.Lock()

    def record_query(
        self,
        query_id: str,
        query_text: str,
        query_embedding: list[float] | None = None,
        intent: str | None = None,
        entities: dict[str, list[str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a query for monitoring."""
        record = {
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_length": len(query_text),
            "word_count": len(query_text.split()),
            "intent": intent or "unknown",
            "entity_count": sum(len(v) for v in (entities or {}).values()),
            "has_job_entity": bool(entities and entities.get("jobs")),
            "has_ws_entity": bool(entities and entities.get("workstations")),
            "has_error_entity": bool(entities and entities.get("error_codes")),
            **(metadata or {}),
        }

        # Store embedding separately (large)
        if query_embedding:
            record["embedding_dim"] = len(query_embedding)
            # Only store embedding hash for drift detection
            record["embedding_hash"] = hash(tuple(query_embedding[:10]))

        with self._lock:
            self._query_buffer.append(record)
            if len(self._query_buffer) >= 1000:
                self._flush_queries()

    def record_response(
        self,
        query_id: str,
        response_text: str,
        specialists_used: list[str],
        confidence: float,
        latency_ms: int,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a response for monitoring."""
        record = {
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_length": len(response_text),
            "word_count": len(response_text.split()),
            "specialists_count": len(specialists_used),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "success": success,
            **(metadata or {}),
        }

        with self._lock:
            self._response_buffer.append(record)
            if len(self._response_buffer) >= 1000:
                self._flush_responses()

    def record_feedback(
        self,
        query_id: str,
        rating: int | None = None,  # 1-5
        helpful: bool | None = None,
        action_taken: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record user feedback for monitoring."""
        record = {
            "query_id": query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "helpful": helpful,
            "action_taken": action_taken,
            **(metadata or {}),
        }

        with self._lock:
            self._feedback_buffer.append(record)
            if len(self._feedback_buffer) >= 500:
                self._flush_feedback()

    def _flush_queries(self) -> None:
        """Flush query buffer to storage."""
        if not self._query_buffer:
            return

        file_path = (
            self.storage_path
            / f"queries_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}.jsonl"
        )
        with open(file_path, "a") as f:
            for record in self._query_buffer:
                f.write(json.dumps(record) + "\n")

        self._query_buffer.clear()

    def _flush_responses(self) -> None:
        """Flush response buffer to storage."""
        if not self._response_buffer:
            return

        file_path = (
            self.storage_path
            / f"responses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}.jsonl"
        )
        with open(file_path, "a") as f:
            for record in self._response_buffer:
                f.write(json.dumps(record) + "\n")

        self._response_buffer.clear()

    def _flush_feedback(self) -> None:
        """Flush feedback buffer to storage."""
        if not self._feedback_buffer:
            return

        file_path = (
            self.storage_path
            / f"feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}.jsonl"
        )
        with open(file_path, "a") as f:
            for record in self._feedback_buffer:
                f.write(json.dumps(record) + "\n")

        self._feedback_buffer.clear()

    def flush_all(self) -> None:
        """Flush all buffers."""
        with self._lock:
            self._flush_queries()
            self._flush_responses()
            self._flush_feedback()

    def get_data(
        self,
        data_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """
        Get stored data for a date range.

        Args:
            data_type: "queries", "responses", or "feedback"
            start_date: Start of range
            end_date: End of range

        Returns:
            List of records
        """
        # Flush buffers first
        self.flush_all()

        records = []
        pattern = f"{data_type}_*.jsonl"

        for file_path in sorted(self.storage_path.glob(pattern)):
            # Check if file is in date range
            file_date_str = file_path.stem.split("_")[1]
            try:
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                if not (start_date.date() <= file_date.date() <= end_date.date()):
                    continue
            except ValueError:
                continue

            with open(file_path) as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record_time = datetime.fromisoformat(record["timestamp"])
                        if start_date <= record_time <= end_date:
                            records.append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue

        return records

    def cleanup_old_data(self, retention_days: int = 30) -> int:
        """
        Remove data files older than retention period.

        Args:
            retention_days: Number of days to keep data

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_str = cutoff_date.strftime("%Y%m%d")

        try:
            for file_path in self.storage_path.glob("*.jsonl"):
                # Parse date from filename: type_YYYYMMDD_HH.jsonl
                try:
                    parts = file_path.stem.split("_")
                    if len(parts) >= 2:
                        file_date_str = parts[1]
                        if file_date_str < cutoff_str:
                            file_path.unlink()
                            deleted_count += 1
                except (IndexError, ValueError, OSError):
                    continue
        except Exception as e:
            logger.error("data_cleanup_error", error=str(e))

        return deleted_count


# ============================================================================
# DRIFT DETECTOR
# ============================================================================


class DriftDetector:
    """
    Detects drift in AI system data using Evidently.
    """

    def __init__(
        self,
        config: MonitoringConfig,
        data_collector: MonitoringDataCollector,
    ):
        self.config = config
        self.data_collector = data_collector
        self.reports_path = Path(config.reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def detect_data_drift(
        self,
        reference_data: list[dict],
        current_data: list[dict],
    ) -> tuple[bool, float, Any | None]:
        """
        Detect drift in query/input data.

        Returns:
            (drift_detected, drift_score, report)
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("evidently_not_available", operation="data_drift")
            return False, 0.0, None

        if len(reference_data) < 50 or len(current_data) < 20:
            logger.info(
                "insufficient_data_for_drift",
                ref=len(reference_data),
                cur=len(current_data),
            )
            return False, 0.0, None

        try:
            # Convert to DataFrames
            ref_df = pd.DataFrame(reference_data)
            cur_df = pd.DataFrame(current_data)

            # Select numeric columns for drift detection
            numeric_cols = ["query_length", "word_count", "entity_count"]
            available_cols = [
                c for c in numeric_cols if c in ref_df.columns and c in cur_df.columns
            ]

            if not available_cols:
                return False, 0.0, None

            # Create drift report
            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                    DataDriftTable(),
                ]
            )

            report.run(
                reference_data=ref_df[available_cols],
                current_data=cur_df[available_cols],
            )

            # Extract results
            result = report.as_dict()
            drift_score = (
                result.get("metrics", [{}])[0].get("result", {}).get("drift_share", 0.0)
            )
            drift_detected = drift_score >= self.config.drift_threshold

            # Save report
            report_path = (
                self.reports_path
                / f"data_drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
            )
            report.save_html(str(report_path))

            return drift_detected, drift_score, report

        except Exception as e:
            logger.error("data_drift_detection_error", error=str(e))
            return False, 0.0, None

    def detect_prediction_drift(
        self,
        reference_data: list[dict],
        current_data: list[dict],
    ) -> tuple[bool, float, Any | None]:
        """
        Detect drift in model predictions/responses.
        """
        if not EVIDENTLY_AVAILABLE:
            return False, 0.0, None

        if len(reference_data) < 50 or len(current_data) < 20:
            return False, 0.0, None

        try:
            ref_df = pd.DataFrame(reference_data)
            cur_df = pd.DataFrame(current_data)

            # Response metrics
            response_cols = [
                "response_length",
                "confidence",
                "latency_ms",
                "specialists_count",
            ]
            available_cols = [
                c for c in response_cols if c in ref_df.columns and c in cur_df.columns
            ]

            if not available_cols:
                return False, 0.0, None

            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                ]
            )

            report.run(
                reference_data=ref_df[available_cols],
                current_data=cur_df[available_cols],
            )

            result = report.as_dict()
            drift_score = (
                result.get("metrics", [{}])[0].get("result", {}).get("drift_share", 0.0)
            )
            drift_detected = drift_score >= self.config.drift_threshold

            report_path = (
                self.reports_path
                / f"prediction_drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
            )
            report.save_html(str(report_path))

            return drift_detected, drift_score, report

        except Exception as e:
            logger.error("prediction_drift_detection_error", error=str(e))
            return False, 0.0, None

    def detect_target_drift(
        self,
        reference_data: list[dict],
        current_data: list[dict],
    ) -> tuple[bool, float, Any | None]:
        """
        Detect drift in user feedback/outcomes.
        """
        if not EVIDENTLY_AVAILABLE:
            return False, 0.0, None

        if len(reference_data) < 30 or len(current_data) < 10:
            return False, 0.0, None

        try:
            ref_df = pd.DataFrame(reference_data)
            cur_df = pd.DataFrame(current_data)

            # Convert boolean to int for analysis
            for df in [ref_df, cur_df]:
                if "helpful" in df.columns:
                    df["helpful_int"] = df["helpful"].map({True: 1, False: 0, None: -1})

            feedback_cols = ["rating", "helpful_int"]
            available_cols = [
                c for c in feedback_cols if c in ref_df.columns and c in cur_df.columns
            ]

            if not available_cols:
                return False, 0.0, None

            # Filter out null values
            ref_df = ref_df.dropna(subset=available_cols)
            cur_df = cur_df.dropna(subset=available_cols)

            if len(ref_df) < 10 or len(cur_df) < 5:
                return False, 0.0, None

            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                ]
            )

            report.run(
                reference_data=ref_df[available_cols],
                current_data=cur_df[available_cols],
            )

            result = report.as_dict()
            drift_score = (
                result.get("metrics", [{}])[0].get("result", {}).get("drift_share", 0.0)
            )
            drift_detected = drift_score >= self.config.drift_threshold

            report_path = (
                self.reports_path
                / f"target_drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
            )
            report.save_html(str(report_path))

            return drift_detected, drift_score, report

        except Exception as e:
            logger.error("target_drift_detection_error", error=str(e))
            return False, 0.0, None

    def cleanup_old_reports(self) -> int:
        """
        Remove old reports exceeding storage limit.

        Returns:
            Number of reports deleted
        """
        deleted_count = 0
        max_reports = self.config.max_reports_stored

        try:
            reports = sorted(
                self.reports_path.glob("*.html"), key=lambda p: p.stat().st_mtime
            )

            if len(reports) > max_reports:
                to_delete = reports[: len(reports) - max_reports]
                for report_path in to_delete:
                    try:
                        report_path.unlink()
                        deleted_count += 1
                    except OSError:
                        continue
        except Exception as e:
            logger.error("report_cleanup_error", error=str(e))

        return deleted_count


# ============================================================================
# MONITORING SERVICE
# ============================================================================


class AIMonitoringService:
    """
    Main service for AI monitoring and drift detection.

    Features:
    - Scheduled drift detection
    - Resource-limited execution
    - Alert generation
    - Report storage
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ):
        self.config = config or DEFAULT_MONITORING_CONFIG
        self.data_collector = MonitoringDataCollector(
            storage_path=str(Path(self.config.reports_path).parent / "monitoring_data")
        )
        self.drift_detector = DriftDetector(self.config, self.data_collector)

        self._alerts: list[DriftAlert] = []
        self._last_run: datetime | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            "ai_monitoring_service_initialized",
            enabled=self.config.enabled,
            schedule=self.config.schedule,
            evidently_available=EVIDENTLY_AVAILABLE,
        )

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the monitoring service.

        Args:
            tg: Optional TaskGroup to run the scheduler loop in
        """
        if not self.config.enabled:
            logger.info("monitoring_disabled")
            return

        if self._running:
            return

        self._running = True

        if self.config.schedule != MonitoringSchedule.MANUAL:
            if tg:
                self._scheduler_task = tg.create_task(
                    self._scheduler_loop(), name="scheduler_loop"
                )
            else:
                self._scheduler_task = track_task(
                    self._scheduler_loop(), name="scheduler_loop"
                )
            logger.info(
                "monitoring_scheduler_started",
                schedule=self.config.schedule,
                method="task_group" if tg else "track_task",
            )

    async def stop(self) -> None:
        """Stop the monitoring service."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task

        # Flush any remaining data
        self.data_collector.flush_all()
        logger.info("monitoring_service_stopped")

    async def _scheduler_loop(self) -> None:
        """Run scheduled monitoring."""
        while self._running:
            try:
                interval = self._get_schedule_interval()
                await asyncio.sleep(interval)

                if self._running:
                    self.run_monitoring()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduler_error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry

    def _get_schedule_interval(self) -> int:
        """Get interval in seconds based on schedule."""
        intervals = {
            MonitoringSchedule.HOURLY: 3600,
            MonitoringSchedule.EVERY_4_HOURS: 14400,
            MonitoringSchedule.DAILY: 86400,
            MonitoringSchedule.WEEKLY: 604800,
        }
        return intervals.get(self.config.schedule, 86400)

    def run_monitoring(self) -> dict[str, Any]:
        """
        Run a complete monitoring cycle.

        Returns:
            Results of drift detection
        """
        start_time = time.time()

        logger.info("monitoring_run_started")

        # Apply resource limits
        self._apply_resource_limits()

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_drift": None,
            "prediction_drift": None,
            "target_drift": None,
            "alerts": [],
            "duration_seconds": 0,
        }

        try:
            # Define time windows
            now = datetime.now(timezone.utc)
            reference_start = now - timedelta(days=self.config.reference_window_days)
            reference_end = now - timedelta(hours=self.config.current_window_hours)
            current_start = reference_end
            current_end = now

            # Data drift
            if self.config.data_drift_enabled:
                ref_queries = self.data_collector.get_data(
                    "queries", reference_start, reference_end
                )
                cur_queries = self.data_collector.get_data(
                    "queries", current_start, current_end
                )

                detected, score, _ = self.drift_detector.detect_data_drift(
                    ref_queries, cur_queries
                )
                results["data_drift"] = {
                    "detected": detected,
                    "score": score,
                    "reference_count": len(ref_queries),
                    "current_count": len(cur_queries),
                }

                if detected and score >= self.config.alert_threshold:
                    self._create_alert(DriftType.DATA, "data_drift_score", score)

            # Prediction drift
            if self.config.prediction_drift_enabled:
                ref_responses = self.data_collector.get_data(
                    "responses", reference_start, reference_end
                )
                cur_responses = self.data_collector.get_data(
                    "responses", current_start, current_end
                )

                detected, score, _ = self.drift_detector.detect_prediction_drift(
                    ref_responses, cur_responses
                )
                results["prediction_drift"] = {
                    "detected": detected,
                    "score": score,
                    "reference_count": len(ref_responses),
                    "current_count": len(cur_responses),
                }

                if detected and score >= self.config.alert_threshold:
                    self._create_alert(
                        DriftType.PREDICTION, "prediction_drift_score", score
                    )

            # Target drift
            if self.config.target_drift_enabled:
                ref_feedback = self.data_collector.get_data(
                    "feedback", reference_start, reference_end
                )
                cur_feedback = self.data_collector.get_data(
                    "feedback", current_start, current_end
                )

                detected, score, _ = self.drift_detector.detect_target_drift(
                    ref_feedback, cur_feedback
                )
                results["target_drift"] = {
                    "detected": detected,
                    "score": score,
                    "reference_count": len(ref_feedback),
                    "current_count": len(cur_feedback),
                }

                if detected and score >= self.config.alert_threshold:
                    self._create_alert(DriftType.TARGET, "target_drift_score", score)

            results["alerts"] = [a.to_dict() for a in self._alerts[-10:]]

            # Cleanup old data and reports
            self.data_collector.cleanup_old_data(
                retention_days=self.config.reference_window_days + 7
            )
            self.drift_detector.cleanup_old_reports()

        except Exception as e:
            logger.error("monitoring_run_error", error=str(e))
            results["error"] = str(e)

        results["duration_seconds"] = time.time() - start_time
        self._last_run = datetime.now(timezone.utc)

        logger.info(
            "monitoring_run_completed",
            duration=results["duration_seconds"],
            alerts=len(results["alerts"]),
        )

        return results

    def _apply_resource_limits(self) -> None:
        """Apply resource limits before running monitoring."""
        limits = self.config.resource_limits

        try:
            # Set nice level (lower priority)
            os.nice(limits.nice_level)
        except (OSError, AttributeError) as e:
            logger.debug(
                "suppressed_exception", error=str(e), exc_info=True
            )  # was: pass

        if limits:
            try:
                # Set memory limit
                mem_limit = limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except (ValueError, AttributeError) as exc:
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)

    def _create_alert(
        self,
        drift_type: DriftType,
        metric_name: str,
        value: float,
    ) -> DriftAlert:
        """Create and store a drift alert."""
        severity = (
            AlertSeverity.CRITICAL
            if value >= 0.5
            else AlertSeverity.ERROR
            if value >= 0.35
            else AlertSeverity.WARNING
        )

        alert = DriftAlert(
            alert_id=f"alert_{int(time.time())}_{drift_type.value}",
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold=self.config.alert_threshold,
            message="{drift_type.value.title()} drift detected: {metric_name} = {value:.3f} (threshold: {self.config.alert_threshold})",
        )

        self._alerts.append(alert)

        logger.warning(
            "drift_alert_created",
            alert_id=alert.alert_id,
            drift_type=drift_type.value,
            severity=severity.value,
            value=value,
        )

        return alert

    def get_alerts(
        self,
        since: datetime | None = None,
        drift_type: DriftType | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[DriftAlert]:
        """Get alerts with optional filtering."""
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if drift_type:
            alerts = [a for a in alerts if a.drift_type == drift_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_status(self) -> dict[str, Any]:
        """Get monitoring service status."""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "schedule": self.config.schedule.value,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "total_alerts": len(self._alerts),
            "recent_alerts": len(
                [
                    a
                    for a in self._alerts
                    if a.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
                ]
            ),
            "evidently_available": EVIDENTLY_AVAILABLE,
        }


# ============================================================================
# SINGLETON AND FACTORY
# ============================================================================

_monitoring_service_instance: AIMonitoringService | None = None


def get_monitoring_service() -> AIMonitoringService | None:
    """Get the singleton monitoring service instance."""
    return _monitoring_service_instance


async def init_monitoring_service(
    config: MonitoringConfig | None = None,
    tg: asyncio.TaskGroup | None = None,
) -> AIMonitoringService:
    """
    Initialize and start the monitoring service.

    Args:
        config: Monitoring configuration
        tg: Optional TaskGroup to start the service in

    Returns:
        Initialized AIMonitoringService
    """
    global _monitoring_service_instance

    _monitoring_service_instance = AIMonitoringService(config=config)
    _monitoring_service_instance.start(tg=tg)

    return _monitoring_service_instance


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EVIDENTLY_AVAILABLE",
    "DriftType",
    "MonitoringSchedule",
    "AlertSeverity",
    "DriftAlert",
    "ResourceLimits",
    "MonitoringConfig",
    "DEFAULT_MONITORING_CONFIG",
    "MonitoringDataCollector",
    "DriftDetector",
    "AIMonitoringService",
    "get_monitoring_service",
    "init_monitoring_service",
]
