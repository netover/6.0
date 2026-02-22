# pylint: skip-file
# mypy: ignore-errors
"""
Observability Configuration Module.

Centralizes configuration for monitoring and observability tools:
- LangFuse: LLM tracing, prompt management, cost tracking
- Evidently: ML monitoring, data drift detection
- Custom metrics: Internal Prometheus-compatible metrics

Usage:
    from resync.core.observability import (
        get_observability_config,
        setup_langfuse,
        setup_evidently,
    )

    # On app startup
    await setup_langfuse()
    setup_evidently()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LangFuseConfig:
    """LangFuse configuration."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    )
    public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    # Sampling
    sample_rate: float = field(
        default_factory=lambda: float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
    )

    # Flush settings
    flush_interval_seconds: int = 5
    batch_size: int = 50

    def is_configured(self) -> bool:
        """Check if LangFuse is properly configured."""
        return self.enabled and bool(self.public_key) and bool(self.secret_key)


@dataclass
class EvidentlyConfig:
    """Evidently configuration."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("EVIDENTLY_ENABLED", "false").lower()
        == "true"
    )

    # Reference data settings
    reference_window_days: int = 7
    current_window_hours: int = 24

    # Drift thresholds
    feature_drift_threshold: float = 0.1
    prediction_drift_threshold: float = 0.15
    data_quality_threshold: float = 0.95

    # Monitoring intervals
    check_interval_minutes: int = 60
    report_retention_days: int = 30

    # Storage
    reports_dir: str = field(
        default_factory=lambda: os.getenv(
            "EVIDENTLY_REPORTS_DIR", "/var/log/resync/evidently"
        )
    )


@dataclass
class ObservabilityConfig:
    """Combined observability configuration."""

    langfuse: LangFuseConfig = field(default_factory=LangFuseConfig)
    evidently: EvidentlyConfig = field(default_factory=EvidentlyConfig)

    # General settings
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    service_name: str = "resync"
    service_version: str = "5.3.8"


# Singleton config
_config: ObservabilityConfig | None = None


def get_observability_config() -> ObservabilityConfig:
    """Get or create observability configuration."""
    global _config
    if _config is None:
        _config = ObservabilityConfig()
    return _config


# =============================================================================
# LANGFUSE SETUP
# =============================================================================

# LangFuse client (lazy init)
_langfuse_client = None


def setup_langfuse() -> bool:
    """
    Initialize LangFuse client.

    Returns:
        True if successfully initialized, False otherwise
    """
    global _langfuse_client

    config = get_observability_config().langfuse

    if not config.is_configured():
        logger.info("langfuse_disabled", reason="not configured")
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            flush_interval=config.flush_interval_seconds,
            max_retries=3,
        )

        # Verify connection
        # Note: Langfuse doesn't have a direct ping, so we just check client creation

        logger.info(
            "langfuse_initialized",
            host=config.host,
            sample_rate=config.sample_rate,
        )
        return True

    except Exception as exc:
        logger.warning(
            "langfuse_not_available",
            hint="pip install langfuse",
            reason=type(exc).__name__,
        )
        return False
    except Exception as e:
        logger.error("langfuse_init_failed", error=str(e))
        return False


def get_langfuse_client():
    """Get the LangFuse client."""
    return _langfuse_client


def shutdown_langfuse():
    """Shutdown LangFuse client gracefully."""
    global _langfuse_client

    if _langfuse_client:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
            logger.info("langfuse_shutdown")
        except Exception as e:
            logger.warning("langfuse_shutdown_error", error=str(e))
        finally:
            _langfuse_client = None


# =============================================================================
# EVIDENTLY SETUP
# =============================================================================

# Evidently monitor (lazy init)
_evidently_monitor = None


class EvidentlyMonitor:
    """
    Evidently-based monitoring for ML/AI pipelines.

    Tracks:
    - Data drift in inputs
    - Prediction drift in outputs
    - Data quality metrics
    - Feature statistics
    """

    def __init__(self, config: EvidentlyConfig):
        self.config = config
        self._reference_data: list[dict[str, Any]] = []
        self._current_data: list[dict[str, Any]] = []
        self._last_check: datetime | None = None
        self._reports: list[dict[str, Any]] = []

        # Ensure reports directory exists
        os.makedirs(config.reports_dir, exist_ok=True)

        # Try to import evidently
        try:
            from evidently import ColumnMapping
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset
            from evidently.report import Report  # type: ignore[import-untyped]

            self._evidently_available = True
            self._Report = Report
            self._DataDriftPreset = DataDriftPreset
            self._DataQualityPreset = DataQualityPreset
            self._ColumnMapping = ColumnMapping

        except ImportError:
            self._evidently_available = False
            logger.warning("evidently_not_installed", hint="pip install evidently")

    def add_reference_data(self, data: dict[str, Any]) -> None:
        """Add data point to reference dataset."""
        data["_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._reference_data.append(data)

        # Trim old data
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.reference_window_days
        )
        self._reference_data = [
            d
            for d in self._reference_data
            if datetime.fromisoformat(d["_timestamp"]) > cutoff
        ]

    def add_current_data(self, data: dict[str, Any]) -> None:
        """Add data point to current dataset."""
        data["_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._current_data.append(data)

        # Trim old data
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.config.current_window_hours
        )
        self._current_data = [
            d
            for d in self._current_data
            if datetime.fromisoformat(d["_timestamp"]) > cutoff
        ]

    def track_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
        success: bool,
        prompt_type: str = "unknown",
    ) -> None:
        """Track an LLM call for monitoring."""
        data = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "model": model,
            "success": 1 if success else 0,
            "prompt_type": prompt_type,
        }
        self.add_current_data(data)

    def track_rag_query(
        self,
        query_length: int,
        num_results: int,
        avg_score: float,
        latency_ms: float,
        collection: str,
    ) -> None:
        """Track a RAG query for monitoring."""
        data = {
            "query_length": query_length,
            "num_results": num_results,
            "avg_score": avg_score,
            "latency_ms": latency_ms,
            "collection": collection,
        }
        self.add_current_data(data)

    def check_drift(self) -> dict[str, Any]:
        """
        Check for data drift between reference and current data.

        Returns:
            Drift report with metrics and alerts
        """
        if not self._evidently_available:
            return {"error": "Evidently not installed"}

        if len(self._reference_data) < 10 or len(self._current_data) < 10:
            return {"error": "Insufficient data for drift detection"}

        try:
            import pandas as pd

            # Convert to DataFrames
            ref_df = pd.DataFrame(self._reference_data)
            cur_df = pd.DataFrame(self._current_data)

            # Remove timestamp column
            ref_df = ref_df.drop(columns=["_timestamp"], errors="ignore")
            cur_df = cur_df.drop(columns=["_timestamp"], errors="ignore")

            # Select numeric columns only
            numeric_cols = ref_df.select_dtypes(include=["number"]).columns.tolist()
            ref_df = ref_df[numeric_cols]
            cur_df = cur_df[numeric_cols]

            # Create drift report
            report = self._Report(metrics=[self._DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)

            # Extract results
            result = report.as_dict()

            # Check thresholds
            drift_detected = False
            drift_columns = []

            for metric in result.get("metrics", []):
                if metric.get("metric") == "DataDriftTable":
                    for col_data in (
                        metric.get("result", {}).get("drift_by_columns", {}).values()
                    ):
                        if col_data.get("drift_detected"):
                            drift_detected = True
                            drift_columns.append(col_data.get("column_name"))

            report_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drift_detected": drift_detected,
                "drift_columns": drift_columns,
                "reference_size": len(ref_df),
                "current_size": len(cur_df),
                "details": result,
            }

            # Save report
            self._save_report(report_data)
            self._last_check = datetime.now(timezone.utc)

            # Log alert if drift detected
            if drift_detected:
                logger.warning(
                    "data_drift_detected",
                    columns=drift_columns,
                    reference_size=len(ref_df),
                    current_size=len(cur_df),
                )

            return report_data

        except Exception as e:
            logger.error("drift_check_failed", error=str(e))
            return {"error": str(e)}

    def _save_report(self, report: dict[str, Any]) -> None:
        """Save report to disk."""
        try:
            import json

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{timestamp}.json"
            filepath = os.path.join(self.config.reports_dir, filename)

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            self._reports.append(
                {
                    "filename": filename,
                    "timestamp": timestamp,
                    "drift_detected": report.get("drift_detected", False),
                }
            )

            # Cleanup old reports
            self._cleanup_old_reports()

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("report_save_failed", error=str(e))

    def _cleanup_old_reports(self) -> None:
        """Remove reports older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.report_retention_days
        )

        import os

        for filename in os.listdir(self.config.reports_dir):
            if not filename.startswith("drift_report_"):
                continue

            filepath = os.path.join(self.config.reports_dir, filename)
            try:
                # Extract timestamp from filename
                parts = filename.replace("drift_report_", "").replace(".json", "")
                file_time = datetime.strptime(parts, "%Y%m%d_%H%M%S")

                if file_time < cutoff:
                    os.remove(filepath)
                    logger.debug("old_report_deleted", filename=filename)
            except Exception as exc:
                logger.debug(
                    "suppressed_exception", error=str(exc), exc_info=True
                )  # was: pass

    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "enabled": self._evidently_available,
            "reference_data_size": len(self._reference_data),
            "current_data_size": len(self._current_data),
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "reports_count": len(self._reports),
            "config": {
                "reference_window_days": self.config.reference_window_days,
                "current_window_hours": self.config.current_window_hours,
                "check_interval_minutes": self.config.check_interval_minutes,
            },
        }


def setup_evidently() -> EvidentlyMonitor | None:
    """
    Initialize Evidently monitor.

    Returns:
        EvidentlyMonitor instance or None if disabled
    """
    global _evidently_monitor

    config = get_observability_config().evidently

    if not config.enabled:
        logger.info("evidently_disabled")
        return None

    _evidently_monitor = EvidentlyMonitor(config)
    logger.info("evidently_initialized", reports_dir=config.reports_dir)

    return _evidently_monitor


def get_evidently_monitor() -> EvidentlyMonitor | None:
    """Get the Evidently monitor."""
    return _evidently_monitor


# =============================================================================
# UNIFIED OBSERVABILITY
# =============================================================================


async def setup_observability() -> dict[str, bool]:
    """
    Initialize all observability components.

    Returns:
        Dict with component initialization status
    """
    results = {
        "langfuse": setup_langfuse(),
        "evidently": setup_evidently() is not None,
    }

    logger.info("observability_setup_complete", **results)
    return results


async def shutdown_observability() -> None:
    """Shutdown all observability components."""
    shutdown_langfuse()
    logger.info("observability_shutdown_complete")


def get_observability_status() -> dict[str, Any]:
    """Get status of all observability components."""
    config = get_observability_config()

    return {
        "langfuse": {
            "enabled": config.langfuse.enabled,
            "configured": config.langfuse.is_configured(),
            "connected": _langfuse_client is not None,
            "host": config.langfuse.host if config.langfuse.is_configured() else None,
        },
        "evidently": {
            "enabled": config.evidently.enabled,
            "active": _evidently_monitor is not None,
            "statistics": _evidently_monitor.get_statistics()
            if _evidently_monitor
            else None,
        },
        "environment": config.environment,
        "service": {
            "name": config.service_name,
            "version": config.service_version,
        },
    }
