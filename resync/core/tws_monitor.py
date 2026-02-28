# pylint
"""TWS monitoring and alerting system.

This module provides real-time monitoring of the TWS environment,
performance metrics collection, and alert generation for anomalies.
"""

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.exceptions import PerformanceError
from resync.core.interfaces import ITWSClient
from resync.core.task_tracker import track_task
from resync.core.teams_integration import get_teams_integration

from .shared_utils import TeamsNotification, create_job_status_notification

logger = structlog.get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for TWS operations."""

    # API Performance
    api_response_times: list[float] = field(default_factory=list)
    api_error_rates: list[float] = field(default_factory=list)

    # Cache Performance
    cache_hit_ratios: list[float] = field(default_factory=list)
    cache_miss_rates: list[float] = field(default_factory=list)

    # LLM Usage
    llm_calls: int = 0
    llm_tokens_used: int = 0
    llm_cost_estimate: float = 0.0

    # Circuit Breaker Status
    circuit_breaker_trips: int = 0
    circuit_breaker_status: str = "closed"

    # Memory Usage
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0

    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = 0.0

@dataclass
class Alert:
    """Alert for system anomalies or issues."""

    alert_id: str
    severity: str  # critical, high, medium, low
    category: str  # api, cache, llm, circuit_breaker, memory, job
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)

class TWSMonitor:
    """TWS monitoring and alerting system."""

    def __init__(self, tws_client: ITWSClient):
        """Initialize TWS monitor.

        Args:
            tws_client: TWS client for data collection
        """
        self.tws_client = tws_client
        # NOTE: Separate sampling cadence (metrics collection) from alert evaluation.
        # This matches the monitoring dashboard sampling interval (5s) while allowing
        # alert evaluation to run less frequently.
        self.sample_interval_seconds: float = 5.0
        # Keep ~36h of history at 30s interval (4320 records)
        # Using deque for O(1) appends and automatic pruning of old records
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=4320)
        # Bounded deque prevents OOM from repeated threshold breaches.
        # 10 000 entries keeps ~83 hours of alerts at the 30 s check interval
        # even without any suppression, which is sufficient for all dashboards.
        self.alerts: deque[Alert] = deque(maxlen=10_000)
        # Minimum seconds between alerts of the same category (suppresses storms).
        self._alert_suppression_seconds: int = 300
        self.alert_check_interval = 30  # seconds
        self._is_monitoring = False
        self._monitoring_task: asyncio.Task | None = None

        # Alert thresholds
        self.alert_thresholds = {
            "api_error_rate": 0.05,  # 5% error rate
            "cache_hit_ratio": 0.80,  # 80% hit ratio
            "llm_cost_daily": 10.0,  # $10 daily budget
            "memory_usage_mb": 500.0,  # 500MB memory limit
            "circuit_breaker_trips": 3,  # 3 trips per hour
        }

        logger.info("tws_monitor_initialized")

    def start_monitoring(self, tg: asyncio.TaskGroup | None = None) -> None:
        """Start continuous monitoring.

        Args:
            tg: Optional TaskGroup to run the monitoring loop in
        """
        if self._is_monitoring:
            logger.warning("Monitoring already started")
            return

        self._is_monitoring = True
        if tg:
            self._monitoring_task = tg.create_task(
                self._monitoring_loop(), name="monitoring_loop"
            )
        else:
            self._monitoring_task = track_task(
                self._monitoring_loop(), name="monitoring_loop"
            )
        logger.info(
            "tws_monitoring_started", method="task_group" if tg else "track_task"
        )

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None
        logger.info("tws_monitoring_stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_alert_check = 0.0
        while self._is_monitoring:
            try:
                # Collect metrics frequently.
                await self._collect_metrics()

                # Evaluate alerts on a slower cadence.
                now = time.monotonic()
                if (now - last_alert_check) >= max(float(self.alert_check_interval), 1.0):
                    await self._check_alerts()
                    last_alert_check = now

                await asyncio.sleep(max(float(self.sample_interval_seconds), 0.1))
            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error(
                    "error_in_tws_monitoring_loop", error=str(e), exc_info=True
                )
                await asyncio.sleep(10)  # Brief pause on error

    async def _collect_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            start_time = time.time()

            # Collect API metrics
            api_response_time = await self._measure_api_response_time()
            api_error_rate = self._calculate_api_error_rate()

            # Collect cache metrics
            cache_hit_ratio = self._measure_cache_performance()

            # Collect LLM metrics
            llm_metrics = self._measure_llm_usage()

            # Collect memory metrics (async — runs psutil on thread pool)
            memory_usage = await self._measure_memory_usage()

            # Create metrics record
            metrics = PerformanceMetrics(
                api_response_times=[api_response_time],
                api_error_rates=[api_error_rate],
                cache_hit_ratios=[cache_hit_ratio],
                llm_calls=llm_metrics.get("calls", 0),
                llm_tokens_used=llm_metrics.get("tokens", 0),
                llm_cost_estimate=llm_metrics.get("cost", 0.0),
                memory_usage_mb=memory_usage,
                uptime_seconds=time.time() - start_time,
            )

            self.metrics_history.append(metrics)

            # Deque handles pruning automatically via maxlen
            # No need for manual list comprehension filtering

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("error_collecting_metrics", error=str(e), exc_info=True)

    async def _measure_api_response_time(self) -> float:
        """Measure API response time."""
        try:
            start_time = time.time()
            await self.tws_client.check_connection()
            return time.time() - start_time
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("exception_caught", error=str(e), exc_info=True)
            return 999.0  # High value indicates error

    def _calculate_api_error_rate(self) -> float:
        """Calculate API error rate from runtime metrics.

        Returns error_rate as a fraction (0.0–1.0).  Falls back to 0.0 if
        metrics are not yet instrumented.
        """
        # FIX P1-03: Was hardcoded 0.0. Now reads from RuntimeMetricsCollector.
        try:
            from resync.core.metrics.runtime_metrics import runtime_metrics

            snapshot = runtime_metrics.get_snapshot()
            system = snapshot.get("system", {})
            total: int = system.get("api_requests_total", 0)
            errors: int = system.get("api_errors_total", 0)
            if total > 0:
                return errors / total
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:  # noqa: BLE001
            logger.warning("api_error_rate_unavailable", error=str(exc))
        return 0.0

    def _measure_cache_performance(self) -> float:
        """Measure cache hit ratio from runtime metrics.

        Returns hit_ratio as a fraction (0.0–1.0).  Falls back to 0.0 if
        metrics are not yet available.
        """
        # FIX P1-03: Was hardcoded 0.85. Now reads from RuntimeMetricsCollector.
        try:
            from resync.core.metrics.runtime_metrics import runtime_metrics

            snapshot = runtime_metrics.get_snapshot()
            rc = snapshot.get("router_cache", {})
            hits: int = rc.get("hits", 0)
            misses: int = rc.get("misses", 0)
            total = hits + misses
            if total > 0:
                return hits / total
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:  # noqa: BLE001
            logger.warning("cache_performance_unavailable", error=str(exc))
        return 0.0

    def _measure_llm_usage(self) -> dict[str, Any]:
        """Measure LLM usage from lightweight_store or runtime metrics.

        Returns a dict with keys: calls, tokens, cost.
        Falls back to zeros when metrics are not yet instrumented.
        """
        # FIX P1-03: Was hardcoded fake data {"calls": 10, "tokens": 1000, "cost": 0.02}.
        # Now reads from the lightweight metrics store where LLM counters are recorded.
        result: dict[str, Any] = {"calls": 0, "tokens": 0, "cost": 0.0}
        try:
            from resync.core.metrics.lightweight_store import get_metrics_store as _get_lw_store

            store = _get_lw_store()
            result["calls"] = int(store.get_counter("llm.requests_total"))
            result["tokens"] = int(store.get_counter("llm.tokens_total"))
            result["cost"] = float(store.get_counter("llm.cost_usd"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:  # noqa: BLE001
            logger.warning("llm_usage_metrics_unavailable", error=str(exc))
        return result

    async def _measure_memory_usage(self) -> float:
        """Measure memory usage without blocking the event loop.

        psutil.Process().memory_info() is a blocking syscall — it is executed
        on the thread pool via asyncio.to_thread to avoid event-loop starvation.
        """
        import psutil

        def _blocking_measure() -> float:
            return psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        return await asyncio.to_thread(_blocking_measure)

    def _should_emit_alert(self, category: str) -> bool:
        """Return True only if no recent unresolved alert exists for this category.

        Prevents duplicate alerts from filling the bounded deque during sustained
        threshold breaches (e.g. high memory every 30 s for hours).
        """
        now = datetime.now(timezone.utc)
        for alert in reversed(self.alerts):
            if alert.category == category and not alert.resolved:
                age_seconds = (now - alert.timestamp).total_seconds()
                return age_seconds > self._alert_suppression_seconds
        return True

    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        alerts_to_add = []

        if not self.metrics_history:
            return

        latest_metrics = self.metrics_history[-1]

        # Check API error rate
        if latest_metrics.api_error_rates:
            avg_error_rate = sum(latest_metrics.api_error_rates) / len(
                latest_metrics.api_error_rates
            )
            if avg_error_rate > self.alert_thresholds["api_error_rate"]:
                alerts_to_add.append(
                    Alert(
                        alert_id=f"api_error_{int(time.time())}",
                        severity="high",
                        category="api",
                        message=(
                            "API error rate exceeded threshold: "
                            f"{avg_error_rate:.2%}"
                        ),
                        timestamp=datetime.now(timezone.utc),
                        details={"error_rate": avg_error_rate},
                    )
                )

        # Check cache hit ratio
        if latest_metrics.cache_hit_ratios:
            avg_hit_ratio = sum(latest_metrics.cache_hit_ratios) / len(
                latest_metrics.cache_hit_ratios
            )
            if avg_hit_ratio < self.alert_thresholds["cache_hit_ratio"]:
                alerts_to_add.append(
                    Alert(
                        alert_id=f"cache_hit_{int(time.time())}",
                        severity="medium",
                        category="cache",
                        message=f"Cache hit ratio below threshold: {avg_hit_ratio:.2%}",
                        timestamp=datetime.now(timezone.utc),
                        details={"hit_ratio": avg_hit_ratio},
                    )
                )

        # Check LLM cost (daily)
        # FIX P1-01: Original formula multiplied by 24 (assuming hourly rate) but
        # llm_cost_estimate is accumulated per *sample* (metrics collection) cycle.
        # The correct multiplier is seconds_per_day / sample_interval_seconds.
        _seconds_per_day = 86400
        interval = max(float(self.sample_interval_seconds), 0.1)
        daily_cost = latest_metrics.llm_cost_estimate * (_seconds_per_day / interval)
        if daily_cost > self.alert_thresholds["llm_cost_daily"]:
            alerts_to_add.append(
                Alert(
                    alert_id=f"llm_cost_{int(time.time())}",
                    severity="medium",
                    category="llm",
                    message=f"LLM cost approaching daily budget: ${daily_cost:.2f}",
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "daily_cost": daily_cost,
                        "budget": self.alert_thresholds["llm_cost_daily"],
                    },
                )
            )

        # Check memory usage
        if latest_metrics.memory_usage_mb > self.alert_thresholds["memory_usage_mb"]:
            alerts_to_add.append(
                Alert(
                    alert_id=f"memory_usage_{int(time.time())}",
                    severity="high",
                    category="memory",
                    message=(
                        "Memory usage exceeded threshold: "
                        f"{latest_metrics.memory_usage_mb:.1f}MB"
                    ),
                    timestamp=datetime.now(timezone.utc),
                    details={"memory_usage_mb": latest_metrics.memory_usage_mb},
                )
            )

        # Add new alerts — skip duplicates within the suppression window to prevent
        # deque exhaustion from sustained threshold breaches (e.g. high memory for hours).
        for alert in alerts_to_add:
            if self._should_emit_alert(alert.category):
                self.alerts.append(alert)
                logger.warning("new_alert_generated", message=alert.message)

                # Send Teams notification for critical alerts
                if alert.severity in ["critical", "high"]:
                    await self._send_teams_notification(alert)
            else:
                logger.debug(
                    "alert_suppressed_within_window",
                    category=alert.category,
                    suppression_seconds=self._alert_suppression_seconds,
                )

    async def _send_teams_notification(self, alert: Alert) -> None:
        """Send alert notification to Microsoft Teams.

        Args:
            alert: Alert to send notification for
        """
        try:
            # Get Teams integration
            teams_integration = get_teams_integration()

            # Create Teams notification
            notification = TeamsNotification(
                title=f"TWS Alert: {alert.category.title()}",
                message=alert.message,
                severity=alert.severity,
                additional_data=alert.details,
            )

            # Send notification
            await teams_integration.send_notification(notification)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "failed_to_send_teams_notification_for_alert",
                error=str(e),
                exc_info=True,
            )

    async def monitor_job_status_change(
        self, job_data: dict[str, Any], instance_name: str
    ) -> None:
        """Monitor job status changes and send notifications for configured statuses.

        Args:
            job_data: Job status data from TWS
            instance_name: Name of the TWS instance
        """
        try:
            # Get Teams integration
            teams_integration = get_teams_integration()

            # Check if job notifications are enabled
            if (
                not teams_integration.config.enabled
                or not teams_integration.config.enable_job_notifications
            ):
                return

            # Check if this instance is being monitored
            if (
                teams_integration.config.monitored_tws_instances
                and instance_name
                not in teams_integration.config.monitored_tws_instances
            ):
                return

            # Check if job status matches filters
            job_status = job_data.get("status", "").upper()
            if job_status in [
                status.upper() for status in teams_integration.config.job_status_filters
            ]:
                # Send notification
                notification = create_job_status_notification(
                    job_data, instance_name, teams_integration.config.job_status_filters
                )

                if notification is None:
                    return

                await teams_integration.send_notification(notification)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "failed_to_process_job_status_change_for_teams_notification",
                error=str(e),
                exc_info=True,
            )

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary with performance metrics and alerts
        """
        try:
            # Calculate averages
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                avg_api_response_time = (
                    (
                        sum(sum(m.api_response_times) for m in self.metrics_history)
                        / sum(len(m.api_response_times) for m in self.metrics_history)
                    )
                    if any(m.api_response_times for m in self.metrics_history)
                    else 0.0
                )

                avg_cache_hit_ratio = (
                    (
                        sum(sum(m.cache_hit_ratios) for m in self.metrics_history)
                        / sum(len(m.cache_hit_ratios) for m in self.metrics_history)
                    )
                    if any(m.cache_hit_ratios for m in self.metrics_history)
                    else 0.0
                )
            else:
                latest_metrics = PerformanceMetrics()
                avg_api_response_time = 0.0
                avg_cache_hit_ratio = 0.0

            # Get recent alerts
            recent_alerts = [
                alert
                for alert in self.alerts
                if not alert.resolved
                and (datetime.now(timezone.utc) - alert.timestamp).total_seconds()
                < 3600  # Last hour
            ]

            return {
                "current_metrics": {
                    "api_response_time_ms": avg_api_response_time * 1000,
                    "cache_hit_ratio": avg_cache_hit_ratio,
                    "llm_calls_today": latest_metrics.llm_calls,
                    "llm_cost_today": latest_metrics.llm_cost_estimate,
                    "memory_usage_mb": latest_metrics.memory_usage_mb,
                    "uptime_seconds": latest_metrics.uptime_seconds,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "alerts": [
                    {
                        "id": alert.alert_id,
                        "severity": alert.severity,
                        "category": alert.category,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                        "details": alert.details,
                    }
                    for alert in recent_alerts
                ],
                "summary": {
                    "total_alerts": len([a for a in self.alerts if not a.resolved]),
                    "critical_alerts": len(
                        [
                            a
                            for a in self.alerts
                            if not a.resolved and a.severity == "critical"
                        ]
                    ),
                    "high_alerts": len(
                        [
                            a
                            for a in self.alerts
                            if not a.resolved and a.severity == "high"
                        ]
                    ),
                },
            }

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Error generating performance report", error=str(e))
            raise PerformanceError(f"Failed to generate performance report: {e}") from e

    def get_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        recent_alerts = sorted(
            [alert for alert in self.alerts if not alert.resolved],
            key=lambda x: x.timestamp,
            reverse=True,
        )[:limit]

        return [
            {
                "id": alert.alert_id,
                "severity": alert.severity,
                "category": alert.category,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "details": alert.details,
            }
            for alert in recent_alerts
        ]

# Global TWS monitor instance
_tws_monitor: TWSMonitor | None = None

async def get_tws_monitor(
    tws_client: ITWSClient, tg: asyncio.TaskGroup | None = None
) -> TWSMonitor:
    """Get global TWS monitor instance.

    Args:
        tws_client: TWS client instance
        tg: Optional TaskGroup to start the monitor in
    Returns:
        TWSMonitor instance
    """
    global _tws_monitor
    if _tws_monitor is None:
        _tws_monitor = TWSMonitor(tws_client)
        _tws_monitor.start_monitoring(tg=tg)
    return _tws_monitor

async def shutdown_tws_monitor() -> None:
    """Shutdown global TWS monitor instance."""
    global _tws_monitor
    if _tws_monitor is not None:
        await _tws_monitor.stop_monitoring()
        _tws_monitor = None

class TWSMonitorInterface:
    """Interface to provide synchronous access to the TWS monitor."""

    def get_performance_report(self) -> dict[str, Any]:
        """Get performance report. Requires async initialization."""
        if _tws_monitor is None:
            # Return a default/empty report if monitor is not initialized
            return {
                "current_metrics": {
                    "api_response_time_ms": 0.0,
                    "cache_hit_ratio": 0.0,
                    "llm_calls_today": 0,
                    "llm_cost_today": 0.0,
                    "memory_usage_mb": 0.0,
                    "uptime_seconds": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "alerts": [],
                "summary": {"total_alerts": 0, "critical_alerts": 0, "high_alerts": 0},
            }
        return _tws_monitor.get_performance_report()

    def get_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent alerts. Requires async initialization."""
        if _tws_monitor is None:
            return []
        return _tws_monitor.get_alerts(limit)

# Global tws_monitor variable that provides synchronous access
_tws_monitor_instance: TWSMonitorInterface | None = None

class _LazyTWSMonitorInterface:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance: TWSMonitorInterface | None = None

    def get_instance(self) -> TWSMonitorInterface:
        if self._instance is None:
            self._instance = TWSMonitorInterface()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)

tws_monitor = _LazyTWSMonitorInterface()

def get_tws_monitor_sync() -> TWSMonitorInterface:
    """Return the singleton instance (preferred over using the proxy directly)."""
    return tws_monitor.get_instance()
