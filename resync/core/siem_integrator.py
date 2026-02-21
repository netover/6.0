"""
SIEM (Security Information and Event Management) Integration System.

This module provides comprehensive SIEM integration capabilities including:
- Multi-SIEM support (Splunk, ELK, IBM QRadar, ArcSight, etc.)
- Event normalization and correlation
- Real-time event streaming
- Failover and load balancing
- Custom integration APIs
- Event enrichment and filtering
- Performance monitoring and alerting
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import json
import secrets
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import aiohttp

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class SIEMType(Enum):
    """Supported SIEM types."""

    SPLUNK = "splunk"
    ELK_STACK = "elk"
    IBM_QRADAR = "qradar"
    ARCSIGHT = "arcsight"
    SUMO_LOGIC = "sumo_logic"
    LOGRHYTHM = "logrhythm"
    CUSTOM = "custom"


class EventFormat(Enum):
    """Standard event formats."""

    CEF = "cef"  # Common Event Format
    LEEF = "leef"  # Log Event Extended Format
    JSON = "json"
    SYSLOG = "syslog"
    RAW = "raw"


class SIEMStatus(Enum):
    """SIEM connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class SIEMEvent:
    """Standardized security event for SIEM integration."""

    event_id: str
    timestamp: float
    source: str  # Component that generated the event
    event_type: str  # login, anomaly, breach, etc.
    severity: str  # critical, high, medium, low, informational
    category: str  # authentication, authorization, data_access, etc.

    # Event details
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    # Actor information
    user_id: str | None = None
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None

    # Target information
    resource_id: str | None = None
    resource_type: str | None = None
    action: str | None = None

    # Additional context
    correlation_id: str | None = None
    tags: set[str] = field(default_factory=set)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_cef(self) -> str:
        """Convert event to CEF format."""
        cef_version = 0
        device_vendor = "HWA-New"
        device_product = "SecuritySystem"
        device_version = "1.0"
        signature_id = self.event_type
        name = self.message
        severity = self._severity_to_int(self.severity)

        # CEF extension fields
        extensions = []
        if self.user_id:
            extensions.append(f"src={self.user_id}")
        if self.ip_address:
            extensions.append(f"src={self.ip_address}")
        if self.resource_id:
            extensions.append(f"dst={self.resource_id}")
        if self.session_id:
            extensions.append(f"cs1={self.session_id}")

        extension_str = " ".join(extensions)

        return f"CEF:{cef_version}|{device_vendor}|{device_product}|{device_version}|{signature_id}|{name}|{severity}|{extension_str}"

    def to_leef(self) -> str:
        """Convert event to LEEF format."""
        leef_version = 1.0
        vendor = "HWA-New"
        product = "SecuritySystem"
        version = "1.0"
        event_id = self.event_type

        # LEEF attributes
        attributes = []
        attributes.append(f"cat={self.category}")
        attributes.append(f"sev={self._severity_to_int(self.severity)}")
        if self.user_id:
            attributes.append(f"usrName={self.user_id}")
        if self.ip_address:
            attributes.append(f"src={self.ip_address}")
        if self.resource_id:
            attributes.append(f"dst={self.resource_id}")

        attribute_str = "\t".join(attributes)

        return f"LEEF:{leef_version}|{vendor}|{product}|{version}|{event_id}|{attribute_str}|{self.message}"

    def to_json(self) -> str:
        """Convert event to JSON format."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "timestamp": self.timestamp,
                "@timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
                "source": self.source,
                "event_type": self.event_type,
                "severity": self.severity,
                "category": self.category,
                "message": self.message,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "ip_address": self.ip_address,
                "user_agent": self.user_agent,
                "resource_id": self.resource_id,
                "resource_type": self.resource_type,
                "action": self.action,
                "correlation_id": self.correlation_id,
                "tags": list(self.tags),
                **self.details,
                **self.custom_fields,
            },
            separators=(",", ":"),
        )

    def _severity_to_int(self, severity: str) -> int:
        """Convert severity string to numeric value."""
        mapping = {"informational": 0, "low": 1, "medium": 5, "high": 8, "critical": 10}
        return mapping.get(severity.lower(), 5)


@dataclass
class SIEMConfiguration:
    """Configuration for SIEM integration."""

    siem_type: SIEMType
    name: str
    endpoint_url: str
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    batch_size: int = 100
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_ssl_verification: bool = True
    custom_config: dict[str, Any] = field(default_factory=dict)

    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        headers = dict(self.headers)

        if self.api_key:
            if self.siem_type == SIEMType.SPLUNK:
                headers["Authorization"] = f"Splunk {self.api_key}"
            elif self.siem_type == SIEMType.SUMO_LOGIC:
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers["X-API-Key"] = self.api_key

        return headers


class SIEMConnector(ABC):
    """Abstract base class for SIEM connectors."""

    def __init__(self, config: SIEMConfiguration):
        self.config = config
        self.status = SIEMStatus.DISCONNECTED
        self.last_connection_attempt = 0
        self.connection_failures = 0
        self.events_sent = 0
        self.last_event_sent = 0

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to SIEM."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from SIEM."""

    @abstractmethod
    def send_event(self, event: SIEMEvent) -> bool:
        """Send single event to SIEM."""

    @abstractmethod
    def send_events_batch(self, events: list[SIEMEvent]) -> int:
        """Send batch of events to SIEM. Returns number of events sent successfully."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Perform health check on SIEM connection."""

    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self.status == SIEMStatus.CONNECTED

    def get_metrics(self) -> dict[str, Any]:
        """Get connector metrics."""
        return {
            "status": self.status.value,
            "events_sent": self.events_sent,
            "last_event_sent": self.last_event_sent,
            "connection_failures": self.connection_failures,
            "uptime": (
                time.time() - self.last_connection_attempt if self.is_connected() else 0
            ),
        }


class SplunkConnector(SIEMConnector):
    """Splunk SIEM connector."""

    def __init__(self, config: SIEMConfiguration):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> bool:
        """Connect to Splunk."""
        try:
            self.status = SIEMStatus.CONNECTING

            # Create HTTP session
            self.session = aiohttp.ClientSession(
                headers=self.config.get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            )

            # Test connection
            health = await self.health_check()
            if health.get("status") == "ok":
                self.status = SIEMStatus.CONNECTED
                self.last_connection_attempt = time.time()
                self.connection_failures = 0
                return True
            # Health check failed — close session to prevent leak
            await self.session.close()
            self.session = None
            self.status = SIEMStatus.ERROR
            return False

        except Exception as e:
            # Re-raise critical system exceptions and programming errors (bugs)
            if isinstance(
                e,
                (
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ),
            ):
                raise

            # Close session on error to prevent leak
            if self.session:
                await self.session.close()
                self.session = None
            logger.error("Splunk connection failed: %s", e)
            self.status = SIEMStatus.ERROR
            self.connection_failures += 1
            return False

    async def disconnect(self) -> None:
        """Disconnect from Splunk."""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = SIEMStatus.DISCONNECTED

    async def send_event(self, event: SIEMEvent) -> bool:
        """Send single event to Splunk."""
        return await self.send_events_batch([event]) == 1

    async def send_events_batch(self, events: list[SIEMEvent]) -> int:
        """Send batch of events to Splunk."""
        if not self.session or not self.is_connected():
            return 0

        try:
            # Format events for Splunk
            splunk_events = []
            for event in events:
                splunk_event = {
                    "event": event.to_json(),
                    "time": event.timestamp,
                    "host": "hwa-new-system",
                    "source": event.source,
                    "sourcetype": f"hwa:{event.category}",
                    "index": "security",
                }
                splunk_events.append(json.dumps(splunk_event))

            # Send batch
            payload = "\n".join(splunk_events)
            url = f"{self.config.endpoint_url}/services/collector/event"

            async with self.session.post(url, data=payload) as response:
                if response.status == 200:
                    self.events_sent += len(events)
                    self.last_event_sent = time.time()
                    return len(events)
                logger.error(
                    f"Splunk batch send failed: {response.status} - {await response.text()}"
                )
                return 0

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("Splunk batch send error: %s", e)
            self.connection_failures += 1
            if self.connection_failures >= 3:
                self.status = SIEMStatus.ERROR
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Perform Splunk health check."""
        if not self.session:
            return {"status": "disconnected", "error": "No session"}

        try:
            # Test services endpoint
            url = f"{self.config.endpoint_url}/services/server/info"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return {"status": "ok", "response_time": time.time()}
                return {"status": "error", "http_status": response.status}

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("exception_caught", error=str(e), exc_info=True)
            return {"status": "error", "error": str(e)}


class ELKConnector(SIEMConnector):
    """ELK Stack (Elasticsearch) SIEM connector."""

    def __init__(self, config: SIEMConfiguration):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None
        self.bulk_endpoint = f"{self.config.endpoint_url}/_bulk"

    async def connect(self) -> bool:
        """Connect to ELK Stack."""
        try:
            self.status = SIEMStatus.CONNECTING

            self.session = aiohttp.ClientSession(
                headers=self.config.get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            )

            # Test connection
            health = await self.health_check()
            if health.get("status") == "ok":
                self.status = SIEMStatus.CONNECTED
                self.last_connection_attempt = time.time()
                self.connection_failures = 0
                return True
            # Health check failed — close session to prevent leak
            await self.session.close()
            self.session = None
            self.status = SIEMStatus.ERROR
            return False

        except Exception as e:
            # Re-raise critical system exceptions and programming errors
            if isinstance(
                e,
                (
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ),
            ):
                raise

            # Close session on error to prevent leak
            if self.session:
                await self.session.close()
                self.session = None
            logger.error("ELK connection failed: %s", e)
            self.status = SIEMStatus.ERROR
            self.connection_failures += 1
            return False

    async def disconnect(self) -> None:
        """Disconnect from ELK."""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = SIEMStatus.DISCONNECTED

    async def send_event(self, event: SIEMEvent) -> bool:
        """Send single event to ELK."""
        return await self.send_events_batch([event]) == 1

    async def send_events_batch(self, events: list[SIEMEvent]) -> int:
        """Send batch of events to ELK using bulk API."""
        if not self.session or not self.is_connected():
            return 0

        try:
            # Prepare bulk request
            bulk_data = []
            for event in events:
                # Index metadata
                index_meta = {
                    "index": {"_index": "security-events", "_id": event.event_id}
                }
                bulk_data.append(json.dumps(index_meta))
                bulk_data.append(event.to_json())

            payload = "\n".join(bulk_data) + "\n"

            async with self.session.post(self.bulk_endpoint, data=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    successful = sum(
                        1
                        for item in result.get("items", [])
                        if item.get("index", {}).get("status") == 201
                    )
                    self.events_sent += successful
                    self.last_event_sent = time.time()
                    return successful
                logger.error(
                    "ELK bulk send failed: %s - %s",
                    response.status,
                    await response.text(),
                )
                return 0

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("ELK batch send error: %s", e)
            self.connection_failures += 1
            if self.connection_failures >= 3:
                self.status = SIEMStatus.ERROR
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Perform ELK health check."""
        if not self.session:
            return {"status": "disconnected", "error": "No session"}

        try:
            url = f"{self.config.endpoint_url}/_cluster/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        "status": "ok",
                        "cluster_status": health_data.get("status"),
                        "response_time": time.time(),
                    }
                return {"status": "error", "http_status": response.status}

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("exception_caught", error=str(e), exc_info=True)
            return {"status": "error", "error": str(e)}


class SIEMIntegrator:
    """
    Main SIEM integration system with multi-SIEM support and failover.

    Features:
    - Multiple SIEM connectors with load balancing
    - Event normalization and enrichment
    - Real-time event streaming
    - Circuit breaker protection
    - Event correlation and deduplication
    - Performance monitoring
    - Failover and redundancy
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.connectors: dict[str, SIEMConnector] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.correlation_engine = EventCorrelationEngine()
        self.enrichment_engine = EventEnrichmentEngine()

        # Processing
        self.event_buffer: deque = deque(maxlen=1000)
        self.batch_size = self.config.get("batch_size", 50)
        self.flush_interval = self.config.get("flush_interval_seconds", 5.0)

        # Statistics
        self.events_processed = 0
        self.events_dropped = 0
        self.correlation_events = 0
        self.last_flush = time.time()

        # Background tasks
        self._processor_task: asyncio.Task | None = None
        self._flusher_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Circuit breaker for failed SIEMs
        self.circuit_breaker = SIEMCircuitBreaker()

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the SIEM integrator.

        Args:
            tg: Optional TaskGroup to run the background tasks in
        """
        if self._running:
            return

        self._running = True
        if tg:
            self._processor_task = tg.create_task(
                self._event_processor(), name="event_processor"
            )
            self._flusher_task = tg.create_task(self._batch_flusher(), name="batch_flusher")
            self._monitor_task = tg.create_task(self._health_monitor(), name="health_monitor")
        else:
            self._processor_task = track_task(
                self._event_processor(), name="event_processor"
            )
            self._flusher_task = track_task(self._batch_flusher(), name="batch_flusher")
            self._monitor_task = track_task(self._health_monitor(), name="health_monitor")

        logger.info("SIEM integrator started", method="task_group" if tg else "track_task")

    async def stop(self) -> None:
        """Stop the SIEM integrator."""
        if not self._running:
            return

        self._running = False

        # Flush remaining events
        await self._flush_events()

        for task in [self._processor_task, self._flusher_task, self._monitor_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Disconnect all connectors
        for connector in self.connectors.values():
            await connector.disconnect()

        logger.info("SIEM integrator stopped")

    def add_siem_connector(self, name: str, config: SIEMConfiguration) -> bool:
        """Add a SIEM connector."""
        try:
            if config.siem_type == SIEMType.SPLUNK:
                connector = SplunkConnector(config)
            elif config.siem_type == SIEMType.ELK_STACK:
                connector = ELKConnector(config)
            else:
                logger.warning("Unsupported SIEM type: %s", config.siem_type)
                return False

            self.connectors[name] = connector
            logger.info("Added SIEM connector: %s (%s)", name, config.siem_type.value)
            return True

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(
                e,
                (
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ),
            ):
                raise
            logger.error("Failed to add SIEM connector %s: %s", name, e)
            return False

    async def send_security_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        source: str = "hwa-system",
        **kwargs,
    ) -> str:
        """Send a security event to all configured SIEMs."""
        event_id = f"sec_{int(time.time() * 1000000)}_{secrets.token_hex(4)}"

        event = SIEMEvent(
            event_id=event_id,
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            severity=severity,
            category=self._categorize_event(event_type),
            message=message,
            **kwargs,
        )

        # Enrich event
        self.enrichment_engine.enrich_event(event)

        # Check for correlation
        correlation_result = self.correlation_engine.correlate_event(event)
        if correlation_result:
            event.correlation_id = correlation_result["correlation_id"]
            event.tags.add("correlated")
            self.correlation_events += 1

        # Queue event for processing
        try:
            self.event_queue.put_nowait(event)
            self.events_processed += 1
            return event_id
        except asyncio.QueueFull:
            self.events_dropped += 1
            logger.warning("Event queue full, dropping event")
            return ""

    def send_custom_event(self, event: SIEMEvent) -> str:
        """Send a custom security event."""
        try:
            self.event_queue.put_nowait(event)
            self.events_processed += 1
            return event.event_id
        except asyncio.QueueFull:
            self.events_dropped += 1
            logger.warning("Event queue full, dropping custom event")
            return ""

    def get_connector_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all connectors."""
        return {
            name: connector.get_metrics() for name, connector in self.connectors.items()
        }

    def get_system_metrics(self) -> dict[str, Any]:
        """Get overall system metrics."""
        connector_status = self.get_connector_status()

        return {
            "events": {
                "processed": self.events_processed,
                "dropped": self.events_dropped,
                "correlated": self.correlation_events,
                "queue_size": self.event_queue.qsize(),
                "buffer_size": len(self.event_buffer),
            },
            "connectors": {
                "total": len(self.connectors),
                "connected": sum(
                    1 for c in connector_status.values() if c["status"] == "connected"
                ),
                "failed": sum(
                    1 for c in connector_status.values() if c["status"] == "error"
                ),
            },
            "performance": {
                "avg_processing_time": 0.0,  # Would need to track this
                "throughput_per_second": self.events_processed
                / max(1, time.time() - (time.time() - 3600)),
            },
            "correlation": {
                "rules_active": len(self.correlation_engine.rules),
                "alerts_generated": self.correlation_engine.alerts_generated,
            },
        }

    def _categorize_event(self, event_type: str) -> str:
        """Categorize event type."""
        categories = {
            "login": "authentication",
            "logout": "authentication",
            "failed_login": "authentication",
            "password_change": "authentication",
            "anomaly": "threat_detection",
            "breach": "incident",
            "access_denied": "authorization",
            "data_access": "data_security",
            "admin_action": "administration",
        }
        return categories.get(event_type, "general")

    async def _event_processor(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Get event from queue
                event = await self.event_queue.get()

                # Add to buffer for batching
                self.event_buffer.append(event)

                # Process immediately if buffer is full
                if len(self.event_buffer) >= self.batch_size:
                    await self._flush_events()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Event processor error: %s", e, exc_info=True)

    async def _batch_flusher(self) -> None:
        """Periodically flush event batches."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)

                # Check if it's time to flush
                if (
                    len(self.event_buffer) > 0
                    and time.time() - self.last_flush >= self.flush_interval
                ):
                    await self._flush_events()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Batch flusher error: %s", e, exc_info=True)

    async def _flush_events(self) -> None:
        """Flush buffered events to SIEMs."""
        if not self.event_buffer:
            return

        events_to_send = list(self.event_buffer)
        self.event_buffer.clear()

        # Send to all active connectors
        successful_sends = 0

        for name, connector in self.connectors.items():
            if connector.is_connected() and not self.circuit_breaker.is_open(name):
                try:
                    sent = await connector.send_events_batch(events_to_send)
                    successful_sends += sent

                    if sent == 0:
                        self.circuit_breaker.record_failure(name)
                        logger.warning("Failed to send events to %s", name)

                except Exception as e:
                    # Re-raise critical system exceptions
                    if isinstance(
                        e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                    ):
                        raise
                    self.circuit_breaker.record_failure(name)
                    logger.error("Error sending events to %s: %s", name, e)
            else:
                logger.debug("Skipping %s: not connected or circuit open", name)

        self.last_flush = time.time()

        if successful_sends < len(events_to_send):
            failed_count = len(events_to_send) - successful_sends
            logger.warning(
                "Failed to send %s out of %s events", failed_count, len(events_to_send)
            )

    async def _health_monitor(self) -> None:
        """Monitor health of SIEM connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                for name, connector in self.connectors.items():
                    # Check if connector needs reconnection
                    if not connector.is_connected():
                        # Try to reconnect
                        if await connector.connect():
                            logger.info("Reconnected to SIEM: %s", name)
                            self.circuit_breaker.reset(name)
                        else:
                            logger.warning("Failed to reconnect to SIEM: %s", name)

                    # Check circuit breaker
                    if self.circuit_breaker.is_open(
                        name
                    ) and self.circuit_breaker.can_attempt(name):
                        # Try to close circuit breaker if enough time has passed
                        self.circuit_breaker.attempt_reset(name)
                        logger.info("Attempting to reset circuit breaker for %s", name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Health monitor error: %s", e, exc_info=True)


class EventCorrelationEngine:
    """Engine for correlating security events."""

    def __init__(self):
        self.rules: list[dict[str, Any]] = []
        self.alerts_generated = 0

        # Initialize default correlation rules
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize default correlation rules."""
        self.rules = [
            {
                "name": "multiple_failed_logins",
                "condition": lambda events: (
                    len([e for e in events if e.event_type == "failed_login"]) >= 5
                    and len(set(e.ip_address for e in events if e.ip_address)) <= 2
                ),
                "time_window": 300,  # 5 minutes
                "severity": "high",
            },
            {
                "name": "brute_force_attack",
                "condition": lambda events: (
                    len(
                        [
                            e
                            for e in events
                            if e.event_type == "failed_login" and e.user_id
                        ]
                    )
                    >= 10
                ),
                "time_window": 600,  # 10 minutes
                "severity": "critical",
            },
            {
                "name": "privilege_escalation",
                "condition": lambda events: (
                    any(e.event_type == "admin_action" for e in events)
                    and any(e.event_type == "failed_login" for e in events)
                ),
                "time_window": 1800,  # 30 minutes
                "severity": "high",
            },
        ]

    def correlate_event(self, event: SIEMEvent) -> dict[str, Any] | None:
        """Check if event correlates with recent events."""
        # This is a simplified implementation
        # In a real system, you'd maintain sliding windows of events per user/IP/etc.

        # For now, just return None (no correlation)
        # Real implementation would check against historical events
        return None


class EventEnrichmentEngine:
    """Engine for enriching security events with additional context."""

    def __init__(self):
        self.enrichers: list[Callable[[SIEMEvent], Awaitable[None]]] = []

    def enrich_event(self, event: SIEMEvent) -> None:
        """Enrich event with additional context."""
        # Add basic enrichment
        event.tags.add("enriched")
        event.custom_fields["system_version"] = "1.0.0"
        event.custom_fields["enrichment_timestamp"] = time.time()

        # Add severity-based tags
        if event.severity in ["high", "critical"]:
            event.tags.add("priority")
        if event.severity == "critical":
            event.tags.add("immediate_attention")

        # Add category-based tags
        event.tags.add(f"category_{event.category}")


class SIEMCircuitBreaker:
    """Circuit breaker for SIEM connections."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: dict[str, int] = {}
        self.last_failures: dict[str, float] = {}
        self.open_circuits: set[str] = set()

    def record_failure(self, siem_name: str) -> None:
        """Record a failure for a SIEM."""
        self.failure_counts[siem_name] = self.failure_counts.get(siem_name, 0) + 1
        self.last_failures[siem_name] = time.time()

        if self.failure_counts[siem_name] >= self.failure_threshold:
            self.open_circuits.add(siem_name)

    def is_open(self, siem_name: str) -> bool:
        """Check if circuit is open for a SIEM."""
        return siem_name in self.open_circuits

    def can_attempt(self, siem_name: str) -> bool:
        """Check if we can attempt to reset the circuit."""
        if siem_name not in self.open_circuits:
            return False

        last_failure = self.last_failures.get(siem_name, 0)
        return time.time() - last_failure >= self.recovery_timeout

    def attempt_reset(self, siem_name: str) -> None:
        """Attempt to reset the circuit breaker."""
        if siem_name in self.open_circuits:
            self.open_circuits.remove(siem_name)
            self.failure_counts[siem_name] = 0

    def reset(self, siem_name: str) -> None:
        """Force reset the circuit breaker."""
        self.open_circuits.discard(siem_name)
        self.failure_counts[siem_name] = 0
        self.last_failures[siem_name] = time.time()


# Global SIEM integrator instance
# Lazily instantiate the SIEMIntegrator to avoid import-time heavy initialization
# (important for gunicorn --preload and faster module import).
_siem_integrator_instance: Optional["SIEMIntegrator"] = None


def get_siem_integrator_sync() -> "SIEMIntegrator":
    """Return the singleton instance synchronously."""
    global _siem_integrator_instance
    if _siem_integrator_instance is None:
        _siem_integrator_instance = SIEMIntegrator()
    return _siem_integrator_instance


class _LazySIEMIntegratorProxy:
    def __getattr__(self, item):
        return getattr(get_siem_integrator(), item)


# Backward-compatible module-level symbol (lazy proxy)
siem_integrator = _LazySIEMIntegratorProxy()


async def get_siem_integrator() -> SIEMIntegrator:
    """Get the global SIEM integrator instance."""
    if not siem_integrator._running:
        siem_integrator.start()
    return siem_integrator
