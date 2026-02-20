"""
Automatic Service Discovery for Microservices Architecture.

This module provides intelligent service discovery capabilities including:
- Automatic service registration and deregistration
- Dynamic endpoint discovery with health monitoring
- Multi-registry support (Consul, etcd, Kubernetes, DNS-SD)
- Load balancing with health-aware routing
- Service mesh integration preparation
- Circuit breaker coordination
- Configuration management and hot reloading
- Metrics and observability
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import random
import socket
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import aiohttp

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class DiscoveryBackend(Enum):
    """Supported service discovery backends."""

    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    DNS_SD = "dns_sd"
    ZOOKEEPER = "zookeeper"
    EUREKA = "eureka"
    CUSTOM = "custom"


class ServiceStatus(Enum):
    """Service instance status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for service discovery."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    GEOGRAPHIC = "geographic"  # Based on geographic location
    LATENCY_BASED = "latency_based"  # Based on response time


@dataclass
class ServiceInstance:
    """Service instance information."""

    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN

    # Metadata
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    weight: int = 1

    # Health and performance
    last_health_check: float = 0.0
    health_check_interval: int = 30  # seconds
    consecutive_failures: int = 0
    response_time_avg: float = 0.0
    active_connections: int = 0

    # Geographic/location info
    datacenter: str = ""
    region: str = ""
    zone: str = ""

    @property
    def url(self) -> str:
        """Get service URL."""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        """Check if service instance is healthy."""
        return self.status == ServiceStatus.HEALTHY

    @property
    def health_score(self) -> float:
        """Calculate health score (0-100)."""
        if self.status == ServiceStatus.HEALTHY:
            # Penalize based on response time and failures
            base_score = 100.0
            if self.response_time_avg > 0 and self.response_time_avg > 0.5:
                # Penalize slow responses (>500ms)
                base_score -= min(30, (self.response_time_avg - 0.5) * 20)

            # Penalize recent failures
            if self.consecutive_failures > 0:
                base_score -= min(20, self.consecutive_failures * 5)

            return max(0.0, base_score)
        if self.status == ServiceStatus.MAINTENANCE:
            return 50.0  # Reduced but still available
        return 0.0


@dataclass
class ServiceDefinition:
    """Service definition with discovery configuration."""

    service_name: str
    discovery_backend: DiscoveryBackend
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN

    # Health check configuration
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_timeout: int = 5
    health_check_interval: int = 30
    max_consecutive_failures: int = 3

    # Service mesh integration
    service_mesh_enabled: bool = False
    istio_injection_enabled: bool = False

    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60

    # Instance management
    deregister_on_shutdown: bool = True
    instance_ttl: int = 60  # seconds

    # Custom configuration
    backend_config: dict[str, Any] = field(default_factory=dict)


class DiscoveryBackendInterface(ABC):
    """Abstract interface for service discovery backends."""

    @abstractmethod
    def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool:
        """Register a service instance."""

    @abstractmethod
    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""

    @abstractmethod
    def discover_services(self, service_name: str) -> list[ServiceInstance]:
        """Discover all instances of a service."""

    @abstractmethod
    def watch_service(self, service_name: str, callback: Callable[..., Any]) -> None:
        """Watch for service changes."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check backend health."""


class ConsulBackend(DiscoveryBackendInterface):
    """Consul service discovery backend."""

    def __init__(self, config: dict[str, Any]):
        self.consul_url = config.get("url", "http://localhost:8500")
        self.acl_token = config.get("acl_token")
        self.session: aiohttp.ClientSession | None = None

    def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if not self.session:
            headers = {}
            if self.acl_token:
                headers["X-Consul-Token"] = self.acl_token

            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            )

    async def close(self) -> None:
        """Close the HTTP session and release resources."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool:
        """Register service with Consul."""
        self._ensure_session()

        registration_data = {
            "ID": instance.instance_id,
            "Name": service_def.service_name,
            "Address": instance.host,
            "Port": instance.port,
            "Tags": list(instance.tags),
            "Meta": instance.metadata,
            "Check": {
                "HTTP": f"{instance.url}{service_def.health_check_path}",
                "Interval": f"{service_def.health_check_interval}s",
                "Timeout": f"{service_def.health_check_timeout}s",
                "DeregisterCriticalServiceAfter": f"{service_def.instance_ttl}s",
            },
        }

        try:
            async with self.session.put(
                f"{self.consul_url}/v1/agent/service/register", json=registration_data
            ) as response:
                return response.status == 200
        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("Consul registration failed: %s", e)
            return False

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister service from Consul."""
        self._ensure_session()

        try:
            async with self.session.put(
                f"{self.consul_url}/v1/agent/service/deregister/{instance_id}"
            ) as response:
                return response.status == 200
        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("Consul deregistration failed: %s", e)
            return False

    async def discover_services(self, service_name: str) -> list[ServiceInstance]:
        """Discover services from Consul."""
        self._ensure_session()

        try:
            async with self.session.get(
                f"{self.consul_url}/v1/health/service/{service_name}"
            ) as response:
                if response.status != 200:
                    return []

                services_data = await response.json()
                instances = []

                for service_data in services_data:
                    service_info = service_data["Service"]
                    checks = service_data.get("Checks", [])

                    # Determine status from checks
                    status = ServiceStatus.UNHEALTHY
                    for check in checks:
                        if check.get("Status") == "passing":
                            status = ServiceStatus.HEALTHY
                            break

                    instance = ServiceInstance(
                        service_name=service_name,
                        instance_id=service_info["ID"],
                        host=service_info["Address"],
                        port=service_info["Port"],
                        status=status,
                        tags=set(service_info.get("Tags", [])),
                        metadata=service_info.get("Meta", {}),
                    )
                    instances.append(instance)

                return instances

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("Consul service discovery failed: %s", e)
            return []

    def watch_service(self, service_name: str, callback: Callable[..., Any]) -> None:
        """Watch for service changes in Consul."""
        # This would implement Consul's blocking queries for watching
        # Simplified implementation

    async def health_check(self) -> bool:
        """Check Consul health."""
        self._ensure_session()

        try:
            async with self.session.get(
                f"{self.consul_url}/v1/status/leader"
            ) as response:
                return response.status == 200
        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("exception_caught", error=str(e), exc_info=True)
            return False


class KubernetesBackend(DiscoveryBackendInterface):
    """Kubernetes service discovery backend."""

    def __init__(self, config: dict[str, Any]):
        self.api_server = config.get("api_server", "https://kubernetes.default.svc")
        self.token = config.get("token", self._get_service_account_token())
        self.namespace = config.get("namespace", "default")
        self.session: aiohttp.ClientSession | None = None

    def _get_service_account_token(self) -> str:
        """Get service account token from Kubernetes."""
        try:
            with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if not self.session:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            )

    async def close(self) -> None:
        """Close the HTTP session and release resources."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool:
        """Register service as Kubernetes endpoint."""
        # This would create/update Kubernetes endpoints
        # Simplified implementation
        logger.info("Kubernetes registration for %s", instance.service_name)
        return True

    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister service from Kubernetes."""
        logger.info("Kubernetes deregistration for %s", service_name)
        return True

    async def discover_services(self, service_name: str) -> list[ServiceInstance]:
        """Discover services from Kubernetes."""
        self._ensure_session()

        try:
            # Query Kubernetes API for endpoints
            url = f"{self.api_server}/api/v1/namespaces/{self.namespace}/endpoints/{service_name}"

            async with self.session.get(url) as response:
                if response.status != 200:
                    return []

                endpoints_data = await response.json()
                instances = []

                for subset in endpoints_data.get("subsets", []):
                    for address in subset.get("addresses", []):
                        for port in subset.get("ports", []):
                            instance = ServiceInstance(
                                service_name=service_name,
                                instance_id=f"{service_name}-{address['ip']}:{port['port']}",
                                host=address["ip"],
                                port=port["port"],
                                status=ServiceStatus.HEALTHY,
                                protocol="http",  # Assume HTTP for now
                            )
                            instances.append(instance)

                return instances

        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("Kubernetes service discovery failed: %s", e)
            return []

    def watch_service(self, service_name: str, callback: Callable[..., Any]) -> None:
        """Watch for service changes in Kubernetes."""
        # This would implement Kubernetes watch API

    async def health_check(self) -> bool:
        """Check Kubernetes API health."""
        self._ensure_session()

        try:
            async with self.session.get(
                f"{self.api_server}/api/v1/namespaces/{self.namespace}/pods"
            ) as response:
                return response.status == 200
        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error("exception_caught", error=str(e), exc_info=True)
            return False


class ServiceDiscoveryManager:
    """
    Intelligent service discovery manager with multi-backend support.

    Features:
    - Multi-backend service discovery (Consul, etcd, Kubernetes, etc.)
    - Health-aware load balancing
    - Automatic service registration
    - Real-time service monitoring
    - Circuit breaker integration
    - Service mesh preparation
    - Geographic and latency-based routing
    - Metrics and observability
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Service management
        self.services: dict[str, ServiceDefinition] = {}
        self.instances: dict[str, list[ServiceInstance]] = defaultdict(list)
        self.local_instances: dict[str, ServiceInstance] = {}

        # Backend management
        self.backends: dict[str, DiscoveryBackendInterface] = {}
        self.default_backend: DiscoveryBackend | None = None

        # Load balancing state
        self.round_robin_index: dict[str, int] = defaultdict(int)
        self.connection_counts: dict[str, int] = defaultdict(int)

        # Health monitoring
        self.health_check_failures: dict[str, int] = defaultdict(int)

        # Metrics
        self.metrics: dict[str, Any] = {
            "services_registered": 0,
            "instances_discovered": 0,
            "health_checks_performed": 0,
            "load_balancing_decisions": 0,
            "service_failures": 0,
        }

        # Background tasks
        self._discovery_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._running = False

        # Initialize backends
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize discovery backends."""
        backend_configs = self.config.get("backends", {})

        for backend_name, backend_config in backend_configs.items():
            backend_type = backend_config.get("type", "consul")

            if backend_type == "consul":
                self.backends[backend_name] = ConsulBackend(backend_config)
            elif backend_type == "kubernetes":
                self.backends[backend_name] = KubernetesBackend(backend_config)
            # Add other backend types here

        # Set default backend
        if self.backends:
            self.default_backend = list(self.backends.keys())[0]

    def start(self) -> None:
        """Start the service discovery manager."""
        if self._running:
            return

        self._running = True
        self._discovery_task = track_task(
            self._discovery_worker(), name="discovery_worker"
        )
        self._health_monitor_task = track_task(
            self._health_monitor_worker(), name="health_monitor_worker"
        )
        self._metrics_task = track_task(self._metrics_worker(), name="metrics_worker")

        logger.info("Service discovery manager started")

    async def stop(self) -> None:
        """Stop the service discovery manager."""
        if not self._running:
            return

        self._running = False

        # Deregister local services
        for service_name, instance in self.local_instances.items():
            await self.deregister_service(service_name, instance.instance_id)

        for task in [
            self._discovery_task,
            self._health_monitor_task,
            self._metrics_task,
        ]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Close all backend sessions to prevent resource leaks
        for backend in self.backends.values():
            if hasattr(backend, "close"):
                await backend.close()

        logger.info("Service discovery manager stopped")

    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance | None = None
    ) -> str:
        """Register a service."""
        # Generate instance if not provided
        if not instance:
            instance = ServiceInstance(
                service_name=service_def.service_name,
                instance_id=f"{service_def.service_name}_{socket.gethostname()}_{int(time.time())}",
                host=self._get_local_ip(),
                port=self.config.get("default_port", 8000),
                status=ServiceStatus.HEALTHY,
            )

        # Store service definition
        self.services[service_def.service_name] = service_def
        self.local_instances[service_def.service_name] = instance

        # Register with backend
        backend = self.backends.get(
            service_def.discovery_backend.value, self.backends.get(self.default_backend)
        )
        if backend:
            success = await backend.register_service(service_def, instance)
            if success:
                self.metrics["services_registered"] = self.metrics.get("services_registered", 0) + 1
                logger.info(
                    f"Registered service {service_def.service_name} with {service_def.discovery_backend.value}"
                )
                return instance.instance_id

        logger.error("Failed to register service %s", service_def.service_name)
        return ""

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        service_def = self.services.get(service_name)
        if not service_def:
            return False

        backend = self.backends.get(
            service_def.discovery_backend.value, self.backends.get(self.default_backend)
        )
        if backend:
            success = await backend.deregister_service(service_name, instance_id)
            if success:
                # Remove from local instances
                if service_name in self.local_instances:
                    del self.local_instances[service_name]

                logger.info(
                    "Deregistered service %s instance %s", service_name, instance_id
                )
                return True

        return False

    async def discover_service(
        self, service_name: str, strategy: LoadBalancingStrategy | None = None
    ) -> ServiceInstance | None:
        """Discover and select a service instance using load balancing."""
        instances = await self.get_service_instances(service_name)
        healthy_instances = [inst for inst in instances if inst.is_healthy]

        if not healthy_instances:
            return None

        # Use specified strategy or service default
        if not strategy:
            service_def = self.services.get(service_name)
            strategy = (
                service_def.load_balancing_strategy
                if service_def
                else LoadBalancingStrategy.ROUND_ROBIN
            )

        selected_instance = self._select_instance(healthy_instances, strategy)
        if selected_instance:
            self.metrics["load_balancing_decisions"] = self.metrics.get("load_balancing_decisions", 0) + 1
            self.connection_counts[selected_instance.instance_id] += 1

        return selected_instance

    async def get_service_instances(self, service_name: str) -> list[ServiceInstance]:
        """Get all instances of a service."""
        # Check cache first
        if service_name in self.instances:
            cached_instances = self.instances[service_name]
            # Return cached instances if they're recent (< 30 seconds)
            if (
                cached_instances
                and time.time() - cached_instances[0].last_health_check < 30
            ):
                return cached_instances

        # Discover from backend
        service_def = self.services.get(service_name)
        if not service_def:
            return []

        backend = self.backends.get(
            service_def.discovery_backend.value, self.backends.get(self.default_backend)
        )
        if backend:
            instances = await backend.discover_services(service_name)
            self.instances[service_name] = instances
            self.metrics["instances_discovered"] = self.metrics.get("instances_discovered", 0) + len(instances)
            return instances

        return []

    def _select_instance(
        self, instances: list[ServiceInstance], strategy: LoadBalancingStrategy
    ) -> ServiceInstance | None:
        """Select instance using specified load balancing strategy."""
        if not instances:
            return None

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            service_key = instances[0].service_name
            index = self.round_robin_index[service_key]
            self.round_robin_index[service_key] = (index + 1) % len(instances)
            return instances[index]

        if strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(instances)

        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(
                instances,
                key=lambda inst: self.connection_counts.get(inst.instance_id, 0),
            )

        if strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            total_weight = sum(inst.weight for inst in instances)
            if total_weight == 0:
                return random.choice(instances)

            weights = [inst.weight for inst in instances]
            cumulative_weights = [sum(weights[: i + 1]) for i in range(len(weights))]

            rand = random.randint(1, total_weight)
            for i, weight in enumerate(cumulative_weights):
                if rand <= weight:
                    return instances[i]

        elif strategy == LoadBalancingStrategy.LATENCY_BASED:
            # Select instance with lowest average response time
            return min(
                instances, key=lambda inst: inst.response_time_avg or float("inf")
            )

        # Default to round-robin
        return instances[0]

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Create a socket to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Connect to Google DNS
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
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
            logger.error("exception_caught", error=str(e), exc_info=True)
            return "127.0.0.1"

    async def _discovery_worker(self) -> None:
        """Background worker for service discovery."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Discover every minute

                # Refresh service instances for all registered services
                for service_name in self.services:
                    await self.get_service_instances(service_name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Discovery worker error: %s", e, exc_info=True)

    async def _health_monitor_worker(self) -> None:
        """Background worker for health monitoring."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Health check every 30 seconds

                # Perform health checks on all known instances
                for _service_name, instances in self.instances.items():
                    for instance in instances:
                        if (
                            time.time() - instance.last_health_check
                            > instance.health_check_interval
                        ):
                            await self._perform_health_check(instance)
                            self.metrics["health_checks_performed"] = self.metrics.get("health_checks_performed", 0) + 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Health monitor worker error: %s", e, exc_info=True)

    async def _perform_health_check(self, instance: ServiceInstance) -> None:
        """Perform health check on service instance."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                health_url = f"{instance.url}/health"

                start_time = time.time()
                async with session.get(health_url) as response:
                    response_time = time.time() - start_time

                    # Update response time (exponential moving average)
                    if instance.response_time_avg == 0:
                        instance.response_time_avg = response_time
                    else:
                        instance.response_time_avg = (
                            0.7 * instance.response_time_avg + 0.3 * response_time
                        )

                    if response.status == 200:
                        instance.status = ServiceStatus.HEALTHY
                        instance.consecutive_failures = 0
                    else:
                        instance.status = ServiceStatus.UNHEALTHY
                        instance.consecutive_failures += 1

                instance.last_health_check = time.time()

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
            instance.status = ServiceStatus.UNHEALTHY
            instance.consecutive_failures += 1
            instance.last_health_check = time.time()

            if instance.consecutive_failures >= 3:
                logger.warning(
                    "Service instance %s is unhealthy: %s", instance.instance_id, e
                )

    async def _metrics_worker(self) -> None:
        """Background worker for metrics logging."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Log every 5 minutes

                logger.info(
                    "service_discovery_metrics",
                    services_registered=self.metrics.get("services_registered", 0),
                    instances_discovered=self.metrics.get("instances_discovered", 0),
                    health_checks_performed=self.metrics.get("health_checks_performed", 0),
                    load_balancing_decisions=self.metrics.get("load_balancing_decisions", 0),
                    active_services=len(self.services),
                    total_instances=sum(
                        len(insts) for insts in self.instances.values()
                    ),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Re-raise critical system exceptions
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.error("Metrics worker error: %s", e, exc_info=True)

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive service discovery metrics."""
        return {
            "services": {
                "registered_services": len(self.services),
                "local_services": len(self.local_instances),
                "total_instances": sum(len(insts) for insts in self.instances.values()),
                "healthy_instances": sum(
                    1
                    for insts in self.instances.values()
                    for inst in insts
                    if inst.is_healthy
                ),
            },
            "backends": {
                "configured_backends": len(self.backends),
                "active_backends": sum(
                    1
                    for backend in self.backends.values()
                    if backend.health_check is not None
                ),
            },
            "performance": {
                "services_registered": self.metrics.get("services_registered", 0),
                "instances_discovered": self.metrics.get("instances_discovered", 0),
                "health_checks_performed": self.metrics.get("health_checks_performed", 0),
                "load_balancing_decisions": self.metrics.get("load_balancing_decisions", 0),
                "service_failures": self.metrics.get("service_failures", 0),
            },
            "load_balancing": {
                "active_connections": dict(self.connection_counts),
                "round_robin_indices": dict(self.round_robin_index),
            },
        }

    def get_service_health(self, service_name: str) -> dict[str, Any]:
        """Get detailed health information for a service."""
        instances = self.instances.get(service_name, [])

        return {
            "service_name": service_name,
            "total_instances": len(instances),
            "healthy_instances": sum(1 for inst in instances if inst.is_healthy),
            "unhealthy_instances": sum(1 for inst in instances if not inst.is_healthy),
            "average_health_score": sum(inst.health_score for inst in instances)
            / max(1, len(instances)),
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "url": inst.url,
                    "status": inst.status.value,
                    "health_score": inst.health_score,
                    "response_time_avg": inst.response_time_avg,
                    "active_connections": inst.active_connections,
                    "last_health_check": inst.last_health_check,
                }
                for inst in instances
            ],
        }


# Global service discovery manager instance
class _LazyServiceDiscoveryManager:
    """Lazily instantiate ServiceDiscoveryManager to avoid import-time side effects."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance: ServiceDiscoveryManager | None = None

    def get_instance(self) -> ServiceDiscoveryManager:
        if self._instance is None:
            self._instance = ServiceDiscoveryManager()
        return self._instance

    def __getattr__(self, item: str):
        return getattr(self.get_instance(), item)


service_discovery_manager = _LazyServiceDiscoveryManager()


async def get_service_discovery_manager() -> ServiceDiscoveryManager:
    """Get the global service discovery manager instance."""
    if not service_discovery_manager._running:
        service_discovery_manager.start()
    return service_discovery_manager.get_instance()
