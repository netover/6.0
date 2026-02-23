# pylint: disable=all
# mypy: no-rerun
"""
Service Discovery Manager — v7.2-prod (Resync) — atualizado (v7.1 + hardening)

Objetivos de produção:
- Lifecycle determinístico (start/stop corretos + cancelamento de tasks + aclose em httpx).
- Reuso de httpx.AsyncClient (pooling) com httpx.Limits.
- Prometheus com baixa cardinalidade por padrão; métricas por instância opcionais.
- OpenTelemetry: spans apenas em operações request-like; workers só logs/métricas.
- Least Connections seguro: borrow_instance como async context manager garante release.
- Circuit breaker mínimo: open_until por instância e filtragem no LB.
- GARBAGE COLLECTION: remove estado e séries por instância quando instâncias somem do backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import random
import re
import socket
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Coroutine

import httpx
import orjson
import structlog
from antidote import inject, injectable
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram, REGISTRY
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.networks import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from resync.core.task_tracker import track_task

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# =============================================================================
# Prometheus helpers (safe on reload/dev)
# =============================================================================


def _get_or_create_metric(factory, name: str, *args, **kwargs):
    try:
        return factory(name, *args, **kwargs)
    except ValueError:
        collectors = getattr(REGISTRY, "_names_to_collectors", {})
        existing = collectors.get(name)
        if existing is None:
            raise
        return existing


# Low-cardinality metrics (recommended for production)
_prom_registrations = _get_or_create_metric(
    Counter,
    "sdm_registrations_total",
    "Total de registros de serviço bem-sucedidos",
    ["backend", "service_name"],
)
_prom_deregistrations = _get_or_create_metric(
    Counter,
    "sdm_deregistrations_total",
    "Total de deregistros de serviço bem-sucedidos",
    ["backend", "service_name"],
)
_prom_discoveries = _get_or_create_metric(
    Counter,
    "sdm_discoveries_total",
    "Total de descobertas (cache hit/miss)",
    ["service_name", "source"],  # cache|backend
)
_prom_instances_discovered = _get_or_create_metric(
    Counter,
    "sdm_instances_discovered_total",
    "Total de instâncias retornadas pelos backends",
    ["service_name"],
)
_prom_health_checks = _get_or_create_metric(
    Counter,
    "sdm_health_checks_total",
    "Total de health checks executados",
    ["service_name", "result"],  # healthy|unhealthy|skipped|error
)
_prom_health_check_duration = _get_or_create_metric(
    Histogram,
    "sdm_health_check_duration_seconds",
    "Duração de health checks por serviço",
    ["service_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
_prom_lb_decisions = _get_or_create_metric(
    Counter,
    "sdm_lb_decisions_total",
    "Total de decisões de load balancing",
    ["service_name", "strategy"],
)
_prom_errors = _get_or_create_metric(
    Counter,
    "sdm_errors_total",
    "Erros por tipo",
    ["error_type"],
)

_prom_active_services = _get_or_create_metric(
    Gauge,
    "sdm_active_services",
    "Quantidade de serviços registrados no manager",
)
_prom_total_instances = _get_or_create_metric(
    Gauge,
    "sdm_total_instances",
    "Total de instâncias no cache (todos os serviços)",
)
_prom_service_instances = _get_or_create_metric(
    Gauge,
    "sdm_service_instances",
    "Instâncias no cache por serviço",
    ["service_name"],
)
_prom_service_healthy_instances = _get_or_create_metric(
    Gauge,
    "sdm_service_healthy_instances",
    "Instâncias saudáveis por serviço (considerando circuit breaker)",
    ["service_name"],
)
_prom_service_active_connections = _get_or_create_metric(
    Gauge,
    "sdm_service_active_connections",
    "Conexões ativas agregadas por serviço",
    ["service_name"],
)
_prom_service_circuit_open = _get_or_create_metric(
    Gauge,
    "sdm_service_circuit_open_instances",
    "Quantidade de instâncias com circuit breaker aberto por serviço",
    ["service_name"],
)

# Optional per-instance metrics (HIGH cardinality) – disabled by default
_prom_instance_active_connections = _get_or_create_metric(
    Gauge,
    "sdm_instance_active_connections",
    "Conexões ativas por instância (alta cardinalidade)",
    ["instance_id"],
)
_prom_instance_circuit_open = _get_or_create_metric(
    Gauge,
    "sdm_instance_circuit_open",
    "Circuit breaker aberto por instância (alta cardinalidade)",
    ["instance_id"],
)

# =============================================================================
# Enums
# =============================================================================


class DiscoveryBackend(str, Enum):
    CONSUL = "consul"
    KUBERNETES = "kubernetes"


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    LATENCY_BASED = "latency_based"
    GEOGRAPHIC = "geographic"  # placeholder fallback


# =============================================================================
# Settings
# =============================================================================


class ServiceDiscoveryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESYNC_DISCOVERY_",
        frozen=True,
        extra="ignore",
    )

    enabled_backends: str = "consul"
    default_backend: DiscoveryBackend = DiscoveryBackend.CONSUL

    # Consul
    consul_url: AnyUrl = "http://localhost:8500"
    consul_acl_token: str = ""

    # Kubernetes (optional)
    kubernetes_api_server: str = "https://kubernetes.default.svc"
    kubernetes_token: str = ""
    kubernetes_namespace: str = "default"
    kubernetes_verify_tls: bool = False

    # Local instance
    default_port: int = Field(default=8000, gt=0, le=65535)
    advertise_host: str = ""  # if set, use as host instead of auto-detect

    # Cache + loops
    discovery_refresh_interval: int = Field(default=60, gt=1)
    health_tick_interval: int = Field(default=5, gt=0)
    backend_health_interval: int = Field(default=60, gt=1)
    metrics_log_interval: int = Field(default=300, gt=1)

    # Concurrency
    discovery_concurrency: int = Field(default=50, gt=0)
    health_check_concurrency: int = Field(default=100, gt=0)

    # HTTP tuning
    http_timeout_default: float = Field(default=10.0, gt=0.0)
    http_max_connections: int = Field(default=200, gt=10)
    http_max_keepalive: int = Field(default=50, gt=5)
    http_keepalive_expiry: float = Field(default=20.0, gt=0.0)

    # Metrics safety
    enable_instance_metrics: bool = False  # HIGH cardinality -> off by default

    def enabled_backend_set(self) -> set[DiscoveryBackend]:
        raw = [x.strip() for x in self.enabled_backends.split(",") if x.strip()]
        out: set[DiscoveryBackend] = set()
        for x in raw:
            try:
                out.add(DiscoveryBackend(x))
            except ValueError:
                logger.warning("unknown_enabled_backend", value=x)
        if not out:
            out.add(DiscoveryBackend.CONSUL)
        return out


def _make_limits(cfg: ServiceDiscoveryConfig) -> httpx.Limits:
    # httpx.Limits controls connection pool sizing (max_connections, keepalive, expiry). :contentReference[oaicite:3]{index=3}
    return httpx.Limits(
        max_connections=cfg.http_max_connections,
        max_keepalive_connections=cfg.http_max_keepalive,
        keepalive_expiry=cfg.http_keepalive_expiry,
    )


# =============================================================================
# Host validation (IP or hostname)
# =============================================================================

_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$")


def _is_valid_hostname(host: str) -> bool:
    host = host.rstrip(".")
    if not host or len(host) > 253:
        return False
    labels = host.split(".")
    if any(not lbl or len(lbl) > 63 for lbl in labels):
        return False
    for lbl in labels:
        if lbl.startswith("-") or lbl.endswith("-"):
            return False
        if not _HOST_LABEL_RE.match(lbl):
            return False
    return True


# =============================================================================
# Models
# =============================================================================


class ServiceInstance(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    service_name: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN

    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    weight: int = Field(default=1, ge=1)

    # runtime
    last_health_check_mono: float = 0.0
    last_health_check_epoch: float = 0.0
    health_check_interval: int = Field(default=30, gt=0)
    consecutive_failures: int = Field(default=0, ge=0)
    response_time_avg: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_host(self) -> "ServiceInstance":
        h = (self.host or "").strip()
        if not h:
            raise ValueError("host não pode ser vazio")
        try:
            ipaddress.ip_address(h)
            self.host = h
            return self
        except ValueError:
            pass
        if not _is_valid_hostname(h):
            raise ValueError(f"host inválido: {h}")
        self.host = h
        return self

    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

    @property
    def health_score(self) -> float:
        if self.status == ServiceStatus.MAINTENANCE:
            return 50.0
        if self.status != ServiceStatus.HEALTHY:
            return 0.0
        score = 100.0
        if self.response_time_avg > 0.5:
            score -= min(30.0, (self.response_time_avg - 0.5) * 20.0)
        if self.consecutive_failures:
            score -= min(20.0, float(self.consecutive_failures * 5))
        return max(0.0, score)


class ServiceDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_name: str = Field(min_length=1)
    discovery_backend: DiscoveryBackend = DiscoveryBackend.CONSUL
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN

    # health
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_timeout: float = Field(default=5.0, gt=0.0)
    health_check_interval: int = Field(default=30, gt=0)
    max_consecutive_failures: int = Field(default=3, gt=0)

    # circuit breaker (minimal)
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, gt=0)
    circuit_breaker_recovery_timeout: int = Field(default=60, gt=0)

    # consul TTL
    instance_ttl: int = Field(default=60, gt=0)
    deregister_on_shutdown: bool = True

    backend_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_health_path(self) -> "ServiceDefinition":
        if self.health_check_enabled:
            p = (self.health_check_path or "").strip()
            if not p.startswith("/"):
                raise ValueError("health_check_path deve começar com '/'")
        return self


# =============================================================================
# Backend Interface
# =============================================================================


class DiscoveryBackendInterface(ABC):
    @abstractmethod
    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool: ...
    @abstractmethod
    async def deregister_service(self, service_name: str, instance_id: str) -> bool: ...
    @abstractmethod
    async def discover_services(self, service_name: str) -> list[ServiceInstance]: ...
    @abstractmethod
    async def watch_service(
        self, service_name: str, callback: Callable[..., Any]
    ) -> None: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...


# =============================================================================
# Consul Backend
# =============================================================================


class ConsulBackend(DiscoveryBackendInterface):
    def __init__(
        self, cfg: ServiceDiscoveryConfig, config: dict[str, Any] | None = None
    ) -> None:
        self._cfg = cfg
        c = config or {}
        self._consul_url = str(c.get("url", str(cfg.consul_url))).rstrip("/")
        self._acl_token = str(c.get("acl_token", cfg.consul_acl_token or "") or "")
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._acl_token:
                headers["X-Consul-Token"] = self._acl_token
            self._client = httpx.AsyncClient(
                base_url=self._consul_url,
                headers=headers,
                timeout=httpx.Timeout(self._cfg.http_timeout_default),
                limits=_make_limits(self._cfg),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool:
        with tracer.start_as_current_span(
            "sdm.backend.consul.register",
            attributes={
                "service.name": service_def.service_name,
                "instance.id": instance.instance_id,
            },
        ) as span:
            client = self._ensure_client()
            payload: dict[str, Any] = {
                "ID": instance.instance_id,
                "Name": service_def.service_name,
                "Address": instance.host,
                "Port": instance.port,
                "Tags": list(instance.tags),
                "Meta": instance.metadata,
            }
            if service_def.health_check_enabled:
                payload["Check"] = {
                    "HTTP": f"{instance.url}{service_def.health_check_path}",
                    "Interval": f"{service_def.health_check_interval}s",
                    "Timeout": f"{service_def.health_check_timeout}s",
                    "DeregisterCriticalServiceAfter": f"{service_def.instance_ttl}s",
                }
            try:
                r = await client.put(
                    "/v1/agent/service/register",
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                ok = r.status_code == 200
                span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
                return ok
            except asyncio.CancelledError:
                raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "consul_register_failed",
                    service=service_def.service_name,
                    error=str(e),
                )
                return False

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        with tracer.start_as_current_span(
            "sdm.backend.consul.deregister",
            attributes={"service.name": service_name, "instance.id": instance_id},
        ) as span:
            client = self._ensure_client()
            try:
                r = await client.put(f"/v1/agent/service/deregister/{instance_id}")
                ok = r.status_code == 200
                span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
                return ok
            except asyncio.CancelledError:
                raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "consul_deregister_failed", service=service_name, error=str(e)
                )
                return False

    async def discover_services(self, service_name: str) -> list[ServiceInstance]:
        with tracer.start_as_current_span(
            "sdm.backend.consul.discover",
            attributes={"service.name": service_name},
        ) as span:
            client = self._ensure_client()
            try:
                r = await client.get(f"/v1/health/service/{service_name}")
                if r.status_code != 200:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {r.status_code}"))
                    return []

                data: list[dict[str, Any]] = orjson.loads(r.content)
                out: list[ServiceInstance] = []
                for entry in data:
                    svc = entry.get("Service") or {}
                    checks = entry.get("Checks") or []
                    status = (
                        ServiceStatus.HEALTHY
                        if any(c.get("Status") == "passing" for c in checks)
                        else ServiceStatus.UNHEALTHY
                    )

                    host = (
                        svc.get("Address")
                        or (entry.get("Node") or {}).get("Address")
                        or ""
                    )
                    port = int(svc.get("Port") or 0)
                    iid = svc.get("ID")
                    if not host or not port or not iid:
                        continue

                    out.append(
                        ServiceInstance(
                            service_name=service_name,
                            instance_id=str(iid),
                            host=str(host),
                            port=port,
                            status=status,
                            tags=set(svc.get("Tags") or []),
                            metadata=svc.get("Meta") or {},
                        )
                    )
                span.set_status(Status(StatusCode.OK))
                return out
            except asyncio.CancelledError:
                raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "consul_discover_failed", service=service_name, error=str(e)
                )
                return []

    async def watch_service(
        self, service_name: str, callback: Callable[..., Any]
    ) -> None:
        raise NotImplementedError("Consul watch_service não implementado nesta versão.")

    async def health_check(self) -> bool:
        client = self._ensure_client()
        try:
            r = await client.get("/v1/status/leader")
            return r.status_code == 200
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("consul_health_failed", error=str(e))
            return False


# =============================================================================
# Kubernetes Backend (optional discovery via Endpoints)
# =============================================================================


class KubernetesBackend(DiscoveryBackendInterface):
    def __init__(
        self, cfg: ServiceDiscoveryConfig, config: dict[str, Any] | None = None
    ) -> None:
        self._cfg = cfg
        c = config or {}
        self._api_server = str(c.get("api_server", cfg.kubernetes_api_server)).rstrip(
            "/"
        )
        self._token = str(c.get("token", cfg.kubernetes_token) or "")
        self._namespace = str(c.get("namespace", cfg.kubernetes_namespace) or "default")
        self._verify = bool(c.get("verify_tls", cfg.kubernetes_verify_tls))
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._api_server,
                headers=headers,
                timeout=httpx.Timeout(self._cfg.http_timeout_default),
                limits=_make_limits(self._cfg),
                verify=self._verify,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance
    ) -> bool:
        return True  # K8s gerencia via manifestos

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        return True

    async def discover_services(self, service_name: str) -> list[ServiceInstance]:
        client = self._ensure_client()
        url = f"/api/v1/namespaces/{self._namespace}/endpoints/{service_name}"
        try:
            r = await client.get(url)
            if r.status_code != 200:
                return []
            data: dict[str, Any] = orjson.loads(r.content)
            out: list[ServiceInstance] = []
            for subset in data.get("subsets") or []:
                for address in subset.get("addresses") or []:
                    ip = address.get("ip")
                    for pinfo in subset.get("ports") or []:
                        port = int(pinfo.get("port") or 0)
                        if not ip or not port:
                            continue
                        out.append(
                            ServiceInstance(
                                service_name=service_name,
                                instance_id=f"{service_name}-{ip}:{port}",
                                host=str(ip),
                                port=port,
                                status=ServiceStatus.HEALTHY,
                                protocol="http",
                            )
                        )
            return out
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("k8s_discover_failed", service=service_name, error=str(e))
            return []

    async def watch_service(
        self, service_name: str, callback: Callable[..., Any]
    ) -> None:
        raise NotImplementedError(
            "Kubernetes watch_service não implementado nesta versão."
        )

    async def health_check(self) -> bool:
        client = self._ensure_client()
        try:
            r = await client.get(f"/api/v1/namespaces/{self._namespace}/pods")
            return r.status_code == 200
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("k8s_health_failed", error=str(e))
            return False


# =============================================================================
# Service Discovery Manager
# =============================================================================


@injectable
class ServiceDiscoveryManager:
    def __init__(self, config: ServiceDiscoveryConfig | None = None) -> None:
        self.config = config or ServiceDiscoveryConfig()

        self.services: dict[str, ServiceDefinition] = {}
        self.instances: dict[str, list[ServiceInstance]] = defaultdict(list)
        self._discovery_cache_time: dict[str, float] = {}

        self.local_instances: dict[str, ServiceInstance] = {}
        self.backends: dict[str, DiscoveryBackendInterface] = {}
        self._backend_health: dict[str, bool] = {}

        self._health_client: httpx.AsyncClient | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # LB state
        self._round_robin_index: dict[str, int] = defaultdict(int)
        self._conn_counts_by_instance: dict[str, int] = defaultdict(int)
        self._conn_counts_by_service: dict[str, int] = defaultdict(int)

        # circuit breaker: instance_id -> open_until (monotonic)
        self._circuit_open_until: dict[str, float] = {}

        # concurrency
        self._disc_sem = (
            asyncio.Semaphore(self.config.discovery_concurrency)
            if self.config.discovery_concurrency > 0
            else None
        )
        self._hc_sem = (
            asyncio.Semaphore(self.config.health_check_concurrency)
            if self.config.health_check_concurrency > 0
            else None
        )

        self._initialize_backends()

    # ---------------------------------------------------------------------
    # Init / backend selection
    # ---------------------------------------------------------------------

    def _initialize_backends(self) -> None:
        enabled = self.config.enabled_backend_set()

        if DiscoveryBackend.CONSUL in enabled:
            self.backends["consul"] = ConsulBackend(self.config)
            self._backend_health["consul"] = False

        if DiscoveryBackend.KUBERNETES in enabled:
            self.backends["kubernetes"] = KubernetesBackend(self.config)
            self._backend_health["kubernetes"] = False

        if not self.backends:
            self.backends["consul"] = ConsulBackend(self.config)
            self._backend_health["consul"] = False

        if self.config.default_backend.value not in self.backends:
            logger.warning(
                "default_backend_not_enabled",
                requested=self.config.default_backend.value,
                using=next(iter(self.backends.keys())),
            )

    def _get_backend(
        self, service_def: ServiceDefinition
    ) -> DiscoveryBackendInterface | None:
        key = service_def.discovery_backend.value
        if key in self.backends:
            return self.backends[key]
        return self.backends.get(self.config.default_backend.value) or next(
            iter(self.backends.values()), None
        )

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        if self._running:
            return

        asyncio.get_running_loop()  # must be called inside running loop

        self._health_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.http_timeout_default),
            limits=_make_limits(self.config),
        )

        self._running = True
        self._tasks.clear()

        coros: list[tuple[Coroutine[Any, Any, None], str]] = [
            (self._discovery_worker(), "sdm_discovery_worker"),
            (self._health_worker(), "sdm_health_worker"),
            (self._backend_health_worker(), "sdm_backend_health_worker"),
            (self._metrics_worker(), "sdm_metrics_worker"),
        ]

        for coro, name in coros:
            task = (
                tg.create_task(coro, name=name) if tg else track_task(coro, name=name)
            )
            self._tasks.append(task)

        logger.info(
            "sdm_started",
            enabled_backends=list(self.backends.keys()),
            default_backend=self.config.default_backend.value,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # deregister locals (parallel)
        async with asyncio.TaskGroup() as tg:
            for svc, inst in list(self.local_instances.items()):
                sd = self.services.get(svc)
                if sd and sd.deregister_on_shutdown:
                    tg.create_task(self.deregister_service(svc, inst.instance_id))

        # cancel tasks
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()

        # close shared health client
        if self._health_client and not self._health_client.is_closed:
            await self._health_client.aclose()
        self._health_client = None

        # close backends
        async with asyncio.TaskGroup() as tg:
            for b in self.backends.values():
                tg.create_task(b.close())

        logger.info("sdm_stopped")

    # ---------------------------------------------------------------------
    # Registration
    # ---------------------------------------------------------------------

    def _get_advertise_host(self) -> str:
        if self.config.advertise_host:
            return self.config.advertise_host.strip()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except OSError:
                pass

        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"

    async def register_service(
        self, service_def: ServiceDefinition, instance: ServiceInstance | None = None
    ) -> str:
        with tracer.start_as_current_span(
            "sdm.register_service",
            attributes={
                "service.name": service_def.service_name,
                "backend": service_def.discovery_backend.value,
            },
        ) as span:
            try:
                if instance is None:
                    host = self._get_advertise_host()
                    instance = ServiceInstance(
                        service_name=service_def.service_name,
                        instance_id=f"{service_def.service_name}_{host}_{int(time.time())}",
                        host=host,
                        port=self.config.default_port,
                        status=ServiceStatus.HEALTHY,
                        health_check_interval=service_def.health_check_interval,
                    )
                else:
                    instance.health_check_interval = service_def.health_check_interval

                self.services[service_def.service_name] = service_def
                self.local_instances[service_def.service_name] = instance
                _prom_active_services.set(len(self.services))

                backend = self._get_backend(service_def)
                if not backend:
                    _prom_errors.labels(error_type="no_backend").inc()
                    span.set_status(Status(StatusCode.ERROR, "no_backend"))
                    return ""

                if await backend.register_service(service_def, instance):
                    _prom_registrations.labels(
                        backend=service_def.discovery_backend.value,
                        service_name=service_def.service_name,
                    ).inc()
                    span.set_status(Status(StatusCode.OK))
                    logger.info(
                        "service_registered",
                        service=service_def.service_name,
                        instance_id=instance.instance_id,
                    )
                    return instance.instance_id

                _prom_errors.labels(error_type="registration_failed").inc()
                span.set_status(Status(StatusCode.ERROR, "registration_failed"))
                return ""
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _prom_errors.labels(error_type="registration_exception").inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "registration_exception",
                    service=service_def.service_name,
                    error=str(e),
                )
                return ""

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        with tracer.start_as_current_span(
            "sdm.deregister_service",
            attributes={"service.name": service_name, "instance.id": instance_id},
        ) as span:
            sd = self.services.get(service_name)
            if not sd:
                span.set_status(Status(StatusCode.ERROR, "service_not_found"))
                return False

            backend = self._get_backend(sd)
            if not backend:
                _prom_errors.labels(error_type="no_backend").inc()
                span.set_status(Status(StatusCode.ERROR, "no_backend"))
                return False

            try:
                if await backend.deregister_service(service_name, instance_id):
                    self.local_instances.pop(service_name, None)
                    _prom_deregistrations.labels(
                        backend=sd.discovery_backend.value, service_name=service_name
                    ).inc()
                    span.set_status(Status(StatusCode.OK))
                    logger.info(
                        "service_deregistered",
                        service=service_name,
                        instance_id=instance_id,
                    )
                    return True

                _prom_errors.labels(error_type="deregistration_failed").inc()
                span.set_status(Status(StatusCode.ERROR, "deregistration_failed"))
                return False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _prom_errors.labels(error_type="deregistration_exception").inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "deregistration_exception", service=service_name, error=str(e)
                )
                return False

    # ---------------------------------------------------------------------
    # Discovery / Cache / Garbage Collection
    # ---------------------------------------------------------------------

    def _gc_dead_instance_state(self, dead_ids: set[str]) -> None:
        """
        Remove estado por instância (previne leak lógico em churn).
        E remove séries Prometheus se enable_instance_metrics=True.

        Prometheus client libs recomendam remove() com a mesma assinatura de labels(). :contentReference[oaicite:4]{index=4}
        """
        for dead_id in dead_ids:
            self._conn_counts_by_instance.pop(dead_id, None)
            self._circuit_open_until.pop(dead_id, None)

            if self.config.enable_instance_metrics:
                with contextlib.suppress(KeyError, ValueError):
                    _prom_instance_active_connections.remove(
                        dead_id
                    )  # remove(*labelvalues)
                with contextlib.suppress(KeyError, ValueError):
                    _prom_instance_circuit_open.remove(dead_id)  # remove(*labelvalues)

    async def get_service_instances(self, service_name: str) -> list[ServiceInstance]:
        """
        Retorna instâncias (cache local) e faz GC de instâncias mortas ao atualizar do backend.
        """
        now = time.monotonic()
        age = now - self._discovery_cache_time.get(service_name, 0.0)

        # cache hit
        if (
            service_name in self.instances
            and self.instances[service_name]
            and age < self.config.discovery_refresh_interval
        ):
            _prom_discoveries.labels(service_name=service_name, source="cache").inc()
            return self.instances[service_name]

        sd = self.services.get(service_name)
        if not sd:
            return []

        backend = self._get_backend(sd)
        if not backend:
            _prom_errors.labels(error_type="no_backend").inc()
            return []

        _prom_discoveries.labels(service_name=service_name, source="backend").inc()

        with tracer.start_as_current_span(
            "sdm.get_service_instances",
            attributes={
                "service.name": service_name,
                "backend": sd.discovery_backend.value,
            },
        ) as span:
            try:
                insts = await backend.discover_services(service_name)
                for inst in insts:
                    inst.health_check_interval = sd.health_check_interval

                # GC: instâncias que sumiram do backend
                old_insts = self.instances.get(service_name, [])
                current_ids = {i.instance_id for i in insts}
                dead_ids = {i.instance_id for i in old_insts} - current_ids
                if dead_ids:
                    self._gc_dead_instance_state(dead_ids)

                # update cache
                self.instances[service_name] = insts
                self._discovery_cache_time[service_name] = time.monotonic()

                _prom_instances_discovered.labels(service_name=service_name).inc(
                    len(insts)
                )
                _prom_total_instances.set(sum(len(v) for v in self.instances.values()))
                _prom_service_instances.labels(service_name=service_name).set(
                    len(insts)
                )

                span.set_status(Status(StatusCode.OK))
                return insts
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _prom_errors.labels(error_type="discovery_exception").inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("discovery_exception", service=service_name, error=str(e))
                return []

    # ---------------------------------------------------------------------
    # Load Balancing
    # ---------------------------------------------------------------------

    def _circuit_is_open(self, instance_id: str) -> bool:
        return time.monotonic() < self._circuit_open_until.get(instance_id, 0.0)

    def _select_instance(
        self,
        insts: list[ServiceInstance],
        strategy: LoadBalancingStrategy,
        service_name: str,
    ) -> ServiceInstance | None:
        if not insts:
            return None
        match strategy:
            case LoadBalancingStrategy.ROUND_ROBIN:
                idx = self._round_robin_index[service_name]
                self._round_robin_index[service_name] = (idx + 1) % len(insts)
                return insts[idx]
            case LoadBalancingStrategy.RANDOM:
                return random.choice(insts)
            case LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(
                    insts,
                    key=lambda i: self._conn_counts_by_instance.get(i.instance_id, 0),
                )
            case LoadBalancingStrategy.WEIGHTED_RANDOM:
                return random.choices(insts, weights=[i.weight for i in insts], k=1)[0]
            case LoadBalancingStrategy.LATENCY_BASED:
                return min(insts, key=lambda i: i.response_time_avg or float("inf"))
            case _:
                return insts[0]

    async def discover_service(
        self, service_name: str, strategy: LoadBalancingStrategy | None = None
    ) -> ServiceInstance | None:
        with tracer.start_as_current_span(
            "sdm.discover_service", attributes={"service.name": service_name}
        ) as span:
            insts = await self.get_service_instances(service_name)
            sd = self.services.get(service_name)

            candidates = [
                i
                for i in insts
                if i.is_healthy and not self._circuit_is_open(i.instance_id)
            ]
            _prom_service_healthy_instances.labels(service_name=service_name).set(
                len(candidates)
            )

            if not candidates:
                span.set_status(Status(StatusCode.ERROR, "no_candidates"))
                return None

            resolved = strategy or (
                sd.load_balancing_strategy if sd else LoadBalancingStrategy.ROUND_ROBIN
            )
            chosen = self._select_instance(candidates, resolved, service_name)

            if chosen:
                _prom_lb_decisions.labels(
                    service_name=service_name, strategy=resolved.value
                ).inc()
                span.add_event(
                    "lb_decision",
                    {"strategy": resolved.value, "instance_id": chosen.instance_id},
                )
                span.set_status(Status(StatusCode.OK))
            return chosen

    def release_instance(self, service_name: str, instance_id: str) -> None:
        c = self._conn_counts_by_instance.get(instance_id, 0)
        if c > 0:
            self._conn_counts_by_instance[instance_id] = c - 1

        sc = self._conn_counts_by_service.get(service_name, 0)
        if sc > 0:
            self._conn_counts_by_service[service_name] = sc - 1
            _prom_service_active_connections.labels(service_name=service_name).set(
                self._conn_counts_by_service[service_name]
            )

        if self.config.enable_instance_metrics:
            _prom_instance_active_connections.labels(instance_id=instance_id).dec()

    @contextlib.asynccontextmanager
    async def borrow_instance(
        self, service_name: str, strategy: LoadBalancingStrategy | None = None
    ):
        """
        Context Manager obrigatório para Least Connections correto (incrementa/decrementa com segurança).
        """
        inst = await self.discover_service(service_name, strategy)
        if inst is not None:
            self._conn_counts_by_instance[inst.instance_id] += 1
            self._conn_counts_by_service[service_name] += 1
            _prom_service_active_connections.labels(service_name=service_name).set(
                self._conn_counts_by_service[service_name]
            )

            if self.config.enable_instance_metrics:
                _prom_instance_active_connections.labels(
                    instance_id=inst.instance_id
                ).inc()

        try:
            yield inst
        finally:
            if inst is not None:
                self.release_instance(service_name, inst.instance_id)

    # ---------------------------------------------------------------------
    # Health checks
    # ---------------------------------------------------------------------

    async def _perform_health_check(
        self, inst: ServiceInstance, sd: ServiceDefinition
    ) -> None:
        if not self._health_client:
            return

        if not sd.health_check_enabled:
            _prom_health_checks.labels(
                service_name=sd.service_name, result="skipped"
            ).inc()
            return

        url = f"{inst.url}{sd.health_check_path}"

        with tracer.start_as_current_span(
            "sdm.health_check",
            attributes={
                "service.name": sd.service_name,
                "instance.id": inst.instance_id,
            },
        ) as span:
            start = time.monotonic()
            try:
                resp = await self._health_client.get(
                    url, timeout=httpx.Timeout(sd.health_check_timeout)
                )
                elapsed = time.monotonic() - start
                _prom_health_check_duration.labels(
                    service_name=sd.service_name
                ).observe(elapsed)

                inst.response_time_avg = (
                    elapsed
                    if inst.response_time_avg == 0
                    else (0.7 * inst.response_time_avg + 0.3 * elapsed)
                )

                if resp.status_code == 200:
                    inst.status = ServiceStatus.HEALTHY
                    inst.consecutive_failures = 0
                    _prom_health_checks.labels(
                        service_name=sd.service_name, result="healthy"
                    ).inc()
                    span.set_status(Status(StatusCode.OK))

                    self._circuit_open_until.pop(inst.instance_id, None)
                    if self.config.enable_instance_metrics:
                        _prom_instance_circuit_open.labels(
                            instance_id=inst.instance_id
                        ).set(0)
                else:
                    inst.status = ServiceStatus.UNHEALTHY
                    inst.consecutive_failures += 1
                    _prom_health_checks.labels(
                        service_name=sd.service_name, result="unhealthy"
                    ).inc()
                    span.set_status(
                        Status(StatusCode.ERROR, f"HTTP {resp.status_code}")
                    )

                # circuit breaker (minimal)
                if (
                    sd.circuit_breaker_enabled
                    and inst.consecutive_failures
                    >= sd.circuit_breaker_failure_threshold
                ):
                    self._circuit_open_until[inst.instance_id] = (
                        time.monotonic() + sd.circuit_breaker_recovery_timeout
                    )
                    if self.config.enable_instance_metrics:
                        _prom_instance_circuit_open.labels(
                            instance_id=inst.instance_id
                        ).set(1)

                inst.last_health_check_mono = time.monotonic()
                inst.last_health_check_epoch = time.time()

                if inst.consecutive_failures >= sd.max_consecutive_failures:
                    logger.warning(
                        "instance_unhealthy",
                        service=sd.service_name,
                        instance_id=inst.instance_id,
                        failures=inst.consecutive_failures,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                elapsed = time.monotonic() - start
                _prom_health_check_duration.labels(
                    service_name=sd.service_name
                ).observe(elapsed)
                _prom_health_checks.labels(
                    service_name=sd.service_name, result="error"
                ).inc()
                _prom_errors.labels(error_type="health_check_exception").inc()

                inst.status = ServiceStatus.UNHEALTHY
                inst.consecutive_failures += 1
                inst.last_health_check_mono = time.monotonic()
                inst.last_health_check_epoch = time.time()

                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "health_check_exception",
                    service=sd.service_name,
                    instance_id=inst.instance_id,
                    error=str(e),
                )

    async def _run_with_sem(
        self,
        sem: asyncio.Semaphore | None,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        if sem is None:
            await coro_factory()
            return
        async with sem:
            await coro_factory()

    # ---------------------------------------------------------------------
    # Workers (no spans; logs/metrics)
    # ---------------------------------------------------------------------

    async def _discovery_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.discovery_refresh_interval)
                async with asyncio.TaskGroup() as tg:
                    for svc in list(self.services.keys()):
                        self._discovery_cache_time.pop(svc, None)

                        async def _refresh(name: str = svc) -> None:
                            await self.get_service_instances(name)

                        tg.create_task(self._run_with_sem(self._disc_sem, _refresh))

            except asyncio.CancelledError:
                break
            except Exception as e:
                _prom_errors.labels(error_type="discovery_worker_error").inc()
                logger.error("discovery_worker_error", error=str(e), exc_info=True)

    async def _health_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.health_tick_interval)
                snapshot = dict(self.instances)

                async with asyncio.TaskGroup() as tg:
                    for svc, insts in snapshot.items():
                        sd = self.services.get(svc)
                        if not sd:
                            continue
                        for inst in list(insts):
                            due = (
                                time.monotonic() - inst.last_health_check_mono
                            ) > float(inst.health_check_interval)
                            if not due:
                                continue

                            async def _hc(
                                i: ServiceInstance = inst, d: ServiceDefinition = sd
                            ) -> None:
                                await self._perform_health_check(i, d)

                            tg.create_task(self._run_with_sem(self._hc_sem, _hc))

                for svc, insts in snapshot.items():
                    opened = sum(
                        1 for i in insts if self._circuit_is_open(i.instance_id)
                    )
                    _prom_service_circuit_open.labels(service_name=svc).set(opened)

            except asyncio.CancelledError:
                break
            except Exception as e:
                _prom_errors.labels(error_type="health_worker_error").inc()
                logger.error("health_worker_error", error=str(e), exc_info=True)

    async def _backend_health_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.backend_health_interval)
                async with asyncio.TaskGroup() as tg:
                    for key, backend in list(self.backends.items()):

                        async def _check(
                            k: str = key, b: DiscoveryBackendInterface = backend
                        ) -> None:
                            ok = await b.health_check()
                            self._backend_health[k] = ok
                            if not ok:
                                logger.warning("backend_unhealthy", backend=k)

                        tg.create_task(_check())

            except asyncio.CancelledError:
                break
            except Exception as e:
                _prom_errors.labels(error_type="backend_health_worker_error").inc()
                logger.error("backend_health_worker_error", error=str(e), exc_info=True)

    async def _metrics_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_log_interval)
                _prom_active_services.set(len(self.services))
                total = sum(len(v) for v in self.instances.values())
                _prom_total_instances.set(total)

                logger.info(
                    "sdm_metrics",
                    active_services=len(self.services),
                    total_instances=total,
                    healthy_instances=sum(
                        1
                        for insts in self.instances.values()
                        for inst in insts
                        if inst.is_healthy
                        and not self._circuit_is_open(inst.instance_id)
                    ),
                    backends_healthy=sum(1 for v in self._backend_health.values() if v),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                _prom_errors.labels(error_type="metrics_worker_error").inc()
                logger.error("metrics_worker_error", error=str(e), exc_info=True)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        return {
            "services_registered": len(self.services),
            "total_instances": sum(len(v) for v in self.instances.values()),
            "backends_status": dict(self._backend_health),
            "active_connections": dict(self._conn_counts_by_service),
            "circuit_breaker": {
                "open_instances": sum(
                    1
                    for _, until in self._circuit_open_until.items()
                    if time.monotonic() < until
                )
            },
        }


# =============================================================================
# FastAPI Dependency (antidote)
# =============================================================================


async def get_service_discovery(
    sdm: ServiceDiscoveryManager = inject.me(),
) -> ServiceDiscoveryManager:
    return sdm
