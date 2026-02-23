"""
Base Interfaces and Models for Service Discovery Backends

Este módulo fornece a interface abstrata e modelos base para backends de service discovery,
permitindo desacoplamento entre o ServiceDiscoveryManager e provedores específicos
(Consul, Kubernetes, Redis, etc.).

Classes:
    BackendInterface: Interface abstrata que todos os backends devem implementar
    BackendHealth: Modelo para status de saúde do backend
    ServiceInstance: Modelo para instância de serviço (reutilizável)
    ServiceDefinition: Modelo para definição de serviço (reutilizável)

Exemplo de uso:
    class MyCustomBackend(BackendInterface):
        async def register_service(self, service_def, instance) -> bool:
            ...

        async def discover_services(self, service_name) -> list[ServiceInstance]:
            ...
"""

from __future__ import annotations

import ipaddress
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field, model_validator


class ServiceStatus(str, Enum):
    """Status possible states for a service instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class ServiceInstance(BaseModel):
    """
    Modelo para instância de serviço.

    Attributes:
        service_name: Nome do serviço
        instance_id: ID único da instância
        host: Endereço IP ou hostname
        port: Porta do serviço
        protocol: Protocolo (http/https)
        status: Status atual da instância
        tags: Tags para categorização
        metadata: Metadados adicionais
        weight: Peso para load balancing
        health_check_interval: Intervalo de health check em segundos
        consecutive_failures: Falhas consecutivas
        response_time_avg: Tempo médio de resposta
    """

    model_config = Field(
        default=BaseModel.Config(validate_assignment=True, arbitrary_types_allowed=True)
    )

    service_name: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN

    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    weight: int = Field(default=1, ge=1)

    # Runtime fields (not persisted)
    last_health_check_mono: float = 0.0
    last_health_check_epoch: float = 0.0
    health_check_interval: int = Field(default=30, gt=0)
    consecutive_failures: int = Field(default=0, ge=0)
    response_time_avg: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_host(self) -> "ServiceInstance":
        """Validate and normalize the host field."""
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
        """Return the full URL for this service instance."""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        """Check if the instance is healthy."""
        return self.status == ServiceStatus.HEALTHY

    @property
    def health_score(self) -> float:
        """Calculate a health score between 0-100."""
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
    """
    Modelo para definição de serviço.

    Attributes:
        service_name: Nome do serviço
        discovery_backend: Backend de discovery a ser usado
        load_balancing_strategy: Estratégia de load balancing
        health_check_enabled: Se health checks estão habilitados
        health_check_path: Path para health check
        health_check_timeout: Timeout para health check
        health_check_interval: Intervalo de health check
        max_consecutive_failures: Máximo de falhas consecutivas antes de marcar como não saudável
        circuit_breaker_enabled: Se circuit breaker está habilitado
        circuit_breaker_failure_threshold: Threshold de falhas para circuit breaker
        circuit_breaker_recovery_timeout: Timeout de recuperação do circuit breaker
        instance_ttl: TTL da instância no backend
        deregister_on_shutdown: Se deve deregistrar ao encerrar
        backend_config: Configurações específicas do backend
    """

    model_config = Field(default=BaseModel.Config(frozen=True))

    service_name: str = Field(min_length=1)
    discovery_backend: str = "consul"  # Backend type
    load_balancing_strategy: str = "round_robin"

    # Health check settings
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_timeout: float = Field(default=5.0, gt=0.0)
    health_check_interval: int = Field(default=30, gt=0)
    max_consecutive_failures: int = Field(default=3, gt=0)

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, gt=0)
    circuit_breaker_recovery_timeout: int = Field(default=60, gt=0)

    # Instance TTL
    instance_ttl: int = Field(default=60, gt=0)
    deregister_on_shutdown: bool = True

    backend_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_health_path(self) -> "ServiceDefinition":
        """Validate health check path."""
        if self.health_check_enabled:
            p = (self.health_check_path or "").strip()
            if not p.startswith("/"):
                raise ValueError("health_check_path deve começar com '/'")
        return self


@dataclass
class BackendHealth:
    """Status de saúde de um backend de discovery."""

    backend_name: str
    is_healthy: bool = True
    last_check_time: float = 0.0
    error_message: str | None = None
    response_time: float = 0.0

    def mark_unhealthy(self, error: str, response_time: float = 0.0) -> None:
        """Mark the backend as unhealthy."""
        self.is_healthy = False
        self.error_message = error
        self.response_time = response_time

    def mark_healthy(self, response_time: float = 0.0) -> None:
        """Mark the backend as healthy."""
        self.is_healthy = True
        self.error_message = None
        self.response_time = response_time


class BackendInterface(ABC):
    """
    Interface abstrata para backends de service discovery.

    Esta interface define os métodos que todos os backends de discovery
    (Consul, Kubernetes, Redis, etc.) devem implementar.

    Attributes:
        backend_name: Nome identificador do backend

    Example:
        class ConsulBackend(BackendInterface):
            async def register_service(self, service_def, instance) -> bool:
                # implementação específica do Consul
                ...
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend."""
        ...

    @abstractmethod
    async def register_service(
        self,
        service_def: ServiceDefinition,
        instance: ServiceInstance,
    ) -> bool:
        """
        Register a service instance with the backend.

        Args:
            service_def: Service definition
            instance: Service instance to register

        Returns:
            True if registration was successful, False otherwise
        """
        ...

    @abstractmethod
    async def deregister_service(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """
        Deregister a service instance from the backend.

        Args:
            service_name: Name of the service
            instance_id: ID of the instance to deregister

        Returns:
            True if deregistration was successful, False otherwise
        """
        ...

    @abstractmethod
    async def discover_services(
        self,
        service_name: str,
    ) -> list[ServiceInstance]:
        """
        Discover all instances of a service.

        Args:
            service_name: Name of the service to discover

        Returns:
            List of service instances
        """
        ...

    @abstractmethod
    async def watch_service(
        self,
        service_name: str,
        callback: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """
        Watch for changes in a service and call the callback on changes.

        Args:
            service_name: Name of the service to watch
            callback: Async callback to call when service changes
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the backend itself is healthy.

        Returns:
            True if the backend is healthy, False otherwise
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any resources held by the backend."""
        ...


def _is_valid_hostname(hostname: str) -> bool:
    """Check if a hostname is valid."""
    if len(hostname) > 253:
        return False
    # Allow alphanumeric, hyphens, and dots
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
    if not all(c in allowed for c in hostname):
        return False
    # Check for valid label lengths
    labels = hostname.split(".")
    return all(0 < len(label) <= 63 for label in labels)


__all__ = [
    "BackendInterface",
    "BackendHealth",
    "ServiceInstance",
    "ServiceDefinition",
    "ServiceStatus",
]
