"""
Backends Package - Service Discovery Backend Interfaces

Este pacote contém interfaces abstratas para diferentes backends de service discovery,
permitindo desacoplamento entre o ServiceDiscoveryManager e os provedores específicos
(Consul, Kubernetes, Redis, etc.).

Exports:
    BackendInterface: Interface abstrata para backends de discovery
    BackendHealth: Modelo para status de saúde do backend
    ServiceInstance: Modelo para instância de serviço
    ServiceDefinition: Modelo para definição de serviço
    ServiceStatus: Enum para status de serviço
"""

from resync.core.backends.base import (
    BackendHealth,
    BackendInterface,
    ServiceDefinition,
    ServiceInstance,
    ServiceStatus,
)

__all__ = [
    "BackendInterface",
    "BackendHealth",
    "ServiceInstance",
    "ServiceDefinition",
    "ServiceStatus",
]

