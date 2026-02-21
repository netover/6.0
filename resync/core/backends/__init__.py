"""
Backends Package - Service Discovery Backend Interfaces

Este pacote contém interfaces abstratas para diferentes backends de service discovery,
permitindo desacoplamento entre o ServiceDiscoveryManager e os provedores específicos
(Consul, Kubernetes, Redis, etc.).

Exports:
    BackendInterface: Interface abstrata para backends de discovery
    BackendHealth: Modelo para status de saúde do backend
    ServiceInstance: Modelo para instância de serviço
"""

from resync.core.backends.base import BackendInterface, BackendHealth, ServiceInstance

__all__ = [
    "BackendInterface",
    "BackendHealth",
    "ServiceInstance",
]
Backends Package - Service Discovery Backend Interfaces

Este pacote contém interfaces abstratas para diferentes backends de service discovery,
permitindo desacoplamento entre o ServiceDiscoveryManager e os provedores específicos
(Consul, Kubernetes, Redis, etc.).

Exports:
    BackendInterface: Interface abstrata para backends de discovery
    BackendHealth: Modelo para status de saúde do backend
    ServiceInstance: Modelo para instância de serviço
"""

from resync.core.backends.base import BackendInterface, BackendHealth, ServiceInstance

__all__ = [
    "BackendInterface",
    "BackendHealth",
    "ServiceInstance",
]

