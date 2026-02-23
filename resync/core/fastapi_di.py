"""FastAPI dependency helpers (compat layer).

This module exists to keep *route modules* stable while the internal wiring
continues to evolve.

Design choice (P0): all providers here **return the dependency object
directly** (no async generators / ``yield``). That avoids patterns like
``await anext(get_service())`` and keeps semantics consistent across the codebase.

- Request-scoped objects (AgentManager, TWS client, KnowledgeGraph) are provided
  by :mod:`resync.core.wiring` and require ``Request``.
- Process-wide singletons (Redis client, TeamsIntegration, AuditQueue) are
  thin wrappers around their canonical modules.

If an optional feature is not installed (e.g. RAG file ingestion), the provider
raises an HTTP 503 with a clear explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from fastapi import HTTPException, Request, status

# ---------------------------------------------------------------------------
# Minimal generic container (kept for backwards compatibility)
# ---------------------------------------------------------------------------

T_co = TypeVar("T_co", covariant=True)


class ServiceFactory(Protocol[T_co]):
    def __call__(self) -> T_co: ...


@dataclass
class ServiceContainer:
    """Simple service container for legacy call sites."""

    services: dict[str, Any]

    def get(self, name: str) -> Any:
        if name not in self.services:
            raise KeyError(f"Service not found: {name}")
        return self.services[name]


_container: ServiceContainer | None = None


def initialize_services(services: dict[str, Any]) -> ServiceContainer:
    """Initialize the global service container."""

    global _container
    _container = ServiceContainer(services)
    return _container


def get_container() -> ServiceContainer:
    if _container is None:
        raise RuntimeError("Service container not initialized")
    return _container


def get_service(name: str) -> Any:
    return get_container().get(name)


# ---------------------------------------------------------------------------
# Canonical dependencies used by the API
# ---------------------------------------------------------------------------

# Request-scoped providers (FastAPI injects Request automatically)
from resync.core.wiring import (  # noqa: E402
    get_agent_manager,
    get_hybrid_router,
    get_idempotency_manager,
    get_knowledge_graph,
    get_llm_service,
    get_tws_client,
    get_a2a_handler,
)


def get_redis_client():
    """Return the *canonical* Redis client singleton.

    The client is initialized during application startup (lifespan) via the
    RedisInitializer. If it was not initialized, we return a 503 instead of
    silently creating a second client (which can be dangerous with
    gunicorn --preload).
    """

    try:
        from resync.core.redis_init import get_redis_client as _get_redis_client

        return _get_redis_client()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Redis client is not initialized. The application startup likely "
                "failed or Redis is disabled in this environment."
            ),
        ) from exc


def get_teams_integration():
    """Return the TeamsIntegration singleton."""

    from resync.core.teams_integration import get_teams_integration as _get

    return _get()


def get_audit_queue():
    """Return the AuditQueue singleton."""

    from resync.core.audit_queue import get_audit_queue as _get

    return _get()


async def get_file_ingestor(request: Request):
    """RAG file-ingestor dependency.

    This feature is optional. If its implementation is not present (or its
    optional dependencies are missing), we raise a 503.
    """

    try:
        from resync.core.wiring import get_file_ingestor as _get

        return _get(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG file ingestion is not enabled in this build. "
                "Install the optional dependencies / module and enable it in settings."
            ),
        ) from exc


__all__ = [
    # legacy container
    "ServiceContainer",
    "initialize_services",
    "get_container",
    "get_service",
    # API dependencies
    "get_agent_manager",
    "get_hybrid_router",
    "get_idempotency_manager",
    "get_knowledge_graph",
    "get_llm_service",
    "get_tws_client",
    "get_teams_integration",
    "get_redis_client",
    "get_audit_queue",
    "get_file_ingestor",
    "get_a2a_handler",
]
