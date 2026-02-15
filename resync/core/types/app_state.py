from __future__ import annotations

"""Enterprise typed state stored on ``app.state.enterprise_state``.

Starlette's ``app.state`` is dynamic. To make the application enterprise-grade and
auditable, we store a *single* immutable-ish state object on ``app.state`` and
type it precisely.

We use a dataclass with ``slots=True`` to:
- keep a single reference (no scattered attributes on app.state),
- reduce accidental typos / attribute drift,
- improve memory/performance slightly,
- make the contract explicit for tests and tooling.

References:
- Starlette lifespan/state docs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request

if TYPE_CHECKING:
    from resync.core.connection_manager import ConnectionManager
    from resync.core.idempotency.manager import IdempotencyManager
    from resync.core.context_store import ContextStore
    from resync.core.interfaces import ITWSClient
    from resync.core.agent_manager import AgentManager
    from resync.core.agent_router import HybridRouter
    # Canonical LLM service type used by enterprise wiring.
    from resync.services.llm_service import LLMService


@dataclass(slots=True)
class EnterpriseState:
    """All application singletons and lifecycle flags.

    Required for serving HTTP traffic. Must be fully initialized during lifespan startup.
    """

    # Domain singletons
    connection_manager: "ConnectionManager"
    knowledge_graph: "ContextStore"
    tws_client: "ITWSClient"
    agent_manager: "AgentManager"
    hybrid_router: "HybridRouter"
    idempotency_manager: "IdempotencyManager"
    llm_service: "LLMService"

    # Lifecycle flags
    startup_complete: bool
    redis_available: bool

    # Test/shutdown marker
    domain_shutdown_complete: bool = False


def enterprise_state_from_app(app: FastAPI) -> EnterpriseState:
    """Return typed enterprise state from app."""
    return cast(EnterpriseState, getattr(app.state, "enterprise_state"))


def enterprise_state_from_request(request: Request) -> EnterpriseState:
    """Return typed enterprise state from request."""
    return enterprise_state_from_app(request.app)
