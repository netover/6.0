from __future__ import annotations

"""
Enterprise-grade application wiring (HTTP DI canonical).

HTTP request path uses ONLY FastAPI's dependency system (Depends / yield).
No contextvar-based DI container is used for HTTP.

This module provides:
- Explicit dependency providers for FastAPI
- A single place to define lifecycle-managed singletons (via lifespan/app.state)
- Clear separation between:
    * Domain singletons (created at startup, stored on app.state)
    * Request-scoped resources (Depends + yield)
    * Pure config singletons (lru_cache via ``get_settings()``)

v6.0.5+: Hardened for "no confusion / no magic" operation.
v6.1.1+: Fixed TYPE_CHECKING imports, shutdown consistency, logging.

Global State:
    All domain singletons live on ``app.state.enterprise_state`` (an
    ``EnterpriseState`` dataclass).  No module-level mutable state exists
    in this module.
"""

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Request
from resync.core.structured_logger import get_logger
from resync.core.types.app_state import (
    EnterpriseState,
    enterprise_state_from_app,
    enterprise_state_from_request,
)
from resync.settings import get_settings

if TYPE_CHECKING:
    from resync.core.agent_manager import AgentManager
    from resync.core.agent_router import HybridRouter
    from resync.core.a2a_handler import A2AHandler
    from resync.core.connection_manager import ConnectionManager
    from resync.core.context_store import ContextStore
    from resync.core.idempotency.manager import IdempotencyManager
    from resync.core.interfaces import IFileIngestor, ITWSClient
    from resync.services.llm_service import LLMService

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# app.state keys (single source of truth for attribute names)
# -----------------------------------------------------------------------------
STATE_CONNECTION_MANAGER: Final[str] = "connection_manager"
STATE_KNOWLEDGE_GRAPH: Final[str] = "knowledge_graph"
STATE_TWS_CLIENT: Final[str] = "tws_client"
STATE_AGENT_MANAGER: Final[str] = "agent_manager"
STATE_HYBRID_ROUTER: Final[str] = "hybrid_router"
STATE_IDEMPOTENCY_MANAGER: Final[str] = "idempotency_manager"
STATE_LLM_SERVICE: Final[str] = "llm_service"
STATE_FILE_INGESTOR: Final[str] = "file_ingestor"
STATE_A2A_HANDLER: Final[str] = "a2a_handler"

# -----------------------------------------------------------------------------
# Enterprise state contract (used by validate_app_state_contract)
# -----------------------------------------------------------------------------

#: Singleton attributes that must be non-None after init.
_REQUIRED_SINGLETONS: Final[tuple[str, ...]] = (
    STATE_CONNECTION_MANAGER,
    STATE_KNOWLEDGE_GRAPH,
    STATE_TWS_CLIENT,
    STATE_AGENT_MANAGER,
    STATE_HYBRID_ROUTER,
    STATE_IDEMPOTENCY_MANAGER,
    STATE_LLM_SERVICE,
    STATE_FILE_INGESTOR,
    STATE_A2A_HANDLER,
)

#: Boolean flags that must be present and typed correctly.
_REQUIRED_FLAGS: Final[tuple[str, ...]] = (
    "startup_complete",
    "redis_available",
)


def validate_app_state_contract(app: FastAPI) -> None:
    """Fail-fast if required enterprise state is missing or incomplete.

    For enterprise correctness, HTTP traffic must only be served after
    lifespan initialises ``app.state.enterprise_state`` with all required
    singletons and flags.

    Called during lifespan startup and also in integration tests.

    Raises:
        RuntimeError: If ``enterprise_state`` is absent or any required
            attribute is missing / has an invalid type.
    """
    if not hasattr(app.state, "enterprise_state"):
        raise RuntimeError(
            "Missing app.state.enterprise_state (lifespan did not initialise it)."
        )

    st = enterprise_state_from_app(app)
    missing: list[str] = []

    for attr in _REQUIRED_SINGLETONS:
        val = getattr(st, attr, None)
        if val is None:
            missing.append(attr)

    for attr in _REQUIRED_FLAGS:
        val = getattr(st, attr, None)
        if not isinstance(val, bool):
            missing.append(attr)

    if missing:
        raise RuntimeError(
            "EnterpriseState contract incomplete; missing/invalid: "
            + ", ".join(sorted(set(missing)))
        )


def init_domain_singletons(app: FastAPI) -> None:
    """Initialise domain singletons and store on ``app.state.enterprise_state``.

    This is the **only** approved singleton mechanism for the HTTP path.
    It is explicit, auditable, and deterministic.

    All heavy imports are performed locally inside this function to avoid
    circular-import issues at module load time.  The classes are only needed
    at runtime when the lifespan calls this function.

    Args:
        app: The FastAPI application instance (from lifespan).
    """
    # --- Local runtime imports (avoids circular deps at module scope) ---
    from resync.core.agent_manager import initialize_agent_manager
    from resync.core.agent_router import HybridRouter
    from resync.core.connection_manager import ConnectionManager
    from resync.core.context_store import ContextStore
    from resync.core.idempotency.manager import IdempotencyManager
    from resync.core.redis_init import get_redis_client
    from resync.services.llm_service import get_llm_service
    from resync.services.mock_tws_service import MockTWSClient
    from resync.services.rag_client import get_rag_client_singleton
    from resync.knowledge.store.pgvector_store import PgVectorStore
    from resync.knowledge.ingestion.embedding_service import MultiProviderEmbeddingService
    from resync.knowledge.ingestion.ingest import IngestService
    from resync.core.file_ingestor import FileIngestor
    from resync.core.a2a_handler import A2AHandler

    settings = get_settings()

    # Connection manager (in-memory, domain singleton)
    connection_manager = ConnectionManager()

    # Knowledge graph / context store
    knowledge_graph = ContextStore()

    # TWS client (external integration; choose mock vs real)
    if getattr(settings, "TWS_MOCK_MODE", False):
        tws = MockTWSClient()
        logger.info("tws_client_mode", mode="mock")
    else:
        # Use the resilient, unified client (async)
        # Note: wiring.py is mostly called in lifespan; we wrap it for consistency
        try:
            # We use an internal helper to get it synchronously if needed or 
            # ensure it's initialized. UnifiedTWSClient manages its own connection.
            from resync.services.tws_unified import UnifiedTWSClient
            tws = UnifiedTWSClient()
            logger.info("tws_client_mode", mode="unified_resilient")
        except Exception as e:
            logger.error("resilient_tws_init_failed", error=str(e))
            # Fallback to standard optimized client if unified fails to init
            from resync.core.factories.tws_factory import get_tws_client_singleton
            tws = get_tws_client_singleton()
            logger.info("tws_client_mode", mode="optimized_fallback")

    # Agent manager depends on settings + TWS reference
    agent_manager = initialize_agent_manager(
        settings_module=settings, tws_client_factory=lambda: tws
    )

    # Router depends on agent manager
    hybrid_router = HybridRouter(agent_manager=agent_manager)

    # Idempotency manager depends on Redis client.
    # If Redis failed during startup checks but strict=False (degraded mode),
    # we use a degraded manager that fails fast with HTTP 503.
    redis_available = False
    try:
        idempotency_manager = IdempotencyManager(get_redis_client())
        redis_available = True
    except RuntimeError:
        # Redis not available — use degraded manager that fails fast.
        # Endpoints requiring idempotency will return HTTP 503 (correct for degraded mode).
        logger.warning(
            "idempotency_manager_degraded_mode",
            hint="Redis unavailable. Idempotent endpoints will return 503.",
        )
        from resync.core.idempotency.degraded import DegradedIdempotencyManager
        idempotency_manager = DegradedIdempotencyManager()  # type: ignore[assignment]

    # LLM service with automatic fallback and circuit breakers
    llm_service = get_llm_service()

    # RAG client singleton (initializes and validates Config/URL)
    get_rag_client_singleton()

    # Initialize RAG File Ingestor
    vector_store = PgVectorStore()
    embedding_service = MultiProviderEmbeddingService()
    ingest_service = IngestService(embedder=embedding_service, store=vector_store)
    file_ingestor = FileIngestor(ingest_service=ingest_service)
    a2a_handler = A2AHandler(agent_manager=agent_manager)

    # Flags initialised to safe defaults; lifespan will flip startup_complete.
    # redis_available is set based on IdempotencyManager initialization above.
    enterprise_state = EnterpriseState(
        connection_manager=connection_manager,
        knowledge_graph=knowledge_graph,
        tws_client=tws,
        agent_manager=agent_manager,
        hybrid_router=hybrid_router,
        idempotency_manager=idempotency_manager,
        llm_service=llm_service,
        file_ingestor=file_ingestor,
        a2a_handler=a2a_handler,
        startup_complete=False,
        redis_available=redis_available,  # ✅ Now accurate based on actual Redis status
        domain_shutdown_complete=False,
    )
    app.state.enterprise_state = enterprise_state

    logger.info(
        "domain_singletons_initialized",
        singletons=list(_REQUIRED_SINGLETONS),
        tws_mode="mock" if getattr(settings, "TWS_MOCK_MODE", False) else "optimized",
    )


# -----------------------------------------------------------------------------
# Shutdown helper
# -----------------------------------------------------------------------------

async def _safe_close(obj: object, label: str) -> None:
    """Call ``shutdown()`` or ``close()`` on *obj* if available, logging errors.

    This centralises the best-effort teardown pattern used by
    ``shutdown_domain_singletons`` to avoid duplicated try/except/pass blocks.
    """
    try:
        if hasattr(obj, "shutdown"):
            res = obj.shutdown()  # type: ignore[attr-defined]
            if inspect.isawaitable(res):
                await res
        elif hasattr(obj, "close"):
            res = obj.close()  # type: ignore[attr-defined]
            if inspect.isawaitable(res):
                await res
        elif hasattr(obj, "aclose"):
            res = obj.aclose()  # type: ignore[attr-defined]
            if inspect.isawaitable(res):
                await res
    except Exception as exc:
        # During shutdown, ALL exceptions are non-fatal. Re-raising here would
        # abort the remaining shutdown steps and leak resources.
        logger.warning(
            "singleton_shutdown_error",
            component=label,
            error=type(exc).__name__,
            detail="Internal server error. Check server logs for details.",
        )


async def shutdown_domain_singletons(app: FastAPI) -> None:
    """Best-effort shutdown for domain singletons.

    Called from the lifespan ``finally`` block.  Must never raise — failures
    are logged at ``warning`` level so they are visible in observability
    tooling (ELK/Grafana) without blocking the shutdown sequence.

    Reads all singletons from ``enterprise_state`` (the same place
    ``init_domain_singletons`` writes to), ensuring teardown consistency.

    Args:
        app: The FastAPI application instance.
    """
    st = enterprise_state_from_app(app)
    st.domain_shutdown_complete = True

    logger.info("starting_graceful_shutdown")

    # 1. Agent Manager (high-level service - may use TWS, LLM, Redis)
    am = getattr(st, STATE_AGENT_MANAGER, None)
    if am is not None:
        await _safe_close(am, "agent_manager")

    # 2. LLM Service
    llm = getattr(st, STATE_LLM_SERVICE, None)
    if llm is not None:
        await _safe_close(llm, "llm_service")

    # 3. File Ingestor (uses vector store, embedding service)
    fi = getattr(st, STATE_FILE_INGESTOR, None)
    if fi is not None:
        await _safe_close(fi, "file_ingestor")

    # 4. Knowledge graph / context store
    kg = getattr(st, STATE_KNOWLEDGE_GRAPH, None)
    if kg is not None:
        await _safe_close(kg, "knowledge_graph")

    # 5. RAG client singleton (optional external integration)
    try:
        from resync.services.rag_client import close_rag_client_singleton

        await close_rag_client_singleton()
        logger.info("rag_client_closed")
    except Exception as exc:
        logger.warning("rag_client_close_error", error=type(exc).__name__, detail=str(exc))

    # 6. Connection manager
    cm = getattr(st, STATE_CONNECTION_MANAGER, None)
    if cm is not None:
        await _safe_close(cm, "connection_manager")

    # 7. TWS client (external connection)
    tws = getattr(st, STATE_TWS_CLIENT, None)
    if tws is not None:
        await _safe_close(tws, "tws_client")

    # 8. Redis client: close last (infrastructure)
    try:
        from resync.core.redis_init import close_redis_client

        await close_redis_client()
        logger.info("redis_client_closed")
    except Exception as exc:
        logger.warning(
            "redis_client_close_error",
            error=type(exc).__name__,
            detail="Internal server error. Check server logs for details.",
        )

    logger.info("domain_singletons_shutdown_completed")


# -----------------------------------------------------------------------------
# FastAPI dependencies (HTTP path)
#
# These are thin read-only accessors into ``enterprise_state``.  They are
# consumed via ``Depends(get_xxx)`` in route handlers.
# -----------------------------------------------------------------------------

def get_connection_manager(request: Request) -> ConnectionManager:
    """Provide the ``ConnectionManager`` singleton for a request."""
    return enterprise_state_from_request(request).connection_manager


def get_knowledge_graph(request: Request) -> ContextStore:
    """Provide the ``ContextStore`` (knowledge graph) singleton for a request."""
    return enterprise_state_from_request(request).knowledge_graph


def get_tws_client(request: Request) -> ITWSClient:
    """Provide the ``ITWSClient`` singleton for a request."""
    return enterprise_state_from_request(request).tws_client


def get_agent_manager(request: Request) -> AgentManager:
    """Provide the ``AgentManager`` singleton for a request."""
    return enterprise_state_from_request(request).agent_manager


def get_hybrid_router(request: Request) -> HybridRouter:
    """Provide the ``HybridRouter`` singleton for a request."""
    return enterprise_state_from_request(request).hybrid_router


def get_idempotency_manager(request: Request) -> IdempotencyManager:
    """Provide the ``IdempotencyManager`` singleton for a request."""
    return enterprise_state_from_request(request).idempotency_manager


def get_llm_service(request: Request) -> LLMService:
    """Provide the ``LLMService`` singleton for a request."""
    return enterprise_state_from_request(request).llm_service


def get_file_ingestor(request: Request) -> IFileIngestor:
    """Provide the ``IFileIngestor`` singleton for a request."""
    return enterprise_state_from_request(request).file_ingestor


def get_a2a_handler(request: Request) -> A2AHandler:
    """Provide the ``A2AHandler`` singleton for a request."""
    return enterprise_state_from_request(request).a2a_handler


# -----------------------------------------------------------------------------
# Request-scoped resource example (Depends-compatible pattern)
# -----------------------------------------------------------------------------

def request_context() -> Iterator[dict[str, str]]:
    """Example request-scoped resource using ``yield`` semantics.

    This follows the FastAPI "Dependencies with yield" pattern:
    the code before ``yield`` runs before the request handler, and
    the code after ``yield`` runs after the response is sent (cleanup).

    Usage in a route::

        @router.get("/example")
        async def example(ctx: dict = Depends(request_context)):
            return ctx

    Replace/extend for real per-request resources (DB session, UoW, etc.).
    """
    ctx = {"request_context": "active"}
    try:
        yield ctx
    finally:
        # Cleanup runs after the response is sent
        pass
