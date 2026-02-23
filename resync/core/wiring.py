from __future__ import annotations

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
    from resync.core.connection_manager import ConnectionManager
    from resync.core.context_store import ContextStore
    from resync.core.file_ingestor import IFileIngestor
    from resync.core.hybrid_router import HybridRouter
    from resync.core.idempotency.manager import IdempotencyManager
    from resync.core.interfaces import ITWSClient
    from resync.core.llm_service import LLMService
    from resync.core.skill_manager import SkillManager
    from resync.services.a2a_handler import A2AHandler

logger = get_logger(__name__)

# Constants for singleton state keys (matches EnterpriseState fields)
STATE_CONNECTION_MANAGER: Final[str] = "connection_manager"
STATE_KNOWLEDGE_GRAPH: Final[str] = "knowledge_graph"
STATE_TWS_CLIENT: Final[str] = "tws_client"
STATE_AGENT_MANAGER: Final[str] = "agent_manager"
STATE_HYBRID_ROUTER: Final[str] = "hybrid_router"
STATE_IDEMPOTENCY_MANAGER: Final[str] = "idempotency_manager"
STATE_LLM_SERVICE: Final[str] = "llm_service"
STATE_FILE_INGESTOR: Final[str] = "file_ingestor"
STATE_A2A_HANDLER: Final[str] = "a2a_handler"
STATE_SKILL_MANAGER: Final[str] = "skill_manager"

_REQUIRED_SINGLETONS: Final[frozenset[str]] = frozenset(
    {
        STATE_CONNECTION_MANAGER,
        STATE_KNOWLEDGE_GRAPH,
        STATE_TWS_CLIENT,
        STATE_AGENT_MANAGER,
        STATE_HYBRID_ROUTER,
        STATE_IDEMPOTENCY_MANAGER,
        STATE_LLM_SERVICE,
        STATE_FILE_INGESTOR,
        STATE_A2A_HANDLER,
        STATE_SKILL_MANAGER,
    }
)


# -----------------------------------------------------------------------------
# App-level startup/shutdown hooks (lifespan)
# -----------------------------------------------------------------------------


async def init_domain_singletons(app: FastAPI) -> None:
    """Initialize core domain services and attach to app.state.

    This is the composition root for the application.  It wires together
    the main service graph, respecting dependency order (e.g. LLM -> Agent).

    Raises:
        RuntimeError: If critical infrastructure (DB/Redis) is missing and
                      cannot be compensated for (e.g. degraded mode).
    """
    settings = get_settings()
    logger.info("initializing_domain_singletons", environment=settings.environment)

    # Import specialized factories locally to avoid top-level cycles
    from resync.core.agent_manager import initialize_agent_manager
    from resync.core.connection_manager import ConnectionManager
    from resync.core.context_store import ContextStore
    from resync.core.file_ingestor import FileIngestor
    from resync.core.hybrid_router import HybridRouter
    from resync.core.idempotency.manager import IdempotencyManager
    from resync.core.llm_service import get_llm_service
    from resync.core.redis_init import get_redis_client
    from resync.services.a2a_handler import A2AHandler
    from resync.services.embedding_service import MultiProviderEmbeddingService
    from resync.services.ingest_service import IngestService
    from resync.services.pg_vector_store import PgVectorStore
    from resync.services.rag_client import get_rag_client_singleton
    from resync.core.factories.tws_factory import get_tws_client_singleton

    # Core Infrastructure
    connection_manager = ConnectionManager()
    knowledge_graph = ContextStore()

    # TWS Client (Mock vs Real)
    if getattr(settings, "TWS_MOCK_MODE", False):
        from resync.services.mock_tws_service import MockTWSClient

        tws: ITWSClient = MockTWSClient()
        logger.info("tws_client_mode", mode="mock_enabled")
    else:
        try:
            tws = get_tws_client_singleton()  # type: ignore
        except Exception as exc:
            # Fallback to optimized client or fail hard?
            # For now, let's log and try once more or propagate.
            logger.error(
                "tws_client_init_failed", error=str(exc), hint="Check TWS credentials."
            )
            # Retry logic or fallback logic could go here.
            # Re-raising ensures we don't start with a broken singleton
            # unless we explicitly support optional TWS.
            # raise RuntimeError("Failed to initialize TWS client") from exc

            # Attempt a safe fallback if appropriate for the domain
            from resync.core.factories.tws_factory import get_tws_client_singleton

            logger.info("tws_client_mode", mode="optimized_fallback")

    # Agent manager depends on settings + TWS reference
    agent_manager = initialize_agent_manager(
        settings_module=settings, tws_client_factory=lambda: tws
    )

    # Skill manager - loads skills metadata (lightweight, no heavy IO)
    from resync.core.skill_manager import SkillManager

    skill_manager = SkillManager()

    # Router depends on agent manager and skill manager
    hybrid_router = HybridRouter(
        agent_manager=agent_manager, skill_manager=skill_manager
    )

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
    llm_service = await get_llm_service()

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
    enterprise_state = EnterpriseState(  # type: ignore
        connection_manager=connection_manager,
        knowledge_graph=knowledge_graph,
        tws_client=tws,
        agent_manager=agent_manager,
        hybrid_router=hybrid_router,
        idempotency_manager=idempotency_manager,
        llm_service=llm_service,
        file_ingestor=file_ingestor,
        a2a_handler=a2a_handler,
        skill_manager=skill_manager,
        startup_complete=False,
        redis_available=redis_available,
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
    """Call shutdown() or close() on *obj* if available, logging errors.

    This centralises the best-effort teardown pattern used by
    shutdown_domain_singletons to avoid duplicated try/except/pass blocks.
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

    Called from the lifespan finally block.  Must never raise — failures
    are logged at warning level so they are visible in observability
    tooling (ELK/Grafana) without blocking the shutdown sequence.

    Reads all singletons from enterprise_state (the same place
    init_domain_singletons writes to), ensuring teardown consistency.

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
        logger.warning(
            "rag_client_close_error", error=type(exc).__name__, detail=str(exc)
        )

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
# These are thin read-only accessors into enterprise_state.  They are
# consumed via Depends(get_xxx) in route handlers.
# -----------------------------------------------------------------------------


def get_connection_manager(request: Request) -> ConnectionManager:
    """Provide the ConnectionManager singleton for a request."""
    return enterprise_state_from_request(request).connection_manager


def get_knowledge_graph(request: Request) -> ContextStore:
    """Provide the ContextStore (knowledge graph) singleton for a request."""
    return enterprise_state_from_request(request).knowledge_graph


def get_tws_client(request: Request) -> ITWSClient:
    """Provide the ITWSClient singleton for a request."""
    return enterprise_state_from_request(request).tws_client


def get_agent_manager(request: Request) -> AgentManager:
    """Provide the AgentManager singleton for a request."""
    return enterprise_state_from_request(request).agent_manager


def get_hybrid_router(request: Request) -> HybridRouter:
    """Provide the HybridRouter singleton for a request."""
    return enterprise_state_from_request(request).hybrid_router


def get_idempotency_manager(request: Request) -> IdempotencyManager:
    """Provide the IdempotencyManager singleton for a request."""
    return enterprise_state_from_request(request).idempotency_manager


def get_llm_service(request: Request) -> LLMService:
    """Provide the LLMService singleton for a request."""
    return enterprise_state_from_request(request).llm_service


def get_file_ingestor(request: Request) -> IFileIngestor:
    """Provide the IFileIngestor singleton for a request."""
    return enterprise_state_from_request(request).file_ingestor


def get_a2a_handler(request: Request) -> A2AHandler:
    """Provide the A2AHandler singleton for a request."""
    return enterprise_state_from_request(request).a2a_handler


def get_skill_manager(request: Request) -> SkillManager:
    """Provide the SkillManager singleton for a request."""
    return enterprise_state_from_request(request).skill_manager


# -----------------------------------------------------------------------------
# Request-scoped resource example (Depends-compatible pattern)
# -----------------------------------------------------------------------------


def request_context() -> Iterator[dict[str, str]]:
    """Example request-scoped resource using yield semantics.

    This follows the FastAPI "Dependencies with yield" pattern:
    the code before yield runs before the request handler, and
    the code after yield runs after the response is sent (cleanup).

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
