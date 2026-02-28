# ruff: noqa: E501
"""
LangGraph Agent Graph Implementation v6.1.0 (Golden Path).

v6.1 enhancements over v6.0:
- Deterministic planner with template-based step decomposition
- ServiceOrchestrator integration for parallel data collection
- Verification loop for post-action confirmation against real TWS state
- Output critique (LLM-based, troubleshoot-only, max 2 retries)
- Transient state reset between turns (checkpoint safety)
- LLM rescue fallback when deterministic plan aborts
- Webhook/alert proactive entry with pre-defined intent

Architecture:
    START
      │
      ▼
    [Router] (resets transient state; skips classification if webhook)
      │
      ├─► (needs_clarification) ─► [Clarify] ─► END
      │
      ├─► (status/troubleshoot/action) ─► [Planner] ─► [Plan Executor] ──┐
      │                                       │    ▲  (loop)              │
      │                                       │    └──────────────────────│
      │                                       └─► (plan_failed) ─► [LLM Rescue]
      │                                                                   │
      ├─► [Query] (RAG) ──────────────────────────────────────────────────┤
      ├─► [General] ──────────────────────────────────────────────────────┤
      │                                                                   │
      ▼                                                                   ▼
    [Synthesizer]
      │
      ├─► (troubleshoot) ─► [Output Critique] ──┐
      │                          │    ▲  (max 2) │
      │                          └────┘          │
      ├──────────────────────────────────────────┤
      ▼                                          ▼
    [Hallucination Check]
      │
      ▼
     END
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, TypedDict

from resync.core.langfuse import PromptType, get_prompt_manager
from resync.core.langgraph.models import Intent, RouterOutput
from resync.core.langgraph.templates import (
    get_clarification_question,
    get_status_translation,
    render_template,
)
from resync.core.metrics import runtime_metrics
from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

def _metric_inc(metric: Any, value: float = 1.0) -> None:
    """Increment metric counters compatible with both wrapper and Prometheus APIs."""
    if hasattr(metric, "increment"):
        metric.increment(value)
        return
    if hasattr(metric, "inc"):
        metric.inc(value)

# Import SemanticCache for router intent caching
try:
    from resync.core.cache.semantic_cache import SemanticCache

    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    SemanticCache = None

# LangGraph imports
try:
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    interrupt = None

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict, total=False):
    """State passed between nodes in the graph."""

    # Input
    message: str
    user_id: str | None
    session_id: str | None
    tws_instance_id: str | None
    conversation_history: list[dict[str, str]]

    # Classification
    intent: Intent
    confidence: float
    entities: dict[str, Any]

    # Clarification
    needs_clarification: bool
    missing_entities: list[str]
    clarification_question: str
    clarification_context: dict[str, Any]

    # Processing
    current_node: str
    retry_count: int
    max_retries: int

    # Tool execution
    tool_name: str | None
    tool_input: dict[str, Any]
    tool_output: str | None
    tool_error: str | None

    # Raw data for synthesis
    raw_data: dict[str, Any]

    # Approval
    requires_approval: bool
    approval_status: str | None

    # Output
    response: str
    metadata: dict[str, Any]
    error: str | None

    # Hallucination
    is_grounded: bool
    hallucination_retry_count: int

    # ---------------------------------------------------------------------
    # v6.1 "Golden Path" (Planner + Orchestrator + Verification + Critique)
    # NOTE: keep these values JSON-serializable because LangGraph checkpointers
    # persist state between turns. See LangGraph persistence docs.
    # ---------------------------------------------------------------------

    # Orchestrator (serialized)
    orchestration_result: dict[str, Any] | None

    # Planner / executor (serialized)
    execution_plan: dict[str, Any] | None
    plan_step_index: int
    plan_failed: bool
    rescue_used: bool

    # Verification loop
    verification_attempts: int
    action_pending_verification: str | None

    # Output critique (high-risk paths only)
    critique_feedback: list[str] | None
    critique_retries: int
    needs_refinement: bool

    # Document Knowledge Graph context (DKG)
    doc_kg_context: str

    # Config-driven limits (set from AgentGraphConfig at graph init)
    max_verification_attempts: int

@dataclass
class AgentGraphConfig:
    """Configuration for the agent graph."""

    max_retries: int = 3
    require_approval_for_actions: bool = True
    default_model: str = "gpt-4o"
    default_temperature: float = 0.7
    enable_hallucination_check: bool = True
    enable_output_critique: bool = True
    max_verification_attempts: int = 3

# Required entities per intent
REQUIRED_ENTITIES = {
    Intent.STATUS: ["job_name"],
    Intent.TROUBLESHOOT: ["job_name"],
    Intent.ACTION: ["job_name", "action_type"],
    Intent.QUERY: [],
    Intent.GENERAL: [],
    Intent.UNKNOWN: [],
}

# Entity patterns for extraction
ENTITY_PATTERNS = {
    "job_name": [
        r"job\s+([A-Z0-9_-]+)",
        r"([A-Z][A-Z0-9_-]{3,})",
        r"processo\s+([A-Z0-9_-]+)",
    ],
    "workstation": [
        r"workstation\s+([A-Z0-9_-]+)",
        r"ws[_-]?([A-Z0-9]+)",
    ],
    "action_type": [
        r"(cancelar|reiniciar|executar|parar|submit|rerun|hold|release)",
    ],
}

# Router Cache Singleton (threshold=0.95 for high precision)
# Thread-safe initialization with asyncio.Lock
import threading as _threading_module  # FIX P2-09: module-level import for sync lock

_router_cache_instance: SemanticCache | None = None
_router_cache_lock: asyncio.Lock | None = None
_router_cache_sync_lock: _threading_module.Lock = _threading_module.Lock()  # FIX P2-09: shared module-level lock

def _get_sync_init_lock() -> _threading_module.Lock:
    """Return the module-level threading.Lock for sync-context initialization."""
    return _router_cache_sync_lock

def _get_router_cache() -> SemanticCache | None:
    """Get or create the router cache singleton (thread-safe with asyncio.Lock).

    Uses double-checked locking pattern for safe lazy initialization
    in both sync and async contexts.
    """
    global _router_cache_instance, _router_cache_lock

    if not SEMANTIC_CACHE_AVAILABLE:
        return None

    # Fast path: already initialized (no lock needed)
    if _router_cache_instance is not None:
        return _router_cache_instance

    # Initialize lock on first use (lazy initialization)
    if _router_cache_lock is None:
        _router_cache_lock = asyncio.Lock()

    # Double-checked locking: first check without lock, then acquire lock
    if _router_cache_instance is None:
        # Use synchronous locking pattern
        # Note: In async context, prefer async_init_router_cache()
        try:
            # Try to acquire lock synchronously (works in sync context)
            # If we're in an async context, we can't use sync lock acquisition
            # Fall back to simple creation (best effort - race condition possible
            # but unlikely in practice due to GIL)
            if _router_cache_instance is None:
                try:
                    _router_cache_instance = SemanticCache(
                        threshold=0.95,
                        default_ttl=3600,
                    )
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    import sys as _sys
                    from resync.core.exception_guard import maybe_reraise_programming_error
                    _exc_type, _exc, _tb = _sys.exc_info()
                    maybe_reraise_programming_error(_exc, _tb)

                    logger.warning("router_cache_init_failed", error=str(e))
                    return None
        except RuntimeError:
            # No running event loop — we're in a sync context (e.g. CLI tool).
            # FIX P2-09: Original code created a new threading.Lock() on every call
            # which means the lock provides no mutual exclusion (each caller gets
            # their own lock object). Use the module-level _sync_init_lock instead.
            import threading as _threading

            _sync_init_lock: _threading.Lock = _get_sync_init_lock()
            with _sync_init_lock:
                if _router_cache_instance is None:
                    try:
                        _router_cache_instance = SemanticCache(
                            threshold=0.95,
                            default_ttl=3600,
                        )
                    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                        import sys as _sys
                        from resync.core.exception_guard import maybe_reraise_programming_error
                        _exc_type, _exc, _tb = _sys.exc_info()
                        maybe_reraise_programming_error(_exc, _tb)

                        logger.warning("router_cache_init_failed", error=str(e))
                        return None

    return _router_cache_instance

async def async_init_router_cache() -> SemanticCache | None:
    """Async version of router cache initialization (preferred for async contexts).

    This is the proper async-safe way to get/create the router cache
    when called from within an async function.
    """
    global _router_cache_instance, _router_cache_lock

    if not SEMANTIC_CACHE_AVAILABLE:
        return None

    # Fast path: already initialized
    if _router_cache_instance is not None:
        return _router_cache_instance

    # Initialize lock on first use
    if _router_cache_lock is None:
        _router_cache_lock = asyncio.Lock()

    # Acquire lock for thread-safe initialization
    async with _router_cache_lock:
        # Double-check after acquiring lock
        if _router_cache_instance is None:
            try:
                _router_cache_instance = SemanticCache(
                    threshold=0.95,
                    default_ttl=3600,
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning("router_cache_init_failed", error=str(e))
                return None

    return _router_cache_instance

def _get_knowledge_graph_or_stub() -> Any:
    """Return the Knowledge Graph client (RAG) or a safe stub.

    The ServiceOrchestrator expects an object with an async method
    ``search`` or similar. Our RAGClient provides ``search``; if it isn't
    available or raises, we fall back to a stub returning no context.
    """

    try:
        from resync.services.rag_client import RAGClient

        return RAGClient()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise

        class _KGStub:
            """Stub that satisfies both RAGClient and Orchestrator interfaces."""

            async def get_relevant_context(self, query: str) -> str | None:
                """Used by ServiceOrchestrator._get_kg_context_safe()."""
                return None

            async def search(self, *args: Any, **kwargs: Any) -> dict:
                """Used by query_handler_node RAG path."""
                return {"results": []}

        return _KGStub()

# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

async def router_node(state: AgentState) -> AgentState:
    """
    Classify intent using structured output.

    Uses Pydantic model for guaranteed JSON parsing.
    """
    import re

    logger.debug("router_node_start", message=state.get("message", "")[:50])

    # ------------------------------------------------------------------
    # v6.1 hardening: reset transient per-turn fields
    #
    # LangGraph checkpointers persist state between turns/super-steps.
    # If a user asks a second question in the same thread, we must NOT
    # accidentally reuse the previous plan/orchestration/verification.
    # ------------------------------------------------------------------
    transient_defaults: dict[str, Any] = {
        "execution_plan": None,
        "plan_step_index": 0,
        "plan_failed": False,
        "rescue_used": False,
        "orchestration_result": None,
        "verification_attempts": 0,
        "action_pending_verification": None,
        "critique_retries": 0,
        "critique_feedback": None,
        "needs_refinement": False,
        "doc_kg_context": "",
        "max_verification_attempts": state.get("max_verification_attempts", 3),
    }
    for k, default in transient_defaults.items():
        state[k] = default  # type: ignore[typeddict-item]

    message = state.get("message", "")
    state["needs_clarification"] = False
    state["missing_entities"] = []

    # Extract entities using patterns
    entities = {}
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities[entity_type] = match.group(1)
                break

    # Merge with clarification context if present
    clarification_ctx = state.get("clarification_context", {})
    if clarification_ctx:
        entities = {**clarification_ctx.get("entities", {}), **entities}
        state["clarification_context"] = {}

    state["entities"] = entities

    # If intent is pre-defined (e.g. webhook/alert), skip classification.
    pre_defined_intent = state.get("intent")
    if (
        pre_defined_intent
        and pre_defined_intent != Intent.UNKNOWN
        and state.get("metadata", {}).get("intent_predefined")
    ):
        state["current_node"] = "router"
        state.setdefault("confidence", 1.0)
        logger.info(
            "router_using_predefined_intent",
            intent=pre_defined_intent.value
            if isinstance(pre_defined_intent, Intent)
            else pre_defined_intent,
            source="metadata.intent_predefined",
        )
        return state

    # --- ROUTER CACHE: Check if we already understand this query ---
    # SECURITY FIX: Get user_id from state to scope cache per-user
    user_id = state.get("user_id")
    router_cache = await async_init_router_cache()
    if router_cache:
        try:
            # SECURITY FIX: Pass user_id to prevent cross-user cache leakage
            cached_intent = await router_cache.check_intent(message, user_id=user_id)

            if cached_intent:
                # Cache hit! Use the cached understanding
                logger.info(
                    "⚡ router_cache_hit",
                    intent=cached_intent.get("intent"),
                    confidence=cached_intent.get("confidence"),
                    user_id=user_id,
                )

                # Increment metric
                _metric_inc(runtime_metrics.router_cache_hits)

                # Populate state with cached data
                intent_value = cached_intent.get("intent")
                if isinstance(intent_value, str):
                    try:
                        state["intent"] = Intent(intent_value)
                    except ValueError:
                        state["intent"] = Intent.UNKNOWN
                else:
                    state["intent"] = intent_value

                state["confidence"] = cached_intent.get("confidence", 1.0)

                # Merge cached entities with extracted ones (extracted takes precedence)
                cached_entities = cached_intent.get("entities", {})
                state["entities"] = {**cached_entities, **entities}

                state["current_node"] = "router"

                # Skip LLM call - return immediately
                return state

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Graceful degradation - continue to LLM if cache fails
            logger.warning("router_cache_check_error", error=str(e))

    # Cache miss - increment metric
    _metric_inc(runtime_metrics.router_cache_misses)

    # Classify intent using LLM with structured output
    try:
        from resync.core.utils.llm import call_llm_structured

        result = await call_llm_structured(
            prompt=f"Classify the intent of: {message}",
            output_model=RouterOutput,
            model=settings.llm_model or "gpt-4o",
        )

        if result:
            state["intent"] = result.intent
            state["confidence"] = result.confidence
            entities.update(result.entities)
            state["entities"] = entities

            # --- ROUTER CACHE: Store high-confidence classifications ---
            # Only cache if confidence is high enough (> 0.8) to avoid propagating errors
            if result.confidence > 0.8 and router_cache:
                try:
                    intent_data = {
                        "intent": result.intent.value
                        if isinstance(result.intent, Intent)
                        else result.intent,
                        "entities": entities,
                        "confidence": result.confidence,
                    }
                    # SECURITY FIX: Pass user_id to prevent cross-user cache leakage
                    await router_cache.store_intent(
                        message, intent_data, user_id=user_id
                    )
                    logger.debug(
                        "router_cache_stored",
                        intent=intent_data["intent"],
                        confidence=result.confidence,
                        user_id=user_id,
                    )
                    # Increment metric
                    _metric_inc(runtime_metrics.router_cache_sets)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as cache_err:
                    import sys as _sys
                    from resync.core.exception_guard import maybe_reraise_programming_error
                    _exc_type, _exc, _tb = _sys.exc_info()
                    maybe_reraise_programming_error(_exc, _tb)

                    # Non-critical - log and continue
                    logger.warning("router_cache_store_error", error=str(cache_err))
        else:
            state = _fallback_router(state)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("router_llm_failed", error=str(e))
        state = _fallback_router(state)

    # Check for missing required entities
    intent = state.get("intent", Intent.GENERAL)
    required = REQUIRED_ENTITIES.get(intent, [])
    missing = [e for e in required if not entities.get(e)]

    if missing and intent in [Intent.STATUS, Intent.TROUBLESHOOT, Intent.ACTION]:
        state["needs_clarification"] = True
        state["missing_entities"] = missing
        state["clarification_context"] = {
            "intent": intent.value if isinstance(intent, Intent) else intent,
            "entities": entities,
            "original_message": message,
        }

    state["current_node"] = "router"

    logger.info(
        "router_complete",
        intent=intent.value if isinstance(intent, Intent) else intent,
        confidence=state.get("confidence", 0),
        needs_clarification=state["needs_clarification"],
    )

    return state

async def document_kg_context_node(state: AgentState) -> AgentState:
    """Enrich state with Document-KG GraphRAG context for downstream LLM calls.

    This node is a no-op when KG_RETRIEVAL_ENABLED is disabled or when
    the import fails. The context is stored in state['doc_kg_context'].

    Runs between router and planner/handlers in the v6.1 graph.
    """
    state["current_node"] = "document_kg_context"
    try:
        from resync.core.document_graphrag import DocumentGraphRAG

        dgr = DocumentGraphRAG()
        ctx = await dgr.build_context(
            tenant=(state.get("entities") or {}).get("tenant", "default"),
            graph_version=int(
                (state.get("entities") or {}).get("graph_version", 1) or 1
            ),
            state=state,
        )
        state["doc_kg_context"] = ctx or ""
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.debug("document_kg_context_skipped", error=str(e))
        state["doc_kg_context"] = ""
    return state

def _fallback_router(state: AgentState) -> AgentState:
    """Fallback using keyword matching."""
    message = state.get("message", "").lower()

    if any(kw in message for kw in ["status", "estado", "workstation"]):
        intent, confidence = Intent.STATUS, 0.7
    elif any(kw in message for kw in ["erro", "error", "falha", "abend"]):
        intent, confidence = Intent.TROUBLESHOOT, 0.7
    elif any(kw in message for kw in ["cancelar", "reiniciar", "executar"]):
        intent, confidence = Intent.ACTION, 0.8
    elif any(kw in message for kw in ["como", "o que", "qual", "documentação"]):
        intent, confidence = Intent.QUERY, 0.6
    else:
        intent, confidence = Intent.GENERAL, 0.5

    state["intent"] = intent
    state["confidence"] = confidence
    return state

def clarification_node(state: AgentState) -> AgentState:
    """Generate clarification question for missing entities."""
    missing = state.get("missing_entities", [])
    intent = state.get("intent", Intent.GENERAL)
    entities = state.get("entities", {})

    if not missing:
        state["response"] = "Como posso ajudar?"
        return state

    # Get first missing entity
    entity = missing[0]
    intent_value = intent.value if isinstance(intent, Intent) else intent

    question = get_clarification_question(
        entity_type=entity,
        language="pt",
        action=intent_value,
        job_name=entities.get("job_name", ""),
    )

    state["clarification_question"] = question
    state["response"] = f"❓ {question}"
    state["current_node"] = "clarification"

    return state

def planner_node(state: AgentState) -> AgentState:
    """Create a deterministic execution plan for high-impact intents.

    - STATUS / TROUBLESHOOT / ACTION use templates (no LLM cost).
    - Other intents bypass planning.
    """
    state["current_node"] = "planner"

    from resync.core.langgraph.plan_templates import create_plan

    intent = state.get("intent", Intent.GENERAL)
    intent_to_template = {
        Intent.STATUS: "status",
        Intent.TROUBLESHOOT: "troubleshoot",
        Intent.ACTION: "action",
    }
    template_key = intent_to_template.get(intent)
    if not template_key:
        state["execution_plan"] = None
        state["plan_step_index"] = 0
        return state

    state["execution_plan"] = create_plan(template_key)
    state["plan_step_index"] = 0

    logger.info(
        "planner_created_plan",
        template=template_key,
        total_steps=state["execution_plan"]["total_steps"],
    )

    return state

def _step_completed(plan: dict[str, Any], step_id: str) -> bool:
    for s in plan.get("steps", []):
        if s.get("id") == step_id:
            return bool(s.get("completed"))
    return False

async def plan_executor_node(state: AgentState) -> AgentState:
    """Execute one step of the plan (looped by the graph)."""

    state["current_node"] = "plan_executor"
    plan = state.get("execution_plan")
    if not plan:
        return state

    steps = plan.get("steps", [])
    idx = int(state.get("plan_step_index", 0) or 0)
    if idx >= len(steps):
        return state

    step = steps[idx]
    # dependency check — must early-return to prevent executing with missing prereqs
    for req in step.get("requires", []) or []:
        if not _step_completed(plan, req):
            step["error"] = f"dependency_not_met:{req}"
            step["completed"] = False
            if step.get("on_failure") == "abort":
                state["plan_failed"] = True
            else:
                # Skip this step and advance
                state["plan_step_index"] = idx + 1
            plan["steps"] = steps
            state["execution_plan"] = plan
            return state

    action = step.get("action")

    try:
        if action == "orchestrator_collect":
            state = await _execute_orchestrator_collect(state, include_logs=True)
        elif action == "orchestrator_collect_light":
            state = await _execute_orchestrator_collect(state, include_logs=False)
        elif action == "analyze_evidence":
            state = _execute_analyze_evidence(state)
        elif action == "validate_action":
            state = await _execute_validate_action(state)
        elif action == "request_approval":
            state = _execute_request_approval(state)
        elif action == "execute_action":
            state = await _execute_tws_action(state)
        elif action == "verify_action":
            state = await _execute_verification_once(state)
        # synthesize_diagnosis / format_status are handled later by synthesizer

        # Verification step may request retry without advancing the plan.
        if action == "verify_action" and (state.get("raw_data", {}) or {}).get(
            "_verification_retry"
        ):
            step["completed"] = False
            step["error"] = None
            # Do not advance step index; the conditional edge will loop back.
            state["execution_plan"] = plan
            state["plan_step_index"] = idx
            return state

        step["completed"] = True
        step["error"] = None

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        step["completed"] = False
        step["error"] = str(e)
        if step.get("on_failure") == "abort":
            state["plan_failed"] = True

    # advance to next step
    state["plan_step_index"] = idx + 1
    plan["steps"] = steps
    state["execution_plan"] = plan
    return state

def _after_plan_executor(state: AgentState) -> str:
    """Decide next hop after executing a plan step."""

    if state.get("plan_failed"):
        # Template failed; use rescue path for troubleshoot only.
        if state.get("intent") == Intent.TROUBLESHOOT and not state.get("rescue_used"):
            return "llm_rescue"
        return "synthesizer"

    plan = state.get("execution_plan")
    if not plan:
        return "synthesizer"

    idx = int(state.get("plan_step_index", 0) or 0)
    total = int(plan.get("total_steps", 0) or 0)
    if idx < total:
        return "plan_executor"
    return "synthesizer"

async def _execute_orchestrator_collect(
    state: AgentState, include_logs: bool
) -> AgentState:
    """Collect status/logs/deps/history in parallel using ServiceOrchestrator."""
    from resync.core.factories import get_tws_client_singleton
    from resync.core.orchestrator import ServiceOrchestrator

    job_name = state.get("entities", {}).get("job_name")
    if not job_name:
        raise ValueError("job_name_missing")

    tws = get_tws_client_singleton()
    kg = _get_knowledge_graph_or_stub()
    orchestrator = ServiceOrchestrator(
        tws_client=tws,
        knowledge_graph=kg,
        max_retries=2,
        timeout_seconds=15 if include_logs else 8,
    )
    result = await orchestrator.investigate_job_failure(
        job_name=job_name,
        include_logs=include_logs,
        include_dependencies=True,
    )

    state["orchestration_result"] = {
        "tws_status": result.tws_status,
        "tws_logs": result.tws_logs,
        "kg_context": result.kg_context,
        "job_dependencies": result.job_dependencies,
        "historical_failures": result.historical_failures,
        "success_rate": result.success_rate,
        "errors": result.errors,
    }

    state["raw_data"] = {
        "job_name": job_name,
        "status": result.tws_status or {},
        "logs_snippet": (result.tws_logs or "")[:1200],
        "dependencies": result.job_dependencies or [],
        "historical_failures": result.historical_failures or [],
        "kg_context": result.kg_context or "",
        "data_completeness": "{result.success_rate:.0%}",
        "orchestrator_errors": result.errors,
    }
    return state

def _execute_analyze_evidence(state: AgentState) -> AgentState:
    """Extract lightweight signals from the collected evidence."""
    raw = state.get("raw_data", {}) or {}
    status = raw.get("status") or {}

    # Normalize return code / abend code if present.
    rc = status.get("return_code") or status.get("rc")
    abend = status.get("abend_code") or status.get("abend")
    raw["signals"] = {
        "status": status.get("status"),
        "return_code": rc,
        "abend_code": abend,
        "has_logs": bool(raw.get("logs_snippet")),
        "deps_count": len(raw.get("dependencies") or []),
        "history_count": len(raw.get("historical_failures") or []),
    }
    state["raw_data"] = raw
    return state

async def _execute_validate_action(state: AgentState) -> AgentState:
    """Validate if an action is allowed for the current job status."""
    from resync.core.factories import get_tws_client_singleton

    entities = state.get("entities", {})
    job_name = entities.get("job_name")
    action_type = entities.get("action_type")
    if not job_name or not action_type:
        raise ValueError("missing_action_entities")

    # Prefer orchestrated status if available.
    current_status = None
    orch = state.get("orchestration_result") or {}
    if isinstance(orch, dict):
        st = orch.get("tws_status") or {}
        if isinstance(st, dict):
            current_status = st.get("status")

    if not current_status:
        tws = get_tws_client_singleton()
        status = await tws.get_job_status(job_name)
        current_status = (status or {}).get("status")

    invalid_combos = {
        "rerun": {"EXECUTING", "WAITING", "RUNNING"},
        "reiniciar": {"EXECUTING", "WAITING", "RUNNING"},
        "cancel": {"SUCC", "COMPLETED"},
        "cancelar": {"SUCC", "COMPLETED"},
    }
    blocked = invalid_combos.get(str(action_type).lower(), set())
    if current_status in blocked:
        raise ValueError(f"action_not_allowed:{action_type}:{current_status}")

    state.setdefault("raw_data", {})["pre_action_status"] = current_status
    return state

def _execute_request_approval(state: AgentState) -> AgentState:
    """Request HITL approval via LangGraph interrupt when available."""
    entities = state.get("entities", {})
    job_name = entities.get("job_name", "unknown")
    action_type = entities.get("action_type", "unknown")

    if interrupt is None:
        state["requires_approval"] = True
        return state

    approval = interrupt(
        {
            "type": "action_approval",
            "action": action_type,
            "job": job_name,
            "message": f"Aprovar '{action_type}' no job {job_name}?",
            "pre_action_status": (state.get("raw_data", {}) or {}).get(
                "pre_action_status"
            ),
        }
    )
    if not approval or not approval.get("approved"):
        raise ValueError("action_not_approved")

    state.setdefault("raw_data", {})["approved_by"] = approval.get("approver")
    return state

async def _execute_tws_action(state: AgentState) -> AgentState:
    """Execute an action against TWS."""
    from resync.core.factories import get_tws_client_singleton

    entities = state.get("entities", {})
    job_name = entities.get("job_name")
    action_type = entities.get("action_type")
    if not job_name or not action_type:
        raise ValueError("missing_action_entities")

    tws = get_tws_client_singleton()
    result = await tws.execute_action(str(action_type), job_name)
    state.setdefault("raw_data", {}).update(
        {"action": action_type, "action_result": result}
    )
    state["action_pending_verification"] = str(action_type)
    return state

async def _execute_verification_once(state: AgentState) -> AgentState:
    """Verify the action had effect in TWS with bounded retries.

    Uses backoff (2s, 4s, 6s...) and sets raw_data['_verification_retry']=True
    to keep the plan executor on the verification step.
    """
    import asyncio

    from resync.core.factories import get_tws_client_singleton

    entities = state.get("entities", {})
    job_name = entities.get("job_name")
    action = state.get("action_pending_verification")
    if not job_name or not action:
        raise ValueError("missing_verification_context")

    attempts = int(state.get("verification_attempts", 0) or 0)
    max_attempts = int(state.get("max_verification_attempts", 3) or 3)
    delay = (attempts + 1) * 2
    await asyncio.sleep(delay)

    tws = get_tws_client_singleton()
    status = await tws.get_job_status(job_name)
    current_status = (status or {}).get("status", "UNKNOWN")

    expected_statuses = {
        "rerun": {"EXECUTING", "WAITING", "READY", "RUNNING"},
        "reiniciar": {"EXECUTING", "WAITING", "READY", "RUNNING"},
        "cancel": {"CANCELLED", "ABEND"},
        "cancelar": {"CANCELLED", "ABEND"},
        "hold": {"HELD"},
        "release": {"WAITING", "READY", "EXECUTING"},
        "submit": {"EXECUTING", "WAITING", "READY"},
    }
    expected = expected_statuses.get(str(action).lower(), set())
    verified = current_status in expected if expected else True

    raw = state.setdefault("raw_data", {})
    raw.update(
        {
            "post_action_status": current_status,
            "action_verified": verified,
            "verification_attempt": attempts + 1,
        }
    )

    if verified:
        state["verification_attempts"] = 0
        raw["_verification_retry"] = False
        return state

    attempts += 1
    state["verification_attempts"] = attempts
    if attempts >= max_attempts:
        raw["verification_exhausted"] = True
        raw["_verification_retry"] = False
    else:
        raw["_verification_retry"] = True
    return state

async def llm_rescue_node(state: AgentState) -> AgentState:
    """Emergency fallback when the deterministic plan aborts.

    For troubleshooting, we fall back to the existing incident pipeline.
    """

    state["current_node"] = "llm_rescue"
    state["rescue_used"] = True
    # Discard the failed plan to avoid confusion.
    state["execution_plan"] = None
    state["plan_step_index"] = 0
    return await troubleshoot_handler_node(state)

async def output_critique_node(state: AgentState) -> AgentState:
    """LLM-based critique for high-risk paths only (troubleshoot/diagnostic)."""

    state["current_node"] = "output_critique"
    retries = int(state.get("critique_retries", 0) or 0)
    if retries >= 2:
        state["needs_refinement"] = False
        return state

    response = state.get("response", "")
    message = state.get("message", "")
    raw_data = state.get("raw_data", {})

    try:
        from resync.core.utils.llm import call_llm

        critique_prompt = (
            "Avalie a resposta abaixo para um operador TWS em produção. "
            "Retorne APENAS JSON com chaves: satisfactory (bool), issues (list[str]), missing (list[str]).\n\n"
            f"PERGUNTA: {message}\n\nRESPOSTA: {response}\n\n"
            f"DADOS: {json.dumps(raw_data, ensure_ascii=False, default=str)[:2000]}\n"
        )
        text = await call_llm(critique_prompt, temperature=0.1)

        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        critique = (
            json.loads(m.group(0))
            if m
            else {"satisfactory": True, "issues": [], "missing": []}
        )

        if not critique.get("satisfactory", True):
            state["critique_retries"] = retries + 1
            state["critique_feedback"] = (critique.get("issues") or []) + (
                critique.get("missing") or []
            )
            state["needs_refinement"] = True
            # Force synthesizer to regenerate with critique feedback.
            state["response"] = ""
        else:
            state["needs_refinement"] = False

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        # Never block a response on critique failures.
        state["needs_refinement"] = False

    return state

def _after_critique(state: AgentState) -> str:
    if state.get("needs_refinement") and (state.get("critique_retries", 0) or 0) < 2:
        return "synthesizer"
    return "hallucination_check"

async def status_handler_node(state: AgentState) -> AgentState:
    """Handle job status queries."""
    from resync.core.factories import get_tws_client_singleton

    state["current_node"] = "status_handler"
    job_name = state.get("entities", {}).get("job_name")

    try:
        tws = get_tws_client_singleton()

        # v6.1: prefer orchestrated enrichment (status + deps + KG) when available
        try:
            from resync.core.orchestrator import ServiceOrchestrator

            kg = _get_knowledge_graph_or_stub()
            orchestrator = ServiceOrchestrator(
                tws_client=tws,
                knowledge_graph=kg,
                max_retries=1,
                timeout_seconds=8,
            )
            result = await orchestrator.investigate_job_failure(
                job_name=job_name,
                include_logs=False,
                include_dependencies=True,
            )

            state["orchestration_result"] = {
                "tws_status": result.tws_status,
                "kg_context": result.kg_context,
                "job_dependencies": result.job_dependencies,
                "historical_failures": result.historical_failures,
                "success_rate": result.success_rate,
                "errors": result.errors,
            }

            status = result.tws_status or {"status": "UNKNOWN"}
            if result.job_dependencies:
                status = {**status, "dependencies": result.job_dependencies}

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            status = await tws.get_job_status(job_name)

        state["raw_data"] = status
        state["tool_name"] = "get_job_status"
        state["tool_output"] = json.dumps(status, ensure_ascii=False, default=str)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("status_handler_error", error=str(e))
        state["error"] = str(e)
        state["raw_data"] = {"error": str(e)}

    return state

async def troubleshoot_handler_node(state: AgentState) -> AgentState:
    """
    Handle troubleshooting using the Incident Response Pipeline.

    Uses the same cognitive model as automatic incident detection:
    Error -> Enrichment (RAG) -> Analysis (LLM) -> Response

    This gives users rich, contextualized responses with:
    - Historical context (similar past incidents)
    - Root cause hypothesis
    - Suggested actions
    """
    state["current_node"] = "troubleshoot_handler"
    job_name = state.get("entities", {}).get("job_name")
    message = state.get("message", "")

    try:
        # v6.1: enrich context in paralelo via ServiceOrchestrator (quando possível)
        enriched_context: dict[str, Any] = {"job_name": job_name} if job_name else {}

        try:
            from resync.core.factories import get_tws_client_singleton
            from resync.core.orchestrator import ServiceOrchestrator

            tws = get_tws_client_singleton()
            kg = _get_knowledge_graph_or_stub()
            orchestrator = ServiceOrchestrator(
                tws_client=tws,
                knowledge_graph=kg,
                max_retries=2,
                timeout_seconds=15,
            )
            orch = await orchestrator.investigate_job_failure(
                job_name=job_name or "",
                include_logs=True,
                include_dependencies=True,
            )

            state["orchestration_result"] = {
                "tws_status": orch.tws_status,
                "tws_logs": orch.tws_logs,
                "kg_context": orch.kg_context,
                "job_dependencies": orch.job_dependencies,
                "historical_failures": orch.historical_failures,
                "success_rate": orch.success_rate,
                "errors": orch.errors,
            }

            enriched_context.update(
                {
                    "tws_status": orch.tws_status,
                    "tws_logs_snippet": (orch.tws_logs or "")[:1200],
                    "job_dependencies": orch.job_dependencies or [],
                    "historical_failures": orch.historical_failures or [],
                    "kg_context": orch.kg_context or "",
                    "orchestrator_errors": orch.errors,
                    "data_completeness": "{orch.success_rate:.0%}",
                }
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Não bloquear troubleshooting se orquestração falhar
            enriched_context["orchestrator_error"] = str(e)

        # Inject DKG context if available
        kg_ctx = state.get("doc_kg_context", "") or ""
        if kg_ctx:
            enriched_context["kg_context"] = kg_ctx

        # Try using the Incident Response Pipeline (v6.0.0)
        from resync.core.langgraph.incident_response import (
            OutputChannel,
            handle_incident,
        )

        result = await handle_incident(
            error=message,
            component="tws" if job_name else "system",
            severity="medium",
            output_channel=OutputChannel.CHAT,
            user_context=enriched_context,
            original_query=message,
        )

        # Use the chat response from incident pipeline
        if result.get("chat_response"):
            state["response"] = result["chat_response"]
            state["raw_data"] = {
                "job_name": job_name,
                "analysis": result.get("analysis", {}),
                "related_incidents": result.get("related_incidents", []),
                "processing_time_ms": result.get("processing_time_ms"),
            }
        else:
            # Fallback to diagnostic subgraph
            state = await _fallback_troubleshoot(state, job_name, message)

        state["tool_name"] = "incident_response"

    except ImportError:
        # incident_response not available, use diagnostic subgraph
        state = await _fallback_troubleshoot(state, job_name, message)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("troubleshoot_error", error=str(e))
        state["error"] = str(e)
        state["raw_data"] = {"error": str(e)}

    return state

async def _fallback_troubleshoot(
    state: AgentState, job_name: str | None, message: str
) -> AgentState:
    """Fallback troubleshooting using diagnostic subgraph."""
    from resync.core.langgraph.subgraphs import get_diagnostic_subgraph

    diagnostic = get_diagnostic_subgraph()

    if diagnostic:
        result = await diagnostic.ainvoke(
            {
                "problem_description": message,
                "job_name": job_name,
                "max_iterations": 3,
            }
        )

        state["raw_data"] = {
            "job_name": job_name,
            "symptoms": result.get("symptoms", []),
            "root_cause": result.get("root_cause"),
            "recommendations": result.get("recommendations", []),
            "solution": result.get("solution"),
        }
    else:
        state["raw_data"] = {
            "job_name": job_name,
            "message": "Diagnostic analysis unavailable",
        }

    state["tool_name"] = "diagnostic_subgraph"
    return state

async def query_handler_node(state: AgentState) -> AgentState:
    """Handle RAG queries."""
    from resync.services.rag_client import RAGClient

    state["current_node"] = "query_handler"
    message = state.get("message", "")

    try:
        rag = RAGClient()
        results = rag.search(query=message, limit=5)

        # Build context from results
        context = "\n".join(
            [r.get("content", "")[:500] for r in results.get("results", [])]
        )

        # Generate response with LLM
        from resync.core.utils.llm import call_llm

        kg_ctx = state.get("doc_kg_context", "") or ""
        prompt = f"""Responda baseado na documentação:

Contexto:
{context}
{kg_ctx}

Pergunta: {message}

Resposta:"""

        response = await call_llm(prompt, temperature=0.3)

        state["response"] = response
        state["raw_data"] = {
            "sources": [r.get("source") for r in results.get("results", [])],
            "response": response,
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("query_error", error=str(e))
        state["error"] = str(e)
        state["response"] = "Não foi possível buscar na documentação."

    return state

async def action_handler_node(state: AgentState) -> AgentState:
    """
    Handle actions using native interrupt() for approval.

    LangGraph 0.3 feature: Uses interrupt() instead of custom approval flow.
    """
    state["current_node"] = "action_handler"

    entities = state.get("entities", {})
    job_name = entities.get("job_name", "unknown")
    action_type = entities.get("action_type", "unknown")

    # Determine risk level
    high_risk_actions = ["cancelar", "cancel", "delete"]
    risk_level = "high" if action_type in high_risk_actions else "medium"

    if interrupt is not None:
        # Use LangGraph 0.3 native interrupt
        approval = interrupt(
            {
                "type": "action_approval",
                "action": action_type,
                "job": job_name,
                "risk_level": risk_level,
                "message": f"Aprovar {action_type} no job {job_name}?",
            }
        )

        if approval.get("approved"):
            # Execute action
            try:
                from resync.core.factories import get_tws_client_singleton

                tws = get_tws_client_singleton()
                result = await tws.execute_action(action_type, job_name)

                state["raw_data"] = {
                    "action": action_type,
                    "job_name": job_name,
                    "result": result,
                    "approved_by": approval.get("approver"),
                }
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                state["error"] = str(e)
                state["raw_data"] = {"error": str(e)}
        else:
            state["raw_data"] = {
                "action": action_type,
                "job_name": job_name,
                "status": "cancelled",
                "reason": "Not approved",
            }
    else:
        # Fallback for older LangGraph versions
        state["requires_approval"] = True
        state["approval_status"] = "pending"
        state["response"] = render_template(
            "action_pending_approval",
            action_type=action_type,
            job_name=job_name,
            risk_level=risk_level,
        )

    return state

async def general_handler_node(state: AgentState) -> AgentState:
    """Handle general conversation."""
    from resync.core.utils.llm import call_llm

    state["current_node"] = "general_handler"
    message = state.get("message", "")

    try:
        prompt_manager = get_prompt_manager()
        agent_prompt = await prompt_manager.get_default_prompt(PromptType.AGENT)

        if agent_prompt:
            kg_ctx = state.get("doc_kg_context", "") or ""
            base_ctx = "Conversa geral sobre TWS."
            if kg_ctx:
                base_ctx = base_ctx + "\n\n" + kg_ctx
            system = agent_prompt.compile(context=base_ctx)
            full_prompt = f"SYSTEM: {system}\n\nUSER: {message}"
        else:
            full_prompt = (
                f"Você é um assistente especialista em TWS/HWA.\n\nUsuário: {message}"
            )

        response = await call_llm(full_prompt, temperature=0.7)
        state["response"] = response
        state["raw_data"] = {"response": response}

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("general_handler_error", error=str(e))
        state["response"] = "Olá! Como posso ajudar com o TWS hoje?"

    return state

def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesize user-friendly response from raw data."""
    state["current_node"] = "synthesizer"

    # Skip if already have good response
    if state.get("response") and not state.get("response", "").startswith("{"):
        return state

    raw_data = state.get("raw_data", {})
    intent = state.get("intent", Intent.GENERAL)
    entities = state.get("entities", {})

    job_name = entities.get("job_name", raw_data.get("job_name", "N/A"))
    status = raw_data.get("status", "")

    # Select template based on intent
    if intent == Intent.STATUS:
        if status in ["SUCC", "SUCCESS", "COMPLETED"]:
            template_name = "status_success"
        else:
            template_name = "status_error"

        state["response"] = render_template(
            template_name,
            job_name=job_name,
            status=get_status_translation(status),
            last_run=raw_data.get("last_run", "N/A"),
            workstation=raw_data.get("workstation", "N/A"),
            return_code=raw_data.get("return_code", "0"),
            additional_info="",
            error_code=raw_data.get("error_code", "N/A"),
            error_message=raw_data.get("error_message", "N/A"),
            recommendation=raw_data.get("recommendation", "Verifique os logs."),
        )

    elif intent == Intent.TROUBLESHOOT:
        recs = raw_data.get("recommendations", [])
        critique_feedback = state.get("critique_feedback") or []
        if critique_feedback:
            # Turn critique feedback into actionable bullet points.
            recs = list(recs) + [f"[Critique] {x}" for x in critique_feedback]
        state["response"] = render_template(
            "troubleshoot_analysis",
            job_name=job_name,
            problem_summary=raw_data.get("root_cause", "Em análise"),
            technical_details=json.dumps(raw_data.get("symptoms", []), indent=2),
            root_cause=raw_data.get("root_cause", "Em análise"),
            recommendations="\n".join(f"- {r}" for r in recs)
            if recs
            else "- Verificar logs",
        )

    elif intent == Intent.ACTION:
        action_type = raw_data.get("action", "Ação")
        action_verified = raw_data.get("action_verified")
        post_status = raw_data.get("post_action_status", "desconhecido")

        if action_verified:
            icon = "✅"
            result_msg = f"Confirmado! Status atual: {post_status}"
        elif raw_data.get("verification_exhausted"):
            icon = "⚠️"
            result_msg = (
                f"Ação executada, mas o status ainda é '{post_status}' após "
                f"{raw_data.get('verification_attempt', '?')} verificações. "
                "Pode levar mais tempo — recomendo verificar manualmente em alguns minutos."
            )
        else:
            # Fallback behavior for non-verified flows
            success = raw_data.get("status") != "cancelled"
            icon = "✅" if success else "❌"
            result_msg = "Sucesso" if success else raw_data.get("reason", "Falha")

        state["response"] = render_template(
            "action_result",
            icon=icon,
            action_type=action_type,
            job_name=job_name,
            result=result_msg,
            details=raw_data.get("message", ""),
        )

    elif not state.get("response"):
        # Generic fallback
        state["response"] = render_template(
            "generic_response", content=json.dumps(raw_data, indent=2)
        )

    # Add metadata
    state["metadata"] = {
        "intent": intent.value if isinstance(intent, Intent) else intent,
        "confidence": state.get("confidence", 0),
        "tool_used": state.get("tool_name"),
    }

    return state

async def hallucination_check_node(state: AgentState) -> AgentState:
    """Check response for hallucinations."""
    from resync.core.langgraph.hallucination_grader import grade_hallucination

    state["current_node"] = "hallucination_check"

    response = state.get("response", "")
    raw_data = state.get("raw_data", {})
    message = state.get("message", "")

    try:
        result = await grade_hallucination(
            documents=[json.dumps(raw_data)],
            generation=response,
            question=message,
        )

        state["is_grounded"] = result.is_grounded

        if not result.is_grounded:
            retry_count = state.get("hallucination_retry_count", 0)
            state["hallucination_retry_count"] = retry_count + 1

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("hallucination_check_error", error=str(e))
        state["is_grounded"] = True  # Assume grounded on error

    return state

# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def _get_next_node(state: AgentState) -> str:
    """Route from router to appropriate handler."""
    if state.get("needs_clarification"):
        return "clarification"

    intent = state.get("intent", Intent.GENERAL)

    routing = {
        Intent.STATUS: "status_handler",
        Intent.TROUBLESHOOT: "troubleshoot_handler",
        Intent.QUERY: "query_handler",
        Intent.ACTION: "action_handler",
        Intent.GENERAL: "general_handler",
        Intent.UNKNOWN: "general_handler",
    }

    return routing.get(intent, "general_handler")

def _get_next_node_v6_1(state: AgentState) -> str:
    """v6.1 Golden Path routing: clarification goes direct, everything else through DKG context."""
    if state.get("needs_clarification"):
        return "clarification"
    return "document_kg_context"

def _after_dkg_context(state: AgentState) -> str:
    """Route from DKG context node to the appropriate handler or planner."""
    intent = state.get("intent", Intent.GENERAL)
    if intent in {Intent.STATUS, Intent.TROUBLESHOOT, Intent.ACTION}:
        return "planner"

    routing = {
        Intent.QUERY: "query_handler",
        Intent.GENERAL: "general_handler",
        Intent.UNKNOWN: "general_handler",
    }
    return routing.get(intent, "general_handler")

def _should_retry(state: AgentState) -> str:
    """Check if we should retry or proceed."""
    if state.get("tool_error") and state.get("retry_count", 0) < state.get(
        "max_retries", 3
    ):
        return "retry"
    return "synthesizer"

def _should_regenerate(state: AgentState) -> str:
    """Check if we should regenerate due to hallucination."""
    if not state.get("is_grounded", True):
        if state.get("hallucination_retry_count", 0) < 2:
            return "regenerate"
    return "end"

# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_tws_agent_graph(
    config: AgentGraphConfig | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """
    Create the TWS agent graph (v6.0.0).

    Args:
        config: Graph configuration
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled StateGraph or FallbackGraph
    """
    config = config or AgentGraphConfig()

    if not LANGGRAPH_AVAILABLE:
        logger.warning("langgraph_unavailable_using_fallback")
        return FallbackGraph(config)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("document_kg_context", document_kg_context_node)
    graph.add_node("clarification", clarification_node)

    # v6.1 Golden Path
    graph.add_node("planner", planner_node)
    graph.add_node("plan_executor", plan_executor_node)
    graph.add_node("llm_rescue", llm_rescue_node)
    if config.enable_output_critique:
        graph.add_node("output_critique", output_critique_node)

    # Direct handlers (kept for simple intents / fallbacks)
    graph.add_node("status_handler", status_handler_node)
    graph.add_node("troubleshoot_handler", troubleshoot_handler_node)
    graph.add_node("query_handler", query_handler_node)
    graph.add_node("action_handler", action_handler_node)
    graph.add_node("general_handler", general_handler_node)
    graph.add_node("synthesizer", synthesizer_node)

    if config.enable_hallucination_check:
        graph.add_node("hallucination_check", hallucination_check_node)

    # Entry point
    graph.set_entry_point("router")

    # Router edges (v6.1 + DKG): clarification goes direct, rest through DKG context
    graph.add_conditional_edges(
        "router",
        _get_next_node_v6_1,
        {
            "clarification": "clarification",
            "document_kg_context": "document_kg_context",
        },
    )

    # DKG context routes to planner or direct handlers
    graph.add_conditional_edges(
        "document_kg_context",
        _after_dkg_context,
        {
            "planner": "planner",
            "query_handler": "query_handler",
            "general_handler": "general_handler",
        },
    )

    # Clarification ends the graph (user needs to respond)
    graph.add_edge("clarification", END)

    # Planner/executor flow
    graph.add_edge("planner", "plan_executor")
    graph.add_conditional_edges(
        "plan_executor",
        _after_plan_executor,
        {
            "plan_executor": "plan_executor",
            "llm_rescue": "llm_rescue",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("llm_rescue", "synthesizer")

    # Direct handlers to synthesizer
    for handler in [
        "query_handler",
        "general_handler",
        "status_handler",
        "troubleshoot_handler",
        "action_handler",
    ]:
        graph.add_edge(handler, "synthesizer")

    # Synthesizer -> (optional critique) -> hallucination -> end
    if config.enable_output_critique and config.enable_hallucination_check:

        def _after_synthesizer(state: AgentState) -> str:
            return (
                "output_critique"
                if state.get("intent") == Intent.TROUBLESHOOT
                else "hallucination_check"
            )

        graph.add_conditional_edges(
            "synthesizer",
            _after_synthesizer,
            {
                "output_critique": "output_critique",
                "hallucination_check": "hallucination_check",
            },
        )

        graph.add_conditional_edges(
            "output_critique",
            _after_critique,
            {
                "synthesizer": "synthesizer",
                "hallucination_check": "hallucination_check",
            },
        )

        graph.add_conditional_edges(
            "hallucination_check",
            _should_regenerate,
            {"regenerate": "router", "end": END},
        )

    elif config.enable_hallucination_check:
        graph.add_edge("synthesizer", "hallucination_check")
        graph.add_conditional_edges(
            "hallucination_check",
            _should_regenerate,
            {"regenerate": "router", "end": END},
        )
    else:
        graph.add_edge("synthesizer", END)

    # Compile
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "agent_graph_created",
        version="6.1.1",
        nodes=len(graph.nodes),
        features=[
            "planner_templates",
            "service_orchestrator",
            "verification_loop",
            "document_knowledge_graph",
            "output_critique" if config.enable_output_critique else "",
            "hallucination_check" if config.enable_hallucination_check else "",
        ],
    )

    return compiled

def create_router_graph() -> Any:
    """Create simplified router-only graph."""
    if not LANGGRAPH_AVAILABLE:
        return FallbackGraph(AgentGraphConfig())

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.set_entry_point("router")
    graph.add_edge("router", END)

    return graph.compile()

# =============================================================================
# FALLBACK IMPLEMENTATION
# =============================================================================

class FallbackGraph:
    """Fallback when LangGraph unavailable."""

    def __init__(self, config: AgentGraphConfig):
        self.config = config

    async def ainvoke(self, state: dict[str, Any]) -> AgentState:
        """Process message through sequential execution."""
        full_state: AgentState = {
            "message": state.get("message", ""),
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id"),
            "tws_instance_id": state.get("tws_instance_id"),
            "conversation_history": state.get("conversation_history", []),
            "intent": Intent.UNKNOWN,
            "confidence": 0.0,
            "entities": {},
            "needs_clarification": False,
            "missing_entities": [],
            "clarification_question": "",
            "clarification_context": state.get("clarification_context", {}),
            "current_node": "start",
            "retry_count": 0,
            "max_retries": self.config.max_retries,
            "requires_approval": False,
            "response": "",
            "metadata": {},
            "error": None,
            "raw_data": {},
            "is_grounded": True,
            "hallucination_retry_count": 0,
            # v6.1 transient fields
            "execution_plan": None,
            "plan_step_index": 0,
            "plan_failed": False,
            "rescue_used": False,
            "orchestration_result": None,
            "verification_attempts": 0,
            "action_pending_verification": None,
            "critique_retries": 0,
            "critique_feedback": None,
            "needs_refinement": False,
            "max_verification_attempts": self.config.max_verification_attempts,
            "doc_kg_context": "",
        }

        try:
            # Router
            full_state = await router_node(full_state)

            if full_state.get("needs_clarification"):
                return clarification_node(full_state)

            # DKG context enrichment (no-op when disabled)
            full_state = await document_kg_context_node(full_state)

            # Handler
            intent = full_state.get("intent", Intent.GENERAL)
            handlers = {
                Intent.STATUS: status_handler_node,
                Intent.TROUBLESHOOT: troubleshoot_handler_node,
                Intent.QUERY: query_handler_node,
                Intent.ACTION: action_handler_node,
                Intent.GENERAL: general_handler_node,
            }

            handler = handlers.get(intent, general_handler_node)
            full_state = await handler(full_state)

            # Synthesize
            full_state = synthesizer_node(full_state)

            # Hallucination check
            if self.config.enable_hallucination_check:
                full_state = await hallucination_check_node(full_state)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("fallback_error", error=str(e))
            full_state["error"] = str(e)
            full_state["response"] = f"Erro: {str(e)}"

        return full_state

    # Alias for compatibility
    invoke = ainvoke

    async def astream(self, state: dict[str, Any]):
        """Stream results."""
        result = await self.ainvoke(state)
        yield result

    async def astream_events(self, state: dict[str, Any], version: str = "v2"):
        """Stream with events (simulated for fallback)."""
        yield {"event": "on_chain_start", "name": "fallback_graph"}
        result = await self.ainvoke(state)
        yield {"event": "on_chain_end", "data": {"output": result}}

# =============================================================================
# STREAMING SUPPORT
# =============================================================================

async def stream_agent_response(
    graph,
    state: dict[str, Any],
    on_node_start: callable = None,
    on_node_end: callable = None,
    on_token: callable = None,
):
    """
    Stream agent response with callbacks.

    Args:
        graph: Compiled graph
        state: Initial state
        on_node_start: Callback(node_name) when node starts
        on_node_end: Callback(node_name, output) when node ends
        on_token: Callback(token) for LLM tokens

    Yields:
        Events from the graph execution
    """
    try:
        async for event in graph.astream_events(state, version="v2"):
            event_type = event.get("event")

            if event_type == "on_chain_start" and on_node_start:
                on_node_start(event.get("name"))

            elif event_type == "on_chain_end" and on_node_end:
                on_node_end(event.get("name"), event.get("data"))

            elif event_type == "on_llm_stream" and on_token:
                chunk = event.get("data", {}).get("chunk", "")
                on_token(chunk)

            yield event

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("streaming_error", error=str(e))
        yield {"event": "error", "error": str(e)}

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AgentState",
    "AgentGraphConfig",
    "Intent",
    "create_tws_agent_graph",
    "create_router_graph",
    "FallbackGraph",
    "stream_agent_response",
    # Individual nodes (for testing)
    "router_node",
    "clarification_node",
    "planner_node",
    "plan_executor_node",
    "llm_rescue_node",
    "output_critique_node",
    "document_kg_context_node",
    "status_handler_node",
    "troubleshoot_handler_node",
    "query_handler_node",
    "action_handler_node",
    "general_handler_node",
    "synthesizer_node",
    "hallucination_check_node",
]
