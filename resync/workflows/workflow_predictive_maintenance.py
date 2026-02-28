# pylint
"""
LangGraph Workflow - Predictive Maintenance

Workflow multi-step complexo para análise preditiva de falhas em jobs TWS.

Passos:
1. Fetch historical data (jobs + metrics)
2. Detect degradation patterns
3. Correlate job slowdown with resource saturation
4. Predict failure timeline (2-4 weeks ahead)
5. Generate actionable recommendations
6. Human review (if confidence < 0.8)
7. Execute preventive actions (optional)

Author: Resync Team
Version: 1.0.0
"""

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

import structlog
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_async_session
from resync.core.utils.llm_factories import LLMFactory
from resync.workflows.nodes import (
    correlate_metrics,
    detect_degradation,
    fetch_job_history,
    fetch_workstation_metrics,
    generate_recommendations,
    notify_operators,
    predict_timeline,
)

# --- Optional Dependencies Check ---
LANGGRAPH_AVAILABLE = False
POSTGRES_SAVER_AVAILABLE = False
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
    POSTGRES_SAVER_AVAILABLE = True
except ImportError:

    class HumanMessage:
        """Placeholder class when langgraph is not available."""

        pass

    class SystemMessage:
        """Placeholder class when langgraph is not available."""

        pass

    END = "END"
    StateGraph = None  # type: ignore
    PostgresSaver = None  # type: ignore

logger = structlog.get_logger(__name__)

# ============================================================================
# CHECKPOINTER SINGLETON (prevents connection pool leak)
# ============================================================================

_checkpointer_pool: Any = None
_checkpointer_instance: Any = None
_pool_lock = asyncio.Lock()

async def get_checkpointer() -> Any | None:
    """
    Returns singleton PostgresSaver with shared connection pool.

    This prevents connection pool exhaustion by reusing a single pool
    across all workflow executions.
    """
    global _checkpointer_pool, _checkpointer_instance

    if _checkpointer_instance is not None:
        return _checkpointer_instance

    async with _pool_lock:
        if _checkpointer_instance is not None:
            return _checkpointer_instance

        if not POSTGRES_SAVER_AVAILABLE or PostgresSaver is None:
            logger.warning("postgres_checkpointer_unavailable")
            return None

        try:
            from psycopg_pool import AsyncConnectionPool

            from resync.settings import settings

            conn_url = settings.database_url.replace("postgresql+asyncpg", "postgresql")

            _checkpointer_pool = AsyncConnectionPool(
                conninfo=conn_url,
                min_size=2,
                max_size=10,
                timeout=30.0,
                max_waiting=50,
            )

            await _checkpointer_pool.open()
            await _checkpointer_pool.wait()

            _checkpointer_instance = PostgresSaver(_checkpointer_pool)
            await _checkpointer_instance.setup()

            logger.info("postgres_checkpointer_initialized", min_size=2, max_size=10)
            return _checkpointer_instance

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("postgres_checkpointer_init_failed", error=str(e))
            return None

async def close_checkpointer() -> None:
    """Close the checkpointer pool. Call on application shutdown."""
    global _checkpointer_pool, _checkpointer_instance

    if _checkpointer_pool:
        try:
            await _checkpointer_pool.close()
            logger.info("postgres_checkpointer_closed")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("postgres_checkpointer_close_failed", error=str(e))
        finally:
            _checkpointer_pool = None
            _checkpointer_instance = None

# ============================================================================
# RATE LIMITING FOR LLM CALLS
# ============================================================================

_llm_semaphore = asyncio.Semaphore(10)

async def rate_limited_llm_call(llm_func, *args, **kwargs):
    """Wrapper for LLM calls with rate limiting."""
    async with _llm_semaphore:
        return await llm_func(*args, **kwargs)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class PredictiveMaintenanceState(TypedDict):
    """State para workflow de Predictive Maintenance."""

    # Input
    job_name: str
    lookback_days: int

    # Fetched data
    job_history: list[dict[str, Any]]
    workstation_metrics: list[dict[str, Any]]

    # Analysis results
    degradation_detected: bool
    degradation_type: str | None
    degradation_severity: float  # 0.0 - 1.0

    # Correlation
    correlation_found: bool
    root_cause: str | None
    contributing_factors: list[str]

    # Prediction
    failure_probability: float  # 0.0 - 1.0
    estimated_failure_date: datetime | None
    confidence: float  # 0.0 - 1.0

    # Recommendations
    recommendations: list[dict[str, Any]]
    preventive_actions: list[dict[str, Any]]

    # Human review
    requires_human_review: bool
    human_approved: bool | None
    human_feedback: str | None

    # Execution
    actions_executed: list[str]
    execution_results: dict[str, Any]

    # Metadata
    workflow_id: str
    started_at: datetime
    completed_at: datetime | None
    status: Literal["running", "pending_review", "completed", "failed"]
    error: str | None

# ============================================================================
# NODES (WORKFLOW STEPS)
# ============================================================================

async def fetch_data_node(
    state: PredictiveMaintenanceState, db: AsyncSession | None = None
) -> PredictiveMaintenanceState:
    """
    Step 1: Fetch historical data.

    Busca:
    - Job execution history (30 days)
    - Workstation metrics (30 days)
    - Joblog patterns (failures)

    Note: Opens its own DB session if not provided to avoid holding
    connection open during LLM processing.
    """
    # Open own session if not provided to avoid holding connection during LLM processing
    if db is None:
        async with get_async_session() as session:
            return await _fetch_data_node_impl(state, session)
    return await _fetch_data_node_impl(state, db)

async def _fetch_data_node_impl(
    state: PredictiveMaintenanceState, db: AsyncSession
) -> PredictiveMaintenanceState:
    """Implementation of fetch_data_node."""
    logger.info(
        "predictive_maintenance.fetch_data",
        job_name=state["job_name"],
        lookback_days=state["lookback_days"],
    )

    try:
        # Fetch job history
        job_history = await fetch_job_history(
            db=db, job_name=state["job_name"], days=state["lookback_days"]
        )

        # Get workstation from job history
        if job_history:
            workstation = job_history[0].get("workstation")

            # Fetch workstation metrics
            workstation_metrics = await fetch_workstation_metrics(
                db=db, workstation=workstation, days=state["lookback_days"]
            )
        else:
            workstation_metrics = []

        return {
            **state,
            "job_history": job_history,
            "workstation_metrics": workstation_metrics,
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("predictive_maintenance.fetch_data_failed", error=str(e))
        return {**state, "status": "failed", "error": f"Failed to fetch data: {str(e)}"}

async def analyze_degradation_node(
    state: PredictiveMaintenanceState,
    llm: Any,  # Typed as Any to support flexible backend (ChatLiteLLM)
) -> PredictiveMaintenanceState:
    """
    Step 2: Detect degradation patterns.

    Analisa:
    - Runtime trends (crescimento > 10%/semana)
    - Failure rate trends (crescimento)
    - Return code patterns
    """
    logger.info("predictive_maintenance.analyze_degradation")

    if not state["job_history"]:
        return {
            **state,
            "degradation_detected": False,
            "degradation_type": None,
            "degradation_severity": 0.0,
        }

    try:
        # Detect degradation using LLM
        degradation_result = await detect_degradation(
            job_history=state["job_history"], llm=llm
        )

        return {
            **state,
            "degradation_detected": degradation_result["detected"],
            "degradation_type": degradation_result.get("type"),
            "degradation_severity": degradation_result.get("severity", 0.0),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("predictive_maintenance.analyze_degradation_failed", error=str(e))
        return {**state, "error": f"Degradation analysis failed: {str(e)}"}

async def correlate_node(
    state: PredictiveMaintenanceState, llm: Any
) -> PredictiveMaintenanceState:
    """
    Step 3: Correlate job degradation with resource metrics.

    Correlação:
    - Job slowdown ↔ CPU saturation
    - Job failures ↔ Memory issues
    - Job errors ↔ Disk space
    """
    logger.info("predictive_maintenance.correlate")

    if not state["degradation_detected"]:
        # No degradation, skip correlation
        return {
            **state,
            "correlation_found": False,
            "root_cause": None,
            "contributing_factors": [],
        }

    try:
        # Correlate using LLM
        correlation_result = await correlate_metrics(
            job_history=state["job_history"],
            workstation_metrics=state["workstation_metrics"],
            degradation_type=state["degradation_type"],
            llm=llm,
        )

        return {
            **state,
            "correlation_found": correlation_result["found"],
            "root_cause": correlation_result.get("root_cause"),
            "contributing_factors": correlation_result.get("factors", []),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("predictive_maintenance.correlate_failed", error=str(e))
        return state

async def predict_node(
    state: PredictiveMaintenanceState, llm: Any
) -> PredictiveMaintenanceState:
    """
    Step 4: Predict failure timeline.

    Predição:
    - Extrapolate trends (linear, exponential)
    - Estimate failure date (quando exceder threshold)
    - Calculate confidence (baseado em R² e data quality)
    """
    logger.info("predictive_maintenance.predict")

    if not state["degradation_detected"]:
        return {
            **state,
            "failure_probability": 0.0,
            "estimated_failure_date": None,
            "confidence": 0.0,
        }

    try:
        # Predict using LLM + statistical analysis
        prediction_result = await predict_timeline(
            job_history=state["job_history"],
            degradation_type=state["degradation_type"],
            degradation_severity=state["degradation_severity"],
            llm=llm,
        )

        return {
            **state,
            "failure_probability": prediction_result["probability"],
            "estimated_failure_date": prediction_result.get("date"),
            "confidence": prediction_result["confidence"],
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("predictive_maintenance.predict_failed", error=str(e))
        return state

async def recommend_node(
    state: PredictiveMaintenanceState, llm: Any
) -> PredictiveMaintenanceState:
    """
    Step 5: Generate recommendations.

    Recommendations:
    - Specific actions (increase CPU, archive data, etc)
    - Priority (critical, high, medium, low)
    - Estimated impact
    - Implementation complexity
    """
    logger.info("predictive_maintenance.recommend")

    if not state["degradation_detected"]:
        return {
            **state,
            "recommendations": [],
            "preventive_actions": [],
        }

    try:
        # Generate recommendations using LLM
        recommendations_result = await generate_recommendations(
            root_cause=state["root_cause"],
            contributing_factors=state["contributing_factors"],
            failure_probability=state["failure_probability"],
            estimated_failure_date=state["estimated_failure_date"],
            llm=llm,
        )

        # Determine if human review is needed
        requires_review = (
            state["confidence"] < 0.8 or state["failure_probability"] > 0.7
        )

        return {
            **state,
            "recommendations": recommendations_result["recommendations"],
            "preventive_actions": recommendations_result["actions"],
            "requires_human_review": requires_review,
            "status": "pending_review" if requires_review else "running",
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error("predictive_maintenance.recommend_failed", error=str(e))
        return state

async def human_review_node(
    state: PredictiveMaintenanceState,
) -> PredictiveMaintenanceState:
    """
    Step 6: Human review (se necessário).

    Este node PAUSA o workflow até receber input humano.
    LangGraph checkpoint permite resumir depois.
    """
    logger.info("predictive_maintenance.human_review", workflow_id=state["workflow_id"])

    # Notify operators
    await notify_operators(
        workflow_id=state["workflow_id"],
        job_name=state["job_name"],
        recommendations=state["recommendations"],
        failure_probability=state["failure_probability"],
        estimated_failure_date=state["estimated_failure_date"],
    )

    # Workflow will pause here
    # Resume when human provides feedback via API
    return {**state, "status": "pending_review"}

async def execute_actions_node(
    state: PredictiveMaintenanceState, db: AsyncSession | None = None
) -> PredictiveMaintenanceState:
    """
    Step 7: Execute preventive actions (optional).

    Actions podem incluir:
    - Adjust workstation limits
    - Archive old data
    - Scale resources
    - Reschedule jobs

    Note: Opens its own DB session if not provided to avoid holding
    connection open during execution.
    """
    # Open own session if not provided to avoid
    # holding connection during action execution.
    if db is None:
        async with get_async_session() as session:
            return await _execute_actions_node_impl(state, session)
    return await _execute_actions_node_impl(state, db)

async def _execute_actions_node_impl(
    state: PredictiveMaintenanceState, db: AsyncSession
) -> PredictiveMaintenanceState:
    """Implementation of execute_actions_node."""
    logger.info("predictive_maintenance.execute_actions")

    if not state.get("human_approved"):
        # Human rejected, skip execution
        return {
            **state,
            "actions_executed": [],
            "execution_results": {},
            "status": "completed",
        }

    # Action execution is simulated until real TWS integration is implemented

    actions_executed = []
    execution_results = {}

    for action in state["preventive_actions"]:
        action_type = action.get("type")

        try:
            # Execute action
            result = await execute_preventive_action(action=action, db=db)

            actions_executed.append(action_type)
            execution_results[action_type] = result

            logger.info(
                "predictive_maintenance.action_executed",
                action=action_type,
                result=result,
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error(
                "predictive_maintenance.action_failed", action=action_type, error=str(e)
            )
            execution_results[action_type] = {"status": "failed", "error": str(e)}

    return {
        **state,
        "actions_executed": actions_executed,
        "execution_results": execution_results,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc),
    }

async def execute_preventive_action(
    action: dict[str, Any], db: AsyncSession
) -> dict[str, Any]:
    """
    Execute a specific preventive action.

    Execution is simulated — returns placeholder result.
    Implement real TWS action execution before enabling in production.
    """
    await asyncio.sleep(0.1)
    return {
        "status": "simulated",
        "details": (
            "Action not yet implemented — enable "
            "PREDICTIVE_MAINTENANCE_EXECUTE=true when ready"
        ),
    }

# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue_after_fetch(
    state: PredictiveMaintenanceState,
) -> Literal["analyze", "end"]:
    """Route after data fetch."""
    if state.get("error"):
        return "end"
    if not state["job_history"]:
        logger.warning("predictive_maintenance.no_data")
        return "end"
    return "analyze"

def should_continue_after_recommend(
    state: PredictiveMaintenanceState,
) -> Literal["human_review", "execute", "end"]:
    """Route after recommendations."""
    if state["requires_human_review"]:
        return "human_review"
    if state["preventive_actions"]:
        return "execute"
    return "end"

def should_continue_after_human_review(
    state: PredictiveMaintenanceState,
) -> Literal["execute", "end"]:
    """Route after human review."""
    if state.get("human_approved"):
        return "execute"
    return "end"

# ============================================================================
# WORKFLOW GRAPH
# ============================================================================

def create_predictive_maintenance_workflow(
    llm: Any, checkpointer: Any | None = None
) -> StateGraph:
    """
    Create the Predictive Maintenance workflow graph.

    Graph structure:

    START
      ↓
    fetch_data
      ↓
    analyze_degradation
      ↓
    correlate
      ↓
    predict
      ↓
    recommend
      ↓
    [human_review] (conditional - if confidence < 0.8)
      ↓
    execute_actions (conditional - if approved)
      ↓
    END
    """

    # Create graph
    workflow = StateGraph(PredictiveMaintenanceState)

    # Add nodes
    # fetch_data_node opens its own DB session internally
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("analyze", lambda state: analyze_degradation_node(state, llm))
    workflow.add_node("correlate", lambda state: correlate_node(state, llm))
    workflow.add_node("predict", lambda state: predict_node(state, llm))
    workflow.add_node("recommend", lambda state: recommend_node(state, llm))
    workflow.add_node("human_review", human_review_node)
    # execute_actions_node opens its own DB session internally
    workflow.add_node("execute", execute_actions_node)

    # Define edges
    workflow.set_entry_point("fetch_data")

    # Conditional routing
    workflow.add_conditional_edges(
        "fetch_data", should_continue_after_fetch, {"analyze": "analyze", "end": END}
    )

    # Linear flow through analysis
    workflow.add_edge("analyze", "correlate")
    workflow.add_edge("correlate", "predict")
    workflow.add_edge("predict", "recommend")

    # Conditional routing after recommendations
    workflow.add_conditional_edges(
        "recommend",
        should_continue_after_recommend,
        {"human_review": "human_review", "execute": "execute", "end": END},
    )

    # Conditional routing after human review
    workflow.add_conditional_edges(
        "human_review",
        should_continue_after_human_review,
        {"execute": "execute", "end": END},
    )

    # Execute always goes to end
    workflow.add_edge("execute", END)

    # Compile with checkpointer for pause/resume
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["human_review"],  # Pause AFTER notification is sent
    )

# ============================================================================
# WORKFLOW RUNNER
# ============================================================================

class WorkflowRequest(BaseModel):
    """Validated request for predictive maintenance workflow."""

    job_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Job name (alphanumeric, underscore, hyphen, dot)",
    )
    lookback_days: int = Field(
        default=30, ge=1, le=90, description="Days of history (max 90 for performance)"
    )
    workflow_id: str | None = Field(
        default=None, max_length=100, description="Existing workflow ID to resume"
    )

    @field_validator("job_name")
    @classmethod
    def validate_job_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", v):
            raise ValueError(
                "job_name must contain only: letters, numbers, underscore, hyphen, dot"
            )
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("job_name cannot contain path separators")
        return v

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v: str | None) -> str | None:
        if v is not None:
            if not re.match(r"^pm_[a-zA-Z0-9_\-\.]+_[a-f0-9]+$", v):
                raise ValueError("workflow_id format invalid. Expected: pm_<job>_<id>")
            if ".." in v or "/" in v:
                raise ValueError("workflow_id cannot contain path separators")
        return v

class ApprovalRequest(BaseModel):
    """Validated request for workflow approval."""

    workflow_id: str = Field(..., max_length=100)
    approved: bool
    feedback: str | None = Field(default=None, max_length=5000)
    user_id: str = Field(..., min_length=1, max_length=100)

    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v: str) -> str:
        if not re.match(r"^pm_[a-zA-Z0-9_\-\.]+_[a-f0-9]+$", v):
            raise ValueError("workflow_id format invalid")
        if ".." in v or "/" in v:
            raise ValueError("workflow_id contains prohibited characters")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-@\.]+$", v):
            raise ValueError("user_id contains invalid characters")
        return v

async def run_predictive_maintenance(
    job_name: str, lookback_days: int = 30, workflow_id: str | None = None
) -> dict[str, Any]:
    """
    Run the Predictive Maintenance workflow.

    Args:
        job_name: Name of the job to analyze
        lookback_days: Days of history to analyze
        workflow_id: Resume existing workflow (if paused for human review)

    Returns:
        Final workflow state
    """
    try:
        request = WorkflowRequest(
            job_name=job_name, lookback_days=lookback_days, workflow_id=workflow_id
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("invalid_workflow_request", job_name=job_name[:20], error=str(e))
        raise ValueError(f"Invalid parameters: {e}")

    job_name = request.job_name
    lookback_days = request.lookback_days
    workflow_id = request.workflow_id

    if not LANGGRAPH_AVAILABLE:
        logger.error("langgraph_missing")
        return {"status": "failed", "error": "LangGraph dependency missing"}

    from resync.settings import settings

    model_name = getattr(settings, "agent_model_name", None) or getattr(
        settings, "llm_model", "gpt-4o"
    )
    llm = LLMFactory.get_langchain_llm(model=model_name)

    # Get dedicated checkpointer with its own connection pool
    checkpointer = await get_checkpointer()

    # Create workflow - db session will be opened per-node
    workflow = create_predictive_maintenance_workflow(
        llm=llm, checkpointer=checkpointer
    )

    if workflow_id:
        # Resume existing workflow
        logger.info("predictive_maintenance.resume", workflow_id=workflow_id)

        config = {"configurable": {"thread_id": workflow_id}}

        # Get current state
        state = await workflow.aget_state(config)

        # Continue from checkpoint
        result = await workflow.ainvoke(state.values, config=config)
    else:
        # Start new workflow
        workflow_id = f"pm_{job_name}_{uuid.uuid4().hex[:12]}"

        logger.info(
            "predictive_maintenance.start", workflow_id=workflow_id, job_name=job_name
        )

        initial_state: PredictiveMaintenanceState = {
            "job_name": job_name,
            "lookback_days": lookback_days,
            "job_history": [],
            "workstation_metrics": [],
            "degradation_detected": False,
            "degradation_type": None,
            "degradation_severity": 0.0,
            "correlation_found": False,
            "root_cause": None,
            "contributing_factors": [],
            "failure_probability": 0.0,
            "estimated_failure_date": None,
            "confidence": 0.0,
            "recommendations": [],
            "preventive_actions": [],
            "requires_human_review": False,
            "human_approved": None,
            "human_feedback": None,
            "actions_executed": [],
            "execution_results": {},
            "workflow_id": workflow_id,
            "started_at": datetime.now(timezone.utc),
            "completed_at": None,
            "status": "running",
            "error": None,
        }

        config = {"configurable": {"thread_id": workflow_id}}

        result = await workflow.ainvoke(initial_state, config=config)

    logger.info(
        "predictive_maintenance.completed",
        workflow_id=workflow_id,
        status=result.get("status"),
    )

    return result

# ============================================================================
# API FOR HUMAN REVIEW
# ============================================================================

async def approve_workflow(
    workflow_id: str,
    approved: bool,
    feedback: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Approve or reject workflow recommendations.

    This resumes the workflow from the human_review checkpoint.

    Args:
        workflow_id: ID of the workflow to approve/reject
        approved: True to approve, False to reject
        feedback: Optional feedback from the approver
        user_id: ID of the user making the approval (for audit trail)
    """
    import os

    if user_id is None:
        user_id = os.getenv("INTERNAL_CALL_USER", "system")

    try:
        request = ApprovalRequest(
            workflow_id=workflow_id,
            approved=approved,
            feedback=feedback,
            user_id=user_id,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("invalid_approval_request", workflow_id=workflow_id, error=str(e))
        raise ValueError(f"Invalid parameters: {e}")

    workflow_id = request.workflow_id

    logger.info(
        "workflow_approval_request",
        workflow_id=workflow_id,
        approved=approved,
        user_id=user_id,
        feedback_length=len(feedback) if feedback else 0,
    )

    # Get dedicated checkpointer with its own connection pool
    checkpointer = await get_checkpointer()

    from resync.settings import settings

    model_name = getattr(settings, "agent_model_name", None) or getattr(
        settings, "llm_model", "gpt-4o"
    )
    llm = LLMFactory.get_langchain_llm(model=model_name)
    workflow = create_predictive_maintenance_workflow(
        llm=llm, checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": workflow_id}}

    # Get current state
    state = await workflow.aget_state(config)

    if not state:
        raise ValueError(f"Workflow {workflow_id} not found")

    # Update state with human decision
    updated_state = {
        **state.values,
        "human_approved": approved,
        "human_feedback": feedback,
        "approved_by": user_id,
    }

    # Resume workflow
    return await workflow.ainvoke(updated_state, config=config)
