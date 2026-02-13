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
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

import structlog
from resync.core.utils.llm_factories import LLMFactory

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = "END"
    class StateGraph:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass


# PostgresSaver requires psycopg3 with libpq - fallback to MemorySaver if unavailable
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore
    POSTGRES_SAVER_AVAILABLE = False

from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_async_session
from resync.workflows.nodes import (
    correlate_metrics,
    detect_degradation,
    fetch_job_history,
    fetch_workstation_metrics,
    generate_recommendations,
    notify_operators,
    predict_timeline,
)

logger = structlog.get_logger(__name__)

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
    state: PredictiveMaintenanceState,
    db: AsyncSession
) -> PredictiveMaintenanceState:
    """
    Step 1: Fetch historical data.

    Busca:
    - Job execution history (30 days)
    - Workstation metrics (30 days)
    - Joblog patterns (failures)
    """
    logger.info(
        "predictive_maintenance.fetch_data",
        job_name=state["job_name"],
        lookback_days=state["lookback_days"]
    )

    try:
        # Fetch job history
        job_history = await fetch_job_history(
            db=db,
            job_name=state["job_name"],
            days=state["lookback_days"]
        )

        # Get workstation from job history
        if job_history:
            workstation = job_history[0].get("workstation")

            # Fetch workstation metrics
            workstation_metrics = await fetch_workstation_metrics(
                db=db,
                workstation=workstation,
                days=state["lookback_days"]
            )
        else:
            workstation_metrics = []

        return {
            **state,
            "job_history": job_history,
            "workstation_metrics": workstation_metrics,
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error(
            "predictive_maintenance.fetch_data_failed",
            error=str(e)
        )
        return {
            **state,
            "status": "failed",
            "error": f"Failed to fetch data: {str(e)}"
        }


async def analyze_degradation_node(
    state: PredictiveMaintenanceState,
    llm: Any # Typed as Any to support flexible backend (ChatLiteLLM)
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
            job_history=state["job_history"],
            llm=llm
        )

        return {
            **state,
            "degradation_detected": degradation_result["detected"],
            "degradation_type": degradation_result.get("type"),
            "degradation_severity": degradation_result.get("severity", 0.0),
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error(
            "predictive_maintenance.analyze_degradation_failed",
            error=str(e)
        )
        return {
            **state,
            "error": f"Degradation analysis failed: {str(e)}"
        }


async def correlate_node(
    state: PredictiveMaintenanceState,
    llm: Any
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
            llm=llm
        )

        return {
            **state,
            "correlation_found": correlation_result["found"],
            "root_cause": correlation_result.get("root_cause"),
            "contributing_factors": correlation_result.get("factors", []),
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error(
            "predictive_maintenance.correlate_failed",
            error=str(e)
        )
        return state


async def predict_node(
    state: PredictiveMaintenanceState,
    llm: Any
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
            llm=llm
        )

        return {
            **state,
            "failure_probability": prediction_result["probability"],
            "estimated_failure_date": prediction_result.get("date"),
            "confidence": prediction_result["confidence"],
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error(
            "predictive_maintenance.predict_failed",
            error=str(e)
        )
        return state


async def recommend_node(
    state: PredictiveMaintenanceState,
    llm: Any
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
            llm=llm
        )

        # Determine if human review is needed
        requires_review = state["confidence"] < 0.8 or state["failure_probability"] > 0.7

        return {
            **state,
            "recommendations": recommendations_result["recommendations"],
            "preventive_actions": recommendations_result["actions"],
            "requires_human_review": requires_review,
            "status": "pending_review" if requires_review else "running",
        }

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        logger.error(
            "predictive_maintenance.recommend_failed",
            error=str(e)
        )
        return state


async def human_review_node(
    state: PredictiveMaintenanceState
) -> PredictiveMaintenanceState:
    """
    Step 6: Human review (se necessário).

    Este node PAUSA o workflow até receber input humano.
    LangGraph checkpoint permite resumir depois.
    """
    logger.info(
        "predictive_maintenance.human_review",
        workflow_id=state["workflow_id"]
    )

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
    return {
        **state,
        "status": "pending_review"
    }


async def execute_actions_node(
    state: PredictiveMaintenanceState,
    db: AsyncSession
) -> PredictiveMaintenanceState:
    """
    Step 7: Execute preventive actions (optional).

    Actions podem incluir:
    - Adjust workstation limits
    - Archive old data
    - Scale resources
    - Reschedule jobs
    """
    logger.info("predictive_maintenance.execute_actions")

    if not state.get("human_approved"):
        # Human rejected, skip execution
        return {
            **state,
            "actions_executed": [],
            "execution_results": {},
            "status": "completed"
        }

    # Action execution is simulated until real TWS integration is implemented

    actions_executed = []
    execution_results = {}

    for action in state["preventive_actions"]:
        action_type = action.get("type")

        try:
            # Execute action
            result = await execute_preventive_action(
                action=action,
                db=db
            )

            actions_executed.append(action_type)
            execution_results[action_type] = result

            logger.info(
                "predictive_maintenance.action_executed",
                action=action_type,
                result=result
            )

        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            logger.error(
                "predictive_maintenance.action_failed",
                action=action_type,
                error=str(e)
            )
            execution_results[action_type] = {
                "status": "failed",
                "error": str(e)
            }

    return {
        **state,
        "actions_executed": actions_executed,
        "execution_results": execution_results,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc)
    }


async def execute_preventive_action(
    action: dict[str, Any],
    db: AsyncSession
) -> dict[str, Any]:
    """
    Execute a specific preventive action.

    Execution is simulated — returns placeholder result.
    Implement real TWS action execution before enabling in production.
    """
    await asyncio.sleep(0.1)
    return {"status": "simulated", "details": "Action not yet implemented — enable PREDICTIVE_MAINTENANCE_EXECUTE=true when ready"}


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue_after_fetch(
    state: PredictiveMaintenanceState
) -> Literal["analyze", "end"]:
    """Route after data fetch."""
    if state.get("error"):
        return "end"
    if not state["job_history"]:
        logger.warning("predictive_maintenance.no_data")
        return "end"
    return "analyze"


def should_continue_after_recommend(
    state: PredictiveMaintenanceState
) -> Literal["human_review", "execute", "end"]:
    """Route after recommendations."""
    if state["requires_human_review"]:
        return "human_review"
    if state["preventive_actions"]:
        return "execute"
    return "end"


def should_continue_after_human_review(
    state: PredictiveMaintenanceState
) -> Literal["execute", "end"]:
    """Route after human review."""
    if state.get("human_approved"):
        return "execute"
    return "end"


# ============================================================================
# WORKFLOW GRAPH
# ============================================================================

def create_predictive_maintenance_workflow(
    llm: Any,
    db: AsyncSession,
    checkpointer: PostgresSaver
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
    workflow.add_node(
        "fetch_data",
        lambda state: fetch_data_node(state, db)
    )
    workflow.add_node(
        "analyze",
        lambda state: analyze_degradation_node(state, llm)
    )
    workflow.add_node(
        "correlate",
        lambda state: correlate_node(state, llm)
    )
    workflow.add_node(
        "predict",
        lambda state: predict_node(state, llm)
    )
    workflow.add_node(
        "recommend",
        lambda state: recommend_node(state, llm)
    )
    workflow.add_node(
        "human_review",
        human_review_node
    )
    workflow.add_node(
        "execute",
        lambda state: execute_actions_node(state, db)
    )

    # Define edges
    workflow.set_entry_point("fetch_data")

    # Conditional routing
    workflow.add_conditional_edges(
        "fetch_data",
        should_continue_after_fetch,
        {
            "analyze": "analyze",
            "end": END
        }
    )

    # Linear flow through analysis
    workflow.add_edge("analyze", "correlate")
    workflow.add_edge("correlate", "predict")
    workflow.add_edge("predict", "recommend")

    # Conditional routing after recommendations
    workflow.add_conditional_edges(
        "recommend",
        should_continue_after_recommend,
        {
            "human_review": "human_review",
            "execute": "execute",
            "end": END
        }
    )

    # Conditional routing after human review
    workflow.add_conditional_edges(
        "human_review",
        should_continue_after_human_review,
        {
            "execute": "execute",
            "end": END
        }
    )

    # Execute always goes to end
    workflow.add_edge("execute", END)

    # Compile with checkpointer for pause/resume
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]  # Pause before human review
    )


# ============================================================================
# WORKFLOW RUNNER
# ============================================================================

async def run_predictive_maintenance(
    job_name: str,
    lookback_days: int = 30,
    workflow_id: str | None = None
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
    # Initialize
    # Use centralized factory for LLM to ensure governance/metrics
    # Original: ChatAnthropic(model="claude-sonnet-4-20250514")
    llm = LLMFactory.get_langchain_llm(model="claude-3-sonnet-20240229") # Updated to valid model ID

    async with get_async_session() as db:
        checkpointer = PostgresSaver(db.connection())

        workflow = create_predictive_maintenance_workflow(
            llm=llm,
            db=db,
            checkpointer=checkpointer
        )

        if workflow_id:
            # Resume existing workflow
            logger.info(
                "predictive_maintenance.resume",
                workflow_id=workflow_id
            )

            config = {"configurable": {"thread_id": workflow_id}}

            # Get current state
            state = await workflow.aget_state(config)

            # Continue from checkpoint
            result = await workflow.ainvoke(
                state.values,
                config=config
            )
        else:
            # Start new workflow
            workflow_id = f"pm_{job_name}_{datetime.now(timezone.utc).timestamp()}"

            logger.info(
                "predictive_maintenance.start",
                workflow_id=workflow_id,
                job_name=job_name
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
                "error": None
            }

            config = {"configurable": {"thread_id": workflow_id}}

            result = await workflow.ainvoke(
                initial_state,
                config=config
            )

        logger.info(
            "predictive_maintenance.completed",
            workflow_id=workflow_id,
            status=result.get("status")
        )

        return result


# ============================================================================
# API FOR HUMAN REVIEW
# ============================================================================

async def approve_workflow(
    workflow_id: str,
    approved: bool,
    feedback: str | None = None
) -> dict[str, Any]:
    """
    Approve or reject workflow recommendations.

    This resumes the workflow from the human_review checkpoint.
    """
    async with get_async_session() as db:
        checkpointer = PostgresSaver(db.connection())

        llm = LLMFactory.get_langchain_llm(model="claude-3-sonnet-20240229")
        workflow = create_predictive_maintenance_workflow(
            llm=llm,
            db=db,
            checkpointer=checkpointer
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
            "human_feedback": feedback
        }

        # Resume workflow
        return await workflow.ainvoke(
            updated_state,
            config=config
        )

