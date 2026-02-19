"""
Agent routes for FastAPI.

v5.4.1 Enhancements (PR-4, PR-5):
- Agent execution endpoint with hybrid routing
- HITL approval endpoints
- Tool catalog listing
- Execution trace retrieval

Author: Resync Team
Version: 5.4.1
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from resync.api.dependencies_v2 import get_logger
from resync.core.langgraph.roma_graph import execute_roma_query
from resync.settings import settings
from resync.api.models.requests import (
    AgentExecuteRequest,
    ApprovalListQuery,
    ApprovalRequest,
    DiagnosticRequest,
    ToolCallRequest,
)
from resync.api.models.responses_v2 import (
    AgentExecuteResponse,
    AgentInfo,
    AgentListResponse,
    ApprovalListResponse,
    ApprovalResponse,
    DiagnosticResponse,
    DiagnosticResult,
    ExecutionTrace,
    PendingApproval,
    ToolResult,
)

router = APIRouter(tags=["Agents"])


# =============================================================================
# AGENT LISTING
# =============================================================================
# Cache for agent configuration
_AGENT_CACHE: dict[str, Any] | None = None
_AGENT_CACHE_TIMESTAMP: float = 0
_AGENT_CACHE_TTL: int = 300  # 5 minutes
_AGENT_CACHE_LOCK = asyncio.Lock()


async def _load_agents_config() -> dict[str, Any]:
    """
    Load agent configuration from YAML with caching and non-blocking I/O.
    """
    global _AGENT_CACHE, _AGENT_CACHE_TIMESTAMP

    async with _AGENT_CACHE_LOCK:
        # Check cache
        now = time.time()
        if _AGENT_CACHE and (now - _AGENT_CACHE_TIMESTAMP < _AGENT_CACHE_TTL):
            return _AGENT_CACHE

        try:
            import yaml
            import aiofiles

            config_path = Path("config/agents.yaml")
            if not config_path.exists():
                return {}

            async with aiofiles.open(config_path, mode="r", encoding="utf-8") as f:
                content = await f.read()

            # Offload CPU-bound YAML parsing to thread
            config = await asyncio.to_thread(yaml.safe_load, content)

            # Update cache
            _AGENT_CACHE = config
            _AGENT_CACHE_TIMESTAMP = now
            return config

        except Exception:
            # In case of error (e.g. malformed YAML), return empty or existing cache
            # logging would be good here but we don't have the logger instance
            return _AGENT_CACHE or {}


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    logger_instance=Depends(get_logger),
) -> AgentListResponse:
    """
    List all available agents.

    Returns agents from config/agents.yaml with their current status.
    """
    try:
        # Load agents from configuration (cached & async)
        config = await _load_agents_config()

        if config and "agents" in config:
            agents = []
            for agent_config in config.get("agents", []):
                agents.append(
                    AgentInfo(
                        id=agent_config.get("id", "unknown"),
                        name=agent_config.get("name", "Unknown Agent"),
                        status="active",
                        description=agent_config.get("goal", ""),
                    )
                )

            return AgentListResponse(agents=agents, total=len(agents))

        # Fallback to default agents
        default_agents = [
            AgentInfo(
                id="tws-troubleshooting",
                name="TWS Troubleshooting Agent",
                status="active",
                description="Diagnose and resolve TWS issues",
            ),
            AgentInfo(
                id="tws-general",
                name="TWS General Assistant",
                status="active",
                description="General TWS assistance and status queries",
            ),
            AgentInfo(
                id="job-analyst",
                name="Job Analyst Agent",
                status="active",
                description="Analyze job execution patterns",
            ),
        ]

        return AgentListResponse(agents=default_agents, total=len(default_agents))

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("list_agents_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list agents",
        ) from e


@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    logger_instance=Depends(get_logger),
) -> AgentInfo:
    """
    Get details for a specific agent.
    """
    agents_response = await list_agents(logger_instance)

    for agent in agents_response.agents:
        if agent.id == agent_id:
            return agent

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Agent '{agent_id}' not found",
    )


# =============================================================================
# AGENT EXECUTION (PR-4)
# =============================================================================


@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent(
    request: AgentExecuteRequest,
    logger_instance=Depends(get_logger),
) -> AgentExecuteResponse:
    """
    Execute an agent with the given goal or message.

    The request is routed through the hybrid router which selects
    the appropriate execution path:
    - RAG-only: Quick knowledge base queries
    - Agentic: Multi-step tasks with tools
    - Diagnostic: Troubleshooting with HITL

    Use routing_mode to force a specific path.

    v6.0: Now uses HybridRouter via DI (includes skill_manager).
    """
    import time

    start_time = time.time()
    trace_id = str(uuid.uuid4())

    try:
        from resync.core.agent_router import RoutingMode

        # Get input
        input_message = request.get_input()
        if not input_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'message' or 'goal' must be provided",
            )

        # Prepare context
        context = {
            "tws_instance_id": request.tws_instance_id,
            "session_id": request.session_id,
        }

        # Determine forced routing mode
        force_mode = None
        if request.routing_mode:
            force_mode = RoutingMode(request.routing_mode)

        # Get router via DI (includes skill_manager)
        # Note: We need Request object for DI, but current signature doesn't have it
        # Using factory as fallback for this endpoint
        from resync.core.agent_router import create_router
        from resync.core.agent_manager import get_agent_manager
        from resync.core.skill_manager import SkillManager

        agent_mgr = get_agent_manager()
        skill_mgr = SkillManager()  # Creates own instance for standalone use
        router_instance = create_router(agent_mgr, skill_manager=skill_mgr)

        result = await router_instance.route(
            message=input_message,
            context=context,
            force_mode=force_mode,
        )

        processing_time = int((time.time() - start_time) * 1000)

        # Build execution trace if requested
        trace = None
        if request.include_trace:
            trace = ExecutionTrace(
                trace_id=trace_id,
                session_id=request.session_id,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration_ms=processing_time,
                input_message=input_message,
                output_message=result.response,
                routing_mode=result.routing_mode.value,
                intent=result.intent,
                confidence=result.confidence,
                handler=result.handler,
                tools_used=result.tools_used,
                success=True,
            )

        logger_instance.info(
            "agent_executed",
            trace_id=trace_id,
            routing_mode=result.routing_mode.value,
            intent=result.intent,
            confidence=result.confidence,
            tools_used=result.tools_used,
            processing_time_ms=processing_time,
        )

        return AgentExecuteResponse(
            response=result.response,
            success=True,
            routing_mode=result.routing_mode.value,
            intent=result.intent,
            confidence=result.confidence,
            handler=result.handler,
            tools_used=result.tools_used,
            processing_time_ms=processing_time,
            session_id=request.session_id,
            trace_id=trace_id,
            trace=trace,
            requires_approval=result.requires_approval,
            approval_id=result.approval_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger_instance.error("agent_execute_error", error=str(e), trace_id=trace_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent execution failed. Check server logs for details.",
        ) from e


@router.post("/roma/execute")
async def execute_roma_agent(
    query: Annotated[
        str, Query(min_length=1, description="Query to execute with ROMA orchestration")
    ],
    use_roma: Annotated[
        bool, Query(description="Must be true to enable ROMA route")
    ] = False,
    logger_instance: Annotated[Any, Depends(get_logger)] = None,
) -> dict[str, Any]:
    """Execute query via ROMA orchestration graph (feature-flagged)."""
    if not use_roma:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set use_roma=true to call this endpoint",
        )

    if not settings.ROMA_ORCHESTRATION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ROMA orchestration disabled",
        )

    try:
        result = await execute_roma_query(query)
        return {
            "success": True,
            "query": query,
            "final_response": result.get("final_response", ""),
            "plan": [task.model_dump() for task in result.get("plan", [])],
            "verification_notes": result.get("verification_notes", []),
            "execution_logs": result.get("execution_logs", []),
        }
    except Exception as e:
        logger_instance.error("roma_execute_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ROMA execution failed",
        ) from e


# =============================================================================
# TOOL MANAGEMENT
# =============================================================================


@router.get("/tools/catalog")
async def list_tools(
    user_role: str = Query("viewer", pattern="^(viewer|operator|admin|system)$"),
    tags: str | None = Query(None, description="Filter by tags (comma-separated)"),
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    List available tools from the catalog.

    Filters by user role to show only accessible tools.
    """
    try:
        from resync.tools import UserRole, get_tool_catalog

        catalog = get_tool_catalog()
        role = UserRole(user_role)

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        # Get filtered tools
        tools = catalog.list_tools(user_role=role, tags=tag_list)

        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "permission": t.permission.value,
                    "requires_approval": t.requires_approval,
                    "tags": t.tags,
                    "timeout_seconds": t.timeout_seconds,
                }
                for t in tools
            ],
            "total": len(tools),
            "user_role": user_role,
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("list_tools_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tools",
        ) from e


@router.post("/tools/execute", response_model=ToolResult)
async def execute_tool(
    request: ToolCallRequest,
    logger_instance=Depends(get_logger),
) -> ToolResult:
    """
    Execute a specific tool directly.

    This bypasses the agent and executes the tool with given parameters.
    Subject to permission checks and HITL requirements.
    """
    import time

    start_time = time.time()

    try:
        from resync.core.specialists.tools import (
            ApprovalRequiredError,
            UserRole,
            get_tool_catalog,
        )

        catalog = get_tool_catalog()
        tool_def = catalog.get(request.tool_name)

        if not tool_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{request.tool_name}' not found",
            )

        # Check permission
        role = UserRole(request.user_role)
        can_exec, reason = catalog.can_execute(request.tool_name, role)
        if not can_exec:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=reason,
            )

        # Add role context to parameters
        params = dict(request.parameters)
        params["_user_role"] = request.user_role

        # Execute tool
        try:
            result = tool_def.function(**params)
            duration = int((time.time() - start_time) * 1000)

            return ToolResult(
                tool_name=request.tool_name,
                success=True,
                result=result,
                duration_ms=duration,
            )

        except ApprovalRequiredError as e:
            # HITL approval required
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                error=f"Approval required: {e.approval_id}",
                trace_id=e.approval_id,
                duration_ms=int((time.time() - start_time) * 1000),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger_instance.error("tool_execute_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool execution failed. Check server logs for details.",
        ) from e


# =============================================================================
# HITL APPROVAL ENDPOINTS (PR-5)
# =============================================================================


@router.get("/approvals", response_model=ApprovalListResponse)
async def list_pending_approvals(
    query: ApprovalListQuery = Depends(),
    logger_instance=Depends(get_logger),
) -> ApprovalListResponse:
    """
    List pending HITL approval requests.

    Operators and admins can review pending tool executions
    that require human approval.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        pending = catalog.get_pending_approvals()

        # Filter by status/tool if requested
        if query.tool_name:
            pending = [p for p in pending if p.tool_name == query.tool_name]

        # Convert to response models
        pending_list = []
        by_tool: dict[str, int] = {}
        by_risk: dict[str, int] = {}

        for trace in pending[query.offset : query.offset + query.limit]:
            # Get tool definition for risk level
            tool_def = catalog.get(trace.tool_name)
            risk = (
                "high"
                if tool_def and tool_def.permission.value == "execute"
                else "medium"
            )

            pending_list.append(
                PendingApproval(
                    trace_id=trace.trace_id,
                    tool_name=trace.tool_name,
                    action_summary=f"Execute {trace.tool_name}",
                    input_params=trace.input_params,
                    user_id=trace.user_id,
                    requested_at=trace.timestamp,
                    risk_level=risk,
                )
            )

            # Count by tool
            by_tool[trace.tool_name] = by_tool.get(trace.tool_name, 0) + 1
            by_risk[risk] = by_risk.get(risk, 0) + 1

        return ApprovalListResponse(
            pending=pending_list,
            total=len(pending),
            by_tool=by_tool,
            by_risk_level=by_risk,
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("list_approvals_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list approvals",
        ) from e


@router.post("/approvals/{trace_id}", response_model=ApprovalResponse)
async def handle_approval(
    trace_id: str,
    request: ApprovalRequest,
    logger_instance=Depends(get_logger),
) -> ApprovalResponse:
    """
    Approve or reject a pending execution.

    If approved, the tool will be executed and the result returned.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()

        if request.action == "approve":
            trace = catalog.approve_execution(trace_id, request.approver_id or "admin")

            if not trace:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Approval request '{trace_id}' not found",
                )

            # Execute the approved tool
            tool_def = catalog.get(trace.tool_name)
            execution_result = None

            if tool_def:
                try:
                    # Execute without approval check (already approved)
                    result = tool_def.function(**trace.input_params)
                    execution_result = ToolResult(
                        tool_name=trace.tool_name,
                        success=True,
                        result=result,
                        trace_id=trace_id,
                    )
                except Exception as e:
                    execution_result = ToolResult(
                        tool_name=trace.tool_name,
                        success=False,
                        error=str(e),
                        trace_id=trace_id,
                    )

            logger_instance.info(
                "approval_granted",
                trace_id=trace_id,
                approver=request.approver_id,
                tool=trace.tool_name,
            )

            return ApprovalResponse(
                trace_id=trace_id,
                action="approve",
                status="approved",
                approved_by=request.approver_id,
                approved_at=datetime.now(timezone.utc),
                reason=request.reason,
                execution_result=execution_result,
            )

        # reject
        trace = catalog.reject_execution(trace_id, request.reason or "Rejected by user")

        if not trace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Approval request '{trace_id}' not found",
            )

        logger_instance.info(
            "approval_rejected",
            trace_id=trace_id,
            approver=request.approver_id,
            reason=request.reason,
        )

        return ApprovalResponse(
            trace_id=trace_id,
            action="reject",
            status="rejected",
            approved_by=request.approver_id,
            reason=request.reason,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("handle_approval_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process approval. Check server logs for details.",
        ) from e


# =============================================================================
# DIAGNOSTIC ENDPOINT (PR-3)
# =============================================================================


@router.post("/diagnose", response_model=DiagnosticResponse)
async def run_diagnostic(
    request: DiagnosticRequest,
    logger_instance=Depends(get_logger),
) -> DiagnosticResponse:
    """
    Run diagnostic troubleshooting for a problem.

    Uses the LangGraph diagnostic graph to iteratively
    diagnose and propose solutions.
    """
    import time

    start_time = time.time()
    trace_id = str(uuid.uuid4())

    try:
        from resync.core.langgraph.diagnostic_graph import (
            DiagnosticConfig,
            diagnose_problem,
        )

        config = DiagnosticConfig(
            max_iterations=request.max_iterations,
            include_historical=request.include_history,
        )

        result = await diagnose_problem(
            problem_description=request.problem_description,
            tws_instance_id=request.tws_instance_id,
            config=config,
        )

        processing_time = int((time.time() - start_time) * 1000)

        diagnostic_result = DiagnosticResult(
            success=result.get("success", False),
            symptoms=result.get("symptoms", []),
            possible_causes=result.get("possible_causes", []),
            root_cause=result.get("root_cause"),
            root_cause_confidence=result.get("confidence", 0.0),
            proposed_solution=result.get("solution"),
            solution_steps=result.get("steps", []),
            risk_level=result.get("risk_level", "medium"),
            recommendations=result.get("recommendations", []),
            requires_action=result.get("requires_action", False),
            trace_id=trace_id,
            processing_time_ms=processing_time,
        )

        formatted = result.get("response", "")
        if not formatted:
            formatted = _format_diagnostic_response(diagnostic_result)

        logger_instance.info(
            "diagnostic_completed",
            trace_id=trace_id,
            success=diagnostic_result.success,
            root_cause=diagnostic_result.root_cause,
            processing_time_ms=processing_time,
        )

        return DiagnosticResponse(
            result=diagnostic_result,
            formatted_response=formatted,
        )

    except Exception as e:
        logger_instance.error("diagnostic_error", error=str(e), trace_id=trace_id)

        # Return error response
        error_result = DiagnosticResult(
            success=False,
            recommendations=["Contact support for assistance"],
            trace_id=trace_id,
        )

        return DiagnosticResponse(
            result=error_result,
            formatted_response=f"Erro no diagnóstico: {e}",
        )


def _format_diagnostic_response(result: DiagnosticResult) -> str:
    """Format diagnostic result as human-readable text."""
    parts = []

    if result.root_cause:
        parts.append(f"**🔍 Causa Raiz:** {result.root_cause}")
        parts.append("(Confiança: {result.root_cause_confidence * 100:.0f}%)")

    if result.proposed_solution:
        parts.append(f"\n**💡 Solução:** {result.proposed_solution}")

    if result.solution_steps:
        parts.append("\n**📋 Passos:**")
        for i, step in enumerate(result.solution_steps, 1):
            parts.append(f"{i}. {step}")

    if result.recommendations:
        parts.append("\n**📌 Recomendações:**")
        for rec in result.recommendations:
            parts.append(f"• {rec}")

    if result.requires_action:
        parts.append(f"\n⚠️ Risco: {result.risk_level}")

    return (
        "\n".join(parts)
        if parts
        else "Diagnóstico concluído sem resultados específicos."
    )


# =============================================================================
# EXECUTION HISTORY
# =============================================================================


@router.get("/traces")
async def list_execution_traces(
    limit: int = Query(100, ge=1, le=1000),
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    List recent execution traces.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        history = catalog.get_execution_history(limit=limit)

        traces = [trace.to_dict() for trace in history]

        return {
            "traces": traces,
            "total": len(traces),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("list_traces_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list traces",
        ) from e


# =============================================================================
# PR-9: ACTIVE RUNS TRACKING
# =============================================================================


@router.get("/runs/active")
async def get_active_runs(
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Get currently active tool runs.

    Returns runs that are queued, in-progress, or blocked on user.
    Useful for real-time UI updates.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        active = catalog.get_active_runs()

        return {
            "runs": [run.to_dict() for run in active],
            "total": len(active),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("get_active_runs_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active runs",
        ) from e


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Request cancellation of an active run.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        success = catalog.cancel_run(run_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found or already completed",
            )

        logger_instance.info("run_cancelled", run_id=run_id)

        return {
            "run_id": run_id,
            "status": "cancelled",
            "message": "Run cancellation requested",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("cancel_run_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel run",
        ) from e


# =============================================================================
# PR-11: UNDO OPERATIONS
# =============================================================================


@router.get("/undo/available")
async def list_undoable_operations(
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    List operations that can be undone.

    Returns trace_ids of operations that have undo support
    and haven't been undone yet.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        undoable = catalog.get_undoable_operations()

        # Get details for each undoable operation
        details = []
        for trace_id in undoable:
            for trace in catalog.get_execution_history():
                if trace.trace_id == trace_id:
                    details.append(
                        {
                            "trace_id": trace_id,
                            "tool_name": trace.tool_name,
                            "timestamp": trace.timestamp.isoformat(),
                            "risk_level": trace.risk_level.value,
                        }
                    )
                    break

        return {
            "undoable": details,
            "total": len(undoable),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("list_undoable_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list undoable operations",
        ) from e


@router.post("/undo/{trace_id}")
async def undo_operation(
    trace_id: str,
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Undo a previous operation.

    Attempts to revert the changes made by the operation
    identified by trace_id.
    """
    try:
        from resync.tools import get_tool_catalog

        catalog = get_tool_catalog()
        success = await catalog.undo_operation(trace_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation '{trace_id}' not found or cannot be undone",
            )

        logger_instance.info("operation_undone", trace_id=trace_id)

        return {
            "trace_id": trace_id,
            "status": "undone",
            "message": "Operation successfully reverted",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("undo_operation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to undo operation",
        ) from e


# =============================================================================
# PR-8: PARALLEL TOOL EXECUTION
# =============================================================================


@router.post("/tools/execute-parallel")
async def execute_tools_in_parallel(
    requests: list[dict[str, Any]],
    strategy: str = Query("smart", pattern="^(concurrent|serial|smart)$"),
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Execute multiple tools with smart parallelization.

    Strategy options:
    - concurrent: All tools run in parallel
    - serial: All tools run sequentially
    - smart: Read-only parallel, stateful serial (default)

    Example request body:
    [
        {"tool_name": "get_job_log", "parameters": {"job_name": "JOB1"}},
        {"tool_name": "get_job_log", "parameters": {"job_name": "JOB2"}},
        {"tool_name": "get_workstation_status", "parameters": {}}
    ]
    """
    try:
        from resync.core.specialists.parallel_executor import execute_tools_parallel

        results = await execute_tools_parallel(requests, strategy=strategy)

        success_count = sum(1 for r in results if r["success"])

        logger_instance.info(
            "parallel_execution_complete",
            tool_count=len(requests),
            strategy=strategy,
            success_count=success_count,
        )

        return {
            "results": results,
            "total": len(results),
            "success_count": success_count,
            "strategy": strategy,
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("parallel_execution_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Parallel execution failed. Check server logs for details.",
        ) from e


# =============================================================================
# PR-10: SUB-AGENT DISPATCH
# =============================================================================


@router.post("/sub-agents/dispatch")
async def dispatch_sub_agent_endpoint(
    prompt: str,
    context: str | None = None,
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Dispatch a read-only sub-agent for search/analysis tasks.

    Sub-agents have restricted access (read-only tools only)
    and cannot spawn other sub-agents.
    """
    try:
        from resync.core.specialists.sub_agent import dispatch_sub_agent

        result = await dispatch_sub_agent(prompt, context)

        logger_instance.info(
            "sub_agent_dispatched",
            agent_id=result.get("agent_id"),
            status=result.get("status"),
            tool_calls=result.get("tool_call_count"),
        )

        return result

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("sub_agent_dispatch_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sub-agent dispatch failed. Check server logs for details.",
        ) from e


@router.post("/sub-agents/dispatch-parallel")
async def dispatch_parallel_sub_agents_endpoint(
    prompts: list[str],
    context: str | None = None,
    max_concurrent: int = Query(5, ge=1, le=10),
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Dispatch multiple sub-agents in parallel.

    Useful for concurrent search/analysis across multiple
    jobs, workstations, or documents.

    Example:
    {
        "prompts": [
            "Analyze job JOB1 errors",
            "Analyze job JOB2 errors",
            "Analyze job JOB3 errors"
        ],
        "context": "Looking for ABEND patterns"
    }
    """
    try:
        from resync.core.specialists.sub_agent import dispatch_parallel_sub_agents

        results = await dispatch_parallel_sub_agents(prompts, context, max_concurrent)

        success_count = sum(1 for r in results if r.get("status") == "completed")

        logger_instance.info(
            "parallel_sub_agents_dispatched",
            agent_count=len(prompts),
            success_count=success_count,
        )

        return {
            "results": results,
            "total": len(results),
            "success_count": success_count,
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("parallel_sub_agents_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Parallel sub-agent dispatch failed. Check server logs for details.",
        ) from e


# =============================================================================
# PR-12: RISK ASSESSMENT
# =============================================================================


@router.post("/tools/assess-risk")
async def assess_tool_risk(
    tool_name: str,
    parameters: dict[str, Any],
    logger_instance=Depends(get_logger),
) -> dict[str, Any]:
    """
    Assess the risk level of a tool execution.

    Returns risk classification based on tool type,
    parameters, and affected systems (PROD, STAGE, etc.).
    """
    try:
        from resync.core.specialists.tools import RiskLevel, get_tool_catalog

        catalog = get_tool_catalog()
        tool = catalog.get(tool_name)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found",
            )

        risk_level = catalog.assess_risk(tool_name, parameters)

        risk_descriptions = {
            RiskLevel.LOW: "Safe operation, can proceed without approval",
            RiskLevel.MEDIUM: "Moderate risk, review recommended",
            RiskLevel.HIGH: "High risk, explicit approval required",
            RiskLevel.CRITICAL: "Critical operation, multiple approvals required",
        }

        return {
            "tool_name": tool_name,
            "risk_level": risk_level.value,
            "description": risk_descriptions.get(risk_level, "Unknown risk"),
            "requires_approval": risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL},
            "tool_permission": tool.permission.value,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("assess_risk_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess risk",
        ) from e
