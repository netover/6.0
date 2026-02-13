"""
Response models for FastAPI endpoints.

v5.4.1 Enhancements (PR-4):
- ToolResult for tool execution results
- ExecutionTrace for full execution tracing
- ApprovalResponse for HITL workflows
- DiagnosticResponse for troubleshooting results

Author: Resync Team
Version: 5.4.1
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# EXISTING MODELS (maintained for compatibility)
# =============================================================================


class WorkstationInfo(BaseModel):
    """Information model for workstation."""

    id: str
    name: str
    status: str
    last_seen: datetime | None = None


class JobInfo(BaseModel):
    """Information model for job."""

    id: str
    name: str
    status: str
    workstation_id: str
    start_time: datetime | None = None
    end_time: datetime | None = None


class SystemStatusResponse(BaseModel):
    """Response model for system status operations."""

    workstations: list[WorkstationInfo]
    jobs: list[JobInfo]
    timestamp: datetime


class AgentInfo(BaseModel):
    """Information model for agent."""

    id: str
    name: str
    status: str
    description: str | None = None


class AgentListResponse(BaseModel):
    """Response model for agent list operations."""

    agents: list[AgentInfo]
    total: int


class FileUploadResponse(BaseModel):
    """Response model for file upload operations."""

    filename: str
    status: str
    file_id: str | None = None
    upload_time: datetime


class AuditFlagInfo(BaseModel):
    """Information model for audit flag."""

    memory_id: str
    status: str
    user_query: str
    agent_response: str
    ia_audit_reason: str | None = None
    ia_audit_confidence: float | None = None
    created_at: datetime
    reviewed_at: datetime | None = None


class AuditMetricsResponse(BaseModel):
    """Response model for audit metrics operations."""

    pending: int
    approved: int
    rejected: int
    total: int


class AuditReviewResponse(BaseModel):
    """Response model for audit review operations."""

    memory_id: str
    action: str
    status: str
    reviewed_at: datetime


class ChatMessageResponse(BaseModel):
    """Response model for chat message operations."""

    message: str
    timestamp: datetime
    agent_id: str | None = None  # Now represents the handler that processed the message
    is_final: bool = False
    metadata: dict[str, Any] | None = None  # Intent classification and routing info


class HealthResponse(BaseModel):
    """Response model for health operations."""

    status: str
    uptime: str
    version: str


class LoginResponse(BaseModel):
    """Response model for login operations."""

    message: str


class StatusResponse(BaseModel):
    """Response model for status operations."""

    workstations: list[Any]
    jobs: list[Any]


class AuditFlagsResponse(BaseModel):
    """Response model for audit flags operations."""

    flags: list[dict[str, Any]]


class RAGUploadResponse(BaseModel):
    """Response model for RAG upload operations."""

    filename: str
    status: str


# =============================================================================
# NEW MODELS FOR AGENTIC EXECUTION (PR-4)
# =============================================================================


class ToolResult(BaseModel):
    """
    Result of a tool execution.
    """

    tool_name: str = Field(..., description="Name of the tool executed")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(None, description="Tool result data")
    error: str | None = Field(None, description="Error message if failed")
    duration_ms: int = Field(0, description="Execution duration in milliseconds")
    trace_id: str | None = Field(None, description="Trace ID for this execution")


class ExecutionStep(BaseModel):
    """
    A single step in an execution trace.
    """

    step_number: int = Field(..., description="Step number in sequence")
    action: str = Field(..., description="Action taken")
    tool_name: str | None = Field(None, description="Tool used if any")
    input_summary: str | None = Field(None, description="Summary of input")
    output_summary: str | None = Field(None, description="Summary of output")
    duration_ms: int = Field(0, description="Step duration")
    success: bool = Field(True, description="Whether step succeeded")
    error: str | None = Field(None, description="Error if step failed")


class ExecutionTrace(BaseModel):
    """
    Complete trace of an agentic execution.

    Provides full observability into what happened.
    """

    trace_id: str = Field(..., description="Unique trace identifier")
    session_id: str | None = Field(None, description="Session ID")
    user_id: str | None = Field(None, description="User ID")

    # Timing
    start_time: datetime = Field(..., description="Execution start time")
    end_time: datetime | None = Field(None, description="Execution end time")
    duration_ms: int = Field(0, description="Total duration")

    # Input/Output
    input_message: str = Field(..., description="Original input")
    output_message: str = Field(..., description="Final output")

    # Routing
    routing_mode: str = Field(..., description="Routing mode used")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Classification confidence")
    handler: str = Field(..., description="Handler that processed request")

    # Execution details
    steps: list[ExecutionStep] = Field(default_factory=list, description="Execution steps")
    tools_used: list[str] = Field(default_factory=list, description="Tools used")
    tool_results: list[ToolResult] = Field(default_factory=list, description="Tool results")

    # Status
    success: bool = Field(True, description="Overall success")
    error: str | None = Field(None, description="Error if failed")

    # HITL
    required_approval: bool = Field(False, description="Whether HITL approval was needed")
    approval_status: str | None = Field(None, description="Approval status if applicable")
    approved_by: str | None = Field(None, description="Approver ID if approved")


class AgentExecuteResponse(BaseModel):
    """
    Response from agent execution.
    """

    response: str = Field(..., description="Agent response")
    success: bool = Field(True, description="Whether execution succeeded")

    # Routing info
    routing_mode: str = Field(..., description="Routing mode used")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Classification confidence")
    handler: str = Field(..., description="Handler used")

    # Tools
    tools_used: list[str] = Field(default_factory=list, description="Tools executed")

    # Metadata
    processing_time_ms: int = Field(0, description="Processing time")
    session_id: str | None = Field(None, description="Session ID")
    trace_id: str | None = Field(None, description="Trace ID for full trace")

    # Full trace (optional)
    trace: ExecutionTrace | None = Field(None, description="Full execution trace")

    # HITL (if approval required)
    requires_approval: bool = Field(False, description="Whether approval is needed")
    approval_id: str | None = Field(None, description="Approval request ID")
    pending_action: dict[str, Any] | None = Field(None, description="Pending action details")


# =============================================================================
# APPROVAL RESPONSE MODELS
# =============================================================================


class PendingApproval(BaseModel):
    """
    A pending approval request.
    """

    trace_id: str = Field(..., description="Trace ID")
    tool_name: str = Field(..., description="Tool requiring approval")
    action_summary: str = Field(..., description="Summary of action")
    input_params: dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    user_id: str | None = Field(None, description="User who requested")
    requested_at: datetime = Field(..., description="When approval was requested")
    expires_at: datetime | None = Field(None, description="When approval expires")
    risk_level: str = Field("medium", description="Risk level")


class ApprovalResponse(BaseModel):
    """
    Response to an approval action.
    """

    trace_id: str = Field(..., description="Trace ID")
    action: str = Field(..., description="Action taken (approve/reject)")
    status: str = Field(..., description="Result status")
    approved_by: str | None = Field(None, description="Approver ID")
    approved_at: datetime | None = Field(None, description="Approval time")
    reason: str | None = Field(None, description="Reason for action")

    # If approved and executed
    execution_result: ToolResult | None = Field(None, description="Execution result if approved")


class ApprovalListResponse(BaseModel):
    """
    Response listing pending approvals.
    """

    pending: list[PendingApproval] = Field(default_factory=list, description="Pending approvals")
    total: int = Field(0, description="Total pending")

    # Statistics
    by_tool: dict[str, int] = Field(default_factory=dict, description="Count by tool")
    by_risk_level: dict[str, int] = Field(default_factory=dict, description="Count by risk")


# =============================================================================
# DIAGNOSTIC RESPONSE MODELS
# =============================================================================


class DiagnosticResult(BaseModel):
    """
    Result of a diagnostic analysis.
    """

    success: bool = Field(..., description="Whether diagnosis succeeded")

    # Diagnosis
    symptoms: list[str] = Field(default_factory=list, description="Identified symptoms")
    possible_causes: list[dict[str, Any]] = Field(
        default_factory=list, description="Possible causes"
    )
    root_cause: str | None = Field(None, description="Most likely root cause")
    root_cause_confidence: float = Field(0.0, description="Confidence in root cause")

    # Solution
    proposed_solution: str | None = Field(None, description="Proposed solution")
    solution_steps: list[str] = Field(default_factory=list, description="Steps to resolve")
    risk_level: str = Field("medium", description="Risk level of solution")

    # Evidence
    documentation_context: list[dict[str, Any]] = Field(
        default_factory=list, description="Relevant docs"
    )
    historical_incidents: list[dict[str, Any]] = Field(
        default_factory=list, description="Similar incidents"
    )
    similar_resolutions: list[dict[str, Any]] = Field(
        default_factory=list, description="Past resolutions"
    )

    # Recommendations
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")

    # Execution
    requires_action: bool = Field(False, description="Whether action is needed")
    requires_approval: bool = Field(False, description="Whether HITL approval needed")
    approval_id: str | None = Field(None, description="Approval ID if needed")

    # Trace
    trace_id: str | None = Field(None, description="Trace ID")
    iterations: int = Field(1, description="Diagnostic iterations")
    processing_time_ms: int = Field(0, description="Processing time")


class DiagnosticResponse(BaseModel):
    """
    Full response from diagnostic endpoint.
    """

    result: DiagnosticResult = Field(..., description="Diagnostic result")

    # Formatted response
    formatted_response: str = Field(..., description="Human-readable response")

    # Session
    session_id: str | None = Field(None, description="Session ID")

    # Full trace (optional)
    trace: ExecutionTrace | None = Field(None, description="Full execution trace")


# =============================================================================
# FEEDBACK RESPONSE MODELS
# =============================================================================


class FeedbackResponse(BaseModel):
    """
    Response to feedback submission.
    """

    feedback_id: str = Field(..., description="Feedback ID")
    trace_id: str = Field(..., description="Associated trace ID")
    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Confirmation message")


class IncidentResolutionResponse(BaseModel):
    """
    Response to incident resolution recording.
    """

    incident_id: str = Field(..., description="Incident ID")
    status: str = Field(..., description="Recording status")
    message: str = Field(..., description="Confirmation message")
    added_to_knowledge_base: bool = Field(False, description="Whether added to KB")


# =============================================================================
# METRICS AND OBSERVABILITY RESPONSE MODELS
# =============================================================================


class MetricsSummary(BaseModel):
    """
    Summary of system metrics.
    """

    time_range_hours: int = Field(..., description="Time range covered")

    # Performance
    total_requests: int = Field(0, description="Total requests")
    success_rate: float = Field(0.0, description="Success rate")
    avg_latency_ms: float = Field(0.0, description="Average latency")
    p95_latency_ms: float = Field(0.0, description="95th percentile latency")

    # Tools
    tool_calls: int = Field(0, description="Total tool calls")
    tool_success_rate: float = Field(0.0, description="Tool success rate")
    tools_by_usage: dict[str, int] = Field(default_factory=dict, description="Usage by tool")

    # Routing
    requests_by_mode: dict[str, int] = Field(
        default_factory=dict, description="Requests by routing mode"
    )
    requests_by_intent: dict[str, int] = Field(
        default_factory=dict, description="Requests by intent"
    )

    # HITL
    approval_requests: int = Field(0, description="Approval requests")
    approval_rate: float = Field(0.0, description="Approval rate")

    # Errors
    error_count: int = Field(0, description="Error count")
    errors_by_type: dict[str, int] = Field(default_factory=dict, description="Errors by type")


class AnomalyInfo(BaseModel):
    """
    Information about a detected anomaly.
    """

    anomaly_id: str = Field(..., description="Anomaly ID")
    detected_at: datetime = Field(..., description="Detection time")
    metric_name: str = Field(..., description="Metric with anomaly")
    expected_value: float = Field(..., description="Expected value")
    actual_value: float = Field(..., description="Actual value")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Anomaly description")


class MetricsResponse(BaseModel):
    """
    Response with system metrics.
    """

    summary: MetricsSummary = Field(..., description="Metrics summary")
    anomalies: list[AnomalyInfo] = Field(default_factory=list, description="Detected anomalies")
    health_status: str = Field("healthy", description="Overall health status")

    # Raw metrics (optional)
    raw_metrics: dict[str, Any] | None = Field(None, description="Raw metric data")


class TraceListResponse(BaseModel):
    """
    Response listing execution traces.
    """

    traces: list[ExecutionTrace] = Field(default_factory=list, description="Execution traces")
    total: int = Field(0, description="Total matching traces")

    # Pagination
    limit: int = Field(100, description="Limit used")
    offset: int = Field(0, description="Offset used")
    has_more: bool = Field(False, description="Whether more results exist")
