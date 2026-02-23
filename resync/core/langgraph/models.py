"""
Structured Output Models for LangGraph v6.0.0.

Pydantic models for type-safe LLM responses.
Used with LangChain's with_structured_output() for guaranteed parsing.

Usage:
    from resync.core.langgraph.models import RouterOutput, DiagnosisOutput

    # With structured output
    result = await llm.with_structured_output(RouterOutput).ainvoke(prompt)
    print(result.intent)  # Type-safe access
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================


class Intent(str, Enum):
    """User intent categories."""

    STATUS = "status"
    TROUBLESHOOT = "troubleshoot"
    QUERY = "query"
    ACTION = "action"
    GENERAL = "general"
    UNKNOWN = "unknown"


class RouterOutput(BaseModel):
    """
    Structured output for intent classification.

    Used by router_node to classify user messages.
    """

    intent: Intent = Field(description="The classified intent of the user message")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the classification (0-1)"
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities from the message (job_name, workstation, etc.)",
    )
    reasoning: str | None = Field(
        default=None, description="Brief explanation of the classification"
    )


class EntityExtractionOutput(BaseModel):
    """Structured output for entity extraction."""

    job_name: str | None = Field(
        default=None, description="Name of the TWS job mentioned"
    )
    workstation: str | None = Field(
        default=None, description="Name of the workstation mentioned"
    )
    action_type: (
        Literal["cancel", "restart", "execute", "hold", "release", "submit"] | None
    ) = Field(default=None, description="Type of action requested")
    error_code: str | None = Field(
        default=None, description="Error code mentioned (e.g., AWSB1234E)"
    )
    time_reference: str | None = Field(
        default=None,
        description="Time reference mentioned (today, yesterday, last hour, etc.)",
    )


# =============================================================================
# TROUBLESHOOTING
# =============================================================================


class SymptomAnalysis(BaseModel):
    """Analysis of a single symptom."""

    symptom: str = Field(description="Description of the symptom")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level"
    )
    related_component: str | None = Field(
        default=None, description="Related system component"
    )


class CauseHypothesis(BaseModel):
    """Hypothesis about root cause."""

    cause: str = Field(description="Description of the possible cause")
    likelihood: Literal["low", "medium", "high"] = Field(
        description="Likelihood of this being the root cause"
    )
    verification_steps: list[str] = Field(
        default_factory=list, description="Steps to verify this hypothesis"
    )


class DiagnosisOutput(BaseModel):
    """
    Structured output for diagnostic analysis.

    Used by diagnostic nodes to analyze problems.
    """

    symptoms: list[SymptomAnalysis] = Field(
        default_factory=list, description="List of identified symptoms"
    )
    possible_causes: list[CauseHypothesis] = Field(
        default_factory=list, description="List of possible root causes"
    )
    root_cause: str | None = Field(default=None, description="Most likely root cause")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the diagnosis")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommended actions"
    )


# =============================================================================
# HALLUCINATION GRADING
# =============================================================================


class HallucinationGrade(BaseModel):
    """
    Structured output for hallucination checking.

    Used to verify responses are grounded in facts.
    """

    is_grounded: bool = Field(
        description="Whether the response is grounded in the provided context"
    )
    decision: Literal["useful", "not_useful", "not_grounded"] = Field(
        description="Decision on the response quality"
    )
    explanation: str = Field(description="Explanation of the grading decision")
    problematic_claims: list[str] = Field(
        default_factory=list,
        description="List of claims that are not supported by context",
    )


# =============================================================================
# ACTION APPROVAL
# =============================================================================


class ActionRequest(BaseModel):
    """Request for an action that requires approval."""

    action_type: str = Field(description="Type of action requested")
    job_name: str = Field(description="Target job name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for the action"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="medium", description="Risk level of the action"
    )
    requires_approval: bool = Field(
        default=True, description="Whether the action requires human approval"
    )


class ApprovalResponse(BaseModel):
    """Response from human approval."""

    approved: bool = Field(description="Whether the action was approved")
    approver: str | None = Field(default=None, description="Who approved the action")
    comments: str | None = Field(
        default=None, description="Additional comments from approver"
    )


# =============================================================================
# SYNTHESIS
# =============================================================================


class SynthesisInput(BaseModel):
    """Input for response synthesis."""

    raw_data: dict[str, Any] = Field(description="Raw data to synthesize")
    intent: Intent = Field(description="Original user intent")
    template_name: str | None = Field(
        default=None, description="Specific template to use"
    )
    language: str = Field(default="pt", description="Output language")


# =============================================================================
# AGENT STATE
# =============================================================================


class AgentStateModel(BaseModel):
    """
    Pydantic model for AgentState validation.

    Can be used to validate state at runtime.
    """

    # Input
    message: str = Field(default="")
    user_id: str | None = None
    session_id: str | None = None
    tws_instance_id: str | None = None

    # Classification
    intent: Intent = Intent.UNKNOWN
    confidence: float = 0.0
    entities: dict[str, Any] = Field(default_factory=dict)

    # Clarification
    needs_clarification: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    clarification_question: str = ""

    # Processing
    current_node: str = "start"
    retry_count: int = 0
    max_retries: int = 3

    # Tool execution
    tool_name: str | None = None
    tool_output: str | None = None
    tool_error: str | None = None

    # Output
    response: str = ""
    error: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Intent
    "Intent",
    "RouterOutput",
    "EntityExtractionOutput",
    # Diagnosis
    "SymptomAnalysis",
    "CauseHypothesis",
    "DiagnosisOutput",
    # Hallucination
    "HallucinationGrade",
    # Actions
    "ActionRequest",
    "ApprovalResponse",
    # Synthesis
    "SynthesisInput",
    # State
    "AgentStateModel",
]
