"""
Specialist Agents Models.

Pydantic models for specialist agent configuration, responses,
and team coordination.

Author: Resync Team
Version: 5.2.3.29
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class SpecialistType(str, Enum):
    """Types of specialist agents."""

    JOB_ANALYST = "job_analyst"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    KNOWLEDGE = "knowledge"


class TeamExecutionMode(str, Enum):
    """Team execution modes for specialist coordination."""

    COORDINATE = "coordinate"  # Orchestrator delegates and synthesizes
    COLLABORATE = "collaborate"  # Agents work together iteratively
    ROUTE = "route"  # Single best agent handles query
    PARALLEL = "parallel"  # All agents run in parallel


class SpecialistConfig(BaseModel):
    """Configuration for a specialist agent."""

    specialist_type: SpecialistType = Field(..., description="Type of specialist")
    enabled: bool = Field(default=True, description="Whether this specialist is active")
    model_name: str = Field(default="gpt-4o", description="LLM model to use")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=2048, ge=100, le=8192, description="Maximum response tokens")
    timeout_seconds: int = Field(default=30, ge=5, le=120, description="Request timeout")
    retry_attempts: int = Field(
        default=3, ge=0, le=5, description="Number of retry attempts on failure"
    )
    custom_instructions: str | None = Field(
        default=None, description="Additional custom instructions"
    )

    model_config = ConfigDict(use_enum_values=True)


class SpecialistResponse(BaseModel):
    """Response from a single specialist agent."""

    specialist_type: SpecialistType = Field(
        ..., description="Type of specialist that generated this response"
    )
    response: str = Field(..., description="The specialist's response text")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score for the response"
    )
    tools_used: list[str] = Field(default_factory=list, description="Tools invoked during analysis")
    processing_time_ms: int = Field(default=0, ge=0, description="Processing time in milliseconds")
    error: str | None = Field(default=None, description="Error message if processing failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    @property
    def is_successful(self) -> bool:
        """Check if response was successful."""
        return self.error is None

    model_config = ConfigDict(use_enum_values=True)


class TeamResponse(BaseModel):
    """Consolidated response from the specialist team."""

    query: str = Field(..., description="Original user query")
    synthesized_response: str = Field(..., description="Final synthesized response")
    specialist_responses: list[SpecialistResponse] = Field(
        default_factory=list, description="Individual specialist responses"
    )
    execution_mode: TeamExecutionMode = Field(
        default=TeamExecutionMode.COORDINATE, description="How specialists were coordinated"
    )
    total_processing_time_ms: int = Field(default=0, ge=0, description="Total processing time")
    specialists_used: list[SpecialistType] = Field(
        default_factory=list, description="Which specialists contributed"
    )
    query_classification: str | None = Field(default=None, description="Detected query type/intent")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Overall confidence")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    @property
    def successful_specialists(self) -> int:
        """Count of specialists that responded successfully."""
        return sum(1 for r in self.specialist_responses if r.is_successful)

    @property
    def failed_specialists(self) -> int:
        """Count of specialists that failed."""
        return sum(1 for r in self.specialist_responses if not r.is_successful)

    model_config = ConfigDict(use_enum_values=True)


class TeamConfig(BaseModel):
    """Configuration for the specialist team."""

    enabled: bool = Field(default=True, description="Whether the team is active")
    execution_mode: TeamExecutionMode = Field(
        default=TeamExecutionMode.COORDINATE, description="Default execution mode"
    )
    orchestrator_model: str = Field(
        default="gpt-4o", description="Model for the orchestrator/planner"
    )
    synthesizer_model: str = Field(default="gpt-4o", description="Model for the synthesizer")
    parallel_execution: bool = Field(
        default=True, description="Run specialists in parallel when possible"
    )
    max_parallel_specialists: int = Field(
        default=4, ge=1, le=10, description="Maximum specialists to run in parallel"
    )
    timeout_seconds: int = Field(default=45, ge=10, le=180, description="Overall team timeout")
    fallback_to_general: bool = Field(
        default=True, description="Fall back to general assistant if all specialists fail"
    )
    specialists: dict[SpecialistType, SpecialistConfig] = Field(
        default_factory=dict, description="Individual specialist configurations"
    )

    model_config = ConfigDict(use_enum_values=True)


class QueryClassification(BaseModel):
    """Classification of a user query for routing."""

    query_type: str = Field(..., description="Detected query type")
    recommended_specialists: list[SpecialistType] = Field(
        default_factory=list, description="Specialists best suited for this query"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Classification confidence")
    requires_graph: bool = Field(default=False, description="Whether query needs dependency graph")
    requires_rag: bool = Field(
        default=False, description="Whether query needs RAG/knowledge search"
    )
    requires_realtime_data: bool = Field(
        default=False, description="Whether query needs live TWS data"
    )
    entities: dict[str, list[str]] = Field(
        default_factory=dict, description="Extracted entities (jobs, workstations, etc.)"
    )

    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

DEFAULT_SPECIALIST_CONFIGS: dict[SpecialistType, SpecialistConfig] = {
    SpecialistType.JOB_ANALYST: SpecialistConfig(
        specialist_type=SpecialistType.JOB_ANALYST,
        model_name="gpt-4o",
        temperature=0.2,  # Lower for factual analysis
        max_tokens=2048,
        timeout_seconds=30,
    ),
    SpecialistType.DEPENDENCY: SpecialistConfig(
        specialist_type=SpecialistType.DEPENDENCY,
        model_name="gpt-4o",
        temperature=0.1,  # Very low for graph traversal
        max_tokens=2048,
        timeout_seconds=30,
    ),
    SpecialistType.RESOURCE: SpecialistConfig(
        specialist_type=SpecialistType.RESOURCE,
        model_name="gpt-4o",
        temperature=0.2,
        max_tokens=1536,
        timeout_seconds=25,
    ),
    SpecialistType.KNOWLEDGE: SpecialistConfig(
        specialist_type=SpecialistType.KNOWLEDGE,
        model_name="gpt-4o",
        temperature=0.4,  # Slightly higher for documentation
        max_tokens=3072,  # More for documentation
        timeout_seconds=35,
    ),
}

DEFAULT_TEAM_CONFIG = TeamConfig(
    enabled=True,
    execution_mode=TeamExecutionMode.COORDINATE,
    orchestrator_model="gpt-4o",
    synthesizer_model="gpt-4o",
    parallel_execution=True,
    max_parallel_specialists=4,
    timeout_seconds=45,
    fallback_to_general=True,
    specialists=DEFAULT_SPECIALIST_CONFIGS,
)
