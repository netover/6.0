"""
Orchestration Schemas

Pydantic models for validation and typing of orchestration structures.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class StepType(str, Enum):
    """Types of steps available."""
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    SCRIPT = "script"
    HUMAN = "human"
    WAIT = "wait"


class StepDependency(BaseModel):
    """Dependency definition for a step."""
    step_id: str
    condition: Optional[str] = None  # python expression, e.g. "output.status == 'success'"


class StepConfig(BaseModel):
    """Configuration for a single step."""
    id: str
    type: StepType
    name: Optional[str] = None
    
    # Execution details
    agent_id: Optional[str] = None  # For AGENT type
    tool_name: Optional[str] = None  # For TOOL type
    prompt_template: Optional[str] = None  # For LLM type
    
    # Inputs
    inputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Control flow
    dependencies: List[StepDependency] = Field(default_factory=list)
    retry_config: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "delay_seconds": 1})
    timeout_seconds: int = 300
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    """Full workflow configuration schema."""
    version: str = "1.0"
    steps: List[StepConfig]
    
    @field_validator("steps")
    def validate_unique_ids(cls, v):
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Step IDs must be unique")
        return v
