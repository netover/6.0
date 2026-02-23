"""Agent management models."""

from enum import Enum

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Types of AI agents."""

    CHAT = "chat"
    TASK = "task"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    SUPPORT = "support"


class AgentConfig(BaseModel):
    """Configuration model for AI agents."""

    id: str = Field(..., description="Unique identifier for the agent.")
    name: str = Field(..., description="The name of the agent.")
    agent_type: AgentType = Field(..., description="The type of AI agent.")
    role: str = Field(..., description="The role of the agent.")
    goal: str = Field(..., description="The primary goal of the agent.")
    backstory: str = Field(..., description="The backstory or context for the agent.")
    tools: list[str] = Field(
        default_factory=list, description="List of tools the agent can use."
    )
    model_name: str = Field(..., description="The name of the LLM model to use.")
    memory: bool = Field(
        default=True, description="Whether the agent should use memory."
    )
    max_rpm: int | None = Field(
        None, description="Maximum requests per minute for the agent."
    )
