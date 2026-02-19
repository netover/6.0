"""Agent validation models with strict validation rules."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import StringConstraints as PydanticStringConstraints

from .common import NumericConstraints, StringConstraints, ValidationPatterns


class AgentType(str, Enum):
    """Valid agent types."""

    LOCAL_SCRIPT = "local_script"
    EXTERNAL_API = "external_api"
    DATABASE_QUERY = "database_query"
    TWS_SPECIALIST = "tws_specialist"
    GENERAL_ASSISTANT = "general_assistant"


class AgentStatus(str, Enum):
    """Agent status values."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ToolCategory(str, Enum):
    """Valid tool categories."""

    TWS_TOOLS = "tws_tools"
    DATABASE_TOOLS = "database_tools"
    API_TOOLS = "api_tools"
    UTILITY_TOOLS = "utility_tools"
    MONITORING_TOOLS = "monitoring_tools"


class AgentConfig(BaseModel):
    """Agent configuration validation model."""

    id: StringConstraints.AGENT_ID = Field(
        ..., description="Unique agent identifier", examples=["tws-troubleshooter-01"]
    )

    name: Annotated[
        str,
        PydanticStringConstraints(
            min_length=NumericConstraints.MIN_AGENT_NAME_LENGTH,
            max_length=NumericConstraints.MAX_AGENT_NAME_LENGTH,
            strip_whitespace=True,
        ),
    ] = Field(
        ...,
        description="Human-readable agent name",
        examples=["TWS Troubleshooting Agent"],
    )

    role: StringConstraints.ROLE_TEXT = Field(
        ...,
        description="Agent's role description",
        examples=["Specialized in TWS system troubleshooting and diagnostics"],
    )

    goal: StringConstraints.ROLE_TEXT = Field(
        ...,
        description="Primary goal of the agent",
        examples=["Resolve TWS system issues and provide diagnostic assistance"],
    )

    backstory: StringConstraints.ROLE_TEXT = Field(
        ...,
        description="Agent's background story",
        examples=["An experienced TWS administrator with deep system knowledge"],
    )

    tools: list[StringConstraints.TOOL_NAME] = Field(
        default_factory=list,
        description="List of tools the agent can use",
        max_length=20,
        examples=[["tws_status_tool", "tws_troubleshooting_tool"]],
    )

    model_name: StringConstraints.MODEL_NAME = Field(
        default="llama3:latest", description="LLM model name", examples=["llama3:latest"]
    )

    memory: bool = Field(default=True, description="Whether the agent has memory enabled")

    verbose: bool = Field(default=False, description="Whether the agent operates in verbose mode")

    type: AgentType = Field(default=AgentType.TWS_SPECIALIST, description="Type of agent")

    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Current agent status")

    description: (
        Annotated[
            str,
            PydanticStringConstraints(
                min_length=NumericConstraints.MIN_AGENT_DESCRIPTION_LENGTH,
                max_length=NumericConstraints.MAX_AGENT_DESCRIPTION_LENGTH,
                strip_whitespace=True,
            ),
        ]
        | None
    ) = Field(None, description="Detailed agent description")

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent configuration",
        max_length=50,
    )

    tags: list[Annotated[str, PydanticStringConstraints(min_length=1, max_length=50)]] = Field(
        default_factory=list, description="Agent tags for categorization", max_length=10
    )

    max_tokens: int | None = Field(
        None, ge=100, le=100000, description="Maximum tokens for model responses"
    )

    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Model temperature setting")

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator("name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent name doesn't contain malicious content."""
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Agent name contains potentially malicious content")
        if ValidationPatterns.COMMAND_INJECTION_PATTERN.search(v):
            raise ValueError("Agent name contains invalid characters")
        return v

    @field_validator("role", "goal", "backstory", "description")
    @classmethod
    def validate_text_content(cls, v: str | None) -> str | None:
        """Validate text fields for malicious content."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Text contains potentially malicious content")
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools_list(cls, v: list[str] | None) -> list[str] | None:
        """Validate tools list."""
        if not v:
            return v
        # Check for duplicate tools
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tools found in list")
        # Validate each tool name
        for tool in v:
            if not tool.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid tool name: {tool}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate tags list."""
        if not v:
            return v
        # Check for duplicate tags
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tags found in list")
        # Validate each tag
        for tag in v:
            if not tag.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid tag format: {tag}")
        return v

    @model_validator(mode="before")
    def validate_model_compatibility(self, values):
        """Validate model compatibility with configuration."""
        if isinstance(values, dict):
            model_name = values.get("model_name")
            max_tokens = values.get("max_tokens")
            if model_name and max_tokens:
                # Add model-specific token limits here
                model_limits = {
                    "llama3:latest": 8192,
                    "gpt-3.5-turbo": 4096,
                    "gpt-4": 8192,
                }
                if model_name in model_limits and max_tokens > model_limits[model_name]:
                    raise ValueError(
                        f"Max tokens {max_tokens} exceeds limit for {model_name} "
                        f"(max: {model_limits[model_name]})"
                    )
        return values


class AgentCreateRequest(AgentConfig):
    """Request model for creating a new agent."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    def validate_create_request(self, values):
        """Validate agent creation request."""
        if isinstance(values, dict):
            # Ensure required fields are provided
            required_fields = ["id", "name", "role", "goal", "backstory"]
            for field in required_fields:
                if not values.get(field):
                    raise ValueError(f"Required field '{field}' is missing or empty")
        return values


class AgentUpdateRequest(BaseModel):
    """Request model for updating an existing agent."""

    name: (
        Annotated[
            str,
            PydanticStringConstraints(
                min_length=NumericConstraints.MIN_AGENT_NAME_LENGTH,
                max_length=NumericConstraints.MAX_AGENT_NAME_LENGTH,
                strip_whitespace=True,
            ),
        ]
        | None
    ) = Field(None, description="Updated agent name")

    role: StringConstraints.ROLE_TEXT | None = Field(None, description="Updated agent role")

    goal: StringConstraints.ROLE_TEXT | None = Field(None, description="Updated agent goal")

    backstory: StringConstraints.ROLE_TEXT | None = Field(
        None, description="Updated agent backstory"
    )

    tools: list[StringConstraints.TOOL_NAME] | None = Field(
        None, description="Updated tools list", max_length=20
    )

    model_name: StringConstraints.MODEL_NAME | None = Field(None, description="Updated model name")

    memory: bool | None = Field(None, description="Updated memory setting")

    verbose: bool | None = Field(None, description="Updated verbose setting")

    type: AgentType | None = Field(None, description="Updated agent type")

    status: AgentStatus | None = Field(None, description="Updated agent status")

    description: (
        Annotated[
            str,
            PydanticStringConstraints(
                min_length=NumericConstraints.MIN_AGENT_DESCRIPTION_LENGTH,
                max_length=NumericConstraints.MAX_AGENT_DESCRIPTION_LENGTH,
                strip_whitespace=True,
            ),
        ]
        | None
    ) = Field(None, description="Updated agent description")

    configuration: dict[str, Any] | None = Field(
        None, description="Updated configuration", max_length=50
    )

    tags: list[Annotated[str, PydanticStringConstraints(min_length=1, max_length=50)]] | None = (
        Field(None, description="Updated tags", max_length=10)
    )

    max_tokens: int | None = Field(None, ge=100, le=100000, description="Updated max tokens")

    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Updated temperature")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @field_validator("name", "role", "goal", "backstory", "description")
    @classmethod
    def validate_text_content(cls, v):
        """Validate text fields for malicious content."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Text contains potentially malicious content")
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools_list(cls, v):
        """Validate tools list if provided."""
        if v is None:
            return v
        if not v:
            return v
        # Check for duplicate tools
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tools found in list")
        # Validate each tool name
        for tool in v:
            if not tool.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid tool name: {tool}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags list if provided."""
        if v is None:
            return v
        if not v:
            return v
        # Check for duplicate tags
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tags found in list")
        # Validate each tag
        for tag in v:
            if not tag.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid tag format: {tag}")
        return v


class AgentQueryParams(BaseModel):
    """Query parameters for agent-related endpoints."""

    agent_id: StringConstraints.AGENT_ID | None = Field(
        None, description="Filter by specific agent ID"
    )

    name: Annotated[str, PydanticStringConstraints(min_length=1, max_length=100)] | None = Field(
        None, description="Filter by agent name (partial match)"
    )

    type: AgentType | None = Field(None, description="Filter by agent type")

    status: AgentStatus | None = Field(None, description="Filter by agent status")

    tags: list[Annotated[str, PydanticStringConstraints(min_length=1, max_length=50)]] | None = (
        Field(None, description="Filter by tags", max_length=5)
    )

    include_inactive: bool = Field(default=False, description="Include inactive agents in results")

    sort_by: str | None = Field(
        default="name",
        pattern=r"^(name|id|type|status|created_at|updated_at)$",
        description="Field to sort results by",
    )

    sort_order: str | None = Field(
        default="asc",
        pattern=r"^(asc|desc)$",
        description="Sort order (ascending or descending)",
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags list."""
        if not v:
            return v
        # Check for duplicate tags
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tags found in list")
        return v


class AgentBulkActionRequest(BaseModel):
    """Request model for bulk agent operations."""

    agent_ids: list[StringConstraints.AGENT_ID] = Field(
        ...,
        description="List of agent IDs to perform action on",
        min_length=1,
        max_length=100,
    )

    action: str = Field(
        ...,
        pattern=r"^(activate|deactivate|delete|export)$",
        description="Bulk action to perform",
    )

    confirmation_token: str | None = Field(
        None, description="Confirmation token for destructive actions"
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("agent_ids")
    @classmethod
    def validate_agent_ids(cls, v):
        """Validate agent IDs list."""
        if not v:
            raise ValueError("Agent IDs list cannot be empty")
        # Check for duplicate IDs
        if len(v) != len(set(v)):
            raise ValueError("Duplicate agent IDs found in list")
        return v
