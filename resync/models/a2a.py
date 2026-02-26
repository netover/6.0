from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

class TaskState(str, Enum):
    """A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"

class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 Request model."""

    jsonrpc: str = "2.0"
    method: str
    params: Optional[dict[str, Any]] = None
    id: Any | None = None

class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 Response model."""

    jsonrpc: str = "2.0"
    result: Any | None = None
    error: Optional[dict[str, Any]] = None
    id: Any | None = None

class AgentCapabilities(BaseModel):
    """Capabilities declared by the agent."""

    actions: list[str] = Field(
        default_factory=list, description="List of JSON-RPC methods supported."
    )
    communication_modes: list[str] = Field(
        default_factory=lambda: ["json-rpc"],
        description="Supported modes: json-rpc, websocket, sse, webhooks.",
    )
    supports_streaming: bool = Field(
        False, description="Whether SSE/streaming is supported."
    )
    supports_push_notifications: bool = Field(
        False, description="Whether webhooks are supported."
    )
    supports_events: bool = Field(True, description="Whether events are published.")
    max_concurrent_tasks: int = Field(10, description="Concurrency limit.")

class AgentContact(BaseModel):
    """Connectivity information for the agent."""

    protocol: str = Field("A2A", description="Protocol name.")
    endpoint: str = Field(..., description="Main JSON-RPC HTTP endpoint.")
    event_endpoint: str | None = Field(None, description="SSE endpoint URL.")
    websocket_endpoint: str | None = Field(
        None, description="WebSocket endpoint URL."
    )
    auth_required: bool = Field(
        False, description="Whether authentication is mandatory."
    )

class AgentCard(BaseModel):
    """The A2A Agent Card (agent.json) for discovery."""

    name: str = Field(..., description="Unique machine-readable name.")
    version: str = Field(..., description="Agent implementation version.")
    description: str = Field(..., description="Human-readable purpose.")
    capabilities: AgentCapabilities = Field(..., description="What the agent can do.")
    contact: AgentContact = Field(..., description="How to reach the agent.")
    protocol_version: str = Field(
        "A2A-2024-11-05", description="A2A protocol version compliance."
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Custom implementation metadata."
    )
