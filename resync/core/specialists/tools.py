from __future__ import annotations

import structlog
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = structlog.get_logger(__name__)


# =============================================================================
# PERMISSIONS & DEFINITIONS
# =============================================================================


class ToolPermission(str, Enum):
    """Tool permission levels."""
    READ_ONLY = "read_only"  # Safe for parallel execution
    READ_WRITE = "read_write"  # Side effects, serial execution recommended
    ADMIN = "admin"  # Requires elevated privileges


class UserRole(str, Enum):
    """User roles for permission checking."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


@dataclass
class ToolDefinition:
    """Definition of a tool available to agents."""
    name: str
    description: str
    function: Callable[..., Awaitable[Any] | Any]
    permission: ToolPermission = ToolPermission.READ_WRITE
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    timeout_seconds: float = 30.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission.value,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
        }


# =============================================================================
# CATALOG
# =============================================================================


class ToolCatalog:
    """Registry for available specialist tools."""

    _instance: ToolCatalog | None = None

    def __new__(cls) -> ToolCatalog:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}  # type: ignore
        return cls._instance

    def __init__(self):
        # Already initialized in __new__
        pass

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name, permission=tool.permission)

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_read_only_tools(self) -> list[ToolDefinition]:
        """Get only read-only tools (safe for parallel execution)."""
        return [
            t for t in self._tools.values()
            if t.permission == ToolPermission.READ_ONLY
        ]


def get_tool_catalog() -> ToolCatalog:
    """Get the singleton tool catalog."""
    return ToolCatalog()


def tool(
    name: str,
    description: str,
    permission: ToolPermission = ToolPermission.READ_WRITE,
    requires_approval: bool = False,
    timeout_seconds: float = 30.0,
    tags: list[str] | None = None,
):
    """Decorator to register a function as a specialist tool."""
    def decorator(func):
        definition = ToolDefinition(
            name=name,
            description=description,
            function=func,
            permission=permission,
            requires_approval=requires_approval,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )
        get_tool_catalog().register(definition)
        return func
    return decorator


# =============================================================================
# COMMON TOOLS (Examples)
# =============================================================================


@tool(
    name="get_job_logs",
    description="Retrieve logs for a specific job execution.",
    permission=ToolPermission.READ_ONLY,
    tags=["logs", "job", "tws"]
)
async def get_job_logs(job_id: str, limit: int = 100) -> str:
    """Tool for retrieving and analyzing job logs."""
    # Placeholder implementation
    return f"Logs for job {job_id} (last {limit} lines)..."
