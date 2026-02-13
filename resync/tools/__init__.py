"""
Resync Tools Module.

v5.8.0: Unified tool system with clear separation of concerns.

Structure:
- registry.py: Tool catalog, decorators, permissions
- definitions/: Input/Output schemas (Pydantic models)
- implementations/: Tool implementations

Usage:
    from resync.tools import tool, ToolPermission, get_tool_catalog
    from resync.tools.definitions import JobLogInput, RAGSearchInput

    @tool(permission=ToolPermission.READ_ONLY)
    def my_tool(query: str) -> dict:
        ...

Author: Resync Team
Version: 5.8.0
"""


try:
    from .registry import (
        ROLE_PERMISSIONS,
        # Exceptions
        ApprovalRequiredError,
        RiskLevel,
        # Catalog
        ToolCatalog,
        ToolDefinition,
        ToolExecutionTrace,
        # Enums
        ToolPermission,
        # Dataclasses
        ToolRun,
        ToolRunStatus,
        UserRole,
        get_tool_catalog,
        # Decorator
        tool,
    )
    
except ImportError:  # pragma: no cover
    # Optional tool system deps (e.g., structlog) may not be installed in minimal/test environments.
    pass

__all__ = [
    # Enums
    "ToolPermission",
    "UserRole",
    "ToolRunStatus",
    "RiskLevel",
    "ROLE_PERMISSIONS",
    # Dataclasses
    "ToolRun",
    "ToolExecutionTrace",
    "ToolDefinition",
    # Catalog
    "ToolCatalog",
    "get_tool_catalog",
    # Decorator
    "tool",
    # Exceptions
    "ApprovalRequiredError",
]
