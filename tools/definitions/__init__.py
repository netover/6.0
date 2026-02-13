"""
Tool Definitions - Input/Output Schemas.

v5.8.0: Centralized schemas for all tools.
"""

from .schemas import (
    # Calendar schemas
    CalendarInput,
    CalendarOutput,
    # Dependency schemas
    DependencyGraphInput,
    DependencyGraphOutput,
    # Error code schemas
    ErrorCodeInput,
    ErrorCodeOutput,
    # Command schemas
    ExecuteCommandInput,
    ExecuteCommandOutput,
    IncidentRecord,
    JobHistoryInput,
    JobHistoryOutput,
    # Job schemas
    JobLogInput,
    JobLogOutput,
    # Metrics schemas
    MetricsInput,
    MetricsOutput,
    # RAG schemas
    RAGSearchInput,
    RAGSearchOutput,
    # Search schemas
    SearchHistoryInput,
    # Workstation schemas
    WorkstationStatusInput,
    WorkstationStatusOutput,
)

# TWS tool definitions (migrated from tool_definitions/)
from .tws import TWSToolReadOnly, TWSStatusTool, TWSTroubleshootingTool  # noqa: F401

__all__ = [
    # Job schemas
    "JobLogInput",
    "JobLogOutput",
    "JobHistoryInput",
    "JobHistoryOutput",
    # Workstation schemas
    "WorkstationStatusInput",
    "WorkstationStatusOutput",
    # Command schemas
    "ExecuteCommandInput",
    "ExecuteCommandOutput",
    # RAG schemas
    "RAGSearchInput",
    "RAGSearchOutput",
    # Dependency schemas
    "DependencyGraphInput",
    "DependencyGraphOutput",
    # Search schemas
    "SearchHistoryInput",
    "IncidentRecord",
    # Metrics schemas
    "MetricsInput",
    "MetricsOutput",
    # Calendar schemas
    "CalendarInput",
    "CalendarOutput",
    # Error code schemas
    "ErrorCodeInput",
    "ErrorCodeOutput",
]
