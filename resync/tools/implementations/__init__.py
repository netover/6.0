"""
Tool Implementations.

v5.8.0: Actual tool logic implementations.

Note: For backward compatibility, tool implementations are still
available in resync.core.specialists.tools. This module re-exports
them for the new unified structure.

Future versions will migrate implementations here completely.
"""

# Re-export from original location for backward compatibility
try:
    from resync.core.specialists.tools import (
        CalendarTool,
        DependencyGraphTool,
        ErrorCodeTool,
        JobLogTool,
        MetricsTool,
        RAGTool,
        SearchHistoryTool,
        TWSCommandTool,
        WorkstationTool,
    )

    __all__ = [
        "RAGTool",
        "JobLogTool",
        "TWSCommandTool",
        "DependencyGraphTool",
        "WorkstationTool",
        "CalendarTool",
        "MetricsTool",
        "ErrorCodeTool",
        "SearchHistoryTool",
    ]
except ImportError:
    # Tools not available yet
    __all__ = []
