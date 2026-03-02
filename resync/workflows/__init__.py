"""
Resync Workflows Module

LangGraph multi-step workflows for predictive maintenance and capacity forecasting.

FIX P0-05: Replaced eager module-level imports with lazy __getattr__ to avoid
ModuleNotFoundError when langgraph is not installed (e.g. lightweight Docker
images, CI without ML dependencies). The public API is unchanged.
"""
from __future__ import annotations

__all__ = [
    "run_predictive_maintenance",
    "approve_workflow",
    "run_capacity_forecast",
]

def __getattr__(name: str):  # noqa: ANN001
    """Lazily import workflow functions to defer langgraph dependency."""
    if name in ("run_predictive_maintenance", "approve_workflow"):
        from .workflow_predictive_maintenance import (
            approve_workflow,
            run_predictive_maintenance,
        )

        _globals = {
            "run_predictive_maintenance": run_predictive_maintenance,
            "approve_workflow": approve_workflow,
        }
        # Cache in module globals so subsequent accesses are free
        import sys
        sys.modules[__name__].__dict__.update(_globals)
        return _globals[name]

    if name == "run_capacity_forecast":
        from .workflow_capacity_forecasting import run_capacity_forecast

        import sys
        sys.modules[__name__].__dict__["run_capacity_forecast"] = run_capacity_forecast
        return run_capacity_forecast

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
