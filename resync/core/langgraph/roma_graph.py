# pylint: skip-file
# mypy: ignore-errors
"""ROMA orchestration graph using LangGraph StateGraph."""

from __future__ import annotations

from typing import Any

from resync.core.langgraph.roma_models import RomaState
from resync.core.langgraph.roma_nodes import (
    aggregator_node,
    atomizer_node,
    executor_node,
    planner_node,
    verifier_node,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "END"
    StateGraph = None


def route_atomizer(state: RomaState) -> str:
    """Route atomic requests directly to aggregator, others to planner."""
    return "aggregator" if state.get("is_atomic") else "planner"


def create_roma_graph() -> Any:
    """Create compiled ROMA graph or fallback implementation."""
    if not LANGGRAPH_AVAILABLE:
        logger.warning("langgraph_not_available_using_roma_fallback")
        return FallbackRomaGraph()

    workflow = StateGraph(RomaState)
    workflow.add_node("atomizer", atomizer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("atomizer")
    workflow.add_conditional_edges(
        "atomizer",
        route_atomizer,
        {"planner": "planner", "aggregator": "aggregator"},
    )
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "aggregator")
    workflow.add_edge("aggregator", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()


class FallbackRomaGraph:
    """Fallback executor if LangGraph is unavailable."""

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        atomized = await atomizer_node(state)
        merged = {**state, **atomized}

        if not merged.get("is_atomic"):
            planned = await planner_node(merged)
            merged.update(planned)
            executed = await executor_node(merged)
            merged.update(executed)

        aggregated = aggregator_node(merged)
        merged.update(aggregated)

        verified = verifier_node(merged)
        merged.update(verified)
        return merged


async def execute_roma_query(user_query: str) -> dict[str, Any]:
    """Execute ROMA orchestration for a query."""
    graph = create_roma_graph()
    initial_state: RomaState = {
        "user_query": user_query,
        "execution_logs": [],
    }
    return await graph.ainvoke(initial_state)
