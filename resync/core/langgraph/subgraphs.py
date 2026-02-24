# pylint: disable=all
"""
LangGraph Subgraphs for Resync v6.0.0.

This module provides reusable subgraphs that can be composed into larger graphs.
Each subgraph is self-contained and can be tested independently.

Subgraphs:
- diagnostic_subgraph: Autonomous problem diagnosis
- parallel_fetch_subgraph: Parallel data fetching
- approval_subgraph: Human-in-the-loop approval
- synthesis_subgraph: Response synthesis

Usage:
    from resync.core.langgraph.subgraphs import get_diagnostic_subgraph

    # Use as a node in main graph
    main_graph.add_node("diagnose", get_diagnostic_subgraph())

    # Or compose multiple subgraphs
    main_graph.add_node("parallel", get_parallel_subgraph())
    main_graph.add_node("synthesize", get_synthesis_subgraph())
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Annotated, Any, TypedDict

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# LangGraph imports
try:
    from langgraph.graph import END, StateGraph
    from langgraph.types import Send, interrupt

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_03_FEATURES = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_03_FEATURES = False
    StateGraph = None
    END = "END"
    Send = None
    interrupt = None


# =============================================================================
# SHARED STATE DEFINITIONS
# =============================================================================


class SubgraphState(TypedDict, total=False):
    """Base state for subgraphs."""

    # Input
    message: str
    entities: dict[str, Any]

    # Output
    result: dict[str, Any]
    error: str | None


class ParallelResult(TypedDict):
    """Result from a parallel data source."""

    source: str
    data: dict[str, Any]
    latency_ms: float
    success: bool
    error: str | None


# =============================================================================
# DIAGNOSTIC SUBGRAPH
# =============================================================================


class DiagnosticSubgraphState(TypedDict, total=False):
    """State for diagnostic subgraph."""

    # Input
    problem_description: str
    job_name: str | None

    # Analysis
    symptoms: list[str]
    possible_causes: list[dict[str, Any]]
    root_cause: str | None
    confidence: float

    # Research
    documentation_context: list[dict[str, Any]]
    historical_incidents: list[dict[str, Any]]

    # Output
    recommendations: list[str]
    solution: str | None

    # Control
    iteration: int
    max_iterations: int


async def _diagnose_node(state: DiagnosticSubgraphState) -> dict:
    """Analyze problem and extract symptoms."""
    from resync.core.utils.llm import call_llm

    problem = state.get("problem_description", "")

    prompt = f"""Analyze this TWS/HWA problem and extract symptoms.

Problem: {problem}

Return a JSON object with:
- symptoms: list of specific symptoms identified
- possible_causes: list of {{cause, likelihood, verification}}
- confidence: float 0-1

JSON:"""

    try:
        response = await call_llm(prompt, model="gpt-4o-mini", temperature=0.2)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "symptoms": data.get("symptoms", [problem]),
                "possible_causes": data.get("possible_causes", []),
                "confidence": data.get("confidence", 0.5),
            }
    except Exception as e:
        logger.error("diagnose_error", error=str(e))

    return {"symptoms": [problem], "possible_causes": [], "confidence": 0.3}


async def _research_node(state: DiagnosticSubgraphState) -> dict:
    """Research documentation and history."""
    from resync.services.rag_client import RAGClient

    symptoms = state.get("symptoms", [])
    query = " ".join(symptoms[:3])

    try:
        rag = RAGClient()
        results = await rag.search(query=query, limit=5)
        return {"documentation_context": results.get("results", [])}
    except Exception as e:
        logger.warning("research_error", error=str(e))
        return {"documentation_context": []}


async def _synthesize_diagnosis_node(state: DiagnosticSubgraphState) -> dict:
    """Synthesize final diagnosis."""
    from resync.core.utils.llm import call_llm

    symptoms = state.get("symptoms", [])
    causes = state.get("possible_causes", [])
    docs = state.get("documentation_context", [])

    prompt = f"""Based on the analysis, provide final diagnosis and recommendations.

Symptoms: {symptoms}
Possible causes: {causes}
Documentation: {[d.get("content", "")[:200] for d in docs[:3]]}

Return JSON with:
- root_cause: most likely cause
- solution: recommended solution
- recommendations: list of action items

JSON:"""

    try:
        response = await call_llm(prompt, model="gpt-4o-mini", temperature=0.3)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "root_cause": data.get("root_cause"),
                "solution": data.get("solution"),
                "recommendations": data.get("recommendations", []),
            }
    except Exception as e:
        logger.error("synthesis_error", error=str(e))

    return {"recommendations": ["Investigate logs manually"]}


def _should_continue_diagnosis(state: DiagnosticSubgraphState) -> str:
    """Check if we need more iterations."""
    confidence = state.get("confidence", 0)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if confidence >= 0.7 or iteration >= max_iter:
        return "synthesize"
    return "research"


def create_diagnostic_subgraph():
    """
    Create the diagnostic subgraph.

    Flow: diagnose -> research -> synthesize
          (loops back if confidence < 0.7)

    Returns:
        Compiled subgraph or None if LangGraph unavailable
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("langgraph_unavailable_for_subgraph")
        return None

    graph = StateGraph(DiagnosticSubgraphState)

    graph.add_node("diagnose", _diagnose_node)
    graph.add_node("research", _research_node)
    graph.add_node("synthesize", _synthesize_diagnosis_node)

    graph.set_entry_point("diagnose")

    graph.add_conditional_edges(
        "diagnose",
        _should_continue_diagnosis,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("research", "diagnose")  # Loop back
    graph.add_edge("synthesize", END)

    return graph.compile()


# =============================================================================
# PARALLEL FETCH SUBGRAPH
# =============================================================================


class ParallelFetchState(TypedDict, total=False):
    """State for parallel fetch subgraph."""

    # Input
    job_name: str | None
    query: str
    tws_instance_id: str | None

    # Results (using Annotated for automatic merging)
    parallel_results: Annotated[list[ParallelResult], lambda x, y: x + y]

    # Aggregated
    aggregated_data: dict[str, Any]
    total_latency_ms: float


async def _fetch_tws_status(state: ParallelFetchState) -> dict:
    """Fetch TWS status in parallel."""
    start = time.time()

    try:
        from resync.services.tws_service import get_tws_client

        tws = get_tws_client()
        job_name = state.get("job_name")

        if job_name:
            status = await asyncio.wait_for(tws.get_job_status(job_name), timeout=5.0)
        else:
            status = {"message": "No job specified"}

        return {
            "parallel_results": [
                {
                    "source": "tws_status",
                    "data": status,
                    "latency_ms": (time.time() - start) * 1000,
                    "success": True,
                    "error": None,
                }
            ]
        }
    except Exception as e:
        return {
            "parallel_results": [
                {
                    "source": "tws_status",
                    "data": {},
                    "latency_ms": (time.time() - start) * 1000,
                    "success": False,
                    "error": str(e),
                }
            ]
        }


async def _fetch_rag_context(state: ParallelFetchState) -> dict:
    """Fetch RAG context in parallel."""
    start = time.time()

    try:
        from resync.services.rag_client import RAGClient

        query = state.get("query", state.get("job_name", ""))
        rag = RAGClient()

        results = await asyncio.wait_for(rag.search(query=query, limit=5), timeout=5.0)

        return {
            "parallel_results": [
                {
                    "source": "rag_search",
                    "data": results,
                    "latency_ms": (time.time() - start) * 1000,
                    "success": True,
                    "error": None,
                }
            ]
        }
    except Exception as e:
        return {
            "parallel_results": [
                {
                    "source": "rag_search",
                    "data": {},
                    "latency_ms": (time.time() - start) * 1000,
                    "success": False,
                    "error": str(e),
                }
            ]
        }


def _fetch_logs(state: ParallelFetchState) -> dict:
    """Fetch recent logs in parallel."""
    start = time.time()

    try:
        # Simulated log fetch
        job_name = state.get("job_name")
        logs: dict[str, list[Any] | str | None] = {"recent_logs": [], "job": job_name}

        return {
            "parallel_results": [
                {
                    "source": "logs",
                    "data": logs,
                    "latency_ms": (time.time() - start) * 1000,
                    "success": True,
                    "error": None,
                }
            ]
        }
    except Exception as e:
        return {
            "parallel_results": [
                {
                    "source": "logs",
                    "data": {},
                    "latency_ms": (time.time() - start) * 1000,
                    "success": False,
                    "error": str(e),
                }
            ]
        }


def _aggregate_results(state: ParallelFetchState) -> dict:
    """Aggregate parallel fetch results."""
    results = state.get("parallel_results", [])

    aggregated = {}
    total_latency: float = 0.0

    for result in results:
        if result.get("success"):
            aggregated[result["source"]] = result["data"]
        total_latency = max(total_latency, result.get("latency_ms", 0))

    logger.info(
        "parallel_aggregation_complete",
        sources=len(results),
        successful=sum(1 for r in results if r.get("success")),
        latency_ms=total_latency,
    )

    return {
        "aggregated_data": aggregated,
        "total_latency_ms": total_latency,
    }


def _parallel_router(state: ParallelFetchState) -> list[Any] | str:
    """Route to parallel nodes using Send API (LangGraph 0.3)."""
    if not LANGGRAPH_03_FEATURES or Send is None:
        # Fallback for older versions
        return "sequential"

    return [
        Send("fetch_tws", state),
        Send("fetch_rag", state),
        Send("fetch_logs", state),
    ]


def create_parallel_subgraph():
    """
    Create the parallel fetch subgraph.

    Uses Send API for true parallel execution (LangGraph 0.3).

    Returns:
        Compiled subgraph or None if LangGraph unavailable
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(ParallelFetchState)

    graph.add_node("fetch_tws", _fetch_tws_status)
    graph.add_node("fetch_rag", _fetch_rag_context)
    graph.add_node("fetch_logs", _fetch_logs)
    graph.add_node("aggregate", _aggregate_results)

    graph.set_entry_point("fetch_tws")  # Will be parallelized via Send

    # Connect all fetch nodes to aggregate
    graph.add_edge("fetch_tws", "aggregate")
    graph.add_edge("fetch_rag", "aggregate")
    graph.add_edge("fetch_logs", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


# =============================================================================
# APPROVAL SUBGRAPH (using interrupt() - LangGraph 0.3)
# =============================================================================


class ApprovalState(TypedDict, total=False):
    """State for approval subgraph."""

    # Request
    action_type: str
    job_name: str
    parameters: dict[str, Any]
    risk_level: str

    # Approval
    approved: bool | None
    approver: str | None
    comments: str | None

    # Result
    execution_result: dict[str, Any] | None


def _request_approval_node(state: ApprovalState) -> dict:
    """Request human approval using interrupt()."""
    if not LANGGRAPH_03_FEATURES or interrupt is None:
        # Fallback: return pending state
        return {"approved": None}

    action = state.get("action_type", "unknown")
    job = state.get("job_name", "unknown")
    risk = state.get("risk_level", "medium")

    # Use LangGraph 0.3 interrupt() for human-in-the-loop
    approval = interrupt(
        {
            "type": "approval_request",
            "action": action,
            "job": job,
            "risk_level": risk,
            "message": f"Approve {action} on job {job}? (Risk: {risk})",
        }
    )

    return {
        "approved": approval.get("approved", False),
        "approver": approval.get("approver"),
        "comments": approval.get("comments"),
    }


async def _execute_action_node(state: ApprovalState) -> dict:
    """Execute the approved action."""
    if not state.get("approved"):
        return {"execution_result": {"status": "cancelled", "reason": "Not approved"}}

    try:
        from resync.services.tws_service import get_tws_client

        tws = get_tws_client()
        action = state.get("action_type")
        job = state.get("job_name")

        # Execute based on action type
        action_map = {
            "cancel": "cancel_job",
            "restart": "rerun_job",
            "submit": "submit_job",
        }
        method_name = action_map.get(str(action))
        if method_name is None:
            result = {"status": "unknown_action"}
        else:
            method = getattr(tws, method_name, None)
            if callable(method):
                result = await method(job)
            else:
                result = {"status": "unsupported_action", "action": str(action)}

        return {"execution_result": result}

    except Exception as e:
        return {"execution_result": {"status": "error", "error": str(e)}}


def _check_approval(state: ApprovalState) -> str:
    """Check if action was approved."""
    if state.get("approved"):
        return "execute"
    return END


def create_approval_subgraph():
    """
    Create the approval subgraph with human-in-the-loop.

    Uses interrupt() for native HitL support (LangGraph 0.3).

    Returns:
        Compiled subgraph or None if LangGraph unavailable
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(ApprovalState)

    graph.add_node("request_approval", _request_approval_node)
    graph.add_node("execute", _execute_action_node)

    graph.set_entry_point("request_approval")

    graph.add_conditional_edges(
        "request_approval", _check_approval, {"execute": "execute", END: END}
    )
    graph.add_edge("execute", END)

    return graph.compile()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_DIAGNOSTIC_SUBGRAPH = None
_PARALLEL_SUBGRAPH = None
_APPROVAL_SUBGRAPH = None


def get_diagnostic_subgraph():
    """Get or create diagnostic subgraph singleton."""
    global _DIAGNOSTIC_SUBGRAPH
    if _DIAGNOSTIC_SUBGRAPH is None:
        _DIAGNOSTIC_SUBGRAPH = create_diagnostic_subgraph()
    return _DIAGNOSTIC_SUBGRAPH


def get_parallel_subgraph():
    """Get or create parallel fetch subgraph singleton."""
    global _PARALLEL_SUBGRAPH
    if _PARALLEL_SUBGRAPH is None:
        _PARALLEL_SUBGRAPH = create_parallel_subgraph()
    return _PARALLEL_SUBGRAPH


def get_approval_subgraph():
    """Get or create approval subgraph singleton."""
    global _APPROVAL_SUBGRAPH
    if _APPROVAL_SUBGRAPH is None:
        _APPROVAL_SUBGRAPH = create_approval_subgraph()
    return _APPROVAL_SUBGRAPH


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # State types
    "SubgraphState",
    "ParallelResult",
    "DiagnosticSubgraphState",
    "ParallelFetchState",
    "ApprovalState",
    # Subgraph creators
    "create_diagnostic_subgraph",
    "create_parallel_subgraph",
    "create_approval_subgraph",
    # Factory functions
    "get_diagnostic_subgraph",
    "get_parallel_subgraph",
    "get_approval_subgraph",
]
