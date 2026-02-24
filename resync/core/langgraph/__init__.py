"""
LangGraph Integration for Resync v6.0.0.

Major refactoring with modern LangGraph 0.3 patterns:
- Structured output with Pydantic models
- Native interrupt() for human-in-the-loop
- Subgraphs for modular composition
- Templates from YAML files
- Native PostgreSQL checkpointer
- Streaming with events

Architecture:
    User Message -> Router Node -> [Handler Nodes] -> Synthesizer -> Response
                          ↓
                   [Subgraphs]
                   - Diagnostic (cyclic diagnosis)
                   - Parallel (concurrent data fetch)
                   - Approval (human-in-the-loop)

Usage:
    from resync.core.langgraph import create_tws_agent_graph

    graph = await create_tws_agent_graph()
    result = await graph.ainvoke({"message": "status do job BATCH001"})

    # With streaming
    async for event in graph.astream_events(state, version="v2"):
        print(event)

    # With checkpointing
    from resync.core.langgraph import get_checkpointer
    checkpointer = await get_checkpointer()
    graph = await create_tws_agent_graph(checkpointer=checkpointer)

Version History:
    - v6.0.0: Complete refactoring with LangGraph 0.3 patterns
    - v5.9.1: Clarification loop, synthesizer
    - v5.4.0: Diagnostic graph
    - v5.2.3.27: Hallucination grader
"""

# Agent Graph
from resync.core.langgraph.agent_graph import (
    AgentGraphConfig,
    AgentState,
    FallbackGraph,
    Intent,
    action_handler_node,
    clarification_node,
    create_router_graph,
    create_tws_agent_graph,
    general_handler_node,
    hallucination_check_node,
    query_handler_node,
    # Individual nodes
    router_node,
    status_handler_node,
    stream_agent_response,
    synthesizer_node,
    troubleshoot_handler_node,
)

# Checkpointer (native PostgreSQL)
from resync.core.langgraph.checkpointer import (
    NATIVE_CHECKPOINTER_AVAILABLE,
    PostgresCheckpointer,
    checkpointer_context,
    close_checkpointer,
    get_checkpointer,
    get_memory_store,
)

# Legacy: Diagnostic Graph (now use subgraphs)
from resync.core.langgraph.diagnostic_graph import (
    DiagnosticConfig,
    DiagnosticPhase,
    DiagnosticState,
    create_diagnostic_graph,
    diagnose_problem,
)

# Hallucination Grader
from resync.core.langgraph.hallucination_grader import (
    GradeAnswer,
    GradeDecision,
    GradeHallucinations,
    HallucinationGrader,
    HallucinationGradeResult,
    get_hallucination_grader,
    get_hallucination_route,
    grade_hallucination,
    is_response_grounded,
)

# v6.0.0: Incident Response Pipeline
from resync.core.langgraph.incident_response import (
    IncidentAnalysis,
    IncidentState,
    OutputChannel,
    Severity,
    create_incident_response_graph,
    get_incident_graph,
    handle_incident,
    integrate_with_anomaly_detector,
    integrate_with_chat,
)

# Pydantic Models for Structured Output
from resync.core.langgraph.models import (
    ActionRequest,
    AgentStateModel,
    ApprovalResponse,
    CauseHypothesis,
    DiagnosisOutput,
    EntityExtractionOutput,
    HallucinationGrade,
    RouterOutput,
    SymptomAnalysis,
    SynthesisInput,
)

# Legacy: Nodes
from resync.core.langgraph.nodes import (
    HumanApprovalNode,
    LLMNode,
    RouterNode,
    ToolNode,
    ValidationNode,
)

# Subgraphs
from resync.core.langgraph.subgraphs import (
    ApprovalState,
    DiagnosticSubgraphState,
    ParallelFetchState,
    ParallelResult,
    SubgraphState,
    create_approval_subgraph,
    create_diagnostic_subgraph,
    create_parallel_subgraph,
    get_approval_subgraph,
    get_diagnostic_subgraph,
    get_parallel_subgraph,
)

# Templates
from resync.core.langgraph.templates import (
    get_action_verb,
    get_clarification_question,
    get_status_translation,
    get_template,
    get_template_string,
    reload_templates,
    render_template,
)


# Convenience function for quick diagnosis
async def quick_diagnose(problem: str, job_name: str | None = None) -> dict:
    """
    Quick diagnosis using the diagnostic subgraph.

    Args:
        problem: Problem description
        job_name: Optional job name

    Returns:
        Diagnosis result dict
    """
    subgraph = get_diagnostic_subgraph()
    if subgraph:
        return await subgraph.ainvoke(
            {
                "problem_description": problem,
                "job_name": job_name,
                "max_iterations": 3,
            }
        )
    return {"error": "Diagnostic subgraph unavailable"}


__all__ = [
    # Agent Graph
    "AgentGraphConfig",
    "AgentState",
    "Intent",
    "create_tws_agent_graph",
    "create_router_graph",
    "FallbackGraph",
    "stream_agent_response",
    # Nodes
    "router_node",
    "clarification_node",
    "status_handler_node",
    "troubleshoot_handler_node",
    "query_handler_node",
    "action_handler_node",
    "general_handler_node",
    "synthesizer_node",
    "hallucination_check_node",
    # Legacy Nodes
    "RouterNode",
    "LLMNode",
    "ToolNode",
    "ValidationNode",
    "HumanApprovalNode",
    # Checkpointer
    "PostgresCheckpointer",
    "get_checkpointer",
    "close_checkpointer",
    "checkpointer_context",
    "get_memory_store",
    "NATIVE_CHECKPOINTER_AVAILABLE",
    # Models
    "RouterOutput",
    "EntityExtractionOutput",
    "DiagnosisOutput",
    "SymptomAnalysis",
    "CauseHypothesis",
    "HallucinationGrade",
    "ActionRequest",
    "ApprovalResponse",
    "SynthesisInput",
    "AgentStateModel",
    # Subgraphs
    "SubgraphState",
    "ParallelResult",
    "DiagnosticSubgraphState",
    "ParallelFetchState",
    "ApprovalState",
    "create_diagnostic_subgraph",
    "create_parallel_subgraph",
    "create_approval_subgraph",
    "get_diagnostic_subgraph",
    "get_parallel_subgraph",
    "get_approval_subgraph",
    # Templates
    "get_template",
    "get_template_string",
    "render_template",
    "reload_templates",
    "get_status_translation",
    "get_clarification_question",
    "get_action_verb",
    # Hallucination Grader
    "HallucinationGrader",
    "GradeHallucinations",
    "GradeAnswer",
    "GradeDecision",
    "HallucinationGradeResult",
    "grade_hallucination",
    "is_response_grounded",
    "get_hallucination_grader",
    "get_hallucination_route",
    # Legacy Diagnostic Graph
    "DiagnosticConfig",
    "DiagnosticPhase",
    "DiagnosticState",
    "create_diagnostic_graph",
    "diagnose_problem",
    "IncidentState",
    "IncidentAnalysis",
    "Severity",
    "OutputChannel",
    "create_incident_response_graph",
    "get_incident_graph",
    "handle_incident",
    "integrate_with_anomaly_detector",
    "integrate_with_chat",
    # Convenience
    "quick_diagnose",
]
