# ruff: noqa: E501
# pylint: disable=all
"""
Incident Response Pipeline for Resync v6.0.0.

Implements a cognitive incident response system that:
1. Intercepts errors from anomaly_detector or chat queries
2. Enriches with historical context (RAG)
3. Analyzes and synthesizes with LLM
4. Outputs to Teams webhook OR chat response

This replaces the reactive model:
    Error -> Webhook

With a cognitive model:
    Error -> Diagnostic Graph -> RAG (History + Docs) -> LLM (Analysis) -> Output

Usage:
    # From anomaly_detector (automatic incident)
    from resync.core.langgraph.incident_response import handle_incident

    await handle_incident(
        error="RedisTimeoutError: Connection refused on port 6379",
        component="redis_pool",
        severity="critical",
        output_channel="teams"  # or "chat"
    )

    # From chat interface (operator query)
    result = await handle_incident(
        error="Job BATCH001 falhou com ABEND",
        component="tws",
        severity="high",
        output_channel="chat",
        user_context={"job_name": "BATCH001", "user_id": "operator1"}
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypedDict

from pydantic import BaseModel, Field

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# LangGraph imports
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"


# =============================================================================
# STATE AND MODELS
# =============================================================================


class Severity(str, Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OutputChannel(str, Enum):
    """Output channels for incident response."""

    TEAMS = "teams"
    CHAT = "chat"
    BOTH = "both"
    LOG_ONLY = "log_only"


class RelatedIncident(BaseModel):
    """A related historical incident."""

    timestamp: datetime
    error_type: str
    component: str
    resolution: str | None = None
    resolution_time_minutes: int | None = None


class IncidentAnalysis(BaseModel):
    """LLM-generated incident analysis."""

    summary: str = Field(description="Executive summary of the incident")
    is_recurring: bool = Field(description="Whether this is a recurring issue")
    recurrence_count: int = Field(default=0, description="Number of similar incidents")
    root_cause_hypothesis: str | None = Field(
        default=None, description="Probable root cause"
    )
    suggested_action: str = Field(description="Recommended immediate action")
    related_documentation: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="Confidence in the analysis")


class IncidentState(TypedDict, total=False):
    """State passed through the incident response graph."""

    # Input
    error: str
    error_type: str
    component: str
    severity: Severity
    timestamp: datetime
    traceback: str | None
    metrics_snapshot: dict[str, Any]

    # User context (for chat channel)
    user_context: dict[str, Any]
    original_query: str | None

    # Enrichment (from RAG)
    related_incidents: list[dict[str, Any]]
    similar_resolutions: list[str]
    documentation_context: list[dict[str, Any]]
    current_metrics: dict[str, Any]

    # Analysis (from LLM)
    analysis: dict[str, Any]

    # Output
    output_channel: OutputChannel
    formatted_message: str
    teams_card: dict[str, Any] | None
    chat_response: str | None

    # Metadata
    processing_time_ms: float
    graph_trace: list[str]


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================


def intercept_node(state: IncidentState) -> dict:
    """
    Node 1: Intercept and normalize the incident.

    Normalizes error data and extracts error type.
    """
    import re

    logger.info(
        "incident_intercept",
        component=state.get("component"),
        severity=state.get("severity"),
    )

    error = state.get("error", "Unknown error")

    # Extract error type from error message
    error_type_match = re.search(
        r"(\w+Error|\w+Exception|\w+Timeout|ABEND|ABND)", error
    )
    error_type = error_type_match.group(1) if error_type_match else "UnknownError"

    # Ensure timestamp
    timestamp = state.get("timestamp") or datetime.now(timezone.utc)

    return {
        "error_type": error_type,
        "timestamp": timestamp,
        "graph_trace": ["intercept"],
    }


async def enrichment_node(state: IncidentState) -> dict:
    """
    Node 2: Enrich with historical context using RAG.

    - Searches for similar incidents in the last 30 days
    - Retrieves related documentation
    - Captures current system metrics
    """
    logger.info("incident_enrichment", error_type=state.get("error_type"))

    error_type = state.get("error_type", "")
    component = state.get("component", "")

    related_incidents = []
    similar_resolutions = []
    documentation_context = []
    current_metrics = {}

    # 1. Search historical incidents via RAG
    try:
        from resync.core.tws_history_rag import search_historical_incidents

        # This function should exist or we create a fallback
        results = await search_historical_incidents(
            query=f"{error_type} {component}",
            days_back=30,
            limit=5,
        )

        for incident in results:
            related_incidents.append(
                {
                    "timestamp": incident.get("timestamp"),
                    "error_type": incident.get("error_type"),
                    "component": incident.get("component"),
                    "resolution": incident.get("resolution"),
                    "resolution_time_minutes": incident.get("mttr"),
                }
            )
            if incident.get("resolution"):
                similar_resolutions.append(incident["resolution"])

    except Exception as e:
        logger.warning("historical_search_failed", error=str(e))
        # Fallback: try generic RAG search
        try:
            from resync.services.rag_client import RAGClient

            rag = RAGClient()
            results = await rag.search(
                query=f"{error_type} {component} error resolution", limit=5
            )

            for doc in results.get("results", []):
                documentation_context.append(
                    {
                        "source": doc.get("source", "unknown"),
                        "content": doc.get("content", "")[:500],
                        "relevance": doc.get("score", 0),
                    }
                )
        except Exception as e2:
            logger.warning("rag_search_failed", error=str(e2))

    # 2. Get current system metrics
    try:
        from resync.api.v1.workstation_metrics_api import get_current_metrics

        current_metrics = await get_current_metrics(component)
    except Exception as e:
        logger.debug("metrics_fetch_failed", error=str(e))
        current_metrics = {"cpu": "N/A", "memory": "N/A", "connections": "N/A"}

    # Update trace
    trace = state.get("graph_trace", [])
    trace.append("enrichment")

    return {
        "related_incidents": related_incidents,
        "similar_resolutions": similar_resolutions,
        "documentation_context": documentation_context,
        "current_metrics": current_metrics,
        "graph_trace": trace,
    }


async def analysis_node(state: IncidentState) -> dict:
    """
    Node 3: Analyze and synthesize with LLM.

    Uses the enriched context to generate:
    - Executive summary
    - Root cause hypothesis
    - Suggested action
    """
    from resync.core.utils.llm import call_llm

    logger.info("incident_analysis", error_type=state.get("error_type"))

    # Build context for LLM
    error = state.get("error", "Unknown")
    error_type = state.get("error_type", "Unknown")
    component = state.get("component", "Unknown")
    severity = state.get("severity", Severity.MEDIUM)
    related = state.get("related_incidents", [])
    _resolutions = state.get("similar_resolutions", [])  # Reserved for future use
    docs = state.get("documentation_context", [])
    metrics = state.get("current_metrics", {})
    kg_context = (state.get("user_context") or {}).get("kg_context", "")

    # Format historical context
    history_text = ""
    if related:
        history_text = f"""
## Incidentes Similares (últimos 30 dias):
{len(related)} ocorrências encontradas.
"""
        for inc in related[:3]:
            history_text += f"- {inc.get('timestamp', 'N/A')}: {inc.get('error_type')} → Resolução: {inc.get('resolution', 'Não documentada')}\n"

    # Format documentation context
    docs_text = ""
    if docs:
        docs_text = "\n## Documentação Relacionada:\n"
        for doc in docs[:2]:
            docs_text += f"- [{doc.get('source')}]: {doc.get('content', '')[:200]}...\n"

    # Format metrics
    metrics_text = ""
    if metrics:
        metrics_text = f"\n## Métricas Atuais:\n{json.dumps(metrics, indent=2)}"

    # Format KG context (Document Knowledge Graph)
    kg_text = ""
    if kg_context:
        kg_text = f"\n## Contexto (Grafo de Conhecimento):\n{kg_context}"

    # Build prompt
    prompt = f"""Você é um Engenheiro de Confiabilidade de Site (SRE) Sênior.

## Erro Atual:
- Tipo: {error_type}
- Componente: {component}
- Severidade: {severity}
- Mensagem: {error}
{history_text}
{docs_text}
{metrics_text}
{kg_text}

## Tarefa:
Analise este incidente e forneça:
1. Um resumo executivo (2-3 frases)
2. Se é um problema recorrente (sim/não e quantas vezes)
3. Hipótese de causa raiz
4. Ação recomendada imediata

Seja conciso, empático e técnico. O público são desenvolvedores e operadores.

Responda em JSON:
{{
    "summary": "...",
    "is_recurring": true/false,
    "recurrence_count": N,
    "root_cause_hypothesis": "...",
    "suggested_action": "...",
    "confidence": 0.0-1.0
}}

JSON:"""

    try:
        response = await call_llm(
            prompt=prompt,
            model=settings.llm_model or "gpt-4o",
            temperature=0.3,
            max_tokens=500,
        )

        # Parse JSON from response
        import re

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {
                "summary": response[:200],
                "is_recurring": len(related) > 0,
                "recurrence_count": len(related),
                "root_cause_hypothesis": "Análise automática não disponível",
                "suggested_action": "Verificar logs do componente",
                "confidence": 0.5,
            }

    except Exception as e:
        logger.error("llm_analysis_failed", error=str(e))
        analysis = {
            "summary": f"Erro {error_type} detectado no componente {component}.",
            "is_recurring": len(related) > 0,
            "recurrence_count": len(related),
            "root_cause_hypothesis": "Análise manual necessária",
            "suggested_action": "Verificar logs e métricas do sistema",
            "confidence": 0.3,
        }

    # Update trace
    trace = state.get("graph_trace", [])
    trace.append("analysis")

    return {
        "analysis": analysis,
        "graph_trace": trace,
    }


def output_node(state: IncidentState) -> dict:
    """
    Node 4: Format and route output.

    Formats the analysis for the appropriate channel:
    - Teams: Adaptive Card with buttons
    - Chat: Markdown response
    - Both: Both formats
    """
    logger.info("incident_output", channel=state.get("output_channel"))

    analysis = state.get("analysis", {})
    error_type = state.get("error_type", "Unknown")
    component = state.get("component", "Unknown")
    severity = state.get("severity", Severity.MEDIUM)

    # Severity emoji
    severity_emoji = {
        Severity.LOW: "ℹ️",
        Severity.MEDIUM: "⚠️",
        Severity.HIGH: "🔶",
        Severity.CRITICAL: "🚨",
    }.get(severity, "⚠️")

    # Format for chat (markdown)
    chat_response = f"""
{severity_emoji} **Incidente: {error_type}**

**Resumo:** {analysis.get("summary", "N/A")}

**Componente:** {component}
**Recorrente:** {"Sim" if analysis.get("is_recurring") else "Não"} ({analysis.get("recurrence_count", 0)} ocorrências)

**Causa Provável:** {analysis.get("root_cause_hypothesis", "Em análise")}

**Ação Recomendada:** {analysis.get("suggested_action", "Verificar logs")}

---
*Confiança da análise: {int(analysis.get("confidence", 0) * 100)}%*
""".strip()

    # Format for Teams (Adaptive Card)
    teams_card = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"{severity_emoji} Incidente: {error_type}",
                            "weight": "Bolder",
                            "size": "Large",
                            "color": "Attention"
                            if severity in [Severity.HIGH, Severity.CRITICAL]
                            else "Default",
                        },
                        {
                            "type": "TextBlock",
                            "text": analysis.get("summary", "N/A"),
                            "wrap": True,
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Componente", "value": component},
                                {
                                    "title": "Severidade",
                                    "value": str(severity.value).upper(),
                                },
                                {
                                    "title": "Recorrente",
                                    "value": f"{'Sim' if analysis.get('is_recurring') else 'Não'} ({analysis.get('recurrence_count', 0)}x)",
                                },
                            ],
                        },
                        {
                            "type": "TextBlock",
                            "text": f"**Causa Provável:** {analysis.get('root_cause_hypothesis', 'Em análise')}",
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": f"**Ação Recomendada:** {analysis.get('suggested_action', 'Verificar logs')}",
                            "wrap": True,
                            "color": "Good",
                        },
                    ],
                    "actions": [
                        {
                            "type": "Action.OpenUrl",
                            "title": "📊 Dashboard",
                            "url": f"{settings.app_base_url or 'http://localhost:8000'}/monitoring",
                        },
                        {
                            "type": "Action.OpenUrl",
                            "title": "📋 Logs",
                            "url": f"{settings.app_base_url or 'http://localhost:8000'}/admin/logs?component={component}",
                        },
                    ],
                },
            }
        ],
    }

    # Update trace
    trace = state.get("graph_trace", [])
    trace.append("output")

    return {
        "formatted_message": chat_response,
        "teams_card": teams_card,
        "chat_response": chat_response,
        "graph_trace": trace,
    }


async def send_notification_node(state: IncidentState) -> dict:
    """
    Node 5: Send notification to configured channels.

    Only executes for Teams/Both channels.
    """
    output_channel = state.get("output_channel", OutputChannel.LOG_ONLY)

    if output_channel in [OutputChannel.TEAMS, OutputChannel.BOTH]:
        try:
            from resync.core.teams_webhook_handler import send_teams_message

            teams_card = state.get("teams_card")
            if teams_card:
                await send_teams_message(teams_card)
                logger.info("teams_notification_sent")
        except Exception as e:
            logger.error("teams_notification_failed", error=str(e))

    # Update trace
    trace = state.get("graph_trace", [])
    trace.append("notification")

    return {"graph_trace": trace}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_incident_response_graph() -> Any | None:
    """
    Create the incident response graph.

    Flow:
        intercept -> enrichment -> analysis -> output -> [send_notification] -> END

    Returns:
        Compiled StateGraph or None if LangGraph unavailable
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("langgraph_unavailable")
        return None

    graph = StateGraph(IncidentState)

    # Add nodes
    graph.add_node("intercept", intercept_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("output", output_node)
    graph.add_node("send_notification", send_notification_node)

    # Define flow
    graph.set_entry_point("intercept")
    graph.add_edge("intercept", "enrichment")
    graph.add_edge("enrichment", "analysis")
    graph.add_edge("analysis", "output")
    graph.add_edge("output", "send_notification")
    graph.add_edge("send_notification", END)

    return graph.compile()


# Cache the graph
_incident_graph: Any | None = None


def get_incident_graph() -> Any | None:
    """Get or create the incident response graph."""
    global _incident_graph
    if _incident_graph is None:
        _incident_graph = create_incident_response_graph()
    return _incident_graph


# =============================================================================
# PUBLIC API
# =============================================================================


async def handle_incident(
    error: str,
    component: str,
    severity: str | Severity = Severity.MEDIUM,
    output_channel: str | OutputChannel = OutputChannel.TEAMS,
    traceback: str | None = None,
    user_context: dict[str, Any] | None = None,
    original_query: str | None = None,
) -> dict[str, Any]:
    """
    Handle an incident through the response pipeline.

    This is the main entry point for both automatic incidents
    (from anomaly_detector) and user queries (from chat).

    Args:
        error: Error message or description
        component: Affected component (redis, database, tws, api, etc.)
        severity: Incident severity (low, medium, high, critical)
        output_channel: Where to send output (teams, chat, both, log_only)
        traceback: Optional full traceback
        user_context: Optional user context for chat queries
        original_query: Original user query (for chat channel)

    Returns:
        Dict with analysis results and formatted responses

    Examples:
        # From anomaly_detector
        await handle_incident(
            error="RedisTimeoutError: Connection refused",
            component="redis_pool",
            severity="critical",
            output_channel="teams"
        )

        # From chat interface
        result = await handle_incident(
            error="Job BATCH001 falhou com ABEND",
            component="tws",
            severity="high",
            output_channel="chat",
            user_context={"job_name": "BATCH001"},
            original_query="O que aconteceu com o job BATCH001?"
        )
        print(result["chat_response"])
    """
    import time

    start_time = time.time()

    # Normalize enums
    if isinstance(severity, str):
        severity = Severity(severity.lower())
    if isinstance(output_channel, str):
        output_channel = OutputChannel(output_channel.lower())

    # Build initial state
    state: IncidentState = {
        "error": error,
        "component": component,
        "severity": severity,
        "traceback": traceback,
        "output_channel": output_channel,
        "user_context": user_context or {},
        "original_query": original_query,
        "graph_trace": [],
    }

    # Get graph
    graph = get_incident_graph()

    if graph is None:
        # Fallback without LangGraph
        logger.warning("using_fallback_incident_handler")
        return await _fallback_handle_incident(state)

    try:
        # Execute graph
        result = await graph.ainvoke(state)

        # Calculate processing time
        result["processing_time_ms"] = (time.time() - start_time) * 1000

        logger.info(
            "incident_handled",
            error_type=result.get("error_type"),
            channel=output_channel.value,
            processing_ms=result["processing_time_ms"],
            trace=result.get("graph_trace"),
        )

        return result

    except Exception as e:
        logger.error("incident_handling_failed", error=str(e))
        return {
            "error": error,
            "component": component,
            "analysis": {"summary": f"Erro ao processar incidente: {str(e)}"},
            "chat_response": f"❌ Erro ao analisar o incidente: {str(e)}",
            "processing_time_ms": (time.time() - start_time) * 1000,
        }


async def _fallback_handle_incident(state: IncidentState) -> dict:
    """Fallback handler when LangGraph is unavailable."""
    # Execute nodes sequentially
    state = {**state, **intercept_node(state)}
    state = {**state, **await enrichment_node(state)}
    state = {**state, **await analysis_node(state)}
    state = {**state, **output_node(state)}
    state = {**state, **await send_notification_node(state)}
    return state


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================


def integrate_with_anomaly_detector() -> Callable[..., Any]:
    """
    Returns a callback function to integrate with anomaly_detector.

    Usage in anomaly_detector.py:
        from resync.core.langgraph.incident_response import integrate_with_anomaly_detector

        incident_callback = integrate_with_anomaly_detector()

        # When anomaly detected:
        await incident_callback(
            error=str(anomaly),
            component="cache",
            severity="high"
        )
    """

    async def callback(
        error: str, component: str, severity: str = "medium", **kwargs: Any
    ) -> dict[str, Any]:
        return await handle_incident(
            error=error,
            component=component,
            severity=severity,
            output_channel=OutputChannel.TEAMS,
            **kwargs,
        )

    return callback


def integrate_with_chat() -> Callable[..., Any]:
    """
    Returns a handler function for chat integration.

    Usage in agent_graph.py troubleshoot_handler:
        from resync.core.langgraph.incident_response import integrate_with_chat

        incident_handler = integrate_with_chat()

        # When user asks about an error:
        response = await incident_handler(
            error=user_message,
            component="tws",
            user_context={"job_name": job_name}
        )
        return response["chat_response"]
    """

    async def handler(
        error: str,
        component: str = "tws",
        user_context: dict[str, Any] | None = None,
        original_query: str | None = None,
    ) -> dict[str, Any]:
        return await handle_incident(
            error=error,
            component=component,
            severity=Severity.MEDIUM,
            output_channel=OutputChannel.CHAT,
            user_context=user_context,
            original_query=original_query,
        )

    return handler


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "Severity",
    "OutputChannel",
    # Models
    "IncidentState",
    "IncidentAnalysis",
    "RelatedIncident",
    # Graph
    "create_incident_response_graph",
    "get_incident_graph",
    # Main API
    "handle_incident",
    # Integration helpers
    "integrate_with_anomaly_detector",
    "integrate_with_chat",
]
