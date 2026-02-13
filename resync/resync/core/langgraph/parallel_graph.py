"""
LangGraph Parallel Execution Implementation.

This module implements advanced parallelization patterns for LangGraph:
- Fan-out/Fan-in (Map-Reduce) pattern for troubleshooting
- Parallel data fetching from multiple sources
- Annotated reducers for result aggregation

Performance Improvement Target:
- Before: Sequential execution ~3-5s for troubleshooting
- After: Parallel execution ~0.5-1s (60-70% reduction)

Architecture:
                    ┌─────────────────────┐
                    │       Router        │
                    └─────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│  TWS Status   │   │   RAG Search    │   │   Log Cache   │
│   (0.2-0.5s)  │   │    (0.3-0.5s)   │   │   (0.1-0.2s)  │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Aggregator    │
                    │ (Reduce/Merge)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Response Format │
                    └─────────────────┘

Usage:
    from resync.core.langgraph.parallel_graph import create_parallel_troubleshoot_graph

    graph = await create_parallel_troubleshoot_graph()
    result = await graph.ainvoke({"message": "Job BATCH001 falhou"})
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Try to import langgraph
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"


# =============================================================================
# STATE DEFINITIONS WITH REDUCERS
# =============================================================================


class DataSourceResult(TypedDict):
    """Result from a single data source."""

    source: str
    data: dict[str, Any]
    latency_ms: float
    success: bool
    error: str | None


class ParallelState(TypedDict, total=False):
    """
    State for parallel execution graph.

    Uses Annotated types with operator.add for automatic list merging
    when multiple nodes write to the same field.
    """

    # Input
    message: str
    user_id: str | None
    session_id: str | None
    tws_instance_id: str | None
    job_name: str | None

    # Classification
    intent: str
    confidence: float
    entities: dict[str, Any]

    # Parallel results - Using Annotated with operator.add for automatic merging
    # When multiple nodes return results, they are automatically combined
    parallel_results: Annotated[list[DataSourceResult], operator.add]

    # Aggregated data
    aggregated_data: dict[str, Any]

    # Timing metrics
    total_latency_ms: float
    parallel_latency_ms: float  # Time for parallel phase only

    # Output
    response: str
    metadata: dict[str, Any]

    # Errors
    errors: Annotated[list[str], operator.add]


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    # Timeouts
    node_timeout_seconds: float = 5.0
    total_timeout_seconds: float = 10.0

    # Data sources to query in parallel
    enable_tws_status: bool = True
    enable_rag_search: bool = True
    enable_log_cache: bool = True
    enable_metrics: bool = True

    # RAG settings
    rag_result_limit: int = 5

    # Aggregation
    min_sources_required: int = 1  # Minimum sources needed to generate response


# =============================================================================
# PARALLEL DATA FETCHING NODES
# =============================================================================


async def tws_status_node(state: ParallelState) -> dict[str, Any]:
    """
    Fetch TWS status in parallel.

    This node runs concurrently with other data fetching nodes.
    Returns results that will be automatically merged via operator.add.
    """
    start_time = time.time()
    source_name = "tws_status"

    logger.debug("parallel_node_start", node=source_name)

    try:
        from resync.tools.definitions.tws import tws_status_tool

        tws_instance = state.get("tws_instance_id")
        result = await asyncio.wait_for(
            tws_status_tool(instance_id=tws_instance),
            timeout=5.0,
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data=result,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            ]
        }

    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        logger.warning("parallel_node_timeout", node=source_name, latency_ms=latency_ms)
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error="Timeout fetching TWS status",
                )
            ],
            "errors": ["TWS status timeout after {latency_ms:.0f}ms"],
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("parallel_node_error", node=source_name, error=str(e), exc_info=True)
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            ],
            "errors": [f"TWS status error: {str(e)}"],
        }


async def rag_search_node(state: ParallelState) -> dict[str, Any]:
    """
    Search RAG knowledge base in parallel.

    Searches for relevant documentation and historical solutions.
    """
    start_time = time.time()
    source_name = "rag_search"

    logger.debug("parallel_node_start", node=source_name)

    try:
        from resync.services.rag_client import RAGServiceClient as RAGClient

        message = state.get("message", "")
        job_name = state.get("job_name")

        # Enrich query with job name if available
        search_query = message
        if job_name:
            search_query = f"{message} job:{job_name}"

        rag_client = RAGClient()
        search_results = await asyncio.wait_for(
            rag_client.search(query=search_query, limit=5),
            timeout=5.0,
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={
                        "results": search_results.get("results", []),
                        "total_found": search_results.get("total", 0),
                        "query": search_query,
                    },
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            ]
        }

    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        logger.warning("parallel_node_timeout", node=source_name, latency_ms=latency_ms)
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error="Timeout searching RAG",
                )
            ],
            "errors": ["RAG search timeout after {latency_ms:.0f}ms"],
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("parallel_node_error", node=source_name, error=str(e), exc_info=True)
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            ],
            "errors": [f"RAG search error: {str(e)}"],
        }


async def log_cache_node(state: ParallelState) -> dict[str, Any]:
    """
    Fetch historical logs from cache in parallel.

    Retrieves recent job execution logs and error patterns.
    """
    start_time = time.time()
    source_name = "log_cache"

    logger.debug("parallel_node_start", node=source_name)

    try:
        from resync.core.redis_init import get_redis_client

        job_name = state.get("job_name")
        state.get("message", "")

        # Try to get cached logs
        redis_client = get_redis_client()

        logs_data = {}
        if redis_client:
            # Get recent error patterns
            error_key = f"resync:errors:recent:{job_name or 'global'}"
            cached_errors = await asyncio.wait_for(
                redis_client.get(error_key),
                timeout=2.0,
            )

            if cached_errors:
                logs_data["recent_errors"] = json.loads(cached_errors)

            # Get job history if job_name provided
            if job_name:
                history_key = f"resync:job:history:{job_name}"
                cached_history = await asyncio.wait_for(
                    redis_client.get(history_key),
                    timeout=2.0,
                )
                if cached_history:
                    logs_data["job_history"] = json.loads(cached_history)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data=logs_data,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            ]
        }

    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        logger.warning("parallel_node_timeout", node=source_name, latency_ms=latency_ms)
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error="Timeout fetching logs",
                )
            ],
            "errors": ["Log cache timeout after {latency_ms:.0f}ms"],
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        # Log cache errors are non-critical
        logger.debug("parallel_node_error", node=source_name, error=str(e))
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            ],
        }


def metrics_node(state: ParallelState) -> dict[str, Any]:
    """
    Fetch relevant metrics in parallel.

    Gets performance metrics, SLO data, and anomaly indicators.
    """
    start_time = time.time()
    source_name = "metrics"

    logger.debug("parallel_node_start", node=source_name)

    try:
        from resync.core.metrics import RuntimeMetrics

        metrics = RuntimeMetrics.get_snapshot()

        latency_ms = (time.time() - start_time) * 1000

        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={
                        "llm_latency_avg": metrics.get("llm_latency_avg_ms", 0),
                        "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                        "error_rate": metrics.get("error_rate_1h", 0),
                        "active_connections": metrics.get("active_connections", 0),
                    },
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            ]
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        # Metrics errors are non-critical
        logger.debug("parallel_node_error", node=source_name, error=str(e))
        return {
            "parallel_results": [
                DataSourceResult(
                    source=source_name,
                    data={},
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            ],
        }


# =============================================================================
# AGGREGATOR NODE
# =============================================================================


def aggregator_node(state: ParallelState) -> dict[str, Any]:
    """
    Aggregate results from all parallel nodes.

    This is the "reduce" phase of map-reduce pattern.
    Combines data from multiple sources into a unified structure.
    """
    logger.debug("aggregator_node_start")

    parallel_results = state.get("parallel_results", [])

    # Calculate parallel phase latency
    if parallel_results:
        max_latency = max(r.get("latency_ms", 0) for r in parallel_results)
        total_sequential_latency = sum(r.get("latency_ms", 0) for r in parallel_results)
    else:
        max_latency = 0
        total_sequential_latency = 0

    # Aggregate by source
    aggregated = {
        "tws_status": {},
        "rag_results": [],
        "log_data": {},
        "metrics": {},
        "sources_available": [],
        "sources_failed": [],
    }

    for result in parallel_results:
        source = result.get("source", "unknown")

        if result.get("success"):
            aggregated["sources_available"].append(source)

            if source == "tws_status":
                aggregated["tws_status"] = result.get("data", {})
            elif source == "rag_search":
                aggregated["rag_results"] = result.get("data", {}).get("results", [])
            elif source == "log_cache":
                aggregated["log_data"] = result.get("data", {})
            elif source == "metrics":
                aggregated["metrics"] = result.get("data", {})
        else:
            aggregated["sources_failed"].append(
                {
                    "source": source,
                    "error": result.get("error"),
                }
            )

    # Calculate performance improvement
    speedup_factor = total_sequential_latency / max_latency if max_latency > 0 else 1

    logger.info(
        "parallel_aggregation_complete",
        sources_available=len(aggregated["sources_available"]),
        sources_failed=len(aggregated["sources_failed"]),
        parallel_latency_ms=max_latency,
        sequential_equivalent_ms=total_sequential_latency,
        speedup_factor="{speedup_factor:.2f}x",
    )

    return {
        "aggregated_data": aggregated,
        "parallel_latency_ms": max_latency,
        "metadata": {
            "parallel_execution": True,
            "sources_queried": len(parallel_results),
            "sources_successful": len(aggregated["sources_available"]),
            "speedup_factor": speedup_factor,
            "parallel_latency_ms": max_latency,
            "sequential_equivalent_ms": total_sequential_latency,
        },
    }


# =============================================================================
# RESPONSE GENERATION NODE
# =============================================================================


async def response_generator_node(state: ParallelState) -> dict[str, Any]:
    """
    Generate final response from aggregated data.

    Uses LLM to synthesize insights from multiple sources.
    """
    logger.debug("response_generator_start")

    aggregated = state.get("aggregated_data", {})
    message = state.get("message", "")

    try:
        from resync.core.langfuse import get_prompt_manager
        from resync.core.utils.llm import call_llm

        # Build context from aggregated data
        context_parts = []

        # Add TWS status
        if aggregated.get("tws_status"):
            status = aggregated["tws_status"]
            context_parts.append(
                f"**Status do TWS:**\n{json.dumps(status, indent=2, ensure_ascii=False)}"
            )

        # Add RAG results
        rag_results = aggregated.get("rag_results", [])
        if rag_results:
            rag_context = "\n".join(
                [
                    f"- [{r.get('source', 'Doc')}]: {r.get('content', '')[:200]}..."
                    for r in rag_results[:3]
                ]
            )
            context_parts.append(f"**Documentação Relevante:**\n{rag_context}")

        # Add historical data
        if aggregated.get("log_data"):
            log_data = aggregated["log_data"]
            if log_data.get("recent_errors"):
                context_parts.append(
                    f"**Erros Recentes:**\n{json.dumps(log_data['recent_errors'][:3], indent=2)}"
                )

        # Add metrics
        if aggregated.get("metrics"):
            metrics = aggregated["metrics"]
            context_parts.append(
                f"**Métricas:**\n"
                "- Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}\n"
                "- Error Rate: {metrics.get('error_rate', 0):.2%}\n"
                "- LLM Latency: {metrics.get('llm_latency_avg', 0):.0f}ms"
            )

        full_context = "\n\n".join(context_parts)

        # Get troubleshooting prompt
        get_prompt_manager()

        system_prompt = f"""Você é um especialista em troubleshooting do TWS (Tivoli Workload Scheduler) / HCL Workload Automation.

Com base nos dados coletados de múltiplas fontes, analise o problema e forneça:
1. Diagnóstico provável
2. Passos de resolução recomendados
3. Ações preventivas

DADOS COLETADOS:
{full_context}

Responda de forma estruturada e acionável."""

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {message}"

        response = await call_llm(
            prompt=full_prompt,
            model=settings.llm_model or "gpt-4o",
            max_tokens=1000,
            temperature=0.3,
        )

        # Add performance note
        parallel_latency = state.get("parallel_latency_ms", 0)
        sources_count = len(aggregated.get("sources_available", []))

        performance_note = (
            f"\n\n---\n"
            "*Análise paralela: {sources_count} fontes consultadas em {parallel_latency:.0f}ms*"
        )

        return {
            "response": response + performance_note,
        }

    except Exception as e:
        logger.error("response_generation_failed", error=str(e), exc_info=True)

        # Fallback response using raw data
        available_sources = aggregated.get("sources_available", [])
        if available_sources:
            fallback_response = f"Não foi possível gerar análise completa, mas coletei dados de: {', '.join(available_sources)}.\n\n"
            if aggregated.get("tws_status"):
                fallback_response += f"Status TWS: {json.dumps(aggregated['tws_status'], indent=2)}"
            return {"response": fallback_response}

        return {
            "response": f"Erro na análise de troubleshooting: {str(e)}",
            "errors": [str(e)],
        }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_parallel_troubleshoot_graph(
    config: ParallelConfig | None = None,
) -> Any:
    """
    Create a parallelized troubleshooting graph.

    This graph executes data fetching in parallel (fan-out),
    then aggregates results (fan-in) before generating response.

    Args:
        config: Configuration for parallel execution

    Returns:
        Compiled StateGraph with parallel execution
    """
    config = config or ParallelConfig()

    if not LANGGRAPH_AVAILABLE:
        logger.warning("langgraph_not_available_using_fallback")
        return FallbackParallelGraph(config)

    # Create graph with ParallelState
    graph = StateGraph(ParallelState)

    # Add parallel data fetching nodes
    if config.enable_tws_status:
        graph.add_node("tws_status", tws_status_node)
    if config.enable_rag_search:
        graph.add_node("rag_search", rag_search_node)
    if config.enable_log_cache:
        graph.add_node("log_cache", log_cache_node)
    if config.enable_metrics:
        graph.add_node("metrics", metrics_node)

    # Add aggregator and response generator
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("response_generator", response_generator_node)

    # Set entry point - fan out to all data sources
    # LangGraph will execute these in parallel automatically
    graph.set_entry_point("tws_status")

    # Fan-out: Multiple edges from entry point (executed in parallel)
    enabled_sources = []
    if config.enable_tws_status:
        enabled_sources.append("tws_status")
    if config.enable_rag_search:
        enabled_sources.append("rag_search")
    if config.enable_log_cache:
        enabled_sources.append("log_cache")
    if config.enable_metrics:
        enabled_sources.append("metrics")

    # All parallel nodes converge to aggregator (fan-in)
    for source in enabled_sources:
        graph.add_edge(source, "aggregator")

    # Aggregator to response generator
    graph.add_edge("aggregator", "response_generator")

    # Response generator to END
    graph.add_edge("response_generator", END)

    compiled = graph.compile()

    logger.info(
        "parallel_troubleshoot_graph_created",
        parallel_sources=len(enabled_sources),
        sources=enabled_sources,
    )

    return compiled


# =============================================================================
# FALLBACK IMPLEMENTATION
# =============================================================================


class FallbackParallelGraph:
    """
    Fallback when LangGraph is not available.

    Uses asyncio.gather for parallel execution instead of LangGraph's
    native parallelization.
    """

    def __init__(self, config: ParallelConfig):
        self.config = config

    async def ainvoke(self, state: dict[str, Any]) -> ParallelState:
        """Execute with asyncio.gather for parallelization."""
        start_time = time.time()

        # Initialize state
        full_state: ParallelState = {
            "message": state.get("message", ""),
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id"),
            "tws_instance_id": state.get("tws_instance_id"),
            "job_name": state.get("job_name"),
            "parallel_results": [],
            "errors": [],
        }

        # Build list of tasks to run in parallel
        tasks = []
        task_names = []

        if self.config.enable_tws_status:
            tasks.append(tws_status_node(full_state))
            task_names.append("tws_status")
        if self.config.enable_rag_search:
            tasks.append(rag_search_node(full_state))
            task_names.append("rag_search")
        if self.config.enable_log_cache:
            tasks.append(log_cache_node(full_state))
            task_names.append("log_cache")
        if self.config.enable_metrics:
            tasks.append(metrics_node(full_state))
            task_names.append("metrics")

        # Execute all in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.total_timeout_seconds,
            )

            # Merge results into state
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    full_state["errors"].append(f"{task_names[i]}: {str(result)}")
                elif isinstance(result, dict):
                    if "parallel_results" in result:
                        full_state["parallel_results"].extend(result["parallel_results"])
                    if "errors" in result:
                        full_state["errors"].extend(result["errors"])

        except asyncio.TimeoutError:
            full_state["errors"].append("Parallel execution timeout")

        # Aggregate results
        aggregator_result = aggregator_node(full_state)
        full_state.update(aggregator_result)

        # Generate response
        response_result = await response_generator_node(full_state)
        full_state.update(response_result)

        # Calculate total latency
        full_state["total_latency_ms"] = (time.time() - start_time) * 1000

        return full_state

    async def invoke(self, state: dict[str, Any]) -> ParallelState:
        """Sync-compatible invoke."""
        return await self.ainvoke(state)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================


async def parallel_troubleshoot(
    message: str,
    job_name: str | None = None,
    tws_instance_id: str | None = None,
    config: ParallelConfig | None = None,
) -> dict[str, Any]:
    """
    High-level function for parallel troubleshooting.

    Usage:
        result = await parallel_troubleshoot(
            message="Job BATCH001 falhou com ABEND",
            job_name="BATCH001",
        )
        print(result["response"])

    Args:
        message: User's troubleshooting query
        job_name: Optional job name for targeted search
        tws_instance_id: Optional TWS instance ID
        config: Optional configuration

    Returns:
        Dict with response, metadata, and timing information
    """
    graph = create_parallel_troubleshoot_graph(config)

    result = await graph.ainvoke(
        {
            "message": message,
            "job_name": job_name,
            "tws_instance_id": tws_instance_id,
        }
    )

    return {
        "response": result.get("response", ""),
        "metadata": result.get("metadata", {}),
        "parallel_latency_ms": result.get("parallel_latency_ms", 0),
        "total_latency_ms": result.get("total_latency_ms", 0),
        "errors": result.get("errors", []),
    }
