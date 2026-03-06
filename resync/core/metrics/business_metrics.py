"""Business metrics collection for Resync.

This module provides business-level KPIs and operational insights
beyond infrastructure metrics, using the internal metrics system.

The metrics are compatible with Prometheus format (via /metrics endpoint)
but use our internal lightweight implementation for flexibility.

v6.3.0: Added TWS job monitoring and LLM usage tracking.

Usage:
    from resync.core.metrics import business_metrics

    # Track TWS job failure
    business_metrics.tws_job_failures_total.labels(
        workstation="PROD01",
        jobstream="DAILY",
        job_name="BACKUP",
        error_code="ABEND0013"
    ).inc()

    # Track LLM tokens
    business_metrics.llm_tokens_consumed.observe(
        1245,
        labels={"provider": "ollama", "model": "qwen2.5:3b", "type": "total"}
    )
"""

from __future__ import annotations

from resync.core.metrics_internal import (
    create_counter,
    create_gauge,
    create_histogram,
)

# =============================================================================
# TWS JOB MONITORING
# =============================================================================

tws_job_failures_total = create_counter(
    "resync_tws_job_failures_total",
    "Total number of TWS job failures detected",
    labels=["workstation", "jobstream", "job_name", "error_code"],
)

tws_job_stuck_total = create_counter(
    "resync_tws_job_stuck_total",
    "Total number of TWS jobs stuck in EXEC state",
    labels=["workstation", "jobstream", "job_name"],
)

tws_job_late_total = create_counter(
    "resync_tws_job_late_total",
    "Total number of TWS jobs past their deadline",
    labels=["workstation", "jobstream", "job_name"],
)

tws_job_duration_seconds = create_histogram(
    "resync_tws_job_duration_seconds",
    "TWS job execution duration in seconds",
    labels=["workstation", "jobstream", "job_name", "status"],
)

tws_active_jobs_gauge = create_gauge(
    "resync_tws_active_jobs",
    "Current number of active TWS jobs",
    labels=["workstation", "status"],
)

# =============================================================================
# LLM USAGE TRACKING
# =============================================================================

llm_requests_total = create_counter(
    "resync_llm_requests_total",
    "Total number of LLM API requests",
    labels=["provider", "model", "status"],  # status: success/error/timeout
)

llm_tokens_consumed = create_histogram(
    "resync_llm_tokens_consumed",
    "Number of tokens consumed per LLM request",
    labels=["provider", "model", "type"],  # type: prompt/completion/total
)

llm_latency_seconds = create_histogram(
    "resync_llm_latency_seconds",
    "LLM request latency in seconds",
    labels=["provider", "model"],
)

llm_cache_hit_rate = create_gauge(
    "resync_llm_cache_hit_rate",
    "LLM semantic cache hit rate (0.0 to 1.0)",
    labels=["model"],
)

# Total accumulated cost (USD) as reported by LiteLLM pricing tables.
# This is a best-effort signal; some providers/models may return 0.
llm_cost_usd_total = create_counter(
    "resync_llm_cost_usd_total",
    "Total accumulated LLM cost in USD (best-effort)",
    labels=["provider", "model"],
)


# Total number of times we fell back to an alternate model due to retryable errors.
llm_fallbacks_total = create_counter(
    "resync_llm_fallbacks_total",
    "Total number of LLM model fallbacks triggered (retryable errors like 429/timeout/5xx)",
    labels=["reason", "from_model", "to_model"],
)

# =============================================================================
# AGENT & ORCHESTRATION
# =============================================================================

agent_executions_total = create_counter(
    "resync_agent_executions_total",
    "Total number of agent executions",
    labels=["agent_name", "status"],  # status: success/error/timeout
)

agent_execution_duration_seconds = create_histogram(
    "resync_agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    labels=["agent_name"],
)

workflow_executions_total = create_counter(
    "resync_workflow_executions_total",
    "Total number of workflow orchestration executions",
    labels=["workflow_id", "strategy", "status"],
)

workflow_step_failures_total = create_counter(
    "resync_workflow_step_failures_total",
    "Total number of failed workflow steps",
    labels=["workflow_id", "step_name", "error_type"],
)

# =============================================================================
# RAG & KNOWLEDGE GRAPH
# =============================================================================

rag_queries_total = create_counter(
    "resync_rag_queries_total",
    "Total number of RAG queries",
    labels=["retriever_type", "status"],  # type: hybrid/vector/bm25
)

rag_retrieval_latency_seconds = create_histogram(
    "resync_rag_retrieval_latency_seconds",
    "RAG document retrieval latency in seconds",
    labels=["retriever_type"],
)

rag_chunks_retrieved = create_histogram(
    "resync_rag_chunks_retrieved",
    "Number of chunks retrieved per RAG query",
    labels=["retriever_type"],
)

kg_extraction_duration_seconds = create_histogram(
    "resync_kg_extraction_duration_seconds",
    "Knowledge graph extraction duration per document",
)

kg_entities_extracted = create_histogram(
    "resync_kg_entities_extracted",
    "Number of entities extracted per document",
)

kg_relationships_extracted = create_histogram(
    "resync_kg_relationships_extracted",
    "Number of relationships extracted per document",
)

# =============================================================================
# USER FEEDBACK & CONTINUAL LEARNING
# =============================================================================

feedback_submissions_total = create_counter(
    "resync_feedback_submissions_total",
    "Total number of user feedback submissions",
    labels=["rating", "source"],  # rating: positive/negative, source: ui/api
)

feedback_negative_semantics_queries = create_counter(
    "resync_feedback_negative_semantics_queries",
    "Queries matching negative feedback patterns",
    labels=["matched_pattern"],
)

threshold_tuning_adjustments = create_counter(
    "resync_threshold_tuning_adjustments",
    "Number of threshold tuning adjustments made",
    labels=["threshold_type", "direction"],  # direction: increase/decrease
)
