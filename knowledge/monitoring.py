"""
Internal metrics for RAG system observability.

Uses internal metrics system instead of prometheus_client.
"""

from resync.core.metrics_internal import (
    create_counter,
    create_gauge,
    create_histogram,
)

# Latency metrics
embed_seconds = create_histogram(
    "rag_embed_seconds",
    "Latency for embedding batches",
)

upsert_seconds = create_histogram(
    "rag_upsert_seconds",
    "Latency for vector upserts",
)

query_seconds = create_histogram(
    "rag_query_seconds",
    "Latency for vector queries",
)

# Job metrics
jobs_total = create_counter(
    "rag_jobs_total",
    "RAG jobs",
    labels=["status"],
)

# Collection metrics
collection_vectors = create_gauge(
    "rag_collection_vectors",
    "Vectors in current read collection",
)
