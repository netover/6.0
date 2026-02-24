"""
Resync Knowledge Module.

v6.0.0: Ontology-Driven Knowledge Graph.

Structure:
- retrieval/: Search and retrieval logic (vector, graph, hybrid)
- ingestion/: Data loading and processing (chunking, embeddings)
- store/: Persistence layer (PGVector, graph databases)
- ontology/: Domain ontology, validation, entity resolution (v6.0.0)

Usage:
    from resync.knowledge.retrieval import HybridRetriever
    from resync.knowledge.ingestion import ChunkingService
    from resync.knowledge.store import PGVectorStore
    from resync.knowledge.ontology import get_ontology_manager, validate_job

Author: Resync Team
Version: 6.0.0
"""

from resync.knowledge import ingestion, ontology, retrieval, store

__version__ = "6.0.0"

__all__ = [
    "retrieval",
    "ingestion",
    "store",
    "ontology",
]
