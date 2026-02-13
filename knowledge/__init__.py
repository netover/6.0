"""
Resync Knowledge Module.

v5.9.2: Ontology-Driven Knowledge Graph.

Structure:
- retrieval/: Search and retrieval logic (vector, graph, hybrid)
- ingestion/: Data loading and processing (chunking, embeddings)
- store/: Persistence layer (PGVector, graph databases)
- ontology/: Domain ontology, validation, entity resolution (v5.9.2)

Usage:
    from resync.knowledge.retrieval import HybridRetriever
    from resync.knowledge.ingestion import ChunkingService
    from resync.knowledge.store import PGVectorStore
    from resync.knowledge.ontology import get_ontology_manager, validate_job

Author: Resync Team
Version: 5.9.2
"""

__all__ = [
    "retrieval",
    "ingestion",
    "store",
    "ontology",
]
