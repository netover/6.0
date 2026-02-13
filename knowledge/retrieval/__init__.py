"""
Knowledge Retrieval Module.

v5.8.0: Unified retrieval across vector, graph, and hybrid approaches.
v5.9.9: Added IReranker interface with gating for CPU optimization.

Components:
- graph.py: Graph-based retrieval
- hybrid.py: Hybrid retrieval (BM25 + Vector)
- hybrid_retriever.py: Full hybrid retriever implementation
- retriever.py: Base retriever
- reranker.py: Result reranking (legacy)
- reranker_interface.py: IReranker interface with gating (v5.9.9)
"""

# Lazy imports to avoid circular dependencies
def get_hybrid_retriever():
    """Get HybridRetriever instance."""
    from .hybrid import HybridRetriever
    return HybridRetriever()

def get_graph_retriever():
    """Get GraphRetriever instance."""
    from .graph import KnowledgeGraph
    return KnowledgeGraph()

def get_reranker():
    """Get IReranker instance (v5.9.9)."""
    from .reranker_interface import create_reranker
    return create_reranker()

def get_gated_reranker():
    """Get reranker with gating policy (v5.9.9)."""
    from .reranker_interface import create_gated_reranker
    return create_gated_reranker()

__all__ = [
    "get_hybrid_retriever",
    "get_graph_retriever",
    "get_reranker",
    "get_gated_reranker",
]
