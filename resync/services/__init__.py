"""
Resync Services Module

v5.4.2: Unified service layer with resilience patterns.
v5.2.3.26: Added advanced KG query techniques.

Services:
- TWS Unified Client: Single access point for TWS with circuit breakers
- LLM Fallback Service: LLM calls with automatic fallback chain
- RAG Client: RAG operations with resilience
- Advanced Graph Queries: Temporal, negation, intersection, verification
"""

from importlib import import_module

from resync.core.factories.tws_factory import get_tws_client as get_tws_client_factory

__all__ = [
    # TWS Unified (v5.4.2 - recommended)
    "UnifiedTWSClient",
    "TWSClientConfig",
    "TWSClientState",
    "TWSClientMetrics",
    "MockTWSClient",
    "get_tws_client",
    "reset_tws_client",
    "tws_client_context",
    "use_mock_tws_client",
    # TWS Graph Service
    "TwsGraphService",
    "get_graph_service",
    "build_job_graph",
    # Advanced Graph Queries (v5.2.3.26)
    "AdvancedGraphQueryService",
    "TemporalGraphManager",
    "NegationQueryEngine",
    "CommonNeighborAnalyzer",
    "EdgeVerificationEngine",
    "TemporalState",
    "NegationResult",
    "IntersectionResult",
    "VerifiedRelationship",
    "RelationConfidence",
    "get_advanced_query_service",
    # LLM Fallback (v5.4.2)
    "LLMService",
    "LLMFallbackConfig",
    "LLMProvider",
    "LLMResponse",
    "LLMMetrics",
    "ModelConfig",
    "FallbackReason",
    "get_llm_service",
    "reset_llm_service",
    "configure_llm_service",
    # Legacy (for backward compatibility)
    "OptimizedTWSClient",
    "get_tws_client_factory",
]

_LAZY_EXPORTS = {
    "AdvancedGraphQueryService": (
        "resync.services.advanced_graph_queries",
        "AdvancedGraphQueryService",
    ),
    "CommonNeighborAnalyzer": (
        "resync.services.advanced_graph_queries",
        "CommonNeighborAnalyzer",
    ),
    "EdgeVerificationEngine": (
        "resync.services.advanced_graph_queries",
        "EdgeVerificationEngine",
    ),
    "IntersectionResult": (
        "resync.services.advanced_graph_queries",
        "IntersectionResult",
    ),
    "NegationQueryEngine": (
        "resync.services.advanced_graph_queries",
        "NegationQueryEngine",
    ),
    "NegationResult": (
        "resync.services.advanced_graph_queries",
        "NegationResult",
    ),
    "RelationConfidence": (
        "resync.services.advanced_graph_queries",
        "RelationConfidence",
    ),
    "TemporalGraphManager": (
        "resync.services.advanced_graph_queries",
        "TemporalGraphManager",
    ),
    "TemporalState": ("resync.services.advanced_graph_queries", "TemporalState"),
    "VerifiedRelationship": (
        "resync.services.advanced_graph_queries",
        "VerifiedRelationship",
    ),
    "get_advanced_query_service": (
        "resync.services.advanced_graph_queries",
        "get_advanced_query_service",
    ),
    "FallbackReason": ("resync.services.llm_fallback", "FallbackReason"),
    "LLMFallbackConfig": ("resync.services.llm_fallback", "LLMFallbackConfig"),
    "LLMMetrics": ("resync.services.llm_fallback", "LLMMetrics"),
    "LLMProvider": ("resync.services.llm_fallback", "LLMProvider"),
    "LLMResponse": ("resync.services.llm_fallback", "LLMResponse"),
    "LLMService": ("resync.services.llm_fallback", "LLMService"),
    "ModelConfig": ("resync.services.llm_fallback", "ModelConfig"),
    "configure_llm_service": (
        "resync.services.llm_fallback",
        "configure_llm_service",
    ),
    "get_llm_service": ("resync.services.llm_fallback", "get_llm_service"),
    "reset_llm_service": ("resync.services.llm_fallback", "reset_llm_service"),
    "TwsGraphService": ("resync.services.tws_graph_service", "TwsGraphService"),
    "build_job_graph": ("resync.services.tws_graph_service", "build_job_graph"),
    "get_graph_service": ("resync.services.tws_graph_service", "get_graph_service"),
    "OptimizedTWSClient": ("resync.services.tws_service", "OptimizedTWSClient"),
    "MockTWSClient": ("resync.services.tws_unified", "MockTWSClient"),
    "TWSClientConfig": ("resync.services.tws_unified", "TWSClientConfig"),
    "TWSClientMetrics": ("resync.services.tws_unified", "TWSClientMetrics"),
    "TWSClientState": ("resync.services.tws_unified", "TWSClientState"),
    "UnifiedTWSClient": ("resync.services.tws_unified", "UnifiedTWSClient"),
    "get_tws_client": ("resync.services.tws_unified", "get_tws_client"),
    "reset_tws_client": ("resync.services.tws_unified", "reset_tws_client"),
    "tws_client_context": ("resync.services.tws_unified", "tws_client_context"),
    "use_mock_tws_client": ("resync.services.tws_unified", "use_mock_tws_client"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
