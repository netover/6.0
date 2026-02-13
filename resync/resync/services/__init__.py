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

from resync.services.advanced_graph_queries import (
    AdvancedGraphQueryService,
    CommonNeighborAnalyzer,
    EdgeVerificationEngine,
    IntersectionResult,
    NegationQueryEngine,
    NegationResult,
    RelationConfidence,
    TemporalGraphManager,
    TemporalState,
    VerifiedRelationship,
    get_advanced_query_service,
)
from resync.services.llm_fallback import (
    FallbackReason,
    LLMFallbackConfig,
    LLMMetrics,
    LLMProvider,
    LLMResponse,
    LLMService,
    ModelConfig,
    configure_llm_service,
    get_llm_service,
    reset_llm_service,
)
from resync.services.tws_client_factory import get_tws_client as get_tws_client_factory
from resync.services.tws_graph_service import (
    TwsGraphService,
    build_job_graph,
    get_graph_service,
)
from resync.services.tws_service import OptimizedTWSClient
from resync.services.tws_unified import (
    MockTWSClient,
    TWSClientConfig,
    TWSClientMetrics,
    TWSClientState,
    UnifiedTWSClient,
    get_tws_client,
    reset_tws_client,
    tws_client_context,
    use_mock_tws_client,
)

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
