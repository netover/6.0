# pylint
# mypy
"""
TWS Knowledge Graph v5.9.3 - Simplified Wrapper

Provides a compatible interface for code that expects the old TWSKnowledgeGraph.
All graph operations are delegated to TwsGraphService which builds graphs on-demand.

Usage:
    from resync.knowledge.retrieval.graph import get_knowledge_graph

    kg = get_knowledge_graph()
    chain = await kg.get_dependency_chain("JOB_X")
    impact = await kg.get_impact_analysis("JOB_X")
"""

from typing import Any

import structlog

from resync.services.tws_graph_service import (
    TwsGraphService,
    get_graph_service,
)

logger = structlog.get_logger(__name__)


class KnowledgeGraphWrapper:
    """
    Wrapper that provides async methods expected by existing code.

    Delegates all operations to TwsGraphService.
    """

    def __init__(self, service: TwsGraphService | None = None):
        self._service = service or get_graph_service()

    def set_tws_client(self, tws_client: Any):
        """Set TWS client."""
        self._service.set_tws_client(tws_client)

    # =========================================================================
    # ASYNC QUERY METHODS
    # =========================================================================

    async def get_dependency_chain(
        self,
        job_id: str,
        max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """Get dependency chain for a job."""
        return await self._service.get_dependency_chain_async(job_id, max_depth)

    async def get_impact_analysis(self, job_id: str) -> dict[str, Any]:
        """Analyze impact if job fails."""
        return await self._service.get_impact_analysis_async(job_id)

    async def get_downstream_jobs(
        self,
        job_id: str,
        max_depth: int = 3,
    ) -> set[str]:
        """Get jobs that depend on this job."""
        return await self._service.get_downstream_jobs(job_id, max_depth)

    async def get_critical_jobs(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Get most critical jobs by centrality."""
        return await self._service.get_critical_jobs_async(top_n)

    async def find_resource_conflicts(
        self,
        job1: str,
        job2: str,
    ) -> list[dict[str, Any]]:
        """Find resource conflicts between jobs."""
        return await self._service.find_resource_conflicts(job1, job2)

    # =========================================================================
    # DEPRECATED WRITE METHODS (no-ops)
    # =========================================================================

    async def add_node(
        self,
        node_id: str,
        node_type: Any,
        name: str | None = None,
        **properties,
    ):
        """DEPRECATED: No-op. Graph built on-demand from TWS API."""
        await self._service.add_node(node_id, str(node_type), name, **properties)

    async def add_edge(
        self,
        source: str,
        target: str,
        relation_type: Any = "DEPENDS_ON",
        **properties,
    ):
        """DEPRECATED: No-op. Graph built on-demand from TWS API."""
        await self._service.add_edge(source, target, str(relation_type), **properties)

    def add_job(
        self,
        job_id: str,
        dependencies: list[str] | None = None,
        **properties,
    ):
        """DEPRECATED: No-op. Graph built on-demand from TWS API."""
        logger.debug("add_job_deprecated", job_id=job_id)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def initialize(self):
        """Initialize (no-op in v5.9.3)."""

    def reload(self):
        """Reload graph (clears cache)."""
        self._service.clear_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "version": "5.9.3",
            "storage": "on-demand",
            "cache": self._service.get_cache_stats(),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics (async version)."""
        return self.get_stats()


# =============================================================================
# MODULE-LEVEL FACTORY FUNCTIONS
# =============================================================================

_wrapper: KnowledgeGraphWrapper | None = None


def get_knowledge_graph() -> KnowledgeGraphWrapper:
    """Get the knowledge graph wrapper (singleton)."""
    global _wrapper
    if _wrapper is None:
        _wrapper = KnowledgeGraphWrapper()
    return _wrapper


# Aliases
get_kg_instance = get_knowledge_graph
get_graph_service = get_graph_service  # Re-export


async def initialize_knowledge_graph() -> KnowledgeGraphWrapper:
    """Initialize and return the knowledge graph."""
    kg = get_knowledge_graph()
    kg.initialize()
    return kg


# Legacy class alias
TWSKnowledgeGraph = KnowledgeGraphWrapper


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # New API
    "TwsGraphService",
    "get_graph_service",
    # Wrapper for compatibility
    "KnowledgeGraphWrapper",
    "TWSKnowledgeGraph",  # Alias
    "get_knowledge_graph",
    "get_kg_instance",  # Alias
    "initialize_knowledge_graph",
]
