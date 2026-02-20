"""
GraphRAG Integration

Integrates SubgraphRetriever, EventDrivenDiscovery, and SmartCacheValidator
with existing Resync systems.

Author: Resync Team
Version: 5.9.9
"""

import structlog

from resync.core.event_driven_discovery import EventDrivenDiscovery
from resync.core.smart_cache_validator import SmartCacheValidator
from resync.core.subgraph_retriever import SubgraphRetriever

logger = structlog.get_logger(__name__)


class GraphRAGIntegration:
    """
    Coordinates GraphRAG features with existing Resync components.

    Provides:
    - SubgraphRetriever for enhanced context retrieval
    - EventDrivenDiscovery for automatic knowledge graph enrichment
    - SmartCacheValidator for event-driven cache validation
    """

    def __init__(
        self,
        llm_service,
        knowledge_graph,
        tws_client,
        redis_client=None,
        enabled: bool = True,
    ):
        """
        Initialize GraphRAG integration.

        Args:
            llm_service: LLM service instance
            knowledge_graph: Knowledge graph instance
            tws_client: TWS client instance
            redis_client: Redis client (optional)
            enabled: Enable/disable GraphRAG features
        """
        self.enabled = enabled

        if not enabled:
            logger.warning("GraphRAG integration disabled")
            self.subgraph_retriever = None
            self.discovery_service = None
            self.cache_validator = None
            return

        # Initialize components
        self.subgraph_retriever = SubgraphRetriever(knowledge_graph)
        self.discovery_service = EventDrivenDiscovery(
            llm_service, knowledge_graph, tws_client, redis_client
        )
        self.cache_validator = SmartCacheValidator(
            tws_client, redis_client, knowledge_graph, self.discovery_service
        )

        logger.info("GraphRAG integration initialized with smart cache validation")

    async def get_enriched_context(
        self, job_name: str, use_subgraph: bool = True
    ) -> dict:
        """
        Get enriched context for a job using GraphRAG.

        Args:
            job_name: Name of the job
            use_subgraph: Use subgraph retrieval (vs simple lookup)

        Returns:
            Structured context dict
        """
        if not self.enabled or not self.subgraph_retriever:
            return {"job": {"name": job_name}}

        try:
            if use_subgraph:
                context = await self.subgraph_retriever.retrieve_job_context(
                    job_name=job_name,
                    depth=2,
                    include_history=True,
                    include_solutions=True,
                )
            else:
                # Fallback to simple context
                context = {"job": {"name": job_name}}

            return context

        except Exception as e:
            logger.error("Failed to get enriched context: %s", e, exc_info=True)
            return {"job": {"name": job_name}}

    async def handle_job_event(
        self, event_type: str, job_name: str, event_details: dict
    ):
        """
        Handle job event - triggers cache validation and/or discovery.

        Called by TWSBackgroundPoller or other event sources.

        Args:
            event_type: Type of event (JOB_ABEND, JOB_FAILED, etc)
            job_name: Name of the job
            event_details: Event metadata
        """
        if not self.enabled:
            return

        # Handle failure events
        if event_type in ("JOB_ABEND", "job_abend", "JOB_FAILED", "job_failed"):
            # 1. Validate cache first
            if self.cache_validator:
                await self.cache_validator.on_job_failed(job_name, event_details)

            # 2. Trigger discovery (validator may have already triggered)
            # Discovery service has its own filters to avoid duplicates
            if self.discovery_service:
                await self.discovery_service.on_job_failed(job_name, event_details)

    async def get_stats(self) -> dict:
        """Get GraphRAG statistics including cache validation."""
        if not self.enabled:
            return {"enabled": False}

        stats = {"enabled": True}

        # Discovery stats
        if self.discovery_service:
            discovery_stats = await self.discovery_service.get_stats()
            stats["discovery"] = discovery_stats

        # Cache validation stats
        if self.cache_validator:
            validation_stats = await self.cache_validator.get_stats()
            stats["cache_validation"] = validation_stats

        return stats

    def clear_cache(self):
        """Clear GraphRAG caches."""
        if self.subgraph_retriever:
            self.subgraph_retriever.clear_cache()
            logger.info("GraphRAG cache cleared")

    def reset_validation_stats(self):
        """Reset cache validation statistics."""
        if self.cache_validator:
            self.cache_validator.reset_stats()
            logger.info("Cache validation stats reset")


# Global instance (initialized on startup)
_graphrag_integration: GraphRAGIntegration | None = None


def initialize_graphrag(
    llm_service, knowledge_graph, tws_client, redis_client=None, enabled: bool = True
):
    """
    Initialize global GraphRAG integration.

    Should be called during application startup.

    Args:
        llm_service: LLM service instance
        knowledge_graph: Knowledge graph instance
        tws_client: TWS client instance
        redis_client: Redis client (optional)
        enabled: Enable/disable GraphRAG
    """
    global _graphrag_integration

    _graphrag_integration = GraphRAGIntegration(
        llm_service=llm_service,
        knowledge_graph=knowledge_graph,
        tws_client=tws_client,
        redis_client=redis_client,
        enabled=enabled,
    )

    logger.info("GraphRAG integration initialized globally")

    return _graphrag_integration


def get_graphrag_integration() -> GraphRAGIntegration | None:
    """
    Get global GraphRAG integration instance.

    Returns:
        GraphRAGIntegration instance or None if not initialized
    """
    return _graphrag_integration


# Convenience functions for direct access


async def get_job_subgraph(job_name: str) -> dict:
    """
    Get job context as subgraph (convenience function).

    Args:
        job_name: Name of the job

    Returns:
        Structured subgraph context
    """
    integration = get_graphrag_integration()

    if not integration:
        logger.warning("GraphRAG not initialized")
        return {"job": {"name": job_name}}

    return await integration.get_enriched_context(job_name)


async def handle_job_failure(job_name: str, event_details: dict):
    """
    Handle job failure event (convenience function).

    Triggers auto-discovery if appropriate.

    Args:
        job_name: Name of failed job
        event_details: Event metadata
    """
    integration = get_graphrag_integration()

    if not integration:
        return

    await integration.handle_job_event("JOB_ABEND", job_name, event_details)
