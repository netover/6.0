"""
TWS Graph Service v5.2.3.26

Builds NetworkX graphs on-demand from TWS API.
Replaces persistent graph storage with fresh data from source of truth.

Features:
- On-demand graph construction from TWS API
- In-memory caching with configurable TTL
- Critical path analysis
- Impact analysis
- Betweenness centrality for bottleneck detection
- v5.2.3.26: Advanced KG queries (temporal, negation, intersection, verification)

Usage:
    from resync.services.tws_graph_service import TwsGraphService, get_graph_service

    # Get singleton instance
    service = get_graph_service(tws_client)

    # Build dependency graph for a job
    graph = await service.get_dependency_graph("JOB_XPTO")

    # Analyze impact
    impact = service.get_impact_analysis(graph, "JOB_XPTO")

    # Find critical path
    path = service.find_critical_path(graph)

    # v5.2.3.26: Advanced queries
    safe_jobs = await service.find_safe_jobs_if_fails("JOB_XPTO")
    history = await service.get_job_status_at_time("JOB_XPTO", some_datetime)

Author: Resync Team
Version: 5.2.3.26
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GraphCacheEntry:
    """Cache entry for a graph."""
    graph: nx.DiGraph
    created_at: float
    scope: str


class TwsGraphService:
    """
    Builds NetworkX graphs on-demand from TWS API.

    This service replaces the persistent graph storage with fresh data
    fetched directly from TWS API. The graph is cached in memory for
    a configurable TTL to avoid excessive API calls.

    Key Principles:
    - TWS API is the single source of truth
    - Graph is built on-demand, not persisted
    - Data is always fresh (within TTL)
    - Memory usage is minimal (~30MB for 100K jobs)
    """

    def __init__(
        self,
        tws_client: Any = None,
        cache_ttl: int = 300,  # 5 minutes default
        max_depth: int = 5,
    ):
        """
        Initialize TwsGraphService.

        Args:
            tws_client: TWS API client
            cache_ttl: Cache time-to-live in seconds
            max_depth: Maximum depth for dependency traversal
        """
        self.tws_client = tws_client
        self.cache_ttl = cache_ttl
        self.max_depth = max_depth
        self._cache: dict[str, GraphCacheEntry] = {}

        logger.info(
            "tws_graph_service_initialized",
            cache_ttl=cache_ttl,
            max_depth=max_depth,
        )

    def set_tws_client(self, tws_client: Any):
        """Set or update the TWS client."""
        self.tws_client = tws_client

    # =========================================================================
    # GRAPH BUILDING
    # =========================================================================

    async def get_dependency_graph(
        self,
        job_id: str,
        depth: int | None = None,
        force_refresh: bool = False,
    ) -> nx.DiGraph:
        """
        Build dependency graph for a job from TWS API.

        Args:
            job_id: Job ID to build graph for
            depth: Maximum traversal depth (default: self.max_depth)
            force_refresh: Force cache refresh

        Returns:
            NetworkX DiGraph with job dependencies
        """
        cache_key = f"job:{job_id}:depth:{depth or self.max_depth}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry.created_at < self.cache_ttl:
                logger.debug("graph_cache_hit", job_id=job_id)
                return entry.graph

        # Build from API
        graph = nx.DiGraph()
        visited = set()
        await self._build_job_dependencies(
            graph, job_id, visited, depth or self.max_depth
        )

        # Cache result
        self._cache[cache_key] = GraphCacheEntry(
            graph=graph,
            created_at=time.time(),
            scope=f"job:{job_id}",
        )

        logger.info(
            "graph_built_from_api",
            job_id=job_id,
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
        )

        return graph

    async def _build_job_dependencies(
        self,
        graph: nx.DiGraph,
        root_job_id: str,
        visited: set[str],
        max_depth: int,
    ):
        """
        Build job dependency graph using parallel BFS level-fetching.
        
        This implementation resolves the N+1 query pattern by fetching all 
        successors/predecessors for a given level in parallel using asyncio.gather.
        """
        current_level = {root_job_id}
        
        for depth in range(max_depth):
            if not current_level:
                break
                
            # Filter out jobs we've already visited to avoid redundant API calls
            to_fetch = [jid for jid in current_level if jid not in visited]
            if not to_fetch:
                break
                
            visited.update(to_fetch)
            
            # Prepare parallel tasks for the current level
            tasks = []
            for jid in to_fetch:
                if jid not in graph:
                    graph.add_node(jid)
                
                # Fetch predecessors and successors in parallel for each job
                tasks.append(self.tws_client.get_current_plan_job_predecessors(jid))
                tasks.append(self.tws_client.get_current_plan_job_successors(jid))
            
            if not tasks:
                break
                
            # Execute all level queries in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            next_level = set()
            
            # Process results (results are alternating preds, succs, preds, succs...)
            for i, jid in enumerate(to_fetch):
                preds_result = results[i * 2]
                succs_result = results[i * 2 + 1]
                
                # Predecessors
                if isinstance(preds_result, list):
                    for pred in preds_result:
                        pred_id = pred.get("jobId") or pred.get("id") or str(pred)
                        if pred_id:
                            graph.add_edge(pred_id, jid, relation="DEPENDS_ON")
                            next_level.add(pred_id)
                elif isinstance(preds_result, Exception):
                    logger.warning("api_preds_failed", job_id=jid, error=str(preds_result))
                
                # Successors
                if isinstance(succs_result, list):
                    for succ in succs_result:
                        succ_id = succ.get("jobId") or succ.get("id") or str(succ)
                        if succ_id:
                            graph.add_edge(jid, succ_id, relation="DEPENDS_ON")
                            next_level.add(succ_id)
                elif isinstance(succs_result, Exception):
                    logger.warning("api_succs_failed", job_id=jid, error=str(succs_result))
            
            current_level = next_level

    async def get_jobstream_graph(
        self,
        jobstream_id: str,
        force_refresh: bool = False,
    ) -> nx.DiGraph:
        """
        Build dependency graph for a job stream.

        Args:
            jobstream_id: Job stream ID
            force_refresh: Force cache refresh

        Returns:
            NetworkX DiGraph with job stream dependencies
        """
        cache_key = f"jobstream:{jobstream_id}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry.created_at < self.cache_ttl:
                return entry.graph

        graph = nx.DiGraph()

        if not self.tws_client:
            return graph

        try:
            # Get job stream details
            js_data = await self.tws_client.get_jobstream(jobstream_id)
            if js_data:
                graph.add_node(jobstream_id, type="jobstream", **js_data)

            # Get predecessors
            preds = await self.tws_client.get_current_plan_jobstream_predecessors(
                jobstream_id
            )
            if preds:
                for pred in preds:
                    pred_id = pred.get("id") or str(pred)
                    graph.add_node(pred_id, type="jobstream")
                    graph.add_edge(pred_id, jobstream_id, relation="DEPENDS_ON")

            # Get successors
            succs = await self.tws_client.get_current_plan_jobstream_successors(
                jobstream_id
            )
            if succs:
                for succ in succs:
                    succ_id = succ.get("id") or str(succ)
                    graph.add_node(succ_id, type="jobstream")
                    graph.add_edge(jobstream_id, succ_id, relation="DEPENDS_ON")

        except Exception as e:
            logger.warning(
                "jobstream_graph_build_failed",
                jobstream_id=jobstream_id,
                error=str(e),
            )

        # Cache result
        self._cache[cache_key] = GraphCacheEntry(
            graph=graph,
            created_at=time.time(),
            scope=f"jobstream:{jobstream_id}",
        )

        return graph

    # =========================================================================
    # GRAPH ANALYSIS
    # =========================================================================

    def find_critical_path(self, graph: nx.DiGraph) -> list[str]:
        """
        Find the critical path (longest path) in a DAG.

        Args:
            graph: NetworkX DiGraph

        Returns:
            List of node IDs in the critical path
        """
        if not graph or graph.number_of_nodes() == 0:
            return []

        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("graph_not_dag", nodes=graph.number_of_nodes())
            return []

        try:
            return nx.dag_longest_path(graph)
        except Exception as e:
            logger.error("critical_path_error", error=str(e))
            return []

    def get_impact_analysis(
        self,
        graph: nx.DiGraph,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Analyze impact if a job fails.

        Args:
            graph: NetworkX DiGraph
            job_id: Job ID to analyze

        Returns:
            Impact analysis dict with affected jobs and severity
        """
        if job_id not in graph:
            return {
                "job_id": job_id,
                "affected_jobs": [],
                "impact_count": 0,
                "severity": "unknown",
                "error": "Job not in graph",
            }

        try:
            # Get all descendants (jobs that depend on this one, directly or indirectly)
            affected = list(nx.descendants(graph, job_id))

            # Determine severity based on impact count
            impact_count = len(affected)
            if impact_count > 20:
                severity = "critical"
            elif impact_count > 10:
                severity = "high"
            elif impact_count > 3:
                severity = "medium"
            else:
                severity = "low"

            return {
                "job_id": job_id,
                "affected_jobs": affected,
                "impact_count": impact_count,
                "severity": severity,
                "direct_successors": list(graph.successors(job_id)),
                "direct_successor_count": graph.out_degree(job_id),
            }

        except Exception as e:
            logger.error("impact_analysis_error", job_id=job_id, error=str(e))
            return {
                "job_id": job_id,
                "affected_jobs": [],
                "impact_count": 0,
                "severity": "error",
                "error": str(e),
            }

    def get_critical_jobs(
        self,
        graph: nx.DiGraph,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find the most critical jobs using betweenness centrality.

        Jobs with high centrality are bottlenecks - if they fail,
        many other jobs are affected.

        Args:
            graph: NetworkX DiGraph
            top_n: Number of top critical jobs to return

        Returns:
            List of dicts with job_id and centrality score
        """
        if not graph or graph.number_of_nodes() == 0:
            return []

        try:
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(graph)

            # Sort by centrality (highest first)
            sorted_jobs = sorted(
                centrality.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Build result
            result = []
            for job_id, score in sorted_jobs[:top_n]:
                impact = self.get_impact_analysis(graph, job_id)
                result.append({
                    "job_id": job_id,
                    "centrality_score": round(score, 4),
                    "impact_count": impact["impact_count"],
                    "severity": impact["severity"],
                    "risk_level": "high" if score > 0.1 else "medium" if score > 0.01 else "low",
                })

            return result

        except Exception as e:
            logger.error("critical_jobs_error", error=str(e))
            return []

    def get_dependency_chain(
        self,
        graph: nx.DiGraph,
        job_id: str,
        direction: str = "both",
    ) -> dict[str, Any]:
        """
        Get the dependency chain for a job.

        Args:
            graph: NetworkX DiGraph
            job_id: Job ID
            direction: "predecessors", "successors", or "both"

        Returns:
            Dict with predecessors and/or successors
        """
        result = {"job_id": job_id}

        if job_id not in graph:
            result["error"] = "Job not in graph"
            return result

        if direction in ("predecessors", "both"):
            result["predecessors"] = list(nx.ancestors(graph, job_id))
            result["predecessor_count"] = len(result["predecessors"])

        if direction in ("successors", "both"):
            result["successors"] = list(nx.descendants(graph, job_id))
            result["successor_count"] = len(result["successors"])

        return result

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def clear_cache(self):
        """Clear the graph cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("graph_cache_cleared", entries_cleared=count)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid = sum(1 for e in self._cache.values() if now - e.created_at < self.cache_ttl)
        expired = len(self._cache) - valid

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "expired_entries": expired,
            "ttl_seconds": self.cache_ttl,
        }

    # =========================================================================
    # CONVENIENCE ASYNC METHODS (for code that expects async interface)
    # =========================================================================

    async def get_dependency_chain_async(
        self,
        job_id: str,
        max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get dependency chain for a job (async convenience method).

        Builds graph on-demand and returns the chain.
        """
        graph = await self.get_dependency_graph(job_id, depth=max_depth)
        chain_data = self.get_dependency_chain(graph, job_id)

        # Convert to list format expected by callers
        chain = []
        for pred in chain_data.get("predecessors", []):
            chain.append({"from": job_id, "to": pred, "type": "DEPENDS_ON"})

        return chain

    async def get_impact_analysis_async(self, job_id: str) -> dict[str, Any]:
        """
        Get impact analysis for a job (async convenience method).

        Builds graph on-demand and returns the analysis.
        """
        graph = await self.get_dependency_graph(job_id)
        return self.get_impact_analysis(graph, job_id)

    async def get_downstream_jobs(
        self,
        job_id: str,
        max_depth: int = 3,
    ) -> set[str]:
        """
        Get jobs that depend on this job.

        Returns set of job IDs that are downstream (successors).
        """
        graph = await self.get_dependency_graph(job_id, depth=max_depth)

        if job_id not in graph:
            return set()

        return set(nx.descendants(graph, job_id))

    def get_critical_jobs_async(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Get critical jobs (async convenience method).

        Note: Requires a pre-built graph. Returns empty if no graph available.
        """
        # Check if we have any cached graph
        if not self._cache:
            logger.warning("get_critical_jobs_no_graph")
            return []

        # Use first available cached graph
        for entry in self._cache.values():
            if time.time() - entry.created_at < self.cache_ttl:
                return self.get_critical_jobs(entry.graph, top_n)

        return []

    async def find_resource_conflicts(
        self,
        job1: str,
        job2: str,
        resource_map: dict[str, set[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find resource conflicts between two jobs using Common Neighbor Analysis.

        v5.2.3.26: Now uses advanced intersection queries.

        Args:
            job1: First job ID
            job2: Second job ID
            resource_map: Optional mapping of job -> resources

        Returns:
            List of conflicts found
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        # Build graphs for both jobs
        graph1 = await self.get_dependency_graph(job1)
        graph2 = await self.get_dependency_graph(job2)

        # Merge graphs
        merged = nx.compose(graph1, graph2)

        # Use advanced query service
        adv_service = get_advanced_query_service(merged)
        result = adv_service.check_resource_conflict(job1, job2, resource_map)

        # Convert to list format
        conflicts = []
        if result["conflict_risk"] != "none":
            conflicts.append({
                "type": "dependency_overlap",
                "jobs": [job1, job2],
                "risk": result["conflict_risk"],
                "common_dependencies": result["common_predecessors"],
                "explanation": result["explanation"],
            })

        return conflicts

    # =========================================================================
    # v5.2.3.26: ADVANCED GRAPH QUERIES
    # =========================================================================

    def get_job_status_at_time(
        self,
        job_id: str,
        at_time: datetime,
    ) -> dict[str, Any]:
        """
        Get job status at a specific point in time.

        Implements Temporal Graph query to resolve conflicting information.

        Args:
            job_id: Job identifier
            at_time: Point in time to query

        Returns:
            Job state at that time
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        adv_service = get_advanced_query_service()
        return adv_service.get_job_status_at(job_id, at_time)

    def record_job_state(
        self,
        job_id: str,
        state: dict[str, Any],
        timestamp: datetime | None = None,
    ):
        """
        Record a job state for temporal tracking.

        Call this when receiving job status updates from TWS API.

        Args:
            job_id: Job identifier
            state: State dict (status, return_code, etc.)
            timestamp: When this state was observed
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        adv_service = get_advanced_query_service()
        adv_service.temporal.record_state(job_id, state, timestamp, source="tws_api")

    def when_did_job_fail(
        self,
        job_id: str,
        since_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Find when a job started failing.

        Implements Temporal Graph query for state change detection.

        Args:
            job_id: Job identifier
            since_hours: How far back to look

        Returns:
            Information about first failure
        """
        from datetime import datetime, timedelta

        from resync.services.advanced_graph_queries import get_advanced_query_service

        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        adv_service = get_advanced_query_service()
        return adv_service.when_did_job_start_failing(job_id, since)

    async def find_safe_jobs_if_fails(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Find jobs that WON'T be affected if this job fails.

        Implements Negation Query using set difference.

        Args:
            job_id: Job that might fail

        Returns:
            Set of unaffected jobs
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        graph = await self.get_dependency_graph(job_id)
        adv_service = get_advanced_query_service(graph)
        return adv_service.find_safe_jobs(job_id)

    async def find_jobs_not_dependent_on(
        self,
        resource_or_job: str,
    ) -> dict[str, Any]:
        """
        Find jobs that do NOT depend on a resource/job.

        Implements Negation Query for exclusion searches.

        Args:
            resource_or_job: Entity to check non-dependence on

        Returns:
            Set of independent jobs
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        graph = await self.get_dependency_graph(resource_or_job)
        adv_service = get_advanced_query_service(graph)
        return adv_service.find_independent_jobs(resource_or_job)

    async def find_shared_bottlenecks(
        self,
        job_list: list[str],
    ) -> dict[str, Any]:
        """
        Find dependencies shared by multiple jobs.

        Implements Common Neighbor Intersection for bottleneck detection.

        Args:
            job_list: List of jobs to analyze

        Returns:
            Shared dependencies ranked by frequency
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        # Build combined graph for all jobs
        merged = nx.DiGraph()
        for job_id in job_list:
            job_graph = await self.get_dependency_graph(job_id)
            merged = nx.compose(merged, job_graph)

        adv_service = get_advanced_query_service(merged)
        return adv_service.find_shared_bottlenecks(job_list)

    def verify_dependency(
        self,
        source: str,
        target: str,
    ) -> dict[str, Any]:
        """
        Verify if a dependency relationship is explicit or inferred.

        Implements Edge Verification to prevent false link hallucination.

        Args:
            source: Source job
            target: Target job

        Returns:
            Verification result with confidence
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        adv_service = get_advanced_query_service()
        return adv_service.verify_dependency(source, target)

    def register_verified_dependency(
        self,
        source: str,
        target: str,
        evidence: list[str] | None = None,
    ):
        """
        Register a verified dependency from TWS API response.

        Call this when TWS API confirms a dependency exists.

        Args:
            source: Source job
            target: Target job
            evidence: Evidence supporting this dependency
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        adv_service = get_advanced_query_service()
        adv_service.register_verified_dependency(source, target, evidence)

    async def comprehensive_job_analysis(
        self,
        job_id: str,
        compare_with: str | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive analysis combining all 4 advanced techniques.

        Args:
            job_id: Job to analyze
            compare_with: Optional second job for interaction analysis

        Returns:
            Complete analysis using temporal, negation, intersection, and verification
        """
        from resync.services.advanced_graph_queries import get_advanced_query_service

        graph = await self.get_dependency_graph(job_id)
        if compare_with:
            graph2 = await self.get_dependency_graph(compare_with)
            graph = nx.compose(graph, graph2)

        adv_service = get_advanced_query_service(graph)
        return adv_service.comprehensive_job_analysis(job_id, compare_with)

    # =========================================================================
    # DEPRECATED WRITE METHODS (no-ops for compatibility)
    # =========================================================================

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str | None = None,
        **properties,
    ):
        """
        Add a node to the graph.

        DEPRECATED: Graph is built on-demand from TWS API.
        This method is a no-op for backward compatibility.
        """
        logger.debug(
            "add_node_deprecated",
            node_id=node_id,
            node_type=node_type,
        )

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str = "DEPENDS_ON",
        **properties,
    ):
        """
        Add an edge to the graph.

        DEPRECATED: Graph is built on-demand from TWS API.
        This method is a no-op for backward compatibility.
        """
        logger.debug(
            "add_edge_deprecated",
            source=source,
            target=target,
            relation_type=relation_type,
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_graph_service: TwsGraphService | None = None


def get_graph_service(tws_client: Any = None) -> TwsGraphService:
    """Get or create the singleton TwsGraphService instance."""
    global _graph_service

    if _graph_service is None:
        _graph_service = TwsGraphService(tws_client=tws_client)
    elif tws_client is not None:
        _graph_service.set_tws_client(tws_client)

    return _graph_service


async def build_job_graph(job_id: str, tws_client: Any = None) -> nx.DiGraph:
    """Convenience function to build a job dependency graph."""
    service = get_graph_service(tws_client)
    return await service.get_dependency_graph(job_id)


def analyze_job_impact(graph: nx.DiGraph, job_id: str) -> dict[str, Any]:
    """Convenience function to analyze job impact."""
    service = get_graph_service()
    return service.get_impact_analysis(graph, job_id)
