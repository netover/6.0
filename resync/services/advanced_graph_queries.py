# pylint
# ruff: noqa: E501
"""
Advanced Knowledge Graph Queries for Resync v5.2.3.26

Implements 4 advanced GraphRAG techniques to solve common RAG failures:

1. TEMPORAL GRAPH - Handles conflicting/versioned information
   - Track job states over time
   - Answer "what was the state at time X?"
   - Resolve version conflicts by timestamp

2. NEGATION QUERIES - Set difference operations
   - Find jobs that do NOT match criteria
   - "Which jobs are NOT dependent on X?"
   - "Jobs that did NOT fail today"

3. COMMON NEIGHBOR INTERSECTION - Find shared dependencies
   - Detect resource conflicts between jobs
   - Find common predecessors/successors
   - Identify shared bottlenecks

4. EDGE VERIFICATION - Prevent false link hallucination
   - Verify explicit vs inferred relationships
   - Check relationship confidence scores
   - Filter co-occurrence from true dependencies

Based on: "Fixing 14 Complex RAG Failures with Knowledge Graphs"
https://medium.com/@fareedkhandev/7125a8837a17

Author: Resync Team
Version: 5.2.3.26
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, TypeVar

import networkx as nx

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
T = TypeVar("T")

class RelationConfidence(str, Enum):
    """Confidence level for a relationship."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    TEMPORAL = "temporal"

@dataclass
class TemporalState:
    """A snapshot of an entity's state at a point in time."""

    entity_id: str
    timestamp: datetime
    state: dict[str, Any]
    source: str = "api"

@dataclass
class VerifiedRelationship:
    """A relationship with verification metadata."""

    source: str
    target: str
    relation_type: str
    confidence: RelationConfidence
    evidence: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_verified: datetime | None = None

@dataclass
class IntersectionResult:
    """Result of a common neighbor analysis."""

    entity_a: str
    entity_b: str
    common_predecessors: set[str]
    common_successors: set[str]
    common_resources: set[str]
    conflict_risk: str
    explanation: str

@dataclass
class NegationResult:
    """Result of a negation/exclusion query."""

    query_description: str
    all_entities: set[str]
    excluded_entities: set[str]
    result_entities: set[str]
    exclusion_reason: str

class TemporalGraphManager:
    """
    Manages temporal states of entities for version conflict resolution.

    Solves the "Conflicting Information" RAG failure:
    - Tracks entity states over time
    - Resolves "what is current?" vs "what was it at time X?"
    - Handles policy/status changes correctly

    Example TWS Use Cases:
    - "What was JOB_X status 2 hours ago?"
    - "When did the job start failing?"
    - "Show status history for the last 24 hours"
    """

    def __init__(self, max_history_per_entity: int = 1000):
        """
        Initialize TemporalGraphManager.

        Args:
            max_history_per_entity: Maximum states to keep per entity (LRU)
        """
        self.max_history = max_history_per_entity
        self._history: dict[str, list[TemporalState]] = defaultdict(list)
        self._last_cleanup = time.time()

    def record_state(
        self,
        entity_id: str,
        state: dict[str, Any],
        timestamp: datetime | None = None,
        source: str = "api",
    ) -> TemporalState:
        """
        Record a new state for an entity.

        Args:
            entity_id: Entity identifier (e.g., job name)
            state: Current state dict (status, return_code, etc.)
            timestamp: When this state was observed (default: now)
            source: Source of this state information

        Returns:
            The created TemporalState
        """
        ts = timestamp or datetime.now(timezone.utc)
        temporal_state = TemporalState(
            entity_id=entity_id, timestamp=ts, state=state.copy(), source=source
        )
        history = self._history[entity_id]
        history.insert(0, temporal_state)
        if len(history) > self.max_history:
            self._history[entity_id] = history[: self.max_history]
        logger.debug(
            "temporal_state_recorded",
            extra={
                "entity_id": entity_id,
                "timestamp": ts.isoformat(),
                "history_size": len(self._history[entity_id]),
            },
        )
        return temporal_state

    def get_state_at(self, entity_id: str, at_time: datetime) -> TemporalState | None:
        """
        Get entity state at a specific point in time.

        Uses the most recent state that was recorded BEFORE at_time.

        Args:
            entity_id: Entity identifier
            at_time: The point in time to query

        Returns:
            TemporalState or None if no history exists
        """
        history = self._history.get(entity_id, [])
        for state in history:
            if state.timestamp <= at_time:
                logger.debug(
                    "temporal_lookup_hit",
                    extra={
                        "entity_id": entity_id,
                        "query_time": at_time.isoformat(),
                        "found_time": state.timestamp.isoformat(),
                    },
                )
                return state
        logger.debug(
            "temporal_lookup_miss",
            extra={"entity_id": entity_id, "query_time": at_time.isoformat()},
        )
        return None

    def get_current_state(self, entity_id: str) -> TemporalState | None:
        """Get the most recent state for an entity."""
        history = self._history.get(entity_id, [])
        return history[0] if history else None

    def get_state_history(
        self,
        entity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[TemporalState]:
        """
        Get state history for an entity within a time range.

        Args:
            entity_id: Entity identifier
            since: Start of time range (inclusive)
            until: End of time range (inclusive)
            limit: Maximum results to return

        Returns:
            List of TemporalState objects (newest first)
        """
        history = self._history.get(entity_id, [])
        result = []
        for state in history:
            if since and state.timestamp < since:
                break
            if until and state.timestamp > until:
                continue
            result.append(state)
            if len(result) >= limit:
                break
        return result

    def find_state_changes(
        self, entity_id: str, field: str, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Find when a specific field changed values.

        Useful for: "When did the job start failing?"

        Args:
            entity_id: Entity identifier
            field: Field name to track (e.g., "status")
            since: Only look at changes since this time

        Returns:
            List of change events with old_value, new_value, timestamp
        """
        history = self.get_state_history(entity_id, since=since, limit=500)
        if len(history) < 2:
            return []
        changes = []
        prev_value = None
        for state in reversed(history):
            current_value = state.state.get(field)
            if prev_value is not None and current_value != prev_value:
                changes.append(
                    {
                        "timestamp": state.timestamp,
                        "field": field,
                        "old_value": prev_value,
                        "new_value": current_value,
                        "source": state.source,
                    }
                )
            prev_value = current_value
        return changes

    def resolve_conflicting_states(
        self, entity_id: str, conflicting_values: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Resolve conflicting information by using the most recent state.

        This is the key solution to the "Conflicting Information" RAG failure.

        Args:
            entity_id: Entity identifier
            conflicting_values: List of potentially conflicting states

        Returns:
            The resolved, current truth
        """
        current = self.get_current_state(entity_id)
        if current:
            logger.info(
                "conflict_resolved_by_temporal",
                extra={
                    "entity_id": entity_id,
                    "resolved_timestamp": current.timestamp.isoformat(),
                },
            )
            return {
                "resolved_value": current.state,
                "resolution_method": "temporal_latest",
                "timestamp": current.timestamp,
                "confidence": "high",
            }
        dated_values = [
            v for v in conflicting_values if "timestamp" in v or "date" in v
        ]
        if dated_values:
            sorted_values = sorted(
                dated_values,
                key=lambda x: x.get("timestamp") or x.get("date"),
                reverse=True,
            )
            return {
                "resolved_value": sorted_values[0],
                "resolution_method": "timestamp_sort",
                "confidence": "medium",
            }
        return {
            "resolved_value": conflicting_values[0] if conflicting_values else None,
            "resolution_method": "first_available",
            "confidence": "low",
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get temporal graph statistics."""
        total_states = sum((len(h) for h in self._history.values()))
        return {
            "entities_tracked": len(self._history),
            "total_states": total_states,
            "avg_states_per_entity": total_states / max(len(self._history), 1),
            "max_history_per_entity": self.max_history,
        }

class NegationQueryEngine:
    """
    Handles negation and exclusion queries using set operations.

    Solves the "Negation Blindness" RAG failure:
    - Vector search finds docs WITH a term, not WITHOUT
    - This engine uses set difference for exclusion queries

    Example TWS Use Cases:
    - "Jobs that are NOT dependent on RESOURCE_X"
    - "Jobs that did NOT fail today"
    - "Workstations NOT affected by the outage"
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        """
        Initialize NegationQueryEngine.

        Args:
            graph: NetworkX graph to query
        """
        self._graph = graph

    def set_graph(self, graph: nx.DiGraph):
        """Update the graph reference."""
        self._graph = graph

    def find_jobs_not_dependent_on(
        self, resource_or_job: str, job_universe: set[str] | None = None
    ) -> NegationResult:
        """
        Find all jobs that do NOT depend on a given resource/job.

        Args:
            resource_or_job: The entity to check non-dependence on
            job_universe: Set of all jobs to consider (default: all in graph)

        Returns:
            NegationResult with jobs not dependent on the entity
        """
        if not self._graph:
            return NegationResult(
                query_description=f"Jobs NOT dependent on {resource_or_job}",
                all_entities=set(),
                excluded_entities=set(),
                result_entities=set(),
                exclusion_reason="No graph available",
            )
        all_jobs = job_universe or set(self._graph.nodes())
        dependent_jobs = set()
        if resource_or_job in self._graph:
            dependent_jobs = set(nx.descendants(self._graph, resource_or_job))
            dependent_jobs.add(resource_or_job)
        not_dependent = all_jobs - dependent_jobs
        logger.info(
            "negation_query_executed",
            extra={
                "target": resource_or_job,
                "total_jobs": len(all_jobs),
                "dependent": len(dependent_jobs),
                "not_dependent": len(not_dependent),
            },
        )
        return NegationResult(
            query_description=f"Jobs NOT dependent on {resource_or_job}",
            all_entities=all_jobs,
            excluded_entities=dependent_jobs,
            result_entities=not_dependent,
            exclusion_reason=f"Excluded {len(dependent_jobs)} jobs that depend on {resource_or_job}",
        )

    def find_jobs_not_in_status(
        self, excluded_statuses: list[str], job_status_map: dict[str, str]
    ) -> NegationResult:
        """
        Find jobs NOT in specified statuses.

        Args:
            excluded_statuses: Statuses to exclude (e.g., ["ABEND", "STUCK"])
            job_status_map: Mapping of job_id -> current status

        Returns:
            NegationResult with jobs not in excluded statuses
        """
        all_jobs = set(job_status_map.keys())
        excluded_jobs = {
            job_id
            for job_id, status in job_status_map.items()
            if status in excluded_statuses
        }
        result_jobs = all_jobs - excluded_jobs
        return NegationResult(
            query_description=f"Jobs NOT in status {excluded_statuses}",
            all_entities=all_jobs,
            excluded_entities=excluded_jobs,
            result_entities=result_jobs,
            exclusion_reason=f"Excluded {len(excluded_jobs)} jobs with status in {excluded_statuses}",
        )

    def find_jobs_not_affected_by(self, failed_job: str) -> NegationResult:
        """
        Find jobs that would NOT be affected if a job fails.

        Useful for impact isolation analysis.

        Args:
            failed_job: The job that failed

        Returns:
            NegationResult with unaffected jobs
        """
        if not self._graph or failed_job not in self._graph:
            return NegationResult(
                query_description=f"Jobs NOT affected by {failed_job} failure",
                all_entities=set(self._graph.nodes()) if self._graph else set(),
                excluded_entities=set(),
                result_entities=set(self._graph.nodes()) if self._graph else set(),
                exclusion_reason=f"Job {failed_job} not in graph",
            )
        all_jobs = set(self._graph.nodes())
        affected_jobs = set(nx.descendants(self._graph, failed_job))
        affected_jobs.add(failed_job)
        unaffected = all_jobs - affected_jobs
        return NegationResult(
            query_description=f"Jobs NOT affected by {failed_job} failure",
            all_entities=all_jobs,
            excluded_entities=affected_jobs,
            result_entities=unaffected,
            exclusion_reason=f"{failed_job} failure would affect {len(affected_jobs)} jobs",
        )

    def find_entities_without_property(
        self,
        property_name: str,
        property_value: Any,
        entity_properties: dict[str, dict[str, Any]],
    ) -> NegationResult:
        """
        Find entities that do NOT have a specific property value.

        Generic negation for any property check.

        Args:
            property_name: Property to check
            property_value: Value to exclude
            entity_properties: Mapping of entity_id -> properties dict

        Returns:
            NegationResult
        """
        all_entities = set(entity_properties.keys())
        entities_with_property = {
            entity_id
            for entity_id, props in entity_properties.items()
            if props.get(property_name) == property_value
        }
        result = all_entities - entities_with_property
        return NegationResult(
            query_description=f"Entities where {property_name} != {property_value}",
            all_entities=all_entities,
            excluded_entities=entities_with_property,
            result_entities=result,
            exclusion_reason=f"Excluded {len(entities_with_property)} entities with {property_name}={property_value}",
        )

class CommonNeighborAnalyzer:
    """
    Finds common neighbors/dependencies between entities.

    Solves the "Common Neighbor Intersection Gap" RAG failure:
    - Detects when two entities share a common dependency
    - Identifies resource conflicts
    - Finds hidden interactions through shared neighbors

    Example TWS Use Cases:
    - "Do JOB_A and JOB_B share any resources?"
    - "What jobs depend on both RESOURCE_X and RESOURCE_Y?"
    - "Find potential conflicts between two job streams"
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        """
        Initialize CommonNeighborAnalyzer.

        Args:
            graph: NetworkX graph to analyze
        """
        self._graph = graph

    def set_graph(self, graph: nx.DiGraph):
        """Update the graph reference."""
        self._graph = graph

    def find_common_predecessors(self, entity_a: str, entity_b: str) -> set[str]:
        """
        Find entities that both A and B depend on.

        Args:
            entity_a: First entity
            entity_b: Second entity

        Returns:
            Set of common predecessors
        """
        if not self._graph:
            return set()
        preds_a = (
            set(nx.ancestors(self._graph, entity_a))
            if entity_a in self._graph
            else set()
        )
        preds_b = (
            set(nx.ancestors(self._graph, entity_b))
            if entity_b in self._graph
            else set()
        )
        return preds_a.intersection(preds_b)

    def find_common_successors(self, entity_a: str, entity_b: str) -> set[str]:
        """
        Find entities that depend on both A and B.

        Args:
            entity_a: First entity
            entity_b: Second entity

        Returns:
            Set of common successors
        """
        if not self._graph:
            return set()
        succs_a = (
            set(nx.descendants(self._graph, entity_a))
            if entity_a in self._graph
            else set()
        )
        succs_b = (
            set(nx.descendants(self._graph, entity_b))
            if entity_b in self._graph
            else set()
        )
        return succs_a.intersection(succs_b)

    def find_common_direct_neighbors(self, entity_a: str, entity_b: str) -> set[str]:
        """
        Find immediate (1-hop) common neighbors.

        Args:
            entity_a: First entity
            entity_b: Second entity

        Returns:
            Set of common direct neighbors
        """
        if not self._graph:
            return set()
        neighbors_a = set()
        neighbors_b = set()
        if entity_a in self._graph:
            neighbors_a = set(self._graph.predecessors(entity_a)) | set(
                self._graph.successors(entity_a)
            )
        if entity_b in self._graph:
            neighbors_b = set(self._graph.predecessors(entity_b)) | set(
                self._graph.successors(entity_b)
            )
        return neighbors_a.intersection(neighbors_b)

    def analyze_interaction(
        self,
        entity_a: str,
        entity_b: str,
        resource_edges: dict[str, set[str]] | None = None,
    ) -> IntersectionResult:
        """
        Full interaction analysis between two entities.

        This is the main method that combines all intersection checks
        to detect potential conflicts or hidden interactions.

        Args:
            entity_a: First entity (e.g., JOB_A)
            entity_b: Second entity (e.g., JOB_B)
            resource_edges: Optional mapping of entity -> resources it uses

        Returns:
            IntersectionResult with full analysis
        """
        common_preds = self.find_common_predecessors(entity_a, entity_b)
        common_succs = self.find_common_successors(entity_a, entity_b)
        common_resources = set()
        if resource_edges:
            resources_a = resource_edges.get(entity_a, set())
            resources_b = resource_edges.get(entity_b, set())
            common_resources = resources_a.intersection(resources_b)
        total_overlap = len(common_preds) + len(common_succs) + len(common_resources)  # noqa: F841
        if common_resources:
            risk = "high"
            explanation = f"RESOURCE CONFLICT: {entity_a} and {entity_b} share {len(common_resources)} resources: {common_resources}. Running simultaneously may cause contention."
        elif common_preds and common_succs:
            risk = "medium"
            explanation = f"DEPENDENCY OVERLAP: Both depend on {len(common_preds)} common jobs and are depended on by {len(common_succs)} common jobs. Scheduling conflicts possible."
        elif common_preds or common_succs:
            risk = "low"
            explanation = f"MINOR OVERLAP: Share {len(common_preds)} predecessors and {len(common_succs)} successors. Limited interaction expected."
        else:
            risk = "none"
            explanation = f"NO INTERACTION: {entity_a} and {entity_b} have no common dependencies or resources. They are independent."
        logger.info(
            "intersection_analysis",
            extra={
                "entity_a": entity_a,
                "entity_b": entity_b,
                "common_predecessors": len(common_preds),
                "common_successors": len(common_succs),
                "common_resources": len(common_resources),
                "risk": risk,
            },
        )
        return IntersectionResult(
            entity_a=entity_a,
            entity_b=entity_b,
            common_predecessors=common_preds,
            common_successors=common_succs,
            common_resources=common_resources,
            conflict_risk=risk,
            explanation=explanation,
        )

    def find_bottleneck_dependencies(self, job_list: list[str]) -> dict[str, int]:
        """
        Find dependencies that appear across multiple jobs.

        Identifies shared bottlenecks in a set of jobs.

        Args:
            job_list: List of job IDs to analyze

        Returns:
            Dict of dependency -> count of jobs using it
        """
        if not self._graph:
            return {}
        dependency_count: dict[str, int] = defaultdict(int)
        for job in job_list:
            if job in self._graph:
                for pred in nx.ancestors(self._graph, job):
                    dependency_count[pred] += 1
        return {dep: count for dep, count in dependency_count.items() if count > 1}

class EdgeVerificationEngine:
    """
    Verifies relationships to prevent false link hallucination.

    Solves the "Inventing False Links" RAG failure:
    - Distinguishes explicit relationships from co-occurrence
    - Tracks evidence for each relationship
    - Provides confidence scores

    Example TWS Use Cases:
    - "Is JOB_A actually dependent on JOB_B, or just mentioned together?"
    - "What evidence supports this dependency?"
    - "Filter to only high-confidence relationships"
    """

    def __init__(self):
        """Initialize EdgeVerificationEngine."""
        self._verified_edges: dict[tuple[str, str, str], VerifiedRelationship] = {}
        self._co_occurrences: dict[tuple[str, str], int] = defaultdict(int)

    def register_explicit_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        evidence: list[str] | None = None,
    ) -> VerifiedRelationship:
        """
        Register an explicitly stated relationship.

        Args:
            source: Source entity
            target: Target entity
            relation_type: Type of relationship (e.g., "DEPENDS_ON")
            evidence: Evidence supporting this relationship

        Returns:
            VerifiedRelationship object
        """
        key = (source, target, relation_type)
        now = datetime.now(timezone.utc)
        if key in self._verified_edges:
            rel = self._verified_edges[key]
            rel.last_verified = now
            if evidence:
                rel.evidence.extend(evidence)
        else:
            rel = VerifiedRelationship(
                source=source,
                target=target,
                relation_type=relation_type,
                confidence=RelationConfidence.EXPLICIT,
                evidence=evidence or [],
                first_seen=now,
                last_verified=now,
            )
            self._verified_edges[key] = rel
        logger.debug(
            "explicit_edge_registered",
            extra={"source": source, "target": target, "relation": relation_type},
        )
        return rel

    def register_co_occurrence(
        self, entity_a: str, entity_b: str, context: str | None = None
    ):
        """
        Register a co-occurrence (entities mentioned together).

        This is NOT treated as a relationship, but tracked to detect
        potential false link attempts.

        Args:
            entity_a: First entity
            entity_b: Second entity
            context: Context where they co-occurred
        """
        key = tuple(sorted([entity_a, entity_b]))
        self._co_occurrences[key] += 1
        logger.debug(
            "co_occurrence_registered",
            extra={"entities": key, "count": self._co_occurrences[key]},
        )

    def verify_relationship(
        self, source: str, target: str, relation_type: str = "DEPENDS_ON"
    ) -> dict[str, Any]:
        """
        Verify if a relationship is explicit or just inferred from co-occurrence.

        This is the key method to prevent false link hallucination.

        Args:
            source: Source entity
            target: Target entity
            relation_type: Type of relationship to check

        Returns:
            Verification result with confidence and evidence
        """
        key = (source, target, relation_type)
        if key in self._verified_edges:
            rel = self._verified_edges[key]
            return {
                "verified": True,
                "confidence": RelationConfidence.EXPLICIT.value,
                "evidence": rel.evidence,
                "first_seen": rel.first_seen,
                "last_verified": rel.last_verified,
                "message": f"VERIFIED: {source} {relation_type} {target} is explicitly stated.",
            }
        reverse_key = (target, source, relation_type)
        if reverse_key in self._verified_edges:
            return {
                "verified": False,
                "confidence": "none",
                "evidence": [],
                "message": f"DIRECTION MISMATCH: Found {target} {relation_type} {source}, not the reverse.",
            }
        co_key = tuple(sorted([source, target]))
        if co_key in self._co_occurrences:
            count = self._co_occurrences[co_key]
            return {
                "verified": False,
                "confidence": "co_occurrence",
                "co_occurrence_count": count,
                "evidence": [],
                "message": f"NOT VERIFIED: {source} and {target} appear together {count} times, but no explicit {relation_type} relationship found. This may be a FALSE LINK.",
            }
        return {
            "verified": False,
            "confidence": "unknown",
            "evidence": [],
            "message": f"UNKNOWN: No information about relationship between {source} and {target}.",
        }

    def get_verified_relationships(
        self, entity: str, direction: str = "both"
    ) -> list[VerifiedRelationship]:
        """
        Get all verified relationships for an entity.

        Args:
            entity: Entity to query
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of verified relationships
        """
        results = []
        for key, rel in self._verified_edges.items():
            source, target, _ = key
            if (
                direction in ("outgoing", "both")
                and source == entity
                or (direction in ("incoming", "both") and target == entity)
            ):
                results.append(rel)
        return results

    def filter_graph_by_confidence(
        self,
        graph: nx.DiGraph,
        min_confidence: RelationConfidence = RelationConfidence.EXPLICIT,
    ) -> nx.DiGraph:
        """
        Create a filtered graph with only high-confidence edges.

        Args:
            graph: Original graph
            min_confidence: Minimum confidence level to include

        Returns:
            Filtered graph
        """
        filtered = nx.DiGraph()
        for source, target, data in graph.edges(data=True):
            rel_type = data.get("relation", "DEPENDS_ON")
            key = (source, target, rel_type)
            if key in self._verified_edges:
                rel = self._verified_edges[key]
                if (
                    rel.confidence == min_confidence
                    or rel.confidence == RelationConfidence.EXPLICIT
                ):
                    filtered.add_edge(source, target, **data)
            elif min_confidence == RelationConfidence.INFERRED:
                filtered.add_edge(source, target, **data)
        logger.info(
            "graph_filtered_by_confidence",
            extra={
                "original_edges": graph.number_of_edges(),
                "filtered_edges": filtered.number_of_edges(),
                "min_confidence": min_confidence.value,
            },
        )
        return filtered

    def get_statistics(self) -> dict[str, Any]:
        """Get verification statistics."""
        return {
            "verified_edges": len(self._verified_edges),
            "co_occurrences_tracked": len(self._co_occurrences),
            "by_confidence": {
                conf.value: sum(
                    (
                        1
                        for rel in self._verified_edges.values()
                        if rel.confidence == conf
                    )
                )
                for conf in RelationConfidence
            },
        }

class AdvancedGraphQueryService:
    """
    Unified service combining all 4 advanced query techniques.

    Provides a single interface for:
    1. Temporal queries (version conflicts)
    2. Negation queries (set difference)
    3. Intersection queries (common neighbors)
    4. Edge verification (false link prevention)
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        """
        Initialize AdvancedGraphQueryService.

        Args:
            graph: Initial NetworkX graph
        """
        self.temporal = TemporalGraphManager()
        self.negation = NegationQueryEngine(graph)
        self.intersection = CommonNeighborAnalyzer(graph)
        self.verification = EdgeVerificationEngine()
        self._graph = graph
        logger.info("advanced_graph_query_service_initialized")

    def set_graph(self, graph: nx.DiGraph):
        """Update the graph for all engines."""
        self._graph = graph
        self.negation.set_graph(graph)
        self.intersection.set_graph(graph)

    def get_job_status_at(self, job_id: str, at_time: datetime) -> dict[str, Any]:
        """Get job status at a specific time."""
        state = self.temporal.get_state_at(job_id, at_time)
        if state:
            return {
                "job_id": job_id,
                "query_time": at_time.isoformat(),
                "status": state.state.get("status"),
                "full_state": state.state,
                "recorded_at": state.timestamp.isoformat(),
                "source": state.source,
            }
        return {
            "job_id": job_id,
            "query_time": at_time.isoformat(),
            "error": "No historical data available",
        }

    def when_did_job_start_failing(
        self, job_id: str, since: datetime | None = None
    ) -> dict[str, Any]:
        """Find when a job transitioned to failure status."""
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
        changes = self.temporal.find_state_changes(job_id, "status", since)
        failure_statuses = {"ABEND", "STUCK", "CANCEL", "ERROR"}
        for change in changes:
            if change["new_value"] in failure_statuses:
                return {
                    "job_id": job_id,
                    "first_failure": change["timestamp"].isoformat(),
                    "previous_status": change["old_value"],
                    "failure_status": change["new_value"],
                    "source": change["source"],
                }
        return {
            "job_id": job_id,
            "message": "No failure transition found in the specified time range",
        }

    def find_safe_jobs(self, failed_job: str) -> dict[str, Any]:
        """Find jobs that won't be affected by a failure."""
        result = self.negation.find_jobs_not_affected_by(failed_job)
        return {
            "query": result.query_description,
            "safe_jobs": list(result.result_entities),
            "safe_count": len(result.result_entities),
            "affected_count": len(result.excluded_entities),
            "explanation": result.exclusion_reason,
        }

    def find_independent_jobs(self, resource_or_job: str) -> dict[str, Any]:
        """Find jobs not dependent on a resource."""
        result = self.negation.find_jobs_not_dependent_on(resource_or_job)
        return {
            "query": result.query_description,
            "independent_jobs": list(result.result_entities),
            "count": len(result.result_entities),
            "explanation": result.exclusion_reason,
        }

    def check_resource_conflict(
        self, job_a: str, job_b: str, resource_map: dict[str, set[str]] | None = None
    ) -> dict[str, Any]:
        """Check for resource conflicts between two jobs."""
        result = self.intersection.analyze_interaction(job_a, job_b, resource_map)
        return {
            "job_a": result.entity_a,
            "job_b": result.entity_b,
            "conflict_risk": result.conflict_risk,
            "common_predecessors": list(result.common_predecessors),
            "common_successors": list(result.common_successors),
            "common_resources": list(result.common_resources),
            "explanation": result.explanation,
        }

    def find_shared_bottlenecks(self, job_list: list[str]) -> dict[str, Any]:
        """Find dependencies shared by multiple jobs."""
        bottlenecks = self.intersection.find_bottleneck_dependencies(job_list)
        sorted_bottlenecks = sorted(
            bottlenecks.items(), key=lambda x: x[1], reverse=True
        )
        return {
            "analyzed_jobs": len(job_list),
            "bottlenecks": [
                {"dependency": dep, "used_by_jobs": count}
                for dep, count in sorted_bottlenecks[:10]
            ],
            "total_shared_dependencies": len(bottlenecks),
        }

    def verify_dependency(self, source: str, target: str) -> dict[str, Any]:
        """Verify if a dependency relationship is real."""
        return self.verification.verify_relationship(source, target, "DEPENDS_ON")

    def register_verified_dependency(
        self, source: str, target: str, evidence: list[str] | None = None
    ):
        """Register a verified dependency from TWS API."""
        self.verification.register_explicit_edge(source, target, "DEPENDS_ON", evidence)

    def comprehensive_job_analysis(
        self, job_id: str, compare_with: str | None = None
    ) -> dict[str, Any]:
        """
        Comprehensive analysis combining all techniques.

        Args:
            job_id: Job to analyze
            compare_with: Optional second job for interaction analysis
        """
        result = {"job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        current_state = self.temporal.get_current_state(job_id)
        if current_state:
            result["current_state"] = current_state.state
            result["state_timestamp"] = current_state.timestamp.isoformat()
        safe_result = self.negation.find_jobs_not_affected_by(job_id)
        result["safe_jobs_if_fails"] = len(safe_result.result_entities)
        result["affected_jobs_if_fails"] = len(safe_result.excluded_entities)
        verified_rels = self.verification.get_verified_relationships(job_id)
        result["verified_dependencies"] = len(verified_rels)
        if compare_with:
            interaction = self.intersection.analyze_interaction(job_id, compare_with)
            result["interaction_with"] = {
                "job": compare_with,
                "conflict_risk": interaction.conflict_risk,
                "common_dependencies": len(interaction.common_predecessors),
                "explanation": interaction.explanation,
            }
        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from all engines."""
        return {
            "temporal": self.temporal.get_statistics(),
            "verification": self.verification.get_statistics(),
            "graph_nodes": self._graph.number_of_nodes() if self._graph else 0,
            "graph_edges": self._graph.number_of_edges() if self._graph else 0,
        }

_advanced_query_service: AdvancedGraphQueryService | None = None

def get_advanced_query_service(
    graph: nx.DiGraph | None = None,
) -> AdvancedGraphQueryService:
    """
    Get singleton AdvancedGraphQueryService instance.

    Args:
        graph: Optional graph to initialize with

    Returns:
        AdvancedGraphQueryService instance
    """
    global _advanced_query_service
    if _advanced_query_service is None:
        _advanced_query_service = AdvancedGraphQueryService(graph)
    elif graph:
        _advanced_query_service.set_graph(graph)
    return _advanced_query_service
