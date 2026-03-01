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
import threading
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from typing import Any

import networkx as nx

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

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
    Manages temporal versions of entity states.

    Notes on concurrency:
    - FastAPI can run with multiple worker threads (and future free-threaded CPython).
    - This manager keeps in-memory mutable state and therefore must be synchronized.
    - History is stored newest-first using deque.appendleft() with maxlen eviction.
    """

    def __init__(self, max_history_per_entity: int = 1000) -> None:
        self.max_history = max_history_per_entity
        # Newest-first per entity; auto-evicts oldest entries.
        self._history: dict[str, deque[TemporalState]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._lock = threading.Lock()

    def record_state(
        self,
        entity_id: str,
        state: dict[str, Any],
        timestamp: datetime | None = None,
        source: str = "api",
    ) -> TemporalState:
        """Record a state snapshot for an entity."""
        ts = timestamp or datetime.now(timezone.utc)
        temporal_state = TemporalState(
            entity_id=entity_id,
            timestamp=ts,
            state=state.copy(),
            source=source,
        )
        with self._lock:
            self._history[entity_id].appendleft(temporal_state)
            history_size = len(self._history[entity_id])
        logger.debug(
            "temporal_state_recorded",
            entity_id=entity_id,
            timestamp=ts.isoformat(),
            history_size=history_size,
        )
        return temporal_state

    def get_state_at(self, entity_id: str, at_time: datetime) -> TemporalState | None:
        """Get the state at (or immediately before) a given time."""
        with self._lock:
            history = list(self._history.get(entity_id, deque()))
        # history is newest-first by insertion; timestamps can be out-of-order, so scan all.
        best: TemporalState | None = None
        for state in history:
            if state.timestamp <= at_time and (best is None or state.timestamp > best.timestamp):
                best = state
        return best

    def get_current_state(self, entity_id: str) -> TemporalState | None:
        """Get the latest recorded state."""
        with self._lock:
            history = self._history.get(entity_id)
            if not history:
                return None
            return history[0]

    def get_state_history(
        self,
        entity_id: str,
        limit: int = 10,
        since: datetime | None = None,
    ) -> list[TemporalState]:
        """Get recent state history for an entity."""
        with self._lock:
            history = list(self._history.get(entity_id, deque()))
        results: list[TemporalState] = []
        for state in history:
            if since is not None and state.timestamp < since:
                # Do not break: timestamps may be out-of-order.
                continue
            results.append(state)
            if len(results) >= limit:
                break
        return results

    def find_state_changes(
        self,
        entity_id: str,
        field: str,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Find when a specific field changed over time."""
        history = self.get_state_history(entity_id, limit=self.max_history, since=since)
        changes: list[dict[str, Any]] = []
        prev_value: Any = None
        for state in reversed(history):  # oldest -> newest for change detection
            current_value = state.state.get(field)
            if prev_value is not None and current_value != prev_value:
                changes.append(
                    {
                        "timestamp": state.timestamp.isoformat(),
                        "old_value": prev_value,
                        "new_value": current_value,
                        "source": state.source,
                    }
                )
            prev_value = current_value
        return changes

    @staticmethod
    def _extract_ts(value: dict[str, Any]) -> datetime | None:
        raw = value.get("timestamp") or value.get("date")
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        return None

    def resolve_conflicting_states(
        self,
        conflicting_states: list[dict[str, Any]],
        strategy: str = "latest",
    ) -> dict[str, Any]:
        """Resolve conflicting state values using a strategy."""
        if not conflicting_states:
            return {}
        if len(conflicting_states) == 1:
            return conflicting_states[0]

        if strategy == "latest":
            dated_values = [v for v in conflicting_states if self._extract_ts(v) is not None]
            if not dated_values:
                return conflicting_states[-1]
            dated_values.sort(key=self._extract_ts)  # type: ignore[arg-type]
            return dated_values[-1]

        if strategy == "merge":
            merged: dict[str, Any] = {}
            # Apply oldest -> newest (best-effort) for determinism
            ordered = sorted(
                conflicting_states,
                key=lambda v: (self._extract_ts(v) is None, self._extract_ts(v) or datetime.min.replace(tzinfo=timezone.utc)),
            )
            for state in ordered:
                merged.update(state)
            return merged

        # Default fallback
        return conflicting_states[-1]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored temporal data."""
        with self._lock:
            entity_count = len(self._history)
            total_states = sum(len(h) for h in self._history.values())
        return {
            "entities_tracked": entity_count,
            "total_states": total_states,
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

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        """
        Initialize NegationQueryEngine.

        Args:
            graph: NetworkX graph to query
        """
        self._graph = graph

    def set_graph(self, graph: nx.DiGraph) -> None:
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

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        """
        Initialize CommonNeighborAnalyzer.

        Args:
            graph: NetworkX graph to analyze
        """
        self._graph = graph

    def set_graph(self, graph: nx.DiGraph) -> None:
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
    Verifies graph edges to reduce false-link hallucination.

    Concurrency:
    - This engine maintains shared mutable in-memory structures.
    - Protect all read-modify-write operations with a lock.
    - Bound internal caches to prevent unbounded memory growth (DoS/OOM).
    """

    def __init__(
        self,
        max_verified_edges: int = 100_000,
        max_co_occurrence_keys: int = 100_000,
        max_co_occurrences_per_key: int = 500,
        max_evidence_per_edge: int = 50,
    ):
        self._lock = threading.Lock()
        self._max_verified_edges = max_verified_edges
        self._max_co_occurrence_keys = max_co_occurrence_keys
        self._max_co_occurrences_per_key = max_co_occurrences_per_key
        self._max_evidence_per_edge = max_evidence_per_edge

        # Use OrderedDict to support FIFO eviction.
        self._verified_edges: OrderedDict[tuple[str, str, str], VerifiedRelationship] = OrderedDict()
        self._co_occurrences: OrderedDict[str, set[str]] = OrderedDict()

    def _evict_verified_edges_if_needed(self) -> None:
        while len(self._verified_edges) > self._max_verified_edges:
            self._verified_edges.popitem(last=False)

    def _evict_co_occurrences_if_needed(self) -> None:
        while len(self._co_occurrences) > self._max_co_occurrence_keys:
            self._co_occurrences.popitem(last=False)

    def register_explicit_edge(
        self,
        source: str,
        target: str,
        relation_type: str = "DEPENDS_ON",
        evidence: list[str] | None = None,
        confidence: RelationConfidence = RelationConfidence.EXPLICIT,
    ) -> VerifiedRelationship:
        """Register a verified explicit relationship."""
        key = (source, target, relation_type)
        ev = (evidence or [])[: self._max_evidence_per_edge]
        with self._lock:
            if key in self._verified_edges:
                rel = self._verified_edges[key]
                # Preserve existing evidence + append new items (bounded).
                rel.evidence.extend(ev)
                if len(rel.evidence) > self._max_evidence_per_edge:
                    rel.evidence[:] = rel.evidence[-self._max_evidence_per_edge :]
                rel.confidence = confidence
                rel.timestamp = datetime.now(timezone.utc)
                self._verified_edges.move_to_end(key)
            else:
                rel = VerifiedRelationship(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    confidence=confidence,
                    evidence=list(ev),
                    timestamp=datetime.now(timezone.utc),
                )
                self._verified_edges[key] = rel
                self._evict_verified_edges_if_needed()
        logger.info(
            "explicit_edge_registered",
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=confidence.value,
        )
        return rel

    def register_co_occurrence(
        self,
        entity_a: str,
        entity_b: str,
        context: str | None = None,
    ) -> None:
        """Register that two entities co-occurred in the same context."""
        if entity_a == entity_b:
            return
        with self._lock:
            s = self._co_occurrences.get(entity_a)
            if s is None:
                s = set()
                self._co_occurrences[entity_a] = s
            s.add(entity_b)
            # Bound set size to avoid unbounded growth per key.
            if len(s) > self._max_co_occurrences_per_key:
                # Drop arbitrary extra elements (set is unordered).
                for _ in range(len(s) - self._max_co_occurrences_per_key):
                    s.pop()
            self._co_occurrences.move_to_end(entity_a)
            self._evict_co_occurrences_if_needed()
        logger.debug(
            "co_occurrence_registered",
            entity_a=entity_a,
            entity_b=entity_b,
            context=context,
        )

    def verify_relationship(
        self,
        source: str,
        target: str,
        relation_type: str = "DEPENDS_ON",
        graph: nx.DiGraph | None = None,
    ) -> VerifiedRelationship | None:
        """Verify if relationship is explicit, inferred, or temporal."""
        key = (source, target, relation_type)
        with self._lock:
            rel = self._verified_edges.get(key)
            if rel is not None:
                return rel

            # Heuristic: co-occurrence suggests inferred relationship but is NOT verified.
            co = self._co_occurrences.get(source, set())
            inferred = target in co

        if inferred:
            return VerifiedRelationship(
                source=source,
                target=target,
                relation_type=relation_type,
                confidence=RelationConfidence.INFERRED,
                evidence=["co_occurrence"],
                timestamp=datetime.now(timezone.utc),
            )

        if graph is not None and graph.has_edge(source, target):
            return VerifiedRelationship(
                source=source,
                target=target,
                relation_type=relation_type,
                confidence=RelationConfidence.TEMPORAL,
                evidence=["graph_edge_present"],
                timestamp=datetime.now(timezone.utc),
            )
        return None

    def get_verified_relationships(
        self,
        entity: str,
        direction: str = "both",
    ) -> list[VerifiedRelationship]:
        """Get all verified relationships for an entity."""
        if direction not in ("incoming", "outgoing", "both"):
            raise ValueError(f"Invalid direction: {direction}")
        results: list[VerifiedRelationship] = []
        with self._lock:
            items = list(self._verified_edges.values())
        for rel in items:
            source = rel.source
            target = rel.target
            if (direction in ("outgoing", "both") and source == entity) or (
                direction in ("incoming", "both") and target == entity
            ):
                results.append(rel)
        return results

    _CONFIDENCE_RANK = {
        RelationConfidence.TEMPORAL: 0,
        RelationConfidence.INFERRED: 1,
        RelationConfidence.EXPLICIT: 2,
    }

    def filter_graph_by_confidence(
        self,
        graph: nx.DiGraph,
        min_confidence: RelationConfidence = RelationConfidence.EXPLICIT,
    ) -> nx.DiGraph:
        """Return a subgraph containing only edges that meet the confidence threshold."""
        filtered = nx.DiGraph()
        min_rank = self._CONFIDENCE_RANK.get(min_confidence, 0)
        with self._lock:
            verified = dict(self._verified_edges)  # snapshot
        for source, target, data in graph.edges(data=True):
            rel_type = data.get("relation", "DEPENDS_ON")
            key = (source, target, rel_type)
            rel = verified.get(key)
            if rel is None:
                # Only include unverified edges when caller explicitly allows inferred/temporal.
                if min_confidence == RelationConfidence.EXPLICIT:
                    continue
                # Treat unverified edges as lowest confidence.
                edge_rank = self._CONFIDENCE_RANK[RelationConfidence.TEMPORAL]
            else:
                edge_rank = self._CONFIDENCE_RANK.get(rel.confidence, -1)
            if edge_rank >= min_rank:
                filtered.add_edge(source, target, **data)
        return filtered

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            verified_edges = len(self._verified_edges)
            co_occurrence_keys = len(self._co_occurrences)
            total_co_occurrences = sum(len(v) for v in self._co_occurrences.values())
        return {
            "verified_edges": verified_edges,
            "co_occurrence_keys": co_occurrence_keys,
            "total_co_occurrences": total_co_occurrences,
            "max_verified_edges": self._max_verified_edges,
            "max_co_occurrence_keys": self._max_co_occurrence_keys,
            "max_co_occurrences_per_key": self._max_co_occurrences_per_key,
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

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        """
        Initialize AdvancedGraphQueryService.

        Args:
            graph: Initial NetworkX graph
        """
        self.temporal = TemporalGraphManager()
        self.negation = NegationQueryEngine(graph)
        self.intersection = CommonNeighborAnalyzer(graph)
        self.verification = EdgeVerificationEngine()
        self._graph_lock = threading.Lock()
        self._graph = graph
        logger.info("advanced_graph_query_service_initialized")

    def set_graph(self, graph: nx.DiGraph) -> None:
        """Atomically update the graph for all engines."""
        with self._graph_lock:
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
_service_lock = threading.Lock()

def get_advanced_query_service(
    graph: nx.DiGraph | None = None,
) -> AdvancedGraphQueryService:
    """Thread-safe singleton accessor for AdvancedGraphQueryService."""
    global _advanced_query_service
    with _service_lock:
        if _advanced_query_service is None:
            _advanced_query_service = AdvancedGraphQueryService(graph)
        elif graph is not None:
            _advanced_query_service.set_graph(graph)
        return _advanced_query_service
