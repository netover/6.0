"""
Context Enrichment for RAG Queries.

Enriches user queries with contextual information from:
- Learning Store (job patterns, failure history)
- Knowledge Graph (relationships, dependencies)
- Historical interactions

This improves RAG retrieval by adding relevant context that
helps match queries to the most relevant documents.

Key Features:
- Automatic entity detection
- Pattern-based context injection
- Knowledge Graph relationship expansion
- Configurable enrichment strategies
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

class EnrichmentType(str, Enum):
    """Types of context enrichment."""

    JOB_PATTERN = "job_pattern"
    FAILURE_HISTORY = "failure_history"
    DEPENDENCY_CONTEXT = "dependency_context"
    RESOURCE_CONTEXT = "resource_context"
    TEMPORAL_CONTEXT = "temporal_context"
    ERROR_CONTEXT = "error_context"

@dataclass
class EnrichmentResult:
    """Result of context enrichment."""

    original_query: str
    enriched_query: str
    enrichments_applied: list[EnrichmentType]
    context_snippets: list[str]
    entities_found: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def was_enriched(self) -> bool:
        return len(self.enrichments_applied) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "enriched_query": self.enriched_query,
            "enrichments_applied": [e.value for e in self.enrichments_applied],
            "context_snippets": self.context_snippets,
            "entities_found": self.entities_found,
            "metadata": self.metadata,
        }

class ContextEnricher:
    """
    Enriches RAG queries with contextual information.

    Process:
    1. Extract entities from query (jobs, workstations, error codes)
    2. Fetch relevant context from Learning Store and KG
    3. Inject context into query as natural language additions
    4. Return enriched query for RAG retrieval
    """

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "job": [
            r"\bjob\s+([A-Z][A-Z0-9_]+)\b",
            r"\b([A-Z][A-Z0-9_]{2,})\s+(?:job|batch|task)\b",
            r"(?:run|execute|submit)\s+([A-Z][A-Z0-9_]+)\b",
        ],
        "job_stream": [
            r"\bstream\s+([A-Z][A-Z0-9_]+)\b",
            r"\b([A-Z][A-Z0-9_]+)\s+stream\b",
            r"(?:in|from)\s+stream\s+([A-Z][A-Z0-9_]+)\b",
        ],
        "workstation": [
            r"\b(?:workstation|ws|server|agent)\s+([A-Z0-9_]+)\b",
            r"\bon\s+([A-Z]{2,3}\d{3,})\b",
        ],
        "error_code": [
            r"\berror\s+(?:code\s+)?(\d{4,})\b",
            r"\bAWS[A-Z]{3}\d{4}[A-Z]?\b",
            r"\brc\s*=?\s*(\d+)\b",
        ],
        "resource": [
            r"\bresource\s+([A-Z0-9_]+)\b",
            r"\b(?:file|database|queue)\s+([A-Z0-9_./]+)\b",
        ],
    }

    # Context templates
    CONTEXT_TEMPLATES = {
        EnrichmentType.JOB_PATTERN: (
            "(Job {job} típico: duração ~{duration}min, "
            "executa às {hour}h)"
        ),
        EnrichmentType.FAILURE_HISTORY: (
            "(Job {job} tem {rate:.0%} taxa de falha, "
            "erros comuns: {errors})"
        ),
        EnrichmentType.DEPENDENCY_CONTEXT: "(Job {job} depende de: {deps})",
        EnrichmentType.RESOURCE_CONTEXT: "(Job {job} usa recursos: {resources})",
        EnrichmentType.TEMPORAL_CONTEXT: "(Contexto temporal: {context})",
        EnrichmentType.ERROR_CONTEXT: (
            "(Erro {code} geralmente resolvido com: "
            "{resolution})"
        ),
    }

    def __init__(
        self,
        enable_learning_store: bool = True,
        enable_knowledge_graph: bool = True,
        max_context_length: int = 500,
        min_confidence: float = 0.5,
    ):
        """
        Initialize context enricher.

        Args:
            enable_learning_store: Use Learning Store for patterns
            enable_knowledge_graph: Use KG for relationships
            max_context_length: Max chars of context to add
            min_confidence: Minimum confidence for including context
        """
        self.enable_learning_store = enable_learning_store
        self.enable_knowledge_graph = enable_knowledge_graph
        self.max_context_length = max_context_length
        self.min_confidence = min_confidence

        self._learning_stores: dict[str, Any] = {}
        self._kg = None

        # Statistics
        self._queries_enriched = 0
        self._total_queries = 0
        self._enrichment_counts: dict[EnrichmentType, int] = {}

    def _get_learning_store(self, instance_id: str):
        """Get Learning Store for instance."""
        if instance_id not in self._learning_stores:
            try:
                from resync.core.tws_multi.learning import TWSLearningStore

                self._learning_stores[instance_id] = TWSLearningStore()
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Could not load learning store: %s", e)
                return None
        return self._learning_stores[instance_id]

    async def _get_kg(self):
        """Get Knowledge Graph instance."""
        if self._kg is None:
            try:
                from resync.knowledge.retrieval.graph import get_kg_instance

                self._kg = await get_kg_instance()
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Could not load knowledge graph: %s", e)
                return None
        return self._kg

    # =========================================================================
    # MAIN ENRICHMENT
    # =========================================================================

    async def enrich_query(
        self,
        query: str,
        instance_id: str = "default",
        user_context: dict[str, Any] | None = None,
    ) -> EnrichmentResult:
        """
        Enrich a query with contextual information.

        Args:
            query: Original user query
            instance_id: TWS instance ID for Learning Store
            user_context: Optional additional context

        Returns:
            EnrichmentResult with enriched query and metadata
        """
        self._total_queries += 1

        # Extract entities from query
        entities = self._extract_entities(query)

        # Collect context snippets
        context_snippets: list[str] = []
        enrichments_applied: list[EnrichmentType] = []
        metadata: dict[str, Any] = {}

        # Enrich from Learning Store
        if self.enable_learning_store:
            ls_context, ls_types = await self._enrich_from_learning_store(
                entities, instance_id
            )
            context_snippets.extend(ls_context)
            enrichments_applied.extend(ls_types)

        # Enrich from Knowledge Graph
        if self.enable_knowledge_graph:
            kg_context, kg_types = await self._enrich_from_knowledge_graph(entities)
            context_snippets.extend(kg_context)
            enrichments_applied.extend(kg_types)

        # Add temporal context if relevant
        temporal_context = self._get_temporal_context(query)
        if temporal_context:
            context_snippets.append(temporal_context)
            enrichments_applied.append(EnrichmentType.TEMPORAL_CONTEXT)

        # Build enriched query
        enriched_query = self._build_enriched_query(query, context_snippets)

        # Update statistics
        if enrichments_applied:
            self._queries_enriched += 1
            for etype in enrichments_applied:
                self._enrichment_counts[etype] = (
                    self._enrichment_counts.get(etype, 0) + 1
                )

        result = EnrichmentResult(
            original_query=query,
            enriched_query=enriched_query,
            enrichments_applied=enrichments_applied,
            context_snippets=context_snippets,
            entities_found=entities,
            metadata=metadata,
        )

        logger.debug(
            "query_enriched",
            original_len=len(query),
            enriched_len=len(enriched_query),
            enrichments=len(enrichments_applied),
            entities=sum(len(v) for v in entities.values()),
        )

        return result

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract entities from text."""
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            found: set[str] = set()
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        found.add(match[0].upper())
                    else:
                        found.add(match.upper())
            if found:
                entities[entity_type] = list(found)

        return entities

    # =========================================================================
    # LEARNING STORE ENRICHMENT
    # =========================================================================

    async def _enrich_from_learning_store(
        self,
        entities: dict[str, list[str]],
        instance_id: str,
    ) -> tuple[list[str], list[EnrichmentType]]:
        """Get context from Learning Store."""
        context_snippets: list[str] = []
        enrichments: list[EnrichmentType] = []

        learning_store = self._get_learning_store(instance_id)
        if not learning_store:
            return context_snippets, enrichments

        jobs = entities.get("job", [])
        job_streams = entities.get("job_stream", ["*"])
        error_codes = entities.get("error_code", [])

        for job_name in jobs:
            for job_stream in job_streams:
                pattern = learning_store.get_job_pattern(job_name, job_stream)
                if pattern and pattern.execution_count >= 5:
                    # Job pattern context
                    if pattern.avg_duration_seconds > 0:
                        context = self.CONTEXT_TEMPLATES[
                            EnrichmentType.JOB_PATTERN
                        ].format(
                            job=job_name,
                            duration=int(pattern.avg_duration_seconds / 60),
                            hour=pattern.typical_start_hour,
                        )
                        context_snippets.append(context)
                        enrichments.append(EnrichmentType.JOB_PATTERN)

                    # Failure history context
                    if pattern.failure_rate > 0.05 and pattern.execution_count >= 10:
                        errors = (
                            ", ".join(pattern.common_failure_reasons[:3]) or "vários"
                        )
                        context = self.CONTEXT_TEMPLATES[
                            EnrichmentType.FAILURE_HISTORY
                        ].format(
                            job=job_name,
                            rate=pattern.failure_rate,
                            errors=errors,
                        )
                        context_snippets.append(context)
                        enrichments.append(EnrichmentType.FAILURE_HISTORY)
                    break  # Found pattern, skip other streams

        # Error resolution context
        for error_code in error_codes:
            resolution = learning_store.get_suggested_resolution(error_code)
            if resolution:
                context = self.CONTEXT_TEMPLATES[EnrichmentType.ERROR_CONTEXT].format(
                    code=error_code,
                    resolution=resolution[:100],
                )
                context_snippets.append(context)
                enrichments.append(EnrichmentType.ERROR_CONTEXT)

        return context_snippets, enrichments

    # =========================================================================
    # KNOWLEDGE GRAPH ENRICHMENT
    # =========================================================================

    async def _enrich_from_knowledge_graph(
        self,
        entities: dict[str, list[str]],
    ) -> tuple[list[str], list[EnrichmentType]]:
        """Get context from Knowledge Graph."""
        context_snippets: list[str] = []
        enrichments: list[EnrichmentType] = []

        kg = await self._get_kg()
        if not kg:
            return context_snippets, enrichments

        jobs = entities.get("job", [])

        for job_name in jobs[:3]:  # Limit to avoid too much context
            try:
                # Get dependencies
                deps = await kg.get_dependency_chain(job_name, max_depth=2)
                if deps and len(deps) > 1:
                    dep_list = ", ".join(deps[:5])
                    context = self.CONTEXT_TEMPLATES[
                        EnrichmentType.DEPENDENCY_CONTEXT
                    ].format(
                        job=job_name,
                        deps=dep_list,
                    )
                    context_snippets.append(context)
                    enrichments.append(EnrichmentType.DEPENDENCY_CONTEXT)

                # Get resources
                resources = await self._get_job_resources(kg, job_name)
                if resources:
                    resource_list = ", ".join(resources[:3])
                    context = self.CONTEXT_TEMPLATES[
                        EnrichmentType.RESOURCE_CONTEXT
                    ].format(
                        job=job_name,
                        resources=resource_list,
                    )
                    context_snippets.append(context)
                    enrichments.append(EnrichmentType.RESOURCE_CONTEXT)

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.debug("KG enrichment failed for %s: %s", job_name, e)
                continue

        return context_snippets, enrichments

    async def _get_job_resources(self, kg, job_name: str) -> list[str]:
        """Get resources used by a job from KG."""
        try:
            resources = await kg.get_jobs_using_resource(job_name)
            return resources if resources else []
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return []

    # =========================================================================
    # TEMPORAL CONTEXT
    # =========================================================================

    def _get_temporal_context(self, query: str) -> str | None:
        """Extract and add temporal context."""
        query_lower = query.lower()

        # Check for time-related queries
        time_patterns = {
            r"\b(?:hoje|today)\b": "hoje, "
            + datetime.now(timezone.utc).strftime("%d/%m/%Y"),
            r"\b(?:ontem|yesterday)\b": "ontem, "
            + (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%d/%m/%Y"),
            r"\b(?:esta semana|this week)\b": "semana atual",
            r"\b(?:manhã|morning)\b": "período matutino (06h-12h)",
            r"\b(?:tarde|afternoon)\b": "período vespertino (12h-18h)",
            r"\b(?:noite|night|overnight)\b": "período noturno (18h-06h)",
            r"\b(?:fim de semana|weekend)\b": "fim de semana (sábado/domingo)",
        }

        for pattern, context in time_patterns.items():
            if re.search(pattern, query_lower):
                return self.CONTEXT_TEMPLATES[EnrichmentType.TEMPORAL_CONTEXT].format(
                    context=context
                )

        return None

    # =========================================================================
    # QUERY BUILDING
    # =========================================================================

    def _build_enriched_query(
        self,
        original_query: str,
        context_snippets: list[str],
    ) -> str:
        """Build final enriched query."""
        if not context_snippets:
            return original_query

        # Join context snippets
        context_text = " ".join(context_snippets)

        # Truncate if too long
        if len(context_text) > self.max_context_length:
            context_text = context_text[: self.max_context_length] + "..."

        # Append context to query
        return f"{original_query} {context_text}"

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get enrichment statistics."""
        return {
            "total_queries": self._total_queries,
            "queries_enriched": self._queries_enriched,
            "enrichment_rate": (
                self._queries_enriched / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
            "enrichment_counts": {
                k.value: v for k, v in self._enrichment_counts.items()
            },
            "config": {
                "enable_learning_store": self.enable_learning_store,
                "enable_knowledge_graph": self.enable_knowledge_graph,
                "max_context_length": self.max_context_length,
            },
        }

# Global instance
_enricher: ContextEnricher | None = None

def get_context_enricher() -> ContextEnricher:
    """Get the global context enricher instance."""
    global _enricher
    if _enricher is None:
        _enricher = ContextEnricher()
    return _enricher

# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

async def enrich_query(
    query: str,
    instance_id: str = "default",
) -> str:
    """
    Simple function to enrich a query.

    Returns just the enriched query string.
    """
    enricher = get_context_enricher()
    result = await enricher.enrich_query(query, instance_id)
    return result.enriched_query

async def enrich_query_with_details(
    query: str,
    instance_id: str = "default",
) -> EnrichmentResult:
    """
    Enrich a query and return full details.
    """
    enricher = get_context_enricher()
    return await enricher.enrich_query(query, instance_id)
