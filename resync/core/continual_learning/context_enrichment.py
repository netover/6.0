# pylint: disable=all
# mypy: no-rerun
"""
Context Enrichment - Enriquece queries RAG com contexto aprendido.

Este módulo usa o Learning Store (padrões de jobs, erros, durações)
para adicionar contexto relevante às queries antes do RAG retrieval:

1. Adiciona informações sobre taxa de falha de jobs
2. Inclui erros comuns relacionados
3. Adiciona contexto de duração esperada
4. Enriquece com padrões de dependência conhecidos
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichmentResult:
    """Result of query enrichment."""

    original_query: str
    enriched_query: str
    context_added: list[str]
    entities_found: dict[str, list[str]]
    enrichment_source: str
    confidence: float = 1.0


@dataclass
class EntityContext:
    """Context information about an entity."""

    entity_id: str
    entity_type: str

    # Job-specific context
    failure_rate: float | None = None
    avg_duration_seconds: float | None = None
    common_errors: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)

    # Workstation-specific context
    active_jobs_count: int | None = None
    last_failure: datetime | None = None

    # General
    last_seen: datetime | None = None
    notes: list[str] = field(default_factory=list)


class ContextEnricher:
    """
    Enriches RAG queries with learned context.

    Uses multiple sources:
    1. TWSLearningStore - Job execution patterns
    2. Knowledge Graph - Relationships and dependencies
    3. Feedback Store - Quality signals
    4. Context Store - Recent interactions
    """

    def __init__(
        self,
        learning_store_factory: Any | None = None,
        knowledge_graph: Any | None = None,
        max_context_length: int = 200,
        enable_job_context: bool = True,
        enable_error_context: bool = True,
        enable_dependency_context: bool = True,
    ):
        """
        Initialize the context enricher.

        Args:
            learning_store_factory: Factory to get instance-specific learning stores
            knowledge_graph: KG instance for relationship context
            max_context_length: Maximum characters to add as context
            enable_job_context: Add job-related context (duration, failure rate)
            enable_error_context: Add error-related context
            enable_dependency_context: Add dependency chain context
        """
        self._learning_store_factory = learning_store_factory
        self._kg = knowledge_graph
        self.max_context_length = max_context_length
        self.enable_job_context = enable_job_context
        self.enable_error_context = enable_error_context
        self.enable_dependency_context = enable_dependency_context

        # Entity extraction patterns
        self._entity_patterns = {
            "job": [
                r"\bjob[:\s]+([A-Z][A-Z0-9_]+)\b",
                r"\b([A-Z][A-Z0-9_]{3,})\s+(?:job|batch|process|processo)\b",
                r"(?:job|processo)\s+(?:chamado|named|called)\s+['\"]?([A-Z][A-Z0-9_]+)['\"]?",
                r"^([A-Z][A-Z0-9_]{3,})$",  # Just the job name alone
            ],
            "workstation": [
                r"\b(?:workstation|ws|server|agent|agente)[:\s]+([A-Z0-9_-]+)\b",
                r"\b([A-Z]{2,3}\d{3,})\b",
            ],
            "error_code": [
                r"\b(AWSB[A-Z0-9]+)\b",
                r"\b(ERR[_-]?\d+)\b",
                r"\berror\s+(?:code\s+)?[:\s]*(\d{4,})\b",
                r"\bcódigo\s+(?:de\s+)?erro[:\s]+(\w+)\b",
            ],
            "job_stream": [
                r"\bjob\s*stream[:\s]+([A-Z][A-Z0-9_]+)\b",
                r"\bstream[:\s]+([A-Z][A-Z0-9_]+)\b",
            ],
        }

        # Query intent patterns
        self._intent_patterns = {
            "failure": [r"\bfail", r"\berro", r"\bfall", r"\babend"],
            "duration": [r"\bdur", r"\btemp", r"\blong", r"\bslow", r"\blento"],
            "dependency": [r"\bdepend", r"\bprecedent", r"\bsuccessor", r"\bchain"],
            "status": [r"\bstatus", r"\bstat", r"\bsituação"],
            "resource": [r"\bresource", r"\blocal", r"\bfile", r"\brecurso"],
        }

    def _get_learning_store(self, instance_id: str):
        """Get learning store for a TWS instance."""
        if self._learning_store_factory:
            return self._learning_store_factory(instance_id)

        # Default: try to import and get from tws_multi
        try:
            from resync.core.tws_multi.learning import TWSLearningStore

            return TWSLearningStore(instance_id)
        except ImportError:
            return None

    async def _get_kg(self):
        """Get Knowledge Graph instance."""
        if self._kg is not None:
            return self._kg

        try:
            from resync.knowledge.retrieval.graph import get_knowledge_graph

            self._kg = await get_knowledge_graph()
            return self._kg
        except ImportError:
            return None

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract entities from text."""
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in self._entity_patterns.items():
            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for m in matches:
                    # Normalize entity ID
                    entity_id = m.upper() if entity_type != "error_code" else m
                    found.add(entity_id)
            if found:
                entities[entity_type] = list(found)

        return entities

    def detect_intent(self, query: str) -> list[str]:
        """Detect query intent based on patterns."""
        query_lower = query.lower()
        intents = []

        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intents.append(intent)
                    break

        return intents

    async def enrich_query(
        self,
        query: str,
        instance_id: str | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> EnrichmentResult:
        """
        Enrich a query with learned context.

        Args:
            query: Original query text
            instance_id: TWS instance ID (for learning store)
            user_context: Additional user-provided context

        Returns:
            EnrichmentResult with original and enriched query
        """
        # Extract entities
        entities = self.extract_entities(query)

        # Detect intent
        intents = self.detect_intent(query)

        # Collect context parts
        context_parts: list[str] = []
        enrichment_source = "none"

        # Get job context
        job_names = entities.get("job", [])
        job_streams = entities.get("job_stream", ["*"])

        if job_names and self.enable_job_context:
            job_context = await self._get_job_context(
                job_names,
                job_streams[0] if job_streams else "*",
                instance_id,
            )
            context_parts.extend(job_context)
            if job_context:
                enrichment_source = "learning_store"

        # Get error context
        error_codes = entities.get("error_code", [])
        if error_codes and self.enable_error_context:
            error_context = await self._get_error_context(error_codes, instance_id)
            context_parts.extend(error_context)
            if error_context and enrichment_source == "none":
                enrichment_source = "learning_store"

        # Get dependency context (if dependency intent detected)
        if "dependency" in intents and job_names and self.enable_dependency_context:
            dep_context = await self._get_dependency_context(job_names)
            context_parts.extend(dep_context)
            if dep_context:
                enrichment_source = (
                    "knowledge_graph" if enrichment_source == "none" else "hybrid"
                )

        # Build enriched query
        if context_parts:
            # Truncate context if too long
            total_context = " | ".join(context_parts)
            if len(total_context) > self.max_context_length:
                total_context = total_context[: self.max_context_length] + "..."

            enriched_query = f"{query} [Contexto: {total_context}]"
        else:
            enriched_query = query

        result = EnrichmentResult(
            original_query=query,
            enriched_query=enriched_query,
            context_added=context_parts,
            entities_found=entities,
            enrichment_source=enrichment_source,
        )

        if context_parts:
            logger.info(
                "query_enriched",
                entities=entities,
                context_count=len(context_parts),
                source=enrichment_source,
            )

        return result

    async def _get_job_context(
        self,
        job_names: list[str],
        job_stream: str,
        instance_id: str | None,
    ) -> list[str]:
        """Get context about jobs from Learning Store."""
        context_parts = []

        if not instance_id:
            instance_id = "default"

        learning_store = self._get_learning_store(instance_id)
        if not learning_store:
            return context_parts

        for job_name in job_names[:3]:  # Limit to 3 jobs
            try:
                pattern = learning_store.get_job_pattern(job_name, job_stream)

                if pattern:
                    # Add failure rate if significant
                    if pattern.failure_rate > 0.05 and pattern.execution_count >= 5:
                        context_parts.append(
                            "Job {job_name} taxa falha {pattern.failure_rate:.0%}"
                        )

                    # Add duration if long-running
                    if pattern.avg_duration_seconds > 1800:  # > 30 min
                        pattern.avg_duration_seconds / 60
                        context_parts.append(
                            "Job {job_name} duração ~{duration_min:.0f}min"
                        )

                    # Add common errors
                    if pattern.common_failure_reasons:
                        top_errors = pattern.common_failure_reasons[:2]
                        context_parts.append(
                            f"Erros comuns {job_name}: {', '.join(top_errors)}"
                        )

            except Exception as e:
                logger.debug("Error getting job context for %s: %s", job_name, e)

        return context_parts

    async def _get_error_context(
        self,
        error_codes: list[str],
        instance_id: str | None,
    ) -> list[str]:
        """Get context about error codes."""
        context_parts = []

        if not instance_id:
            instance_id = "default"

        learning_store = self._get_learning_store(instance_id)
        if not learning_store:
            return context_parts

        for error_code in error_codes[:3]:
            try:
                # Get suggested resolution
                resolution = learning_store.get_suggested_resolution(error_code)

                if resolution:
                    context_parts.append(
                        f"Erro {error_code}: resolução comum '{resolution[:50]}'"
                    )

            except Exception as e:
                logger.debug("Error getting error context for %s: %s", error_code, e)

        return context_parts

    async def _get_dependency_context(
        self,
        job_names: list[str],
    ) -> list[str]:
        """Get dependency context from Knowledge Graph."""
        context_parts = []

        kg = await self._get_kg()
        if not kg:
            return context_parts

        for job_name in job_names[:2]:
            try:
                # Get dependency chain
                chain = await kg.get_dependency_chain(job_name)

                if chain and len(chain) > 1:
                    # Show immediate dependencies
                    deps = chain[:3]
                    context_parts.append(f"{job_name} depende de: {' → '.join(deps)}")

                # Get downstream jobs
                downstream = await kg.get_downstream_jobs(job_name, max_depth=1)

                if downstream:
                    jobs = list(downstream)[:3]
                    context_parts.append(f"Jobs após {job_name}: {', '.join(jobs)}")

            except Exception as e:
                logger.debug("Error getting dependency context for %s: %s", job_name, e)

        return context_parts

    async def get_entity_context(
        self,
        entity_id: str,
        entity_type: str,
        instance_id: str | None = None,
    ) -> EntityContext | None:
        """
        Get comprehensive context for a specific entity.

        Used for detailed entity information in responses.
        """
        context = EntityContext(
            entity_id=entity_id,
            entity_type=entity_type,
        )

        if entity_type == "job":
            learning_store = self._get_learning_store(instance_id or "default")

            if learning_store:
                pattern = learning_store.get_job_pattern(entity_id, "*")

                if pattern:
                    context.failure_rate = pattern.failure_rate
                    context.avg_duration_seconds = pattern.avg_duration_seconds
                    context.common_errors = pattern.common_failure_reasons
                    context.last_seen = pattern.last_execution

            # Get dependencies from KG
            kg = await self._get_kg()
            if kg:
                try:
                    chain = await kg.get_dependency_chain(entity_id)
                    context.dependencies = chain or []

                    downstream = await kg.get_downstream_jobs(entity_id, max_depth=1)
                    context.dependents = list(downstream) if downstream else []
                except Exception as exc:
                    logger.debug(
                        "suppressed_exception", error=str(exc), exc_info=True
                    )  # was: pass

        return context

    def format_context_for_response(
        self,
        context: EntityContext,
    ) -> str:
        """Format entity context for inclusion in response."""
        parts = []

        if context.entity_type == "job":
            if context.failure_rate is not None and context.failure_rate > 0:
                parts.append("Taxa de falha: {context.failure_rate:.1%}")

            if context.avg_duration_seconds is not None:
                if context.avg_duration_seconds > 3600:
                    parts.append(
                        "Duração média: {context.avg_duration_seconds / 3600:.1f}h"
                    )
                elif context.avg_duration_seconds > 60:
                    parts.append(
                        "Duração média: {context.avg_duration_seconds / 60:.0f}min"
                    )

            if context.dependencies:
                deps = context.dependencies[:3]
                parts.append(f"Dependências: {' → '.join(deps)}")

            if context.common_errors:
                errors = context.common_errors[:2]
                parts.append(f"Erros comuns: {', '.join(errors)}")

        return " | ".join(parts) if parts else ""


# Singleton instance
_enricher: ContextEnricher | None = None


def get_context_enricher() -> ContextEnricher:
    """Get global context enricher instance."""
    global _enricher
    if _enricher is None:
        _enricher = ContextEnricher()
    return _enricher


async def enrich_query(
    query: str,
    instance_id: str | None = None,
) -> str:
    """
    Convenience function to enrich a query.

    Returns enriched query string (or original if no context found).
    """
    enricher = get_context_enricher()
    result = await enricher.enrich_query(query, instance_id)
    return result.enriched_query
