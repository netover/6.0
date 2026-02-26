# pylint: skip-file
# mypy: ignore-errors
"""
Hybrid RAG - Knowledge Graph + Vector Search Query Router.

Routes queries to the appropriate system based on query intent:
- Graph queries (dependencies, conflicts, impact) → Knowledge Graph (NetworkX on-demand)
- Documentation queries (how-to, explanations) → RAG (pgvector)
- Complex queries → Both systems, then merge results

This solves the core limitation of pure RAG systems: inability to
perform multi-hop reasoning and relationship traversal.

Architecture:
    Query → Intent Classifier → Router
                                  ├─→ Knowledge Graph → Graph Facts
                                  ├─→ RAG (pgvector) → Semantic Documents
                                  └─→ Context Merger → LLM → Response

v5.9.3: Graph now built on-demand from TWS API via TwsGraphService.
"""

import asyncio
from inspect import isawaitable
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections import OrderedDict
from functools import lru_cache

from resync.core.structured_logger import get_logger
from resync.core.utils.prompt_formatter import OpinionBasedPromptFormatter
from resync.knowledge.retrieval.graph import get_knowledge_graph

logger = get_logger(__name__)

# =============================================================================
# QUERY INTENT CLASSIFICATION
# =============================================================================

class QueryIntent(str, Enum):
    """Types of queries that can be handled."""

    # Graph-only queries
    DEPENDENCY_CHAIN = "dependency_chain"  # "What are the dependencies of X?"
    IMPACT_ANALYSIS = "impact_analysis"  # "What happens if X fails?"
    RESOURCE_CONFLICT = "resource_conflict"  # "Can X and Y run together?"
    CRITICAL_JOBS = "critical_jobs"  # "What are the most critical jobs?"
    JOB_LINEAGE = "job_lineage"  # "Show the full lineage of X"

    # RAG-only queries
    DOCUMENTATION = "documentation"  # "How do I configure X?"
    EXPLANATION = "explanation"  # "What is X?"
    TROUBLESHOOTING = "troubleshooting"  # "How to fix error X?"

    # Hybrid queries (both systems)
    ROOT_CAUSE = "root_cause"  # "Why did X fail?" (events + docs)
    JOB_DETAILS = "job_details"  # "Tell me about job X" (graph + docs)
    COMPARISON = "comparison"  # "Compare X and Y"

    # Default
    GENERAL = "general"  # Unclear intent

@dataclass
class QueryClassification:
    """Result of query classification."""

    intent: QueryIntent
    confidence: float
    entities: dict[str, list[str]]  # {"jobs": [...], "workstations": [...]}
    use_graph: bool
    use_rag: bool
    graph_query_type: str | None = None

# Intent patterns (Portuguese + English)
INTENT_PATTERNS = {
    QueryIntent.DEPENDENCY_CHAIN: [
        r"(?:quais|what).+(?:dependências|dependencies|depends)",
        r"(?:de que|what).+(?:depende|depend)",
        r"(?:cadeia|chain).+(?:dependência|dependency)",
        r"(?:listar?|list).+(?:predecessores|predecessors)",
        r"(?:upstream|montante).+(?:jobs?|trabalhos?)",
    ],
    QueryIntent.IMPACT_ANALYSIS: [
        r"(?:o que acontece|what happens).+(?:falhar?|fails?|fail)",
        r"(?:impacto|impact).+(?:se|if).+(?:falhar?|fails?)",
        r"(?:quais|which|what).+(?:afetados?|affected)",
        r"(?:downstream|jusante).+(?:jobs?|trabalhos?)",
        r"(?:análise de impacto|impact analysis)",
    ],
    QueryIntent.RESOURCE_CONFLICT: [
        r"(?:podem|can).+(?:executar|run).+(?:juntos?|together|simultaneously)",
        r"(?:conflito|conflict).+(?:recursos?|resources?)",
        r"(?:compartilham|share).+(?:recursos?|resources?)",
        r"(?:exclusivos?|exclusive).+(?:recursos?|resources?)",
        r"(?:concorrência|concurrency)",
    ],
    QueryIntent.CRITICAL_JOBS: [
        r"(?:mais|most).+(?:críticos?|critical)",
        r"(?:jobs?|trabalhos?).+(?:importantes?|important)",
        r"(?:gargalos?|bottlenecks?)",
        r"(?:centralidade|centrality)",
        r"(?:alto risco|high risk)",
    ],
    QueryIntent.JOB_LINEAGE: [
        r"(?:linhagem|lineage).+(?:completa|full|complete)",
        r"(?:ancestrais?|ancestors?)",
        r"(?:hierarquia|hierarchy)",
        r"(?:árvore|tree).+(?:dependência|dependency)",
    ],
    QueryIntent.ROOT_CAUSE: [
        r"(?:por que|why).+(?:falhou|failed)",
        r"(?:causa raiz|root cause)",
        r"(?:motivo|reason).+(?:falha|failure)",
        r"(?:análise|analysis).+(?:erro|error)",
        r"(?:investigar?|investigate)",
    ],
    QueryIntent.DOCUMENTATION: [
        r"(?:como|how).+(?:configur[aoei]|configure)",
        r"(?:como|how).+(?:instal[aoei]|install)",
        r"(?:documentação|documentation)",
        r"(?:manual|guide)",
        r"(?:passo a passo|step by step)",
    ],
    QueryIntent.TROUBLESHOOTING: [
        r"(?:como|how).+(?:resolver|fix|solve)",
        r"(?:erro|error).+(?:solução|solution)",
        r"(?:problema|problem).+(?:resolver|resolve)",
        r"(?:não funciona|not working)",
    ],
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    "jobs": [
        r"\bjob[:\s]+([A-Z][A-Z0-9_]+)",
        r"\b([A-Z][A-Z0-9_]{2,})\b",  # Uppercase with underscores
    ],
    "workstations": [
        r"\b(?:workstation|ws|servidor)[:\s]+([A-Z0-9_]+)",
        r"\b([A-Z]{2,3}\d{3,})\b",  # e.g., WS001
    ],
    "resources": [
        r"\bresource[:\s]+([A-Z0-9_]+)",
    ],
}

@lru_cache(maxsize=128)
def _compile_patterns():
    return {
        intent: [re.compile(p, re.IGNORECASE) for p in patterns]
        for intent, patterns in INTENT_PATTERNS.items()
    }

class QueryClassifier:
    """
    Classifies queries to determine routing strategy.

    Uses regex patterns first (fast), then falls back to LLM
    if confidence is low (more accurate but slower).

    Features:
    - LRU cache for LLM results (prevents memory leak)
    - Configurable cache size (default 1000)
    - Query normalization for better cache hits
    """

    # LLM classification prompt
    LLM_ROUTER_PROMPT = """You are a query classifier for a
TWS/HWA workload automation system.
Classify the user query into ONE of these categories:

CATEGORIES:
- DEPENDENCY_CHAIN: Questions about job dependencies, predecessors,
  what a job depends on
- IMPACT_ANALYSIS: Questions about what happens if something fails, downstream effects
- RESOURCE_CONFLICT: Questions about concurrent execution, resource sharing, conflicts
- CRITICAL_JOBS: Questions about most important jobs, bottlenecks, critical paths
- JOB_LINEAGE: Questions about job ancestry, full dependency tree, hierarchy
- ROOT_CAUSE: Questions about why something failed, root cause analysis
- DOCUMENTATION: Questions about how to configure, install, setup procedures
- TROUBLESHOOTING: Questions about fixing errors, solving problems
- EXPLANATION: Questions asking what something is, definitions
- JOB_DETAILS: Questions asking for information about a specific job
- COMPARISON: Questions comparing two or more jobs or systems
- GENERAL: Anything else that doesn't fit above categories

USER QUERY: {query}

Respond with ONLY the category name (e.g., "DEPENDENCY_CHAIN"). No explanation."""

    # Intent map at class level for efficiency
    _INTENT_MAP = {
        "DEPENDENCY_CHAIN": QueryIntent.DEPENDENCY_CHAIN,
        "IMPACT_ANALYSIS": QueryIntent.IMPACT_ANALYSIS,
        "RESOURCE_CONFLICT": QueryIntent.RESOURCE_CONFLICT,
        "CRITICAL_JOBS": QueryIntent.CRITICAL_JOBS,
        "JOB_LINEAGE": QueryIntent.JOB_LINEAGE,
        "ROOT_CAUSE": QueryIntent.ROOT_CAUSE,
        "DOCUMENTATION": QueryIntent.DOCUMENTATION,
        "TROUBLESHOOTING": QueryIntent.TROUBLESHOOTING,
        "EXPLANATION": QueryIntent.EXPLANATION,
        "JOB_DETAILS": QueryIntent.JOB_DETAILS,
        "COMPARISON": QueryIntent.COMPARISON,
        "GENERAL": QueryIntent.GENERAL,
    }

    # Default cache size
    _DEFAULT_CACHE_SIZE = 1000

    def __init__(
        self,
        llm_service: Any | None = None,
        use_llm_fallback: bool = True,
        cache_max_size: int = _DEFAULT_CACHE_SIZE,
    ):
        """
        Initialize classifier.

        Args:
            llm_service: Optional LLM service for fallback classification
            use_llm_fallback: Whether to use LLM when regex fails
            cache_max_size: Maximum LRU cache size (default 1000)
        """
        self._llm = llm_service
        self._use_llm_fallback = use_llm_fallback
        self._cache_max_size = cache_max_size

        # LRU Cache using OrderedDict
        self._intent_cache: OrderedDict[str, QueryIntent] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    def _normalize_for_cache(self, query: str) -> str:
        """
        Normalize query for consistent cache keys.

        - Lowercase
        - Remove extra whitespace
        - Hash for fixed-size key
        """
        # Normalize: lowercase, collapse whitespace, limit length
        normalized = " ".join(query.lower().split())[:200]
        return hashlib.blake2b(normalized.encode(), digest_size=16).hexdigest()

    async def _get_from_cache(self, key: str) -> QueryIntent | None:
        """Get from LRU cache and update access order."""
        async with self._cache_lock:
            if key in self._intent_cache:
                # Move to end (most recently used)
                self._intent_cache.move_to_end(key)
                return self._intent_cache[key]
        return None

    async def _add_to_cache(self, key: str, intent: QueryIntent) -> None:
        """Add to LRU cache with eviction."""
        async with self._cache_lock:
            if key in self._intent_cache:
                # Update existing and move to end
                self._intent_cache.move_to_end(key)
                self._intent_cache[key] = intent
                return

            # Add new entry
            self._intent_cache[key] = intent

            # Evict oldest entries if over limit
            while len(self._intent_cache) > self._cache_max_size:
                self._intent_cache.popitem(last=False)  # Remove oldest (first)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._intent_cache),
            "max_size": self._cache_max_size,
            "utilization": len(self._intent_cache) / self._cache_max_size
            if self._cache_max_size > 0
            else 0,
        }

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self._intent_cache.clear()

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query to determine how to route it.

        Args:
            query: User query text

        Returns:
            QueryClassification with intent and routing info
        """
        query_lower = query.lower()

        # Extract entities first
        entities = self._extract_entities(query)

        # Match intent patterns
        best_intent = QueryIntent.GENERAL
        best_confidence = 0.0

        for matched_intent, compiled_patterns in _compile_patterns().items():
            for pattern in compiled_patterns:
                if pattern.search(query_lower):
                    # Higher confidence if entity found
                    confidence = 0.8 if entities.get("jobs") else 0.6
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = matched_intent
                        break

        # Determine routing
        use_graph, use_rag, graph_query = self._determine_routing(best_intent)

        return QueryClassification(
            intent=best_intent,
            confidence=best_confidence,
            entities=entities,
            use_graph=use_graph,
            use_rag=use_rag,
            graph_query_type=graph_query,
        )

    async def classify_async(self, query: str) -> QueryClassification:
        """
        Classify query with LLM fallback for low-confidence matches.

        This is the async version that can use LLM when regex fails.

        Args:
            query: User query text

        Returns:
            QueryClassification with intent and routing info
        """
        # First try regex classification
        result = self.classify(query)

        # If high confidence or LLM disabled, return regex result
        if result.confidence >= 0.6 or not self._use_llm_fallback:
            return result

        # Low confidence - try LLM classification
        llm_intent = await self._classify_with_llm(query)

        if llm_intent and llm_intent != QueryIntent.GENERAL:
            # LLM found a better match
            use_graph, use_rag, graph_query = self._determine_routing(llm_intent)

            # Re-extract entities if regex didn't find them, but LLM still confidently matched graph intent.
            # In our current setup, we just use the regex entities or empty dict,
            # as the prompt doesn't extract entities yet.
            new_entities = result.entities

            return QueryClassification(
                intent=llm_intent,
                confidence=0.75,  # LLM classification confidence
                entities=new_entities,
                use_graph=use_graph,
                use_rag=use_rag,
                graph_query_type=graph_query,
            )

        # LLM also didn't find a match, return original
        return result

    async def _classify_with_llm(self, query: str) -> QueryIntent | None:
        """
        Use LLM to classify query with LRU caching.

        Args:
            query: User query text

        Returns:
            Classified intent or None if failed
        """
        # Normalize query for cache key
        cache_key = self._normalize_for_cache(query)

        # Check LRU cache first
        cached = await self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug("llm_cache_hit", query=query[:30])
            return cached

        # Get LLM service
        llm = await self._get_llm()
        if llm is None:
            await self._add_to_cache(cache_key, QueryIntent.GENERAL)
            return None

        try:
            prompt = self.LLM_ROUTER_PROMPT.format(query=query)
            response_text = await asyncio.wait_for(llm.generate(prompt), timeout=10.0)

            # Parse robust
            lines = response_text.strip().split("\n")
            category = lines[0].strip().upper().replace(" ", "_") if lines else "GENERAL"

            if category not in self._INTENT_MAP:
                logger.warning("llm_invalid_category", category=category)
                category = "GENERAL"

            # Map to QueryIntent using class-level map
            intent = self._INTENT_MAP[category]

            # Add to LRU cache
            await self._add_to_cache(cache_key, intent)

            logger.debug(
                "llm_classification",
                query=query[:50],
                category=category,
                intent=intent.value,
                cache_size=len(self._intent_cache),
            )

            return intent

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("llm_classification_failed", error=str(e))
            return None

    async def _get_llm(self):
        """Get LLM service (lazy load)."""
        if self._llm is None:
            try:
                from resync.services.llm_service import get_llm_service

                service = get_llm_service()
                if isawaitable(service):
                    self._llm = await asyncio.wait_for(service, timeout=30.0)
                else:
                    self._llm = service
            except ImportError:
                logger.warning("llm_service_not_available")
                return None
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
                logger.warning("llm_service_initialization_failed", error=str(exc))
                return None
        return self._llm

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        """Extract entities from query."""
        entities = {}

        for entity_type, patterns in ENTITY_PATTERNS.items():
            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                found.update(m.upper() for m in matches)
            if found:
                entities[entity_type] = list(found)

        return entities

    def _determine_routing(self, intent: QueryIntent) -> tuple[bool, bool, str | None]:
        """Determine which systems to query."""

        # Graph-only intents
        graph_only = {
            QueryIntent.DEPENDENCY_CHAIN: "dependency_chain",
            QueryIntent.IMPACT_ANALYSIS: "impact_analysis",
            QueryIntent.RESOURCE_CONFLICT: "resource_conflict",
            QueryIntent.CRITICAL_JOBS: "critical_jobs",
            QueryIntent.JOB_LINEAGE: "lineage",
        }

        if intent in graph_only:
            return True, False, graph_only[intent]

        # RAG-only intents
        rag_only = {
            QueryIntent.DOCUMENTATION,
            QueryIntent.EXPLANATION,
            QueryIntent.TROUBLESHOOTING,
        }

        if intent in rag_only:
            return False, True, None

        # Hybrid intents
        hybrid = {
            QueryIntent.ROOT_CAUSE: "event_chain",
            QueryIntent.JOB_DETAILS: "job_info",
            QueryIntent.COMPARISON: "comparison",
        }

        if intent in hybrid:
            return True, True, hybrid[intent]

        # Default: RAG only
        return False, True, None

# =============================================================================
# HYBRID QUERY EXECUTOR
# =============================================================================

class HybridRAG:
    """
    Hybrid Knowledge Graph + RAG query executor.

    Routes queries to appropriate systems and merges results.
    """

    def __init__(
        self,
        rag_retriever: Any | None = None,
        llm_service: Any | None = None,
        use_llm_router: bool = True,
    ):
        """
        Initialize hybrid RAG.

        Args:
            rag_retriever: Existing RAG retriever (pgvector-based)
            llm_service: LLM service for response generation
            use_llm_router: Use LLM for query routing when regex fails
        """
        self._rag = rag_retriever
        self._llm = llm_service
        self._use_llm_router = use_llm_router
        self._classifier = QueryClassifier(
            llm_service=llm_service, use_llm_fallback=use_llm_router
        )
        self._kg = None  # Lazy load

        # Opinion-Based Prompting for +30-50% context adherence improvement
        self._prompt_formatter = OpinionBasedPromptFormatter()

    async def _get_kg(self):
        """Get knowledge graph (lazy load)."""
        if self._kg is None:
            self._kg = await get_knowledge_graph()
            await self._kg.initialize()
        return self._kg

    def _get_rag(self):
        """Get RAG retriever (lazy load)."""
        if self._rag is None:
            try:
                from resync.knowledge.retrieval.retriever import get_retriever

                self._rag = get_retriever()
            except ImportError:
                logger.warning("RAG retriever not available")
        return self._rag

    async def _get_llm(self):
        """Get LLM service (lazy load)."""
        if self._llm is None:
            try:
                from resync.services.llm_service import get_llm_service

                service = get_llm_service()
                if isawaitable(service):
                    self._llm = await asyncio.wait_for(service, timeout=30.0)
                else:
                    self._llm = service
            except ImportError:
                logger.warning("LLM service not available")
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
                logger.warning("llm_service_initialization_failed", error=str(exc))
                return None
        return self._llm

    # =========================================================================
    # MAIN QUERY INTERFACE
    # =========================================================================

    async def query(
        self,
        query_text: str,
        context: dict[str, Any] | None = None,
        generate_response: bool = True,
        enable_continual_learning: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a hybrid query.

        Args:
            query_text: User query
            context: Additional context (user, session, etc.)
            generate_response: Whether to generate LLM response
            enable_continual_learning: Enable context enrichment and active learning

        Returns:
            Dict with graph_results, rag_results, and optional response
        """
        # === CONTINUAL LEARNING: Context Enrichment ===
        enriched_query = query_text
        enrichment_context = []
        if enable_continual_learning:
            try:
                from resync.core.continual_learning import get_context_enricher

                enricher = get_context_enricher()
                instance_id = context.get("instance_id") if context else None
                enrichment_result = await enricher.enrich_query(query_text, instance_id)
                enriched_query = enrichment_result.enriched_query
                enrichment_context = enrichment_result.context_added
                if enrichment_context:
                    logger.info(
                        "query_enriched",
                        original_len=len(query_text),
                        enriched_len=len(enriched_query),
                        context_items=len(enrichment_context),
                    )
            except ImportError:
                logger.debug("continual_learning_module_not_available")
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("context_enrichment_error", error=str(e))
        # === END Context Enrichment ===

        # Classify query - use async version for LLM fallback
        if self._use_llm_router:
            classification = await self._classifier.classify_async(query_text)
        else:
            classification = self._classifier.classify(query_text)

        logger.info(
            "query_classified",
            intent=classification.intent.value,
            confidence=classification.confidence,
            use_graph=classification.use_graph,
            use_rag=classification.use_rag,
        )

        result = {
            "query": query_text,
            "enriched_query": enriched_query if enriched_query != query_text else None,
            "classification": {
                "intent": classification.intent.value,
                "confidence": classification.confidence,
                "entities": classification.entities,
            },
            "graph_results": None,
            "rag_results": None,
            "response": None,
            "continual_learning": {
                "enrichment_context": enrichment_context,
                "needs_review": False,
                "review_reasons": [],
            },
        }

        # Execute graph query if needed
        if classification.use_graph:
            try:
                result["graph_results"] = await asyncio.wait_for(
                    self._execute_graph_query(classification, query_text),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("graph_query_timeout")
                result["graph_results"] = {"error": "timeout"}

        # Execute RAG query if needed (use enriched query for better retrieval)
        if classification.use_rag:
            # Use Fusion RAG for GENERAL or TROUBLESHOOTING intents (ambiguous queries)
            if classification.intent in {
                QueryIntent.GENERAL,
                QueryIntent.TROUBLESHOOTING,
            }:
                result["rag_results"] = await self._execute_fusion_rag_query(
                    enriched_query, classification.entities
                )
            else:
                result["rag_results"] = await self._execute_rag_query(
                    enriched_query, classification.entities
                )

        # Generate response if requested
        if generate_response:
            result["response"] = await self._generate_response(
                query_text, result, classification
            )

            # === CONTINUAL LEARNING: Active Learning Check ===
            if enable_continual_learning and result["response"]:
                try:
                    from resync.core.continual_learning import check_for_review

                    # Get RAG similarity score
                    rag_similarity = 0.0
                    if (
                        result["rag_results"]
                        and isinstance(result["rag_results"], list)
                        and "score" in result["rag_results"][0]
                    ):
                        rag_similarity = float(result["rag_results"][0].get("score", 0))

                    needs_review, warning = await check_for_review(
                        query=query_text,
                        response=str(result["response"]),
                        classification_confidence=classification.confidence,
                        rag_similarity_score=rag_similarity,
                        entities_found=classification.entities,
                    )

                    result["continual_learning"]["needs_review"] = needs_review
                    if warning:
                        result["continual_learning"]["warning"] = warning

                except ImportError:
                    logger.debug("continual_learning_module_not_available")
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    logger.warning("active_learning_check_error", error=str(e))
            # === END Active Learning Check ===

        return result

    # =========================================================================
    # GRAPH QUERY EXECUTION
    # =========================================================================

    async def _execute_graph_query(
        self, classification: QueryClassification, query_text: str
    ) -> dict[str, Any]:
        """Execute knowledge graph query based on classification."""
        kg = await self._get_kg()

        jobs = classification.entities.get("jobs", [])
        query_type = classification.graph_query_type

        if query_type == "dependency_chain" and jobs:
            result = await asyncio.wait_for(kg.get_dependency_chain(jobs[0]), timeout=10.0)
            return {"type": "dependency_chain", "job": jobs[0], "chain": result}

        if query_type == "impact_analysis" and jobs:
            result = await asyncio.wait_for(kg.get_impact_analysis(jobs[0]), timeout=10.0)
            return {"type": "impact_analysis", **result}

        if query_type == "resource_conflict" and len(jobs) >= 2:
            result = await asyncio.wait_for(kg.find_resource_conflicts(jobs[0], jobs[1]), timeout=10.0)
            return {
                "type": "resource_conflict",
                "job_a": jobs[0],
                "job_b": jobs[1],
                "conflicts": result,
            }

        if query_type == "critical_jobs":
            result = await asyncio.wait_for(kg.get_critical_jobs(), timeout=10.0)
            return {"type": "critical_jobs", "jobs": result}

        if query_type == "lineage" and jobs:
            result = await asyncio.wait_for(kg.get_full_lineage(jobs[0]), timeout=10.0)
            return {"type": "lineage", **result}

        if query_type == "job_info" and jobs:
            # Get comprehensive job info
            chain = await asyncio.wait_for(kg.get_dependency_chain(jobs[0]), timeout=10.0)
            impact = await asyncio.wait_for(kg.get_impact_analysis(jobs[0]), timeout=10.0)
            downstream = await asyncio.wait_for(kg.get_downstream_jobs(jobs[0]), timeout=10.0)

            return {
                "type": "job_info",
                "job": jobs[0],
                "dependencies": chain,
                "impact": impact,
                "downstream_jobs": downstream,
            }

        # Default: return graph statistics
        stats = await kg.get_statistics()
        return {"type": "statistics", **stats}

    # =========================================================================
    # RAG QUERY EXECUTION
    # =========================================================================

    async def _execute_rag_query(
        self, query_text: str, entities: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Execute RAG query using pgvector."""
        rag = self._get_rag()

        if rag is None:
            return {"error": "RAG not available", "documents": []}

        try:
            # Build filter if entities found
            if entities.get("jobs"):
                # Could filter by job names in metadata
                pass

            # Execute semantic search
            results = await asyncio.wait_for(rag.retrieve(query_text, top_k=5), timeout=10.0)

            return {
                "type": "semantic_search",
                "query": query_text,
                "documents": results,
            }

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("rag_query_failed", error=str(e))
            return {"error": str(e), "documents": []}

    # =========================================================================
    # FUSION RAG - Query Expansion + RRF Merging
    # =========================================================================

    async def _expand_query(self, query: str, num_variations: int = 3) -> list[str]:
        """
        Generate query variations for Fusion RAG.

        Uses LLM to create semantically similar but differently phrased queries.
        This improves recall by capturing different aspects of the user's intent.

        Args:
            query: Original user query
            num_variations: Number of variations to generate (default: 3)

        Returns:
            List of query variations (including original)
        """
        llm = await self._get_llm()
        if llm is None:
            # Fallback: return original query only
            return [query]

        try:
            prompt = f"""Generate {num_variations} different ways to ask
the following question about TWS/HWA workload automation.
Each variation should capture the same intent but use different words or phrasing.

Original question: {query}

Provide ONLY the variations, one per line, without numbering or explanation."""

            response = await asyncio.wait_for(llm.generate(prompt), timeout=10.0)
            variations = [
                line.strip() for line in response.strip().split("\n") if line.strip()
            ]

            # Always include original query
            all_queries = [query] + variations[:num_variations]

            logger.debug(
                "query_expansion",
                original=query[:50],
                num_variations=len(all_queries) - 1,
            )

            return all_queries

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("query_expansion_failed", error=str(e))
            return [query]

    def _reciprocal_rank_fusion(
        self,
        results_list: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Merge multiple result sets using Reciprocal Rank Fusion (RRF).

        RRF formula: score(doc) = sum(1 / (k + rank_i))
        where rank_i is the position of doc in result set i.

        Documents appearing high in multiple result sets get boosted.

        Args:
            results_list: List of result sets, each is a list of documents
            k: RRF constant (default: 60, standard value from literature)

        Returns:
            Merged and re-ranked documents with 'fusion_score' field
        """
        # Build document index by unique ID
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for result_set in results_list:
            for rank, doc in enumerate(result_set, start=1):
                # Use document ID or content hash as key
                doc_id = (
                    doc.get("id")
                    or doc.get("document_id")
                    or str(hash(doc.get("content", "")[:100]))
                )

                # RRF score contribution
                rrf_score = 1.0 / (k + rank)

                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_map[doc_id] = doc

        # Sort by fusion score
        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build final result list
        fused_results = []
        for doc_id, score in ranked:
            doc = dict(doc_map[doc_id])
            doc["fusion_score"] = round(score, 4)
            fused_results.append(doc)

        logger.debug(
            "rrf_fusion",
            input_sets=len(results_list),
            unique_docs=len(fused_results),
        )

        return fused_results

    async def _execute_fusion_rag_query(
        self, query_text: str, entities: dict[str, list[str]]
    ) -> dict[str, Any]:
        """
        Execute Fusion RAG: expand query + parallel search + RRF merge.

        This improves recall for ambiguous queries by:
        1. Generating query variations
        2. Searching with each variation
        3. Merging results using Reciprocal Rank Fusion

        Args:
            query_text: Original user query
            entities: Extracted entities (for filtering)

        Returns:
            Fusion RAG results with merged documents
        """
        rag = self._get_rag()
        if rag is None:
            return {"error": "RAG not available", "documents": []}

        try:
            # Step 1: Expand query
            queries = await self._expand_query(query_text, num_variations=3)

            # Step 2: Parallel search for all variations
            tasks = [rag.retrieve(q, top_k=5) for q in queries]
            try:
                search_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.error("fusion_rag_timeout", query=query_text[:50])
                return {"error": "Fusion RAG timeout", "documents": []}
            
            all_results = []
            for q, results in zip(queries, search_results, strict=True):
                if isinstance(results, Exception):
                    logger.warning("fusion_search_failed", query=q[:30], error=str(results))
                else:
                    all_results.append(results)

            if not all_results:
                return {"error": "All fusion searches failed", "documents": []}

            # Step 3: Merge with RRF
            fused_docs = self._reciprocal_rank_fusion(all_results)

            return {
                "type": "fusion_search",
                "original_query": query_text,
                "query_variations": queries,
                "documents": fused_docs[:10],  # Top 10 after fusion
                "fusion_stats": {
                    "num_queries": len(queries),
                    "total_results": sum(len(r) for r in all_results),
                    "unique_after_fusion": len(fused_docs),
                },
            }

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("fusion_rag_failed", error=str(e))
            # Fallback to standard RAG
            return await self._execute_rag_query(query_text, entities)

    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================

    async def _generate_response(
        self,
        query_text: str,
        results: dict[str, Any],
        classification: QueryClassification,
    ) -> str:
        """
        Generate natural language response from results.

        Uses Opinion-Based Prompting for +30-50% improvement in context adherence.
        Research shows reformulating questions as "what does X say" improves
        accuracy from 33% → 73% (120% improvement).
        """
        llm = await self._get_llm()

        if llm is None:
            return self._format_results_as_text(results)

        # Build context from results
        context_parts = []

        if results.get("graph_results"):
            context_parts.append(
                "Knowledge Graph Results:\n"
                f"{self._format_graph_results(results['graph_results'])}"
            )

        if results.get("rag_results") and results["rag_results"].get("documents"):
            context_parts.append(
                f"Documentation:\n{self._format_rag_results(results['rag_results'])}"
            )

        context = "\n\n".join(context_parts)

        # Use Opinion-Based Prompting for better context adherence
        # This technique forces LLM to prioritize provided context over training data
        formatted = self._prompt_formatter.format_rag_prompt(
            query=query_text,
            context=context,
            source_name="TWS documentation and knowledge base",
            strict_mode=True,  # Enforce context-only responses
            language="en",  # Default to English
        )

        try:
            # Generate with opinion-based prompts
            # Expected improvement: +30-50% accuracy, -60% hallucination rate
            return await asyncio.wait_for(
                llm.generate(f"{formatted['system']}\n\n{formatted['user']}"),
                timeout=30.0,
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("response_generation_failed", error=str(e))
            return self._format_results_as_text(results)

    def _format_graph_results(self, results: dict[str, Any]) -> str:
        """Format graph results as text."""
        result_type = results.get("type", "unknown")

        if result_type == "dependency_chain":
            chain = results.get("chain", [])
            if not chain:
                return f"Job {results.get('job')} has no dependencies."

            deps = [f"  → {c['to']}" for c in chain]
            return f"Dependencies for {results.get('job')}:\n" + "\n".join(deps)

        if result_type == "impact_analysis":
            count = results.get("affected_count", 0)
            jobs = results.get("affected_jobs", [])
            severity = results.get("severity", "unknown")

            if count == 0:
                return f"Job {results.get('job')} has no downstream dependents."

            return (
                f"Impact Analysis for {results.get('job')}:\n"
                f"- Affected jobs: {count}\n"
                f"- Severity: {severity}\n"
                f"- Jobs: {', '.join(jobs[:10])}"
            )

        if result_type == "resource_conflict":
            conflicts = results.get("conflicts", [])
            if not conflicts:
                return (
                    f"Jobs {results.get('job_a')} and {results.get('job_b')} "
                    f"have no resource conflicts."
                )

            conflict_list = [
                f"  - {c['name']} ({c['conflict_type']})" for c in conflicts
            ]
            return (
                "Resource conflicts between "
                f"{results.get('job_a')} and {results.get('job_b')}:\n"
                + "\n".join(conflict_list)
            )

        if result_type == "critical_jobs":
            jobs = results.get("jobs", [])
            if not jobs:
                return "No critical jobs identified."

            lines = [
                (
                    f"  {i + 1}. {j['job']} "
                    f"(centrality: {j['centrality_score']}, "
                    f"risk: {j['risk_level']})"
                )
                for i, j in enumerate(jobs[:10])
            ]
            return "Most critical jobs:\n" + "\n".join(lines)

        return str(results)

    def _format_rag_results(self, results: dict[str, Any]) -> str:
        """Format RAG results as text."""
        docs = results.get("documents", [])
        if not docs:
            return "No relevant documentation found."

        formatted = []
        for i, doc in enumerate(docs[:3], 1):
            text = doc.get("text", doc.get("content", ""))[:500]
            source = doc.get("source", doc.get("metadata", {}).get("source", "unknown"))
            formatted.append(f"[{i}] {source}:\n{text}...")

        return "\n\n".join(formatted)

    def _format_results_as_text(self, results: dict[str, Any]) -> str:
        """Format all results as plain text (fallback)."""
        parts = []

        if results.get("graph_results"):
            parts.append(self._format_graph_results(results["graph_results"]))

        if results.get("rag_results"):
            parts.append(self._format_rag_results(results["rag_results"]))

        return "\n\n".join(parts) if parts else "No results found."

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_hybrid_rag: HybridRAG | None = None
_hybrid_rag_lock = asyncio.Lock()

async def get_hybrid_rag() -> HybridRAG:
    """Get or create the singleton HybridRAG instance."""
    global _hybrid_rag
    async with _hybrid_rag_lock:
        if _hybrid_rag is None:
            _hybrid_rag = HybridRAG()
        return _hybrid_rag

async def hybrid_query(
    query_text: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Execute a hybrid KG+RAG query.

    Convenience function for one-off queries.

    Args:
        query_text: User query
        context: Optional context

    Returns:
        Query results with graph_results, rag_results, and response
    """
    rag = await get_hybrid_rag()
    return await rag.query(query_text, context)
