# pylint: skip-file
# mypy: ignore-errors
"""
Embedding-Based Intent Router.

v5.3.17 - Fast intent classification using embedding similarity.

This router classifies queries by comparing embeddings to pre-computed
intent examples, avoiding the latency of LLM-based classification.

Architecture:
    Query → Embedding → Similarity Search → Intent Classification
                                ↓
                        Confidence Check → [High: Use Intent | Low: LLM Fallback]

Benefits:
- 10-20ms classification vs 200-500ms for LLM
- Deterministic and consistent results
- No token costs for classification
- Graceful fallback to LLM when uncertain

Usage:
    router = EmbeddingRouter()
    result = await router.classify("Quais as dependências do job XPTO?")
    # result.intent = "dependency_chain"
    # result.confidence = 0.92
"""

import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class RouterIntent(str, Enum):
    """Supported intents for classification."""

    # Graph-oriented intents
    DEPENDENCY_CHAIN = "dependency_chain"
    IMPACT_ANALYSIS = "impact_analysis"
    RESOURCE_CONFLICT = "resource_conflict"
    CRITICAL_JOBS = "critical_jobs"
    JOB_LINEAGE = "job_lineage"

    # RAG-oriented intents
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    TROUBLESHOOTING = "troubleshooting"
    ERROR_LOOKUP = "error_lookup"

    # Hybrid intents
    ROOT_CAUSE = "root_cause"
    JOB_DETAILS = "job_details"

    # General
    GENERAL = "general"
    GREETING = "greeting"
    CHITCHAT = "chitchat"


# Pre-defined examples for each intent (bilingual: PT/EN)
INTENT_EXAMPLES = {
    RouterIntent.DEPENDENCY_CHAIN: [
        "Quais são as dependências do job?",
        "What are the dependencies of this job?",
        "De que esse job depende?",
        "What does this job depend on?",
        "Lista os predecessores",
        "Show me the predecessor jobs",
        "Cadeia de dependências",
        "Upstream jobs",
        "Jobs que precisam rodar antes",
    ],
    RouterIntent.IMPACT_ANALYSIS: [
        "O que acontece se esse job falhar?",
        "What happens if this job fails?",
        "Qual o impacto da falha?",
        "Impact analysis",
        "Quais jobs serão afetados?",
        "Which jobs will be affected?",
        "Downstream impact",
        "Análise de impacto",
        "Consequências da falha",
    ],
    RouterIntent.RESOURCE_CONFLICT: [
        "Esses jobs podem rodar juntos?",
        "Can these jobs run together?",
        "Conflito de recursos",
        "Resource conflict",
        "Compartilham recursos?",
        "Do they share resources?",
        "Concorrência entre jobs",
        "Recursos exclusivos",
    ],
    RouterIntent.CRITICAL_JOBS: [
        "Quais são os jobs mais críticos?",
        "What are the most critical jobs?",
        "Jobs de alto risco",
        "High risk jobs",
        "Gargalos do sistema",
        "System bottlenecks",
        "Jobs importantes",
        "Centralidade no grafo",
    ],
    RouterIntent.JOB_LINEAGE: [
        "Mostra a linhagem completa",
        "Show full lineage",
        "Hierarquia do job",
        "Job hierarchy",
        "Árvore de dependências",
        "Dependency tree",
        "Ancestrais e descendentes",
    ],
    RouterIntent.DOCUMENTATION: [
        "Como configuro isso?",
        "How do I configure this?",
        "Onde está a documentação?",
        "Where is the documentation?",
        "Passo a passo para",
        "Step by step guide",
        "Manual de instalação",
        "Setup instructions",
        "Como faço para",
    ],
    RouterIntent.EXPLANATION: [
        "O que é isso?",
        "What is this?",
        "Explica o conceito",
        "Explain the concept",
        "O que significa",
        "What does it mean",
        "Para que serve",
        "What is it used for",
    ],
    RouterIntent.TROUBLESHOOTING: [
        "Como resolver esse erro?",
        "How to fix this error?",
        "Não está funcionando",
        "It's not working",
        "Problema com",
        "Problem with",
        "Falha ao executar",
        "Failed to execute",
        "Erro ao rodar",
    ],
    RouterIntent.ERROR_LOOKUP: [
        "O que significa o erro RC 12?",
        "What does error RC 12 mean?",
        "Código de erro",
        "Error code",
        "BATCHMAN error",
        "Mensagem de erro",
        "Return code",
        "Significado do erro",
    ],
    RouterIntent.ROOT_CAUSE: [
        "Por que o job falhou?",
        "Why did the job fail?",
        "Causa raiz",
        "Root cause",
        "Motivo da falha",
        "Reason for failure",
        "Investigar falha",
        "Investigate failure",
    ],
    RouterIntent.JOB_DETAILS: [
        "Me conta sobre o job",
        "Tell me about this job",
        "Informações do job",
        "Job information",
        "Status do job",
        "Job status",
        "Detalhes do job",
        "Job details",
    ],
    RouterIntent.GREETING: [
        "Olá",
        "Hello",
        "Oi",
        "Hi",
        "Bom dia",
        "Good morning",
        "Boa tarde",
        "Good afternoon",
    ],
    RouterIntent.CHITCHAT: [
        "Como você está?",
        "How are you?",
        "Tudo bem?",
        "What's up?",
        "Obrigado",
        "Thank you",
        "Valeu",
        "Thanks",
    ],
}


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    intent: RouterIntent
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)
    used_llm_fallback: bool = False
    classification_time_ms: float = 0.0


class EmbeddingRouter:
    """
    Fast intent classification using embedding similarity.

    Pre-computes embeddings for intent examples at initialization,
    then classifies new queries by finding the most similar examples.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        use_llm_fallback: bool = True,
        cache_dir: str | None = None,
    ):
        """
        Initialize the embedding router.

        Args:
            confidence_threshold: Minimum confidence to accept classification
            use_llm_fallback: Fall back to LLM for low confidence
            cache_dir: Directory to cache pre-computed embeddings
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._embedding_model = None
        self._intent_embeddings: dict[RouterIntent, list[np.ndarray]] = {}
        self._initialized = False

        logger.info(
            "embedding_router_created",
            threshold=confidence_threshold,
            fallback=use_llm_fallback,
        )

    def _get_embedding_model(self):
        """Get embedding model (lazy load)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.getenv(
                    "ROUTER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
                self._embedding_model = SentenceTransformer(model_name)
                logger.info("router_embedding_model_loaded: %s", model_name)
            except ImportError:
                logger.error("sentence_transformers_not_available")
                raise
        return self._embedding_model

    def initialize(self):
        """Pre-compute embeddings for all intent examples."""
        if self._initialized:
            return

        import time

        start = time.perf_counter()

        # Try to load from cache
        if self._load_from_cache():
            self._initialized = True
            logger.info(
                "intent_embeddings_loaded_from_cache",
                intents=len(self._intent_embeddings),
            )
            return

        # Compute embeddings
        model = self._get_embedding_model()

        for intent, examples in INTENT_EXAMPLES.items():
            embeddings = model.encode(examples, convert_to_numpy=True)
            self._intent_embeddings[intent] = [emb for emb in embeddings]

        # Save to cache
        self._save_to_cache()

        self._initialized = True
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "intent_embeddings_computed",
            intents=len(self._intent_embeddings),
            examples=sum(len(v) for v in self._intent_embeddings.values()),
            time_ms=elapsed,
        )

    def _load_from_cache(self) -> bool:
        """Try to load embeddings from cache."""
        if not self.cache_dir:
            return False

        cache_file = self.cache_dir / "intent_embeddings.npz"
        if not cache_file.exists():
            return False

        try:
            data = np.load(cache_file, allow_pickle=True)
            for intent in RouterIntent:
                if intent.value in data:
                    self._intent_embeddings[intent] = list(data[intent.value])
            return True
        except Exception as e:
            logger.warning("cache_load_failed: %s", e)
            return False

    def _save_to_cache(self):
        """Save embeddings to cache."""
        if not self.cache_dir:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "intent_embeddings.npz"

            data = {
                intent.value: np.array(embeddings)
                for intent, embeddings in self._intent_embeddings.items()
            }
            np.savez(cache_file, **data)
            logger.info("intent_embeddings_cached")
        except Exception as e:
            logger.warning("cache_save_failed: %s", e)

    async def classify(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """
        Classify query intent using embedding similarity.

        Args:
            query: User query to classify
            context: Optional context (not used in embedding mode)

        Returns:
            ClassificationResult with intent and confidence
        """
        import time

        start = time.perf_counter()

        # Ensure initialized
        if not self._initialized:
            self.initialize()

        # Get query embedding
        model = self._get_embedding_model()
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Compute similarity to all intent examples
        intent_scores: dict[RouterIntent, float] = {}

        for intent, example_embeddings in self._intent_embeddings.items():
            # Compute cosine similarity to each example
            similarities = [
                self._cosine_similarity(query_embedding, ex_emb)
                for ex_emb in example_embeddings
            ]
            # Take max similarity as intent score
            intent_scores[intent] = max(similarities) if similarities else 0.0

        # Get best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]

        elapsed = (time.perf_counter() - start) * 1000

        # Check confidence
        if best_score < self.confidence_threshold:
            if self.use_llm_fallback:
                # Fall back to LLM
                return await self._llm_classify(query, intent_scores, elapsed)
            # Return low confidence result
            best_intent = RouterIntent.GENERAL

        return ClassificationResult(
            intent=best_intent,
            confidence=best_score,
            all_scores={k.value: v for k, v in intent_scores.items()},
            used_llm_fallback=False,
            classification_time_ms=elapsed,
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _llm_classify(
        self,
        query: str,
        embedding_scores: dict[RouterIntent, float],
        embedding_time_ms: float,
    ) -> ClassificationResult:
        """Fall back to LLM for classification."""
        import time

        start = time.perf_counter()

        try:
            from resync.services.llm_service import get_llm_service

            llm = await get_llm_service()

            # Create prompt with top candidates from embedding
            top_candidates = sorted(
                embedding_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]

            candidates_str = ", ".join(c[0].value for c in top_candidates)

            prompt = f"""Classify this query into ONE of these intents:
{candidates_str}, general

Query: {query}

Respond with ONLY the intent name, nothing else."""

            response = await llm.generate(prompt, max_tokens=20)
            response = response.strip().lower()

            # Parse response
            for intent in RouterIntent:
                if intent.value in response:
                    elapsed = (time.perf_counter() - start) * 1000
                    return ClassificationResult(
                        intent=intent,
                        confidence=0.8,  # LLM gives moderate confidence
                        all_scores={k.value: v for k, v in embedding_scores.items()},
                        used_llm_fallback=True,
                        classification_time_ms=embedding_time_ms + elapsed,
                    )

        except Exception as e:
            logger.warning("llm_fallback_failed: %s", e)

        # Default to general
        return ClassificationResult(
            intent=RouterIntent.GENERAL,
            confidence=0.5,
            all_scores={k.value: v for k, v in embedding_scores.items()},
            used_llm_fallback=True,
            classification_time_ms=embedding_time_ms,
        )

    def get_info(self) -> dict[str, Any]:
        """Get router information."""
        return {
            "initialized": self._initialized,
            "confidence_threshold": self.confidence_threshold,
            "use_llm_fallback": self.use_llm_fallback,
            "intents_count": len(self._intent_embeddings),
            "examples_count": sum(len(v) for v in self._intent_embeddings.values()),
            "model_loaded": self._embedding_model is not None,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_embedding_router: EmbeddingRouter | None = None


def get_embedding_router() -> EmbeddingRouter:
    """Get or create embedding router instance."""
    global _embedding_router
    if _embedding_router is None:
        _embedding_router = EmbeddingRouter(
            cache_dir=os.getenv("ROUTER_CACHE_DIR", os.path.join(tempfile.gettempdir(), "resync", "router_cache")),
        )
    return _embedding_router


async def classify_intent(
    query: str,
    context: dict[str, Any] | None = None,
) -> ClassificationResult:
    """
    Convenience function for intent classification.

    Args:
        query: User query
        context: Optional context

    Returns:
        ClassificationResult
    """
    router = get_embedding_router()
    return await router.classify(query, context)


__all__ = [
    "RouterIntent",
    "ClassificationResult",
    "EmbeddingRouter",
    "INTENT_EXAMPLES",
    "get_embedding_router",
    "classify_intent",
]
