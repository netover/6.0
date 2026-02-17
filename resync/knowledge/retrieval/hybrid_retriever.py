"""
Hybrid Retriever for Resync v5.2.3.24

Combines multiple retrieval strategies for optimal TWS job search:
1. Vector Search - Semantic similarity (concepts like "slow jobs")
2. BM25 Search - Keyword/exact match (job codes like "AWSBH001")
3. Cross-Encoder Reranking - Final quality filtering

This is CRITICAL for TWS operators who need to find exact job codes,
not just semantically similar content.

v5.2.3.22: Added dynamic weight adjustment based on query type.
v5.2.3.23: Added field boosting for TWS-specific metadata.
v5.2.3.24: Added query classification cache and performance metrics.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import math
import os
import re
import time
import asyncio
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

try:
    from filelock import FileLock, Timeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False

from .reranker_interface import (
    IReranker,
    RerankGatingConfig,
    RerankGatingPolicy,
    create_reranker,
)

logger = logging.getLogger(__name__)

INDEX_STORAGE_PATH = os.environ.get(
    "BM25_INDEX_PATH",
    os.path.join("data", "bm25_index.bin.gz")
)


# =============================================================================
# BM25 IMPLEMENTATION
# =============================================================================


@dataclass
class BM25Index:
    """
    BM25 (Best Match 25) index for keyword-based retrieval.

    Optimized for TWS job names and technical identifiers.

    v5.2.3.23: Added field boosting for TWS-specific metadata.
    """

    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization

    # v5.2.3.23: Field boost weights for TWS domain
    # Higher boost = more importance in ranking
    field_boosts: dict[str, float] = field(default_factory=lambda: {
        "job_name": 4.0,       # Job name is critical for exact match
        "error_code": 3.5,     # Error codes (RC, ABEND) are high priority
        "workstation": 3.0,    # Workstation/server names
        "job_stream": 2.5,     # Job stream/schedule names
        "message_id": 2.5,     # TWS message IDs (EQQQ...)
        "resource": 2.0,       # Resource names
        "title": 1.5,          # Document titles
        "content": 1.0,        # Default content weight
    })

    # Index storage
    documents: list[dict[str, Any]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0

    # Inverted index: term -> [(doc_idx, term_freq), ...]
    inverted_index: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    # Document frequency: term -> num_docs_containing_term
    doc_freq: dict[str, int] = field(default_factory=dict)

    # v5.2.3.23: TWS-specific patterns for error code extraction
    ERROR_CODE_PATTERNS = [
        re.compile(r"RC[=:]\s*(\d+)", re.IGNORECASE),
        re.compile(r"ABEND\s+([A-Z0-9]+)", re.IGNORECASE),
        re.compile(r"EQQQ(\w+)", re.IGNORECASE),
        re.compile(r"AWSB(\w+)", re.IGNORECASE),
        re.compile(r"(?:error|erro)[:\s]+(\w+)", re.IGNORECASE),
    ]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.

        Optimized for TWS job names:
        - Preserves underscores and hyphens within tokens
        - Case-insensitive
        - Handles alphanumeric codes

        v5.2.3.22: Enhanced for TWS patterns (RC codes, ABEND, message IDs)
        """
        if not text:
            return []

        # Lowercase
        text = text.lower()

        # === v5.2.3.22: PRE-PROCESS TWS PATTERNS ===

        # Normalize RC codes: "RC=8" -> "rc_8 rc8" for better matching
        text = re.sub(r"rc[=:]\s*(\d+)", r"rc_\1 rc\1", text)

        # Normalize ABEND codes: "ABEND S0C7" -> "abend_s0c7 s0c7"
        text = re.sub(r"abend\s+([a-z0-9]+)", r"abend_\1 \1", text)

        # Keep EQQQ message IDs together: "EQQQ001I" stays as-is
        # Keep AWSB job prefixes: "AWSBH001" stays as-is

        # === END TWS PATTERNS ===

        # Split on whitespace and common separators, but preserve _ and -
        # This keeps job names like "AWSBH001_BACKUP" as one token
        tokens = re.findall(r"[a-z0-9_\-]+", text)

        # Also add individual parts for compound names
        expanded = []
        for token in tokens:
            expanded.append(token)
            # Split on underscore to also index parts
            if "_" in token:
                expanded.extend(token.split("_"))
            # Split on hyphen too
            if "-" in token:
                expanded.extend(token.split("-"))

        return [t for t in expanded if len(t) >= 2]

    def build_index(self, documents: list[dict[str, Any]], text_field: str = "content") -> None:
        """
        Build BM25 index from documents with field boosting.

        v5.2.3.23: Implements field-specific boost weights for TWS domain.

        Args:
            documents: List of documents with text content
            text_field: Field name containing searchable text
        """
        self.documents = documents
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = []

        total_length = 0

        for doc_idx, doc in enumerate(documents):
            # Track term frequencies with boost applied
            boosted_term_freqs: dict[str, float] = defaultdict(float)

            # Get metadata
            metadata = doc.get("metadata", {}) or {}

            # === INDEX CONTENT FIELD ===
            text = doc.get(text_field, "") or ""
            if isinstance(text, dict):
                text = str(text.get("text", text.get("content", "")))

            content_boost = self.field_boosts.get("content", 1.0)
            for token in self._tokenize(text):
                boosted_term_freqs[token] += content_boost

            # === INDEX BOOSTED METADATA FIELDS ===

            # Job name (highest boost)
            job_name = metadata.get("job_name", "") or ""
            if job_name:
                job_boost = self.field_boosts.get("job_name", 4.0)
                for token in self._tokenize(job_name):
                    boosted_term_freqs[token] += job_boost

            # Workstation
            workstation = metadata.get("workstation", "") or ""
            if workstation:
                ws_boost = self.field_boosts.get("workstation", 3.0)
                for token in self._tokenize(workstation):
                    boosted_term_freqs[token] += ws_boost

            # Job stream
            job_stream = metadata.get("job_stream", "") or ""
            if job_stream:
                stream_boost = self.field_boosts.get("job_stream", 2.5)
                for token in self._tokenize(job_stream):
                    boosted_term_freqs[token] += stream_boost

            # Resource
            resource = metadata.get("resource", "") or ""
            if resource:
                res_boost = self.field_boosts.get("resource", 2.0)
                for token in self._tokenize(resource):
                    boosted_term_freqs[token] += res_boost

            # Title
            title = metadata.get("title", "") or doc.get("title", "") or ""
            if title:
                title_boost = self.field_boosts.get("title", 1.5)
                for token in self._tokenize(title):
                    boosted_term_freqs[token] += title_boost

            # === v5.2.3.23: EXTRACT AND BOOST ERROR CODES ===
            full_text = f"{text} {job_name} {workstation}"
            error_codes = self._extract_error_codes(full_text)
            if error_codes:
                error_boost = self.field_boosts.get("error_code", 3.5)
                for code in error_codes:
                    for token in self._tokenize(code):
                        boosted_term_freqs[token] += error_boost

            # === v5.2.3.23: EXTRACT AND BOOST MESSAGE IDS ===
            message_ids = self._extract_message_ids(full_text)
            if message_ids:
                msg_boost = self.field_boosts.get("message_id", 2.5)
                for msg_id in message_ids:
                    for token in self._tokenize(msg_id):
                        boosted_term_freqs[token] += msg_boost

            # Calculate document length (sum of boosted frequencies)
            doc_length = sum(boosted_term_freqs.values())
            self.doc_lengths.append(int(doc_length))
            total_length += doc_length

            # Update inverted index with boosted frequencies
            seen_terms = set()
            for term, boosted_freq in boosted_term_freqs.items():
                self.inverted_index[term].append((doc_idx, boosted_freq))
                if term not in seen_terms:
                    self.doc_freq[term] += 1
                    seen_terms.add(term)

        # Calculate average document length
        self.avg_doc_length = total_length / len(documents) if documents else 0.0

        logger.info(
            f"BM25 index built with field boosting: {len(documents)} docs, "
            f"{len(self.inverted_index)} unique terms, "
            "avg_length={self.avg_doc_length:.1f}"
        )

    def _extract_error_codes(self, text: str) -> list[str]:
        """
        Extract TWS error codes from text.

        v5.2.3.23: Identifies RC codes, ABEND codes, and other error patterns.
        """
        codes = []
        for pattern in self.ERROR_CODE_PATTERNS:
            matches = pattern.findall(text)
            codes.extend(matches)
        return codes

    def _extract_message_ids(self, text: str) -> list[str]:
        """
        Extract TWS message IDs from text.

        v5.2.3.23: Identifies EQQQ and AWSB message patterns.
        """
        # Pattern for TWS message IDs (EQQQ001I, AWSBH001, etc.)
        pattern = re.compile(r"\b(EQQQ\w+|AWSB\w+|IEF\w+)\b", re.IGNORECASE)
        return pattern.findall(text)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Search index using BM25 scoring.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.documents)
        scores = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            # IDF calculation
            df = self.doc_freq[term]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # Score each document containing this term
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]

                # BM25 score formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)

                scores[doc_idx] += idf * numerator / denominator

        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save(self, path: str) -> bool:
        """
        Persist BM25 index to disk with compression.

        Args:
            path: Path to save the index file (.bin.gz)

        Returns:
            True if saved successfully, False otherwise
        """
        if not FILELOCK_AVAILABLE:
            logger.warning("filelock not available, skipping BM25 index save")
            return False

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            lock_path = f"{path}.lock"
            lock = FileLock(lock_path, timeout=10)

            try:
                with lock.acquire(timeout=10):
                    import joblib

                    with gzip.open(path, "wb") as f:
                        joblib.dump(self, f)

                    logger.info(
                        "bm25_index_saved",
                        path=path,
                        size_bytes=os.path.getsize(path),
                        num_docs=len(self.documents),
                        num_terms=len(self.inverted_index)
                    )
                    return True

            except Timeout:
                logger.error("bm25_index_save_timeout", path=path)
                return False

        except Exception as e:
            logger.error("bm25_index_save_failed", error=str(e), path=path)
            return False

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """
        Load BM25 index from disk with auto-recovery.

        Args:
            path: Path to the index file

        Returns:
            BM25Index instance (loaded or empty)
        """
        if not os.path.exists(path):
            logger.info("bm25_index_not_found_will_build", path=path)
            return cls()

        if not FILELOCK_AVAILABLE:
            logger.warning("filelock not available, building fresh index")
            return cls()

        lock_path = f"{path}.lock"

        try:
            import joblib

            lock = FileLock(lock_path, timeout=5)

            try:
                with lock.acquire(timeout=5):
                    with gzip.open(path, "rb") as f:
                        index = joblib.load(f)

                    logger.info(
                        "bm25_index_loaded",
                        path=path,
                        size_bytes=os.path.getsize(path),
                        num_docs=len(index.documents),
                        num_terms=len(index.inverted_index)
                    )
                    return index

            except Timeout:
                logger.warning("bm25_index_locked_using_empty", path=path)
                return cls()

        except (EOFError, OSError, ImportError, gzip.BadGzipFile) as e:
            logger.warning(
                "bm25_index_corrupted_rebuilding",
                error=str(e),
                path=path
            )
            try:
                if os.path.exists(path):
                    os.remove(path)
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception as cleanup_error:
                logger.error("bm25_index_cleanup_failed", error=str(cleanup_error))

            return cls()

        except MemoryError:
            logger.error(
                "bm25_index_oom_using_empty",
                path=path
            )
            return cls()

        except Exception as e:
            logger.warning(
                "bm25_index_load_failed_unknown",
                error=str(e),
                error_type=type(e).__name__,
                path=path
            )
            return cls()


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================


class Embedder(Protocol):
    """Protocol for embedding text into vectors."""

    def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Protocol for vector storage."""

    def query(
        self,
        vector: list[float],
        top_k: int,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        ef_search: int | None = None,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]: ...

    def get_all_documents(
        self, collection: str | None = None, limit: int = 10000
    ) -> list[dict[str, Any]]: ...


# =============================================================================
# v5.2.3.24: QUERY CLASSIFICATION TYPES AND METRICS
# =============================================================================


class QueryType(str, Enum):
    """Query classification types for weight selection."""

    EXACT_MATCH = "exact_match"  # Job codes, RC codes, etc.
    SEMANTIC = "semantic"  # How-to, explanations, concepts
    MIXED = "mixed"  # Both exact and semantic elements
    DEFAULT = "default"  # Unclear classification


@dataclass
class QueryClassificationResult:
    """Result of query classification with metadata."""

    query_type: QueryType
    vector_weight: float
    bm25_weight: float
    cached: bool = False
    classification_time_ms: float = 0.0


@dataclass
class QueryClassificationCache:
    """
    LRU Cache for query classifications.

    v5.2.3.24: Prevents redundant pattern matching for repeated queries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: OrderedDict[str, tuple[QueryClassificationResult, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def _make_key(self, query: str) -> str:
        """Create cache key from query."""
        normalized = " ".join(query.lower().split())[:200]
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    def get(self, query: str) -> QueryClassificationResult | None:
        """Get classification from cache if valid."""
        key = self._make_key(query)

        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self._ttl_seconds:
            del self._cache[key]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)

        # Return with cached flag
        return QueryClassificationResult(
            query_type=result.query_type,
            vector_weight=result.vector_weight,
            bm25_weight=result.bm25_weight,
            cached=True,
            classification_time_ms=0.0,
        )

    def put(self, query: str, result: QueryClassificationResult) -> None:
        """Store classification in cache."""
        key = self._make_key(query)

        # Remove oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


# =============================================================================
# HYBRID RETRIEVER CONFIGURATION
# =============================================================================


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""

    # Weight for vector search (0-1)
    vector_weight: float = 0.5

    # Weight for BM25 search (0-1)
    bm25_weight: float = 0.5

    # v5.2.3.22: Enable automatic weight adjustment based on query type
    auto_weight: bool = True

    # Number of candidates to fetch from each retriever before fusion
    candidate_multiplier: int = 3

    # Enable cross-encoder reranking
    enable_reranking: bool = True

    # Number of final results after reranking
    rerank_top_k: int = 5

    # Minimum score threshold for reranked results
    rerank_threshold: float = 0.3

    # BM25 index settings
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # v5.2.3.23: Field boost weights for BM25 indexing
    field_boosts: dict[str, float] = field(default_factory=lambda: {
        "job_name": 4.0,
        "error_code": 3.5,
        "workstation": 3.0,
        "job_stream": 2.5,
        "message_id": 2.5,
        "resource": 2.0,
        "title": 1.5,
        "content": 1.0,
    })

    # v5.2.3.24: Cache and metrics configuration
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour
    metrics_enabled: bool = True

    # v5.9.9: Rerank gating for CPU optimization
    # Only activate reranking when retrieval confidence is low
    rerank_gating_enabled: bool = True
    rerank_score_low_threshold: float = 0.35  # Activate if top1 < threshold
    rerank_margin_threshold: float = 0.05     # Activate if top1-top2 < margin
    rerank_max_candidates: int = 10           # Max docs to rerank

    @classmethod
    def from_settings(cls) -> HybridRetrieverConfig:
        """Create config from application settings."""
        try:
            from resync.settings import settings

            # v5.2.3.23: Load field boosts from settings if available
            field_boosts = {
                "job_name": getattr(settings, "hybrid_boost_job_name", 4.0),
                "error_code": getattr(settings, "hybrid_boost_error_code", 3.5),
                "workstation": getattr(settings, "hybrid_boost_workstation", 3.0),
                "job_stream": getattr(settings, "hybrid_boost_job_stream", 2.5),
                "message_id": getattr(settings, "hybrid_boost_message_id", 2.5),
                "resource": getattr(settings, "hybrid_boost_resource", 2.0),
                "title": getattr(settings, "hybrid_boost_title", 1.5),
                "content": getattr(settings, "hybrid_boost_content", 1.0),
            }

            return cls(
                vector_weight=getattr(settings, "hybrid_vector_weight", 0.5),
                bm25_weight=getattr(settings, "hybrid_bm25_weight", 0.5),
                auto_weight=getattr(settings, "hybrid_auto_weight", True),
                field_boosts=field_boosts,
                cache_enabled=getattr(settings, "hybrid_cache_enabled", True),
                cache_max_size=getattr(settings, "hybrid_cache_max_size", 1000),
                cache_ttl_seconds=getattr(settings, "hybrid_cache_ttl_seconds", 3600),
                metrics_enabled=getattr(settings, "hybrid_metrics_enabled", True),
                # v5.9.9: Gating settings
                rerank_gating_enabled=getattr(settings, "rerank_gating_enabled", True),
                rerank_score_low_threshold=getattr(settings, "rerank_score_low_threshold", 0.35),
                rerank_margin_threshold=getattr(settings, "rerank_margin_threshold", 0.05),
                rerank_max_candidates=getattr(settings, "rerank_max_candidates", 10),
            )
        except ImportError:
            return cls()


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25.

    This is critical for TWS operations:
    - Vector search finds semantically related content ("jobs that failed yesterday")
    - BM25 finds exact matches ("AWSBH001_BACKUP")

    Uses Reciprocal Rank Fusion (RRF) to combine results.

    v5.2.3.22: Added dynamic weight adjustment based on query type.
    v5.2.3.23: Added field boosting and more TWS patterns.
    """

    # v5.2.3.23: Pre-compiled patterns for exact match detection
    # These patterns identify TWS-specific codes that benefit from exact BM25 matching
    EXACT_MATCH_PATTERNS = [
        # Job name patterns
        re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9_]*\b"),  # AWSBH001, BATCH001 (min 2 digits)
        re.compile(r"\bAWSB[A-Z0-9]+\b", re.IGNORECASE),  # AWSBH001 (TWS job prefix)
        re.compile(r"\b[A-Z]{2,}_[A-Z0-9_]+_\d+\b"),  # BATCH_DAILY_001 (compound with number)
        re.compile(r"\bJOB[=:]\s*[A-Z0-9_]+", re.IGNORECASE),  # JOB=X, JOB: X (explicit job ref)

        # Error codes
        re.compile(r"\bRC[=:]\s*\d+\b", re.IGNORECASE),  # RC=8, RC: 12
        re.compile(r"\bABEND\s+[A-Z0-9]+\b", re.IGNORECASE),  # ABEND S0C7
        re.compile(r"\bS[0-9A-F]{3}\b"),  # S0C7, S0C4 (system ABEND codes)
        re.compile(r"\bU\d{4}\b"),  # U0001 (user ABEND codes)
        re.compile(r"\bCC[=:]\s*\d+\b", re.IGNORECASE),  # CC=4, CC=8 (condition codes)

        # Infrastructure codes
        re.compile(r"\b[A-Z]{2}\d{3,}\b"),  # WS001, SRV123 (2 letters + 3+ digits)
        re.compile(r"\bLPAR\d+\b", re.IGNORECASE),  # LPAR1, LPAR2
        re.compile(r"\bSYSPLEX\w+\b", re.IGNORECASE),  # SYSPLEX names

        # TWS message IDs
        re.compile(r"\bEQQQ[A-Z0-9]+\b", re.IGNORECASE),  # EQQQ001I (TWS messages)
        re.compile(r"\bIEF\d{3}[A-Z]\b", re.IGNORECASE),  # IEF450I (JES messages)
        re.compile(r"\bIKJ\d{5}[A-Z]\b", re.IGNORECASE),  # TSO messages
        re.compile(r"\bCSV\d{3}[A-Z]\b", re.IGNORECASE),  # CSV messages

        # Dataset/file patterns
        re.compile(r"\b[A-Z]+\.[A-Z0-9]+\.[A-Z0-9]+\b"),  # HLQ.MLQ.LLQ (dataset names)
        re.compile(r"\bDSN[=:]\s*\S+", re.IGNORECASE),  # DSN=...

        # Time windows
        re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # 14:30, 14:30:00 (time)
        re.compile(r"\b(?:ODATE|SCHED_TIME)[=:]\s*\S+", re.IGNORECASE),  # ODATE=...
    ]

    # v5.2.3.23: Patterns indicating semantic/conceptual queries
    SEMANTIC_PATTERNS = [
        # Questions (PT/EN)
        re.compile(r"\b(como|how|why|por\s*que|quando|when|onde|where)\b", re.IGNORECASE),

        # Actions/Help requests
        re.compile(r"\b(resolver|fix|solve|solucionar|corrigir)\b", re.IGNORECASE),
        re.compile(r"\b(configurar|configure|setup|instalar|install)\b", re.IGNORECASE),
        re.compile(r"\b(ajuda|help|suporte|support)\b", re.IGNORECASE),

        # Explanations
        re.compile(r"\b(explicar?|explain|what\s+is|o\s+que\s+[eé]|significa)\b", re.IGNORECASE),
        re.compile(r"\b(definição|definition|conceito|concept)\b", re.IGNORECASE),

        # Best practices
        re.compile(r"\b(melhores?\s+práticas?|best\s+practices?)\b", re.IGNORECASE),
        re.compile(r"\b(recomenda[çc][ãa]o|recommendation|dica|tip)\b", re.IGNORECASE),

        # Documentation
        re.compile(r"\b(documentação|documentation|manual|guia|guide|tutorial)\b", re.IGNORECASE),
        re.compile(r"\b(exemplo|example|demonstra[çc][ãa]o|demo)\b", re.IGNORECASE),

        # Analysis
        re.compile(r"\b(analis[ea]r?|analyze|investigar?|investigate)\b", re.IGNORECASE),
        re.compile(r"\b(comparar?|compare|diferen[çc]a|difference)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        config: HybridRetrieverConfig | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.config = config or HybridRetrieverConfig()

        # BM25 index (built lazily)
        self.bm25_index: BM25Index | None = None
        self._index_built = False

        # Background save task reference to prevent garbage collection
        self._pending_save_task: asyncio.Task | None = None

        # v5.9.9: Use IReranker interface instead of direct cross-encoder
        self._reranker: IReranker = create_reranker(enabled=self.config.enable_reranking)

        # v5.9.9: Gating policy for CPU optimization
        gating_config = RerankGatingConfig(
            enabled=self.config.rerank_gating_enabled,
            score_low_threshold=self.config.rerank_score_low_threshold,
            margin_threshold=self.config.rerank_margin_threshold,
            max_candidates=self.config.rerank_max_candidates,
        )
        self._gating_policy = RerankGatingPolicy(config=gating_config)

        # v5.2.3.24: Initialize cache
        self._classification_cache: QueryClassificationCache | None = None
        if self.config.cache_enabled:
            self._classification_cache = QueryClassificationCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )

    def _classify_query(self, query: str) -> QueryClassificationResult:
        """
        Classify query and determine weights with caching.

        v5.2.3.24: Uses cache for repeated queries, tracks metrics.

        Returns:
            QueryClassificationResult with type and weights.
        """
        start_time = time.time()

        # Check cache first
        if self._classification_cache:
            cached = self._classification_cache.get(query)
            if cached:
                logger.debug("Query classification cache hit: %s...", query[:50])
                return cached

        # Perform classification
        has_exact = any(p.search(query) for p in self.EXACT_MATCH_PATTERNS)
        has_semantic = any(p.search(query) for p in self.SEMANTIC_PATTERNS)

        if has_exact and not has_semantic:
            query_type = QueryType.EXACT_MATCH
            weights = (0.2, 0.8)
        elif has_semantic and not has_exact:
            query_type = QueryType.SEMANTIC
            weights = (0.8, 0.2)
        elif has_exact and has_semantic:
            query_type = QueryType.MIXED
            weights = (0.4, 0.6)
        else:
            query_type = QueryType.DEFAULT
            weights = (self.config.vector_weight, self.config.bm25_weight)

        classification_time = (time.time() - start_time) * 1000

        result = QueryClassificationResult(
            query_type=query_type,
            vector_weight=weights[0],
            bm25_weight=weights[1],
            cached=False,
            classification_time_ms=classification_time,
        )

        # Store in cache
        if self._classification_cache:
            self._classification_cache.put(query, result)

        logger.debug(
            f"Query classified as {query_type.value}: {query[:50]}... "
            f"(weights: v={weights[0]:.1f}, b={weights[1]:.1f})"
        )

        return result

    async def _ensure_bm25_index(self, collection: str | None = None) -> None:
        """Build BM25 index if not already built, using persistent storage."""
        if self._index_built:
            return

        index_path = INDEX_STORAGE_PATH

        try:
            import asyncio

            logger.info("Attempting to load persisted BM25 index", path=index_path)

            self.bm25_index = await asyncio.to_thread(
                BM25Index.load, index_path
            )

            if self.bm25_index and self.bm25_index.documents:
                self._index_built = True
                logger.info(
                    "BM25 index loaded from disk",
                    num_docs=len(self.bm25_index.documents)
                )
                return

        except Exception as e:
            logger.warning(
                "BM25 index load failed, will rebuild",
                error=str(e)
            )

        logger.info("Building fresh BM25 index from database")

        try:
            from resync.knowledge.store import get_vector_store
            store = get_vector_store()

            documents = await store.get_all_documents(collection=collection)

            if documents:
                self.bm25_index = BM25Index(
                    k1=self.config.bm25_k1,
                    b=self.config.bm25_b,
                    field_boosts=self.config.field_boosts,
                )
                self.bm25_index.build_index(documents)
                self._index_built = True

                self._pending_save_task = asyncio.create_task(self._save_index_async(index_path))

                logger.info(
                    f"BM25 index built and saved: {len(documents)} docs"
                )
            else:
                logger.warning("No documents found for BM25 indexing")

        except Exception as e:
            logger.error("Failed to build BM25 index: %s", e)

    async def _save_index_async(self, path: str) -> None:
        """Save BM25 index in background (non-blocking)."""
        try:
            import asyncio
            await asyncio.sleep(2)

            if self.bm25_index:
                success = self.bm25_index.save(path)
                if success:
                    logger.info("BM25 index persisted successfully")
                else:
                    logger.warning("BM25 index persist failed (non-critical)")
        except Exception as e:
            logger.error("BM25 index async save failed: %s", e)

    def _reciprocal_rank_fusion(
        self,
        results_list: list[list[dict[str, Any]]],
        weights: list[float],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Combine multiple result sets using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each result set

        Args:
            results_list: List of result sets to combine
            weights: Weight for each result set
            k: RRF parameter (default 60)

        Returns:
            Combined and sorted results
        """
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Track scores and best result for each document
        doc_scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict] = {}

        for results, weight in zip(results_list, weights, strict=False):
            for rank, doc in enumerate(results, start=1):
                # Use document ID or content hash as key
                doc_id = doc.get("id") or doc.get("doc_id") or hash(str(doc.get("content", "")))
                doc_id = str(doc_id)

                # RRF score contribution
                doc_scores[doc_id] += weight / (k + rank)

                # Keep the best metadata
                if doc_id not in doc_data or doc.get("score", 0) > doc_data[doc_id].get("score", 0):
                    doc_data[doc_id] = doc

        # Sort by fused score and return
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc["rrf_score"] = rrf_score
            results.append(doc)

        return results

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform vector-based semantic search."""
        try:
            vec = self.embedder.embed(query)

            # Calculate adaptive ef_search
            from resync.knowledge.config import CFG

            ef = CFG.ef_search_base + int(math.log2(max(10, top_k)) * 8)
            ef = min(ef, CFG.ef_search_max)

            return self.store.query(
                vector=vec,
                top_k=top_k,
                collection=collection,
                filters=filters,
                ef_search=ef,
                with_vectors=False,
            )

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Vector search failed: %s", e)
            return []

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not self.bm25_index:
            await self._ensure_bm25_index(collection)

        if not self.bm25_index or not self.bm25_index.documents:
            return []

        try:
            results = self.bm25_index.search(query, top_k)

            # Convert to document format
            hits = []
            for doc_idx, score in results:
                doc = self.bm25_index.documents[doc_idx].copy()
                doc["bm25_score"] = score
                hits.append(doc)

            return hits

        except Exception as e:
            logger.error("BM25 search failed: %s", e)
            return []

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str | None = None,
        enable_reranking: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Combines vector search (semantic) and BM25 (keyword) with optional
        cross-encoder reranking.

        v5.2.3.22: Now uses dynamic weight adjustment based on query type.
        v5.2.3.24: Added performance metrics tracking.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters for vector search
            collection: Vector store collection name
            enable_reranking: Override config reranking setting (None = use config)

        Returns:
            List of relevant documents
        """
        # v5.2.3.24: Get classification with cache
        classification = self._classify_query(query)
        vector_weight = classification.vector_weight
        bm25_weight = classification.bm25_weight

        # Calculate how many candidates to fetch from each retriever
        candidate_k = top_k * self.config.candidate_multiplier

        # Run both searches
        vector_results = await self._vector_search(query, candidate_k, collection, filters)

        bm25_results = await self._bm25_search(query, candidate_k, collection)

        # Log retrieval stats with weights
        logger.debug(
            f"Hybrid search: vector={len(vector_results)}, bm25={len(bm25_results)}, "
            "weights=(v:{vector_weight:.1f}, b:{bm25_weight:.1f}), "
            f"type={classification.query_type.value}"
        )

        # If one search returned nothing, use the other
        if not vector_results and not bm25_results:
            return []

        if not vector_results:
            results = bm25_results[:top_k]
        elif not bm25_results:
            results = vector_results[:top_k]
        else:
            # Fuse results using RRF with dynamic weights
            results = self._reciprocal_rank_fusion(
                [vector_results, bm25_results],
                [vector_weight, bm25_weight],
            )

        # Determine reranking setting
        do_rerank = enable_reranking if enable_reranking is not None else self.config.enable_reranking

        # v5.9.9: Apply gated reranking using IReranker interface
        if do_rerank and len(results) > 1:
            # Extract scores for gating decision
            scores = [
                doc.get("rrf_score", 0) or doc.get("score", 0) or doc.get("vector_score", 0)
                for doc in results
            ]

            # Check if reranking should be activated
            should_rerank, reason = self._gating_policy.should_rerank(scores)

            if should_rerank:
                # Limit candidates to max_candidates for CPU efficiency
                pool = results[:self._gating_policy.config.max_candidates]
                final_k = min(top_k, self.config.rerank_top_k)

                # Use async reranker interface
                results = await self._reranker.rerank(query, pool, top_k=final_k)

                # If rerank returned fewer than expected, append remaining
                if len(results) < top_k:
                    remaining = [
                        doc for doc in results[self._gating_policy.config.max_candidates:]
                        if doc not in results
                    ]
                    results.extend(remaining[:top_k - len(results)])

                logger.debug("Gated rerank activated: reason=%s, candidates=%s", reason, len(pool))
            else:
                # Skip rerank, just truncate
                results = results[:top_k]
                logger.debug("Gated rerank skipped: reason=%s", reason)
        else:
            results = results[:top_k]

        return results


__all__ = [
    "BM25Index",
    "HybridRetriever",
    "HybridRetrieverConfig",
]
