# pylint
# mypy
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

import asyncio
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)

from resync.knowledge.interfaces import Embedder, Retriever, VectorStore

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
INDEX_STORAGE_PATH = os.environ.get(
    "BM25_INDEX_PATH", os.path.join("data", "bm25_index.bin.gz")
)

# Magic bytes for secure BM25 index format (JSON + gzip)
# Format: magic_bytes (8 bytes) + version (2 bytes) + compressed_json_data
BM25_INDEX_MAGIC = b"RESYNCBM"
BM25_INDEX_VERSION = b"01"


# Fallback to joblib for backward compatibility with existing indexes
def _try_load_joblib(path: str):
    """Attempt to load index using joblib (backward compatibility)."""
    import joblib

    try:
        with gzip.open(path, "rb") as f:
            index = joblib.load(f)
        # Validate basic structure
        if not hasattr(index, "documents") or not hasattr(index, "inverted_index"):
            raise ValueError("Invalid joblib index structure")
        logger.warning(
            "bm25_index_loaded_joblib_compat",
            extra={
                "path": path,
                "num_docs": len(index.documents),
                "num_terms": len(index.inverted_index),
            },
        )
        return index
    except Exception as e:
        logger.warning("bm25_index_joblib_load_failed", extra={"error": str(e)})
        raise


def _validate_index_data(data: dict) -> bool:
    """Validate the loaded index data has required fields."""
    required_fields = [
        "k1",
        "b",
        "field_boosts",
        "documents",
        "doc_lengths",
        "avg_doc_length",
        "inverted_index",
        "doc_freq",
    ]
    return all(field in data for field in required_fields)


def _load_secure_json(path: str):
    """Load BM25 index from JSON format with validation."""
    try:
        with gzip.open(path, "rb") as f:
            # Read and verify magic bytes
            magic = f.read(8)
            if magic != BM25_INDEX_MAGIC:
                raise ValueError(
                    f"Invalid magic bytes: {magic.decode('utf-8', errors='replace')}"
                )

            # Read version
            version = f.read(2)
            if version != BM25_INDEX_VERSION:
                raise ValueError(
                    f"Unsupported version: {version.decode('utf-8', errors='replace')}"
                )

            # Read and decompress JSON data
            json_data = f.read()
            data = json.loads(json_data.decode("utf-8"))

        # Validate structure
        if not _validate_index_data(data):
            raise ValueError("Invalid index data structure - missing required fields")

        # Reconstruct BM25Index from validated dict
        index = BM25Index(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            field_boosts=data.get("field_boosts", {}),
            documents=data.get("documents", []),
            doc_lengths=data.get("doc_lengths", []),
            avg_doc_length=data.get("avg_doc_length", 0.0),
            inverted_index=data.get("inverted_index", {}),
            doc_freq=data.get("doc_freq", {}),
        )

        logger.info(
            "bm25_index_loaded_secure",
            extra={
                "path": path,
                "size_bytes": os.path.getsize(path),
                "num_docs": len(index.documents),
                "num_terms": len(index.inverted_index),
            },
        )
        return index
    except (json.JSONDecodeError, ValueError, OSError) as e:
        raise ValueError(f"Failed to load secure index: {e}") from e


@dataclass
class BM25Index:
    """
    BM25 (Best Match 25) index for keyword-based retrieval.

    Optimized for TWS job names and technical identifiers.

    v5.2.3.23: Added field boosting for TWS-specific metadata.
    """

    k1: float = 1.5
    b: float = 0.75
    field_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "job_name": 4.0,
            "error_code": 3.5,
            "workstation": 3.0,
            "job_stream": 2.5,
            "message_id": 2.5,
            "resource": 2.0,
            "title": 1.5,
            "content": 1.0,
        }
    )
    documents: list[dict[str, Any]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    inverted_index: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    doc_freq: dict[str, int] = field(default_factory=dict)
    ERROR_CODE_PATTERNS = [
        re.compile("RC[=:]\\s*(\\d+)", re.IGNORECASE),
        re.compile("ABEND\\s+([A-Z0-9]+)", re.IGNORECASE),
        re.compile("EQQQ(\\w+)", re.IGNORECASE),
        re.compile("AWSB(\\w+)", re.IGNORECASE),
        re.compile("(?:error|erro)[:\\s]+(\\w+)", re.IGNORECASE),
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
        text = text.lower()
        text = re.sub("rc[=:]\\s*(\\d+)", "rc_\\1 rc\\1", text)
        text = re.sub("abend\\s+([a-z0-9]+)", "abend_\\1 \\1", text)
        tokens = re.findall("[a-z0-9_\\-]+", text)
        expanded = []
        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(token.split("_"))
            if "-" in token:
                expanded.extend(token.split("-"))
        return [t for t in expanded if len(t) >= 2]

    def build_index(
        self, documents: list[dict[str, Any]], text_field: str = "content"
    ) -> None:
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
            boosted_term_freqs: dict[str, float] = defaultdict(float)
            metadata = doc.get("metadata", {}) or {}
            text = doc.get(text_field, "") or ""
            if isinstance(text, dict):
                text = str(text.get("text", text.get("content", "")))
            content_boost = self.field_boosts.get("content", 1.0)
            for token in self._tokenize(text):
                boosted_term_freqs[token] += content_boost
            job_name = metadata.get("job_name", "") or ""
            if job_name:
                job_boost = self.field_boosts.get("job_name", 4.0)
                for token in self._tokenize(job_name):
                    boosted_term_freqs[token] += job_boost
            workstation = metadata.get("workstation", "") or ""
            if workstation:
                ws_boost = self.field_boosts.get("workstation", 3.0)
                for token in self._tokenize(workstation):
                    boosted_term_freqs[token] += ws_boost
            job_stream = metadata.get("job_stream", "") or ""
            if job_stream:
                stream_boost = self.field_boosts.get("job_stream", 2.5)
                for token in self._tokenize(job_stream):
                    boosted_term_freqs[token] += stream_boost
            resource = metadata.get("resource", "") or ""
            if resource:
                res_boost = self.field_boosts.get("resource", 2.0)
                for token in self._tokenize(resource):
                    boosted_term_freqs[token] += res_boost
            title = metadata.get("title", "") or doc.get("title", "") or ""
            if title:
                title_boost = self.field_boosts.get("title", 1.5)
                for token in self._tokenize(title):
                    boosted_term_freqs[token] += title_boost
            full_text = f"{text} {job_name} {workstation}"
            error_codes = self._extract_error_codes(full_text)
            if error_codes:
                error_boost = self.field_boosts.get("error_code", 3.5)
                for code in error_codes:
                    for token in self._tokenize(code):
                        boosted_term_freqs[token] += error_boost
            message_ids = self._extract_message_ids(full_text)
            if message_ids:
                msg_boost = self.field_boosts.get("message_id", 2.5)
                for msg_id in message_ids:
                    for token in self._tokenize(msg_id):
                        boosted_term_freqs[token] += msg_boost
            doc_length = sum(boosted_term_freqs.values())
            self.doc_lengths.append(int(doc_length))
            total_length += doc_length  # type: ignore[assignment]
            seen_terms = set()
            for term, boosted_freq in boosted_term_freqs.items():
                self.inverted_index[term].append((doc_idx, boosted_freq))
                if term not in seen_terms:
                    self.doc_freq[term] += 1
                    seen_terms.add(term)
        self.avg_doc_length = total_length / len(documents) if documents else 0.0
        logger.info(
            "BM25 index built with field boosting: "
            f"{len(documents)} docs, {len(self.inverted_index)} unique terms, "
            f"avg_length={self.avg_doc_length:.1f}"
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
        pattern = re.compile("\\b(EQQQ\\w+|AWSB\\w+|IEF\\w+)\\b", re.IGNORECASE)
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
        scores: dict[int, float] = defaultdict(float)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            df = self.doc_freq[term]
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                scores[doc_idx] += idf * numerator / denominator
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save(self, path: str) -> bool:
        """
        Persist BM25 index to disk with compression and secure JSON format.

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
                    # Convert index to dict for JSON serialization
                    index_data = {
                        "k1": self.k1,
                        "b": self.b,
                        "field_boosts": self.field_boosts,
                        "documents": self.documents,
                        "doc_lengths": self.doc_lengths,
                        "avg_doc_length": self.avg_doc_length,
                        "inverted_index": dict(self.inverted_index),
                        "doc_freq": dict(self.doc_freq),
                    }
                    json_bytes = json.dumps(index_data, ensure_ascii=False).encode(
                        "utf-8"
                    )

                    # Write with magic bytes and version header
                    with gzip.open(path, "wb") as f:
                        f.write(BM25_INDEX_MAGIC)
                        f.write(BM25_INDEX_VERSION)
                        f.write(json_bytes)

                    logger.info(
                        "bm25_index_saved",
                        extra={
                            "path": path,
                            "size_bytes": os.path.getsize(path),
                            "num_docs": len(self.documents),
                            "num_terms": len(self.inverted_index),
                        },
                    )
                    return True
            except Timeout:
                logger.error("bm25_index_save_timeout", extra={"path": path})
                return False
        except Exception as e:
            logger.error(
                "bm25_index_save_failed", extra={"error": str(e), "path": path}
            )
            return False

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """
        Load BM25 index from disk with auto-recovery and security validation.

        Args:
            path: Path to the index file

        Returns:
            BM25Index instance (loaded or empty)
        """
        if not os.path.exists(path):
            logger.info("bm25_index_not_found_will_build", extra={"path": path})
            return cls()
        if not FILELOCK_AVAILABLE:
            logger.warning("filelock not available, building fresh index")
            return cls()
        lock_path = f"{path}.lock"
        try:
            lock = FileLock(lock_path, timeout=5)
            try:
                with lock.acquire(timeout=5):
                    # Try secure JSON loading first
                    try:
                        index = _load_secure_json(path)
                        return index
                    except ValueError as secure_error:
                        # Fall back to joblib for backward compatibility
                        logger.warning(
                            "bm25_index_trying_joblib_fallback",
                            extra={"reason": str(secure_error)},
                        )
                        index = _try_load_joblib(path)
                        return index
            except Timeout:
                logger.warning("bm25_index_locked_using_empty", extra={"path": path})
                return cls()
        except (EOFError, OSError, ImportError, gzip.BadGzipFile) as e:
            logger.warning(
                "bm25_index_corrupted_rebuilding", extra={"error": str(e), "path": path}
            )
            try:
                if os.path.exists(path):
                    os.remove(path)
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception as cleanup_error:
                logger.error(
                    "bm25_index_cleanup_failed", extra={"error": str(cleanup_error)}
                )
            return cls()
        except MemoryError:
            logger.error("bm25_index_oom_using_empty", extra={"path": path})
            return cls()
        except Exception as e:
            logger.warning(
                "bm25_index_load_failed_unknown",
                extra={"error": str(e), "error_type": type(e).__name__, "path": path},
            )
            return cls()


class QueryType(str, Enum):
    """Query classification types for weight selection."""

    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    MIXED = "mixed"
    DEFAULT = "default"


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
        self._cache: OrderedDict[str, tuple[QueryClassificationResult, float]] = (
            OrderedDict()
        )
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
        if time.time() - timestamp > self._ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
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
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""

    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    auto_weight: bool = True
    candidate_multiplier: int = 3
    enable_reranking: bool = True
    rerank_top_k: int = 5
    rerank_threshold: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    field_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "job_name": 4.0,
            "error_code": 3.5,
            "workstation": 3.0,
            "job_stream": 2.5,
            "message_id": 2.5,
            "resource": 2.0,
            "title": 1.5,
            "content": 1.0,
        }
    )
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600
    metrics_enabled: bool = True
    rerank_gating_enabled: bool = True
    rerank_score_low_threshold: float = 0.35
    rerank_margin_threshold: float = 0.05
    rerank_max_candidates: int = 10

    @classmethod
    def from_settings(cls) -> HybridRetrieverConfig:
        """Create config from application settings."""
        try:
            from resync.settings import settings

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
                rerank_gating_enabled=getattr(settings, "rerank_gating_enabled", True),
                rerank_score_low_threshold=getattr(
                    settings, "rerank_score_low_threshold", 0.35
                ),
                rerank_margin_threshold=getattr(
                    settings, "rerank_margin_threshold", 0.05
                ),
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

    EXACT_MATCH_PATTERNS = [
        re.compile("\\b[A-Z]{2,}[0-9]{2,}[A-Z0-9_]*\\b"),
        re.compile("\\bAWSB[A-Z0-9]+\\b", re.IGNORECASE),
        re.compile("\\b[A-Z]{2,}_[A-Z0-9_]+_\\d+\\b"),
        re.compile("\\bJOB[=:]\\s*[A-Z0-9_]+", re.IGNORECASE),
        re.compile("\\bRC[=:]\\s*\\d+\\b", re.IGNORECASE),
        re.compile("\\bABEND\\s+[A-Z0-9]+\\b", re.IGNORECASE),
        re.compile("\\bS[0-9A-F]{3}\\b"),
        re.compile("\\bU\\d{4}\\b"),
        re.compile("\\bCC[=:]\\s*\\d+\\b", re.IGNORECASE),
        re.compile("\\b[A-Z]{2}\\d{3,}\\b"),
        re.compile("\\bLPAR\\d+\\b", re.IGNORECASE),
        re.compile("\\bSYSPLEX\\w+\\b", re.IGNORECASE),
        re.compile("\\bEQQQ[A-Z0-9]+\\b", re.IGNORECASE),
        re.compile("\\bIEF\\d{3}[A-Z]\\b", re.IGNORECASE),
        re.compile("\\bIKJ\\d{5}[A-Z]\\b", re.IGNORECASE),
        re.compile("\\bCSV\\d{3}[A-Z]\\b", re.IGNORECASE),
        re.compile("\\b[A-Z]+\\.[A-Z0-9]+\\.[A-Z0-9]+\\b"),
        re.compile("\\bDSN[=:]\\s*\\S+", re.IGNORECASE),
        re.compile("\\b\\d{1,2}:\\d{2}(?::\\d{2})?\\b"),
        re.compile("\\b(?:ODATE|SCHED_TIME)[=:]\\s*\\S+", re.IGNORECASE),
    ]
    SEMANTIC_PATTERNS = [
        re.compile(
            "\\b(como|how|why|por\\s*que|quando|when|onde|where)\\b", re.IGNORECASE
        ),
        re.compile("\\b(resolver|fix|solve|solucionar|corrigir)\\b", re.IGNORECASE),
        re.compile(
            "\\b(configurar|configure|setup|instalar|install)\\b", re.IGNORECASE
        ),
        re.compile("\\b(ajuda|help|suporte|support)\\b", re.IGNORECASE),
        re.compile(
            "\\b(explicar?|explain|what\\s+is|o\\s+que\\s+[eé]|significa)\\b",
            re.IGNORECASE,
        ),
        re.compile("\\b(definição|definition|conceito|concept)\\b", re.IGNORECASE),
        re.compile("\\b(melhores?\\s+práticas?|best\\s+practices?)\\b", re.IGNORECASE),
        re.compile("\\b(recomenda[çc][ãa]o|recommendation|dica|tip)\\b", re.IGNORECASE),
        re.compile(
            "\\b(documentação|documentation|manual|guia|guide|tutorial)\\b",
            re.IGNORECASE,
        ),
        re.compile("\\b(exemplo|example|demonstra[çc][ãa]o|demo)\\b", re.IGNORECASE),
        re.compile(
            "\\b(analis[ea]r?|analyze|investigar?|investigate)\\b", re.IGNORECASE
        ),
        re.compile("\\b(comparar?|compare|diferen[çc]a|difference)\\b", re.IGNORECASE),
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
        self.bm25_index: BM25Index | None = None
        self._index_built = False
        self._pending_save_task: asyncio.Task | None = None
        self._reranker: IReranker = create_reranker(
            enabled=self.config.enable_reranking
        )
        gating_config = RerankGatingConfig(
            enabled=self.config.rerank_gating_enabled,
            score_low_threshold=self.config.rerank_score_low_threshold,
            margin_threshold=self.config.rerank_margin_threshold,
            max_candidates=self.config.rerank_max_candidates,
        )
        self._gating_policy = RerankGatingPolicy(config=gating_config)
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
        if self._classification_cache:
            cached = self._classification_cache.get(query)
            if cached:
                logger.debug("Query classification cache hit: %s...", query[:50])
                return cached
        has_exact = any((p.search(query) for p in self.EXACT_MATCH_PATTERNS))
        has_semantic = any((p.search(query) for p in self.SEMANTIC_PATTERNS))
        if has_exact and (not has_semantic):
            query_type = QueryType.EXACT_MATCH
            weights = (0.2, 0.8)
        elif has_semantic and (not has_exact):
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
        if self._classification_cache:
            self._classification_cache.put(query, result)
        logger.debug(
            "Query classified as "
            f"{query_type.value}: {query[:50]}... "
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

            logger.info(
                "Attempting to load persisted BM25 index", extra={"path": index_path}
            )
            self.bm25_index = await asyncio.to_thread(BM25Index.load, index_path)
            if self.bm25_index and self.bm25_index.documents:
                self._index_built = True
                logger.info(
                    "BM25 index loaded from disk",
                    extra={"num_docs": len(self.bm25_index.documents)},
                )
                return
        except Exception as e:
            logger.warning(
                "BM25 index load failed, will rebuild", extra={"error": str(e)}
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
                import asyncio
                await asyncio.to_thread(self.bm25_index.build_index, documents)
                self._index_built = True
                self._pending_save_task = asyncio.create_task(
                    self._save_index_async(index_path)
                )
                logger.info(f"BM25 index built and saved: {len(documents)} docs")
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
                # Use asyncio.to_thread to avoid blocking the event loop
                success = await asyncio.to_thread(self.bm25_index.save, path)
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
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        weights = [w / total_weight for w in weights]
        doc_scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict] = {}
        # Ensure both lists have the same length to avoid issues
        min_len = min(len(results_list), len(weights))
        results_list = results_list[:min_len]
        weights = weights[:min_len]

        for results, weight in zip(results_list, weights):
            for rank, doc in enumerate(results, start=1):
                doc_id = (
                    doc.get("id")
                    or doc.get("doc_id")
                    or hash(str(doc.get("content", "")))
                )
                doc_id = str(doc_id)
                doc_scores[doc_id] += weight / (k + rank)
                if doc_id not in doc_data or doc.get("score", 0) > doc_data[doc_id].get(
                    "score", 0
                ):
                    doc_data[doc_id] = doc
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
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Perform vector-based semantic search."""
        try:
            vec = await self.embedder.embed(query, timeout=timeout)
            from resync.knowledge.config import CFG

            ef = CFG.ef_search_base + int(math.log2(max(10, top_k)) * 8)
            ef = min(ef, CFG.ef_search_max)
            return await self.store.query(
                vector=vec,
                top_k=top_k,
                collection=collection,
                filters=filters,
                ef_search=ef,
                with_vectors=False,
                timeout=timeout,
            )
        except Exception as e:
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Vector search failed: %s", e)
            return []

    async def _bm25_search(
        self, query: str, top_k: int, collection: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword search."""
        if not self.bm25_index:
            await self._ensure_bm25_index(collection)
        if not self.bm25_index or not self.bm25_index.documents:
            return []
        try:
            import asyncio
            results = await asyncio.to_thread(self.bm25_index.search, query, top_k)
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
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents using hybrid search.
        """
        try:
            async with asyncio.timeout(timeout):
                classification = self._classify_query(query)
                vector_weight = classification.vector_weight
                bm25_weight = classification.bm25_weight
                candidate_k = top_k * self.config.candidate_multiplier
                async with asyncio.TaskGroup() as tg:
                    v_task = tg.create_task(
                        self._vector_search(
                            query, candidate_k, collection, filters, timeout=timeout / 2
                        )
                    )
                    b_task = tg.create_task(
                        self._bm25_search(query, candidate_k, collection)
                    )
                vector_results = v_task.result()
                bm25_results = b_task.result()
                logger.debug(
                    "Hybrid search: "
                    f"vector={len(vector_results)}, bm25={len(bm25_results)}, "
                    f"weights=(v:{vector_weight:.1f}, b:{bm25_weight:.1f}), "
                    f"type={classification.query_type.value}"
                )
                if not vector_results and (not bm25_results):
                    return []
                if not vector_results:
                    results = bm25_results[:top_k]
                elif not bm25_results:
                    results = vector_results[:top_k]
                else:
                    results = self._reciprocal_rank_fusion(
                        [vector_results, bm25_results], [vector_weight, bm25_weight]
                    )
                do_rerank = (
                    enable_reranking
                    if enable_reranking is not None
                    else self.config.enable_reranking
                )
                if do_rerank and len(results) > 1:
                    scores = [
                        doc.get("rrf_score", 0)
                        or doc.get("score", 0)
                        or doc.get("vector_score", 0)
                        for doc in results
                    ]
                    should_rerank, reason = self._gating_policy.should_rerank(scores)
                    if should_rerank:
                        original_results = results
                        pool = results[: self._gating_policy.config.max_candidates]
                        final_k = min(top_k, self.config.rerank_top_k)
                        # Ensure rerank is awaited if it's async (which it is now)
                        results = await self._reranker.rerank(query, pool, top_k=final_k)
                        if len(results) < top_k:
                            seen_ids = {doc.get("id") for doc in results}
                            remaining = [
                                doc
                                for doc in original_results[
                                    self._gating_policy.config.max_candidates :
                                ]
                                if doc.get("id") not in seen_ids
                            ]
                            results.extend(remaining[: top_k - len(results)])
                        logger.debug(
                            "Gated rerank activated: reason=%s, candidates=%s",
                            reason,
                            len(pool),
                        )
                    else:
                        results = results[:top_k]
                        logger.debug("Gated rerank skipped: reason=%s", reason)
                else:
                    results = results[:top_k]
                return results
        except asyncio.TimeoutError:
            logger.warning("Retrieval timeout", query=query[:50])
            return []
        except Exception as e:
            logger.error("Retrieval failed", error=str(e))
            return []


__all__ = ["BM25Index", "HybridRetriever", "HybridRetrieverConfig"]
