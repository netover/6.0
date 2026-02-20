"""
Configuration for the pgvector-based RAG system.

Defines environment variables and defaults for PostgreSQL connection, embedding model, and search parameters.

SECURITY (v5.4.1):
- DATABASE_URL has no default password
- Production requires explicit configuration
- Development falls back to localhost without credentials
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _bool(env: str, default: bool = False) -> bool:
    v = os.getenv(env)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def _get_database_url() -> str:
    """
    Get DATABASE_URL with security validation.

    - Production: MUST be set via environment variable
    - Development: Falls back to localhost (no password in default)
    """
    url = os.getenv("DATABASE_URL")
    env = os.getenv("ENVIRONMENT", "development").lower()

    if url:
        # Warn if using obvious default password
        if "password@" in url or ":password@" in url:
            logger.warning(
                "insecure_database_url: DATABASE_URL contains default password - change for production"
            )
        return url

    # No DATABASE_URL set
    if env == "production":
        raise ValueError(
            "DATABASE_URL must be set in production. "
            "Example: postgresql://user:pass@host:5432/dbname"
        )

    # Development fallback - no password in default
    logger.info("using_dev_database_url: url=postgresql://localhost:5432/resync")
    return "postgresql://localhost:5432/resync"


@dataclass(frozen=True)
class RagConfig:
    """Configuration model for RAG with pgvector."""

    # PostgreSQL connection - NO default password
    database_url: str = None  # type: ignore  # Set in __post_init__

    # Collection names (stored in collection_name column)
    collection_write: str = os.getenv("RAG_COLLECTION_WRITE", "knowledge_v1")
    collection_read: str = os.getenv("RAG_COLLECTION_READ", "knowledge_v1")

    # Embedding settings
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1536"))

    # Search parameters
    max_top_k: int = int(os.getenv("RAG_MAX_TOPK", "50"))

    # HNSW index parameters (used during table creation)
    hnsw_m: int = int(os.getenv("RAG_HNSW_M", "16"))
    hnsw_ef_construction: int = int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "256"))

    # Search tuning
    ef_search_base: int = int(os.getenv("RAG_EF_SEARCH_BASE", "64"))
    ef_search_max: int = int(os.getenv("RAG_EF_SEARCH_MAX", "128"))
    max_neighbors: int = int(os.getenv("RAG_MAX_NEIGHBORS", "32"))

    # Reranking (legacy)
    enable_rerank: bool = _bool("RAG_RERANKER_ON", False)

    # Cross-encoder reranking (v5.3.17+)
    enable_cross_encoder: bool = _bool("RAG_CROSS_ENCODER_ON", True)
    cross_encoder_model: str = os.getenv(
        "RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-small"
    )
    cross_encoder_top_k: int = int(os.getenv("RAG_CROSS_ENCODER_TOP_K", "5"))
    cross_encoder_threshold: float = float(
        os.getenv("RAG_CROSS_ENCODER_THRESHOLD", "0.3")
    )

    # v5.9.9: Rerank gating for CPU optimization
    # Only activate reranking when retrieval confidence is low
    rerank_gating_enabled: bool = _bool("RERANK_GATING_ENABLED", True)
    rerank_score_low_threshold: float = float(
        os.getenv("RERANK_SCORE_LOW_THRESHOLD", "0.35")
    )
    rerank_margin_threshold: float = float(os.getenv("RERANK_MARGIN_THRESHOLD", "0.05"))
    rerank_max_candidates: int = int(os.getenv("RERANK_MAX_CANDIDATES", "10"))

    def __post_init__(self) -> None:
        # Frozen dataclass workaround for dynamic default
        object.__setattr__(self, "database_url", _get_database_url())


CFG = RagConfig()
