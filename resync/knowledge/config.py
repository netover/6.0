"""
Configuration for the pgvector-based RAG system.

Defines environment variables and defaults for PostgreSQL
connection, embedding model, and search parameters.

SECURITY (v5.4.1):
- DATABASE_URL has no default password
- Production requires explicit configuration
- Development falls back to localhost without credentials

Version: 6.0.0 - Refactored with Pydantic v2 for type safety
"""

import os
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

# Use structlog as specified in the stack
import structlog

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

logger = structlog.get_logger(__name__)


def _get_database_url() -> str:
    """
    Get DATABASE_URL with security validation.

    Security measures:
    - Production MUST set via environment variable
    - Development falls back to localhost (no password)
    - Passwords validated against common weak passwords
    - URLs validated for correct format
    - Sanitized logging to prevent credential exposure

    Returns:
        Validated database URL

    Raises:
        ValueError: If DATABASE_URL invalid or missing in production
    """
    url = os.getenv("DATABASE_URL")
    env = os.getenv("ENVIRONMENT", "development").strip().lower()

    if url:
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.hostname]):
                raise ValueError(
                    "Invalid DATABASE_URL format. Must be: "
                    "postgresql://user:pass@host:5432/dbname"
                )
        except Exception as e:
            raise ValueError(f"DATABASE_URL parsing failed: {e}") from e

        # Production security checks
        if env == "production":
            # Validate password strength
            if parsed.password:
                weak_passwords = {
                    "password",
                    "admin",
                    "123456",
                    "postgres",
                    "root",
                    "test",
                    "default",
                    "changeme",
                }
                if parsed.password.lower() in weak_passwords:
                    raise ValueError(
                        "Weak database password detected. "
                        "Use strong password in production."
                    )

        # Log sanitized URL (never expose credentials)
        sanitized = _sanitize_db_url(url)
        logger.info(
            "database_url_configured",
            url=sanitized,
            environment=env,
        )
        return url

    # No DATABASE_URL set
    if env == "production":
        raise ValueError(
            "DATABASE_URL must be set in production. "
            "Example: postgresql+asyncpg://user:pass@host:5432/dbname"
        )

    # Development fallback - no password in default
    default_url = "postgresql://localhost:5432/resync"
    logger.info(
        "using_dev_database_url",
        url="postgresql://localhost:5432/resync",
        environment=env,
    )
    return default_url


def _sanitize_db_url(url: str) -> str:
    """
    Sanitize DATABASE_URL for logging by replacing password with '***'.

    Args:
        url: Database connection URL

    Returns:
        Sanitized URL safe for logging
    """
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with asterisks
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            sanitized = parsed._replace(netloc=netloc)
            return urlunparse(sanitized)
    except Exception:
        # If parsing fails, return generic message
        return "[DATABASE_URL_PARSE_ERROR]"
    return url


def _bool(env: str, default: bool = False) -> bool:
    """
    Parse boolean from environment variable.

    Accepts: "1", "true", "yes", "on" (case-insensitive) as True
    Accepts: "0", "false", "no", "off" (case-insensitive) as False
    Empty string or whitespace returns default

    Args:
        env: Environment variable name
        default: Default value if not set or empty

    Returns:
        Boolean value
    """
    v = os.getenv(env)

    # Not set or None
    if v is None:
        return default

    # Empty or whitespace - handle safely
    v = v.strip()
    if not v:
        return default

    # Parse as boolean
    v_lower = v.lower()
    if v_lower in {"1", "true", "yes", "on"}:
        return True
    elif v_lower in {"0", "false", "no", "off"}:
        return False
    else:
        logger.warning(
            "invalid_boolean_env_var",
            name=env,
            value=v,
            default=default,
            message=f"Unrecognized boolean value '{v}', using default {default}",
        )
        return default


def _parse_int(env: str, default: int) -> int:
    """Parse integer from environment variable with validation."""
    try:
        return int(os.getenv(env, str(default)))
    except (ValueError, TypeError):
        logger.warning(
            "invalid_int_env_var",
            name=env,
            default=default,
            message=f"Invalid integer value in {env}, using default {default}",
        )
        return default


def _parse_float(env: str, default: float) -> float:
    """Parse float from environment variable with validation."""
    try:
        return float(os.getenv(env, str(default)))
    except (ValueError, TypeError):
        logger.warning(
            "invalid_float_env_var",
            name=env,
            default=default,
            message=f"Invalid float value in {env}, using default {default}",
        )
        return default


class RagConfig:
    """
    Configuration model for RAG with pgvector.

    Uses Pydantic v2 BaseSettings for automatic environment variable loading,
    validation, and type safety. All settings are immutable (frozen) after creation.

    Environment variables are loaded automatically with RAG_ prefix.
    Example: RAG_MAX_TOPK=100 sets max_top_k=100
    """

    model_config = SettingsConfigDict(
        frozen=True,  # Immutable after creation
        validate_default=True,  # Validate default values
        env_prefix="RAG_",  # Auto-load from RAG_* env vars
        case_sensitive=False,
    )

    # PostgreSQL connection - loaded from DATABASE_URL
    database_url: str = Field(
        default_factory=_get_database_url,
        description="PostgreSQL connection URL",
    )

    # Collection names (stored in collection_name column)
    collection_write: str = Field(
        default="knowledge_v1",
        description="Collection name for writes",
    )
    collection_read: str = Field(
        default="knowledge_v1",
        description="Collection name for reads",
    )

    # Embedding settings
    embed_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (OpenAI or compatible)",
    )
    embed_dim: int = Field(
        default=1536,
        ge=128,  # Minimum embedding dimension
        le=4096,  # Maximum embedding dimension
        description="Embedding vector dimension",
    )

    # Search parameters with validation
    max_top_k: int = Field(
        default=50,
        ge=1,
        le=1000,  # Prevent OOM from excessive results
        description="Maximum number of results to return",
    )

    # HNSW index parameters with valid ranges
    hnsw_m: int = Field(
        default=16,
        ge=4,  # HNSW minimum
        le=64,  # HNSW maximum for stability
        description="HNSW M parameter (connections per node)",
    )
    hnsw_ef_construction: int = Field(
        default=256,
        ge=10,
        le=10000,
        description="HNSW ef_construction parameter (build quality)",
    )

    # Search tuning with cross-validation
    ef_search_base: int = Field(
        default=64,
        ge=1,
        le=1000,
        description="Base ef_search parameter (minimum search effort)",
    )
    ef_search_max: int = Field(
        default=128,
        ge=1,
        le=2000,
        description="Maximum ef_search parameter (maximum search effort)",
    )
    max_neighbors: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Maximum neighbors to consider",
    )

    # Legacy reranking
    enable_rerank: bool = Field(
        default=False,
        description="Enable legacy reranking (deprecated)",
    )

    # Cross-encoder reranking (v5.3.17+)
    enable_cross_encoder: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    cross_encoder_model: str = Field(
        default="BAAI/bge-reranker-small",
        description="Cross-encoder model for reranking",
    )
    cross_encoder_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to rerank",
    )
    cross_encoder_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for cross-encoder",
    )

    # v5.9.9: Rerank gating for CPU optimization
    rerank_gating_enabled: bool = Field(
        default=True,
        description="Enable conditional reranking based on retrieval confidence",
    )
    rerank_score_low_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Score below which reranking is triggered",
    )
    rerank_margin_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Minimum margin between top-2 scores to skip reranking",
    )
    rerank_max_candidates: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum candidates to pass to reranker",
    )

    @model_validator(mode="after")
    def validate_ef_search_order(self) -> "RagConfig":
        """Validate that ef_search_max >= ef_search_base."""
        if self.ef_search_max < self.ef_search_base:
            raise ValueError(
                f"ef_search_max ({self.ef_search_max}) must be >= "
                f"ef_search_base ({self.ef_search_base})"
            )
        return self

    @model_validator(mode="after")
    def validate_rerank_thresholds(self) -> "RagConfig":
        """Validate rerank threshold consistency."""
        if self.rerank_gating_enabled:
            if self.rerank_score_low_threshold <= 0.0:
                raise ValueError(
                    "rerank_score_low_threshold must be > 0.0 when gating enabled"
                )
            if self.rerank_margin_threshold >= 1.0:
                raise ValueError(
                    "rerank_margin_threshold must be < 1.0"
                )
        return self


# Global configuration instance - loaded once at module import
# Thread-safe and immutable after creation
CFG = RagConfig()
