"""
Configuration for the pgvector-based RAG system.

Version: 7.0.0 — Python 3.14 + Pydantic v2 BaseSettings (HARDENED)

This module provides immutable, validated configuration for the RAG system
with proper security controls, lazy initialization, and production-grade
error handling.

Environment variables:
    ENVIRONMENT: deployment environment (development|staging|production)
    DATABASE_URL: PostgreSQL connection URL (required in production)
    RAG_*: all other settings use the RAG_ prefix (e.g. RAG_EMBED_MODEL)

Intended stack:
    - asyncpg / SQLAlchemy async / sqlmodel + pgvector
    - sentence-transformers / torch / transformers
    - langchain / litellm / langgraph
    - structlog + OpenTelemetry + Prometheus + Sentry
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Final, Self
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import structlog
from pydantic import AliasChoices, Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["RagConfig", "get_config"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# stdlib logger interface has better typing than plain structlog.get_logger
logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported database URL schemes for async drivers
VALID_DB_SCHEMES: Final[frozenset[str]] = frozenset(
    {
        "postgresql+asyncpg",   # asyncpg driver (recommended)
        "postgresql+psycopg",   # psycopg3 async driver
        "postgresql",           # sync fallback (dev only, never in prod)
    }
)

# Weak passwords to reject in production (lowercase)
WEAK_PASSWORDS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "admin",
        "123456",
        "postgres",
        "root",
        "test",
        "changeme",
        "default",
        "secret",
        "pass",
        "db",
        "database",
        "administrator",
        "letmein",
        "welcome",
        "qwerty",
        "abc123",
    }
)

# Embedding dimension bounds (based on common models)
MIN_EMBED_DIM: Final[int] = 128    # minimum for meaningful embeddings
MAX_EMBED_DIM: Final[int] = 4096   # text-embedding-3-large max

# HNSW index parameter bounds (pgvector recommendations)
MIN_HNSW_M: Final[int] = 4
MAX_HNSW_M: Final[int] = 64
MIN_EF_CONSTRUCTION: Final[int] = 10
MAX_EF_CONSTRUCTION: Final[int] = 10000

# Default dev DB URL (never used in production)
DEV_DATABASE_URL: Final[str] = "postgresql+asyncpg://localhost:5432/resync"

# Query-string keys that often carry secrets
SENSITIVE_QS_KEYS: Final[frozenset[str]] = frozenset(
    {"password", "pass", "passwd", "secret", "token", "api_key", "apikey", "sslpassword", "sslkey"}
)

# Conservative identifier pattern for collection/table names (if quiser, pode mover para regex)
# Mantive simples com pattern em Field, abaixo.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_environment() -> str:
    """Return normalized deployment environment."""
    return os.getenv("ENVIRONMENT", "development").strip().lower() or "development"


def _is_weak_password(password: str) -> bool:
    """
    Check if password is weak.

    This is defense-in-depth; timing here não é exposto como endpoint.
    """
    if len(password) < 12:
        return True
    return password.lower() in WEAK_PASSWORDS


def _sanitize_db_url(url: str) -> str:
    """
    Redact secrets from DB URL for safe logging.

    - Redacts password in netloc.
    - Redacts known sensitive query params.
    - Handles IPv6 hostnames.
    - Never raises; falls back to placeholder.
    """
    try:
        parsed = urlparse(url)

        # Netloc com password redigida
        netloc = parsed.netloc
        if parsed.password is not None:
            username = parsed.username or ""
            hostname = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port is not None else ""

            # IPv6 vem sem colchetes em hostname
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"

            if username:
                userinfo = f"{username}:***@"
            else:
                userinfo = "***@"

            netloc = f"{userinfo}{hostname}{port}"

        # Redigir query params sensíveis
        if parsed.query:
            items = []
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                if k.lower() in SENSITIVE_QS_KEYS:
                    items.append((k, "***"))
                else:
                    items.append((k, v))
            new_query = urlencode(items, doseq=True)
            parsed = parsed._replace(query=new_query)

        return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        return "[DATABASE_URL_REDACTED]"


def _validate_database_url_raw(url: str, environment: str) -> str:
    """
    Validate DATABASE_URL format and security requirements.

    Returns the original URL if valid, raises ValueError otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        logger.error(
        "database_url_parse_failed",
            hint="Check DATABASE_URL format (postgresql+asyncpg://user:pass@host:5432/dbname)",
        )
        raise ValueError(
            "DATABASE_URL parsing failed. Expected format: "
            "postgresql+asyncpg://user:pass@host:5432/dbname"
        ) from None

    if not parsed.scheme:
        raise ValueError("DATABASE_URL missing scheme. Use postgresql+asyncpg://")

    if parsed.scheme not in VALID_DB_SCHEMES:
        raise ValueError(
            f"DATABASE_URL scheme '{parsed.scheme}' not supported. "
            f"Valid schemes: {', '.join(sorted(VALID_DB_SCHEMES))}"
        )

    if not parsed.hostname:
        raise ValueError("DATABASE_URL missing hostname")

    if parsed.port is not None and not (1 <= parsed.port <= 65535):
        raise ValueError(f"DATABASE_URL port {parsed.port} outside valid range (1-65535)")

    if not parsed.path or parsed.path in ("/", ""):
        raise ValueError("DATABASE_URL missing database name (path component)")

    if environment == "production":
        if not parsed.username:
            raise ValueError("DATABASE_URL must include username in production")
        if not parsed.password:
            raise ValueError("DATABASE_URL must include password in production")
        if parsed.scheme == "postgresql":
            raise ValueError(
                "Sync postgresql:// scheme not allowed in production. "
                "Use postgresql+asyncpg:// for async operations."
            )
        if _is_weak_password(parsed.password):
            raise ValueError(
                "DATABASE_URL password is too weak. "
                "Use a password with at least 12 characters."
            )

    return url


def _default_database_url_for_env() -> SecretStr:
    """
    Dev fallback for database_url.

    In production we return empty and let the validator fail-fast.
    """
    env = _current_environment()
    if env == "production":
        # Empty: validator vai recusar com mensagem clara.
        return SecretStr("")
    return SecretStr(DEV_DATABASE_URL)


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class RagConfig(BaseSettings):
    """
    RAG/pgvector configuration (immutable, validated, async-stack aware).

    Loads settings from:
        - ENVIRONMENT / RAG_ENVIRONMENT
        - DATABASE_URL (sem prefixo)
        - RAG_* for all other fields

    Safe to share across async WebSocket handlers, metric collectors, and RAG services.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        validate_default=True,
        env_prefix="RAG_",
        case_sensitive=False,
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Ambiente
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("ENVIRONMENT", "RAG_ENVIRONMENT"),
        description="Deployment environment (development|staging|production).",
    )

    # DB URL — sempre validada via field_validator, independentemente da origem.
    database_url: SecretStr = Field(
        default_factory=_default_database_url_for_env,
        validation_alias="DATABASE_URL",  # ignora env_prefix
        description="PostgreSQL connection URL. Use postgresql+asyncpg://... for async.",
    )

    # Coleções (pgvector / tabelas)
    collection_write: str = Field(
        default="knowledge_v1",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Collection/table name for write operations (embeddings table).",
    )
    collection_read: str = Field(
        default="knowledge_v1",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Collection/table name for read operations (blue/green, etc.).",
    )

    # Embeddings
    embed_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="Embedding model name (e.g. OpenAI text-embedding-3-small).",
    )
    embed_dim: int = Field(
        default=1536,
        ge=MIN_EMBED_DIM,
        le=MAX_EMBED_DIM,
        description=(
            f"Embedding vector dimensions ({MIN_EMBED_DIM}-{MAX_EMBED_DIM}). "
            "Must match model output."
        ),
    )

    # Top-k / vizinhança
    max_top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of results for similarity search.",
    )

    # HNSW index
    hnsw_m: int = Field(
        default=16,
        ge=MIN_HNSW_M,
        le=MAX_HNSW_M,
        description=(
            f"HNSW M parameter: connections per node ({MIN_HNSW_M}-{MAX_HNSW_M}). "
            "Higher → better recall, more memory."
        ),
    )
    hnsw_ef_construction: int = Field(
        default=256,
        ge=MIN_EF_CONSTRUCTION,
        le=MAX_EF_CONSTRUCTION,
        description=(
            f"HNSW ef_construction ({MIN_EF_CONSTRUCTION}-{MAX_EF_CONSTRUCTION}). "
            "Higher → better index, slower build."
        ),
    )

    ef_search_base: int = Field(
        default=64,
        ge=1,
        le=1000,
        description="Base ef_search for HNSW queries (minimum search effort).",
    )
    ef_search_max: int = Field(
        default=128,
        ge=1,
        le=2000,
        description="Maximum ef_search for HNSW queries (maximum search effort).",
    )
    max_neighbors: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Maximum neighbors for graph traversal operations.",
    )

    # Reranking
    enable_rerank: bool = Field(
        default=False,
        description="Enable two-stage reranking pipeline.",
    )
    enable_cross_encoder: bool = Field(
        default=True,
        description="Enable cross-encoder for semantic reranking.",
    )
    cross_encoder_model: str = Field(
        default="BAAI/bge-reranker-small",
        min_length=1,
        description="HuggingFace cross-encoder model for reranking.",
    )
    cross_encoder_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to pass through cross-encoder.",
    )
    cross_encoder_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cross-encoder score to include a result.",
    )

    # Adaptive rerank gating
    rerank_gating_enabled: bool = Field(
        default=True,
        description="Enable adaptive gating to skip reranking when unnecessary.",
    )
    rerank_score_low_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Score below which reranking is triggered. "
            "If top score > threshold, rerank may be skipped."
        ),
    )
    rerank_margin_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description=(
            "Margin threshold for rerank decision. "
            "Rerank if top scores are within this margin."
        ),
    )
    rerank_max_candidates: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of candidates considered for reranking.",
    )

    # -----------------------------------------------------------------------
    # Field validators
    # -----------------------------------------------------------------------

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: SecretStr, info: ValidationInfo) -> SecretStr:
        """
        Always validate database_url, regardless of source (env or default).

        Fecha o bypass original: DATABASE_URL do ambiente não passa mais "cru".
        """
        env = str(info.data.get("environment", "development")).strip().lower() or "development"
        raw = v.get_secret_value().strip()

        if not raw:
            if env == "production":
                raise ValueError(
                    "DATABASE_URL must be set in production. "
                    "Format: postgresql+asyncpg://user:pass@host:5432/dbname"
                )
            # fallback seguro dev
            logger.info(
                "using_dev_database_url",
                url=_sanitize_db_url(DEV_DATABASE_URL),
                environment=env,
            )
            return SecretStr(DEV_DATABASE_URL)

        validated = _validate_database_url_raw(raw, env)
        logger.info("database_url_configured", url=_sanitize_db_url(validated), environment=env)
        return SecretStr(validated)

    # -----------------------------------------------------------------------
    # Model validators (cross-field)
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        """
        Validate cross-field constraints.

        - ef_search_max >= ef_search_base
        - rerank thresholds consistent when gating enabled
        - cross_encoder_top_k <= rerank_max_candidates <= max_top_k
        - max_neighbors <= max_top_k
        - hnsw_ef_construction > hnsw_m
        """
        if self.ef_search_max < self.ef_search_base:
            raise ValueError(
                f"ef_search_max ({self.ef_search_max}) must be >= "
                f"ef_search_base ({self.ef_search_base})"
            )

        if self.rerank_gating_enabled:
            if self.rerank_score_low_threshold <= 0.0:
                raise ValueError(
                    "rerank_score_low_threshold must be > 0.0 when rerank gating is enabled"
                )
            if self.rerank_margin_threshold >= 1.0:
                raise ValueError("rerank_margin_threshold must be < 1.0")

        if self.cross_encoder_top_k > self.rerank_max_candidates:
            raise ValueError(
                f"cross_encoder_top_k ({self.cross_encoder_top_k}) "
                f"cannot exceed rerank_max_candidates ({self.rerank_max_candidates})"
            )

        if self.rerank_max_candidates > self.max_top_k:
            raise ValueError(
                f"rerank_max_candidates ({self.rerank_max_candidates}) "
                f"cannot exceed max_top_k ({self.max_top_k})"
            )

        if self.max_neighbors > self.max_top_k:
            raise ValueError(
                f"max_neighbors ({self.max_neighbors}) "
                f"cannot exceed max_top_k ({self.max_top_k})"
            )

        if self.hnsw_ef_construction <= self.hnsw_m:
            raise ValueError("hnsw_ef_construction should be > hnsw_m for sane HNSW config")

        return self


# ---------------------------------------------------------------------------
# Singleton factory (no import-time side effects)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_config() -> RagConfig:
    """
    Lazy, cached singleton for RagConfig.

    Use as FastAPI dependency:
        from config import get_config, RagConfig

        async def dep(cfg: RagConfig = Depends(get_config)) -> RagConfig:
            return cfg

    In tests, call get_config.cache_clear() para resetar.
    """
    return RagConfig()
