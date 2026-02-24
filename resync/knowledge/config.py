"""
Configuration for the pgvector-based RAG system.
Version: 6.0.0 — Pydantic v2 BaseSettings (FIXED)
"""

import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # ← BaseSettings ADICIONADO

logger = structlog.get_logger(__name__)


def _get_database_url() -> str:
    """Valida e retorna DATABASE_URL com sanitização para logging."""
    url = os.getenv("DATABASE_URL")
    env = os.getenv("ENVIRONMENT", "development").strip().lower()

    if url:
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.hostname]):
                raise ValueError("Invalid DATABASE_URL format.")
        except Exception as e:
            raise ValueError(f"DATABASE_URL parsing failed: {e}") from e

        if env == "production" and parsed.password:
            weak = {"password", "admin", "123456", "postgres", "root", "test", "changeme"}
            if parsed.password.lower() in weak:
                raise ValueError("Weak database password detected in production.")

        logger.info("database_url_configured", url=_sanitize_db_url(url), environment=env)
        return url

    if env == "production":
        raise ValueError(
            "DATABASE_URL must be set in production. "
            "Example: postgresql+asyncpg://user:pass@host:5432/dbname"
        )

    default_url = "postgresql://localhost:5432/resync"
    logger.info("using_dev_database_url", url=default_url, environment=env)
    return default_url


def _sanitize_db_url(url: str) -> str:
    """Redact password from URL for safe logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        return "[DATABASE_URL_PARSE_ERROR]"
    return url  # sem senha — seguro retornar


def _bool(env: str, default: bool = False) -> bool:
    """Parse boolean env var. Aceita: 1/true/yes/on; 0/false/no/off."""
    v = os.getenv(env)
    if v is None:
        return default
    v = v.strip()
    if not v:
        return default
    v_lower = v.lower()
    if v_lower in {"1", "true", "yes", "on"}:
        return True
    if v_lower in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid_boolean_env_var", name=env, value=v, default=default)
    return default


class RagConfig(BaseSettings):  # ← HERANÇA CORRIGIDA
    """
    Configuração RAG com pgvector.
    Carrega automaticamente env vars com prefixo RAG_ (exceto DATABASE_URL).
    Imutável após criação (frozen=True).
    """

    model_config = SettingsConfigDict(
        frozen=True,
        validate_default=True,
        env_prefix="RAG_",
        case_sensitive=False,
    )

    # DATABASE_URL lido sem prefixo RAG_ via default_factory
    database_url: str = Field(
        default_factory=_get_database_url,
        description="PostgreSQL connection URL (set via DATABASE_URL env var)",
    )

    collection_write: str = Field(default="knowledge_v1")
    collection_read: str = Field(default="knowledge_v1")

    embed_model: str = Field(default="text-embedding-3-small")
    embed_dim: int = Field(default=1536, ge=128, le=4096)

    max_top_k: int = Field(default=50, ge=1, le=1000)

    hnsw_m: int = Field(default=16, ge=4, le=64)
    hnsw_ef_construction: int = Field(default=256, ge=10, le=10000)

    ef_search_base: int = Field(default=64, ge=1, le=1000)
    ef_search_max: int = Field(default=128, ge=1, le=2000)
    max_neighbors: int = Field(default=32, ge=1, le=100)

    enable_rerank: bool = Field(default=False)
    enable_cross_encoder: bool = Field(default=True)
    cross_encoder_model: str = Field(default="BAAI/bge-reranker-small")
    cross_encoder_top_k: int = Field(default=5, ge=1, le=50)
    cross_encoder_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    rerank_gating_enabled: bool = Field(default=True)
    rerank_score_low_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rerank_margin_threshold: float = Field(default=0.05, ge=0.0, le=0.5)
    rerank_max_candidates: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_ef_search_order(self) -> "RagConfig":
        """ef_search_max deve ser >= ef_search_base."""
        if self.ef_search_max < self.ef_search_base:
            raise ValueError(
                f"ef_search_max ({self.ef_search_max}) must be >= "
                f"ef_search_base ({self.ef_search_base})"
            )
        return self

    @model_validator(mode="after")
    def validate_rerank_thresholds(self) -> "RagConfig":
        """Valida consistência dos thresholds de reranking."""
        if self.rerank_gating_enabled:
            if self.rerank_score_low_threshold <= 0.0:
                raise ValueError("rerank_score_low_threshold must be > 0.0 when gating enabled")
            if self.rerank_margin_threshold >= 1.0:
                raise ValueError("rerank_margin_threshold must be < 1.0")
        return self


# Singleton global — imutável, thread-safe após criação
CFG: RagConfig = RagConfig()
