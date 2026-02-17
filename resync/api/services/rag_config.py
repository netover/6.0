"""
RAG Service Configuration.

Configures the RAG system including:
- PostgreSQL/pgvector connection settings
- Embedding model settings
- Chunking parameters

SECURITY (v5.4.1):
- DATABASE_URL has no default password
- Production requires explicit configuration
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _get_secure_database_url() -> str:
    """
    Get DATABASE_URL with security validation.

    - Production: MUST be set via environment variable
    - Development: Falls back to localhost (no password)
    """
    url = os.getenv("DATABASE_URL")
    env = os.getenv("ENVIRONMENT", "development").lower()

    if url:
        # Warn if using obvious default password
        if ":password@" in url:
            logger.warning(
                "insecure_database_url_detected",
                hint="DATABASE_URL contains default password - change for production",
            )
        return url

    # No DATABASE_URL set
    if env == "production":
        raise ValueError(
            "DATABASE_URL must be set in production. "
            "Example: postgresql://user:pass@host:5432/dbname"
        )

    # Development fallback - no password in URL
    return "postgresql://localhost:5432/resync"


@dataclass
class RAGConfig:
    """Configuration for RAG service with pgvector."""

    # PostgreSQL settings - NO default password
    database_url: str = field(default_factory=_get_secure_database_url)
    collection_name: str = field(
        default_factory=lambda: os.getenv("RAG_COLLECTION", "resync_documents")
    )

    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = 1536

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Service settings
    use_mock: bool = field(
        default_factory=lambda: os.getenv("RAG_USE_MOCK", "true").lower() == "true"
    )
    upload_dir: str = field(default_factory=lambda: os.getenv("RAG_UPLOAD_DIR", "uploads"))

    # Search settings
    default_top_k: int = 10
    max_top_k: int = 100

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        return cls()

    def is_pgvector_configured(self) -> bool:
        """Check if pgvector is properly configured."""
        return bool(self.database_url) and not self.use_mock


# Global configuration instance
_config: RAGConfig | None = None


def get_rag_config() -> RAGConfig:
    """Get or create RAG configuration."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def reset_rag_config():
    """Reset configuration (for testing)."""
    global _config
    _config = None
