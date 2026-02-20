"""
Database Configuration - PostgreSQL Unified Stack.

Production-ready database configuration with PostgreSQL as the only backend.
Supports PostgreSQL extensions:
- pgvector: Vector similarity search for RAG
- pg_trgm: Full-text search with trigrams

All operations are async-only (no psycopg2 sync driver).

Environment Variables:
- DATABASE_URL: Full PostgreSQL connection string
- DATABASE_HOST: PostgreSQL host (default: localhost)
- DATABASE_PORT: PostgreSQL port (default: 5432)
- DATABASE_NAME: Database name (default: resync)
- DATABASE_USER: Username (default: resync)
- DATABASE_PASSWORD: Password
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DatabaseDriver(str, Enum):
    """Database driver enumeration."""

    POSTGRESQL = "postgresql"  # Only supported backend


@dataclass
class DatabaseConfig:
    """
    PostgreSQL database configuration.

    PostgreSQL is the only supported database for production and development.
    All operations use asyncpg for async execution.
    """

    driver: DatabaseDriver = DatabaseDriver.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    name: str = "resync"
    user: str = "resync"
    password: str = ""

    # Connection pool settings
    # Otimizado para <100 req/s, 20 usuários simultâneos
    # Total máximo de conexões: pool_size + max_overflow = 15
    pool_size: int = 5  # Conexões mantidas abertas (reduzido de 10)
    max_overflow: int = 10  # Conexões adicionais sob demanda (reduzido de 20)
    pool_timeout: int = 30  # Timeout aguardando conexão disponível
    pool_recycle: int = 1800  # Reciclar conexões após 30min (reduzido de 1h)
    pool_pre_ping: bool = True  # Verificar conexões antes de usar

    # SSL settings
    ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full

    @property
    def url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        password = self.password or os.getenv("DATABASE_PASSWORD", "")
        return f"postgresql+asyncpg://{self.user}:{password}@{self.host}:{self.port}/{self.name}"

    @property
    def alembic_url(self) -> str:
        """
        Get database URL for Alembic migrations.

        Uses asyncpg driver - Alembic must be configured for async mode.
        See alembic.ini and env.py for async configuration.
        """
        password = self.password or os.getenv("DATABASE_PASSWORD", "")
        return f"postgresql+asyncpg://{self.user}:{password}@{self.host}:{self.port}/{self.name}"

    @property
    def raw_url(self) -> str:
        """
        Get raw PostgreSQL URL (without driver prefix).

        Useful for direct psql connections or third-party tools.
        """
        password = self.password or os.getenv("DATABASE_PASSWORD", "")
        return (
            f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.name}"
        )

    def get_pool_options(self) -> dict:
        """Get connection pool options."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
        }


def get_database_config() -> DatabaseConfig:
    """
    Get database configuration from environment.

    Returns:
        DatabaseConfig: Configured database settings
    """
    # Check for full DATABASE_URL first
    url = os.getenv("DATABASE_URL")

    if url:
        _validate_database_url_security(url)
        return _parse_database_url(url)

    # Build from individual environment variables
    return DatabaseConfig(
        driver=DatabaseDriver.POSTGRESQL,
        host=os.getenv("DATABASE_HOST", "localhost"),
        port=int(os.getenv("DATABASE_PORT", "5432")),
        name=os.getenv("DATABASE_NAME", "resync"),
        user=os.getenv("DATABASE_USER", "resync"),
        password=os.getenv("DATABASE_PASSWORD", ""),
        # Keep env defaults aligned with the dataclass defaults (optimized values)
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "1800")),
        ssl_mode=os.getenv("DATABASE_SSL_MODE", "prefer"),
    )


def _parse_database_url(url: str) -> DatabaseConfig:
    """
    Parse a database URL into DatabaseConfig.

    Args:
        url: PostgreSQL connection URL

    Returns:
        DatabaseConfig: Parsed configuration
    """
    config = DatabaseConfig()

    # Handle postgres:// vs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    parsed = urlparse(url)

    config.host = parsed.hostname or "localhost"
    config.port = parsed.port or 5432
    config.name = parsed.path.lstrip("/") if parsed.path else "resync"
    config.user = parsed.username or "resync"
    config.password = parsed.password or ""

    return config


def _is_production_env() -> bool:
    """
    Best-effort env detection without importing Settings (avoid circular deps).
    """
    raw = (
        os.getenv("APP_ENVIRONMENT")
        or os.getenv("ENVIRONMENT")
        or os.getenv("RESYNC_ENVIRONMENT")
        or os.getenv("ENV")
        or ""
    )
    v = raw.strip().lower()
    return v in {"prod", "production"}


def _validate_database_url_security(url: str) -> None:
    """
    Block obviously insecure DATABASE_URL credentials in production.
    We do NOT require a password (some deployments use IAM/certs), but if one
    is present, it must not be a common/default password.
    """
    if not _is_production_env():
        return

    # Normalize postgres://
    normalized = url
    if normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql://", 1)

    parsed = urlparse(normalized)
    password: Optional[str] = parsed.password

    if not password:
        return

    insecure = {
        "password",
        "admin",
        "postgres",
        "root",
        "123456",
        "12345678",
        "change_me",
        "changeme",
    }
    if password.strip().lower() in insecure:
        raise ValueError(
            "DATABASE_URL contains an insecure database password; set a strong password/secret in production"
        )


# Singleton config instance
_config: DatabaseConfig | None = None


def get_config() -> DatabaseConfig:
    """Get singleton database config."""
    global _config
    if _config is None:
        _config = get_database_config()
    return _config
