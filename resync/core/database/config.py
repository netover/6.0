"""
Database Configuration - PostgreSQL Unified Stack.

Production-ready database configuration with PostgreSQL as the only backend.
Supports PostgreSQL extensions:
- pgvector: Vector similarity search for RAG
- pg_trgm: Full-text search with trigrams

All operations are async-only (no psycopg2 sync driver).

Environment Variables:
- APP_DATABASE_URL: PostgreSQL connection string
- DATABASE_HOST: PostgreSQL host (default: localhost)
- DATABASE_PORT: PostgreSQL port (default: 5432)
- DATABASE_NAME: Database name (default: resync)
- DATABASE_USER: Username (default: resync)
- DATABASE_PASSWORD: Password
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote_plus, urlparse

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
    password: str = field(default="", repr=False)

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
        password = self.password or os.getenv("APP_DATABASE_PASSWORD", "")
        encoded = quote_plus(password) if password else ""
        return f"postgresql+asyncpg://{self.user}:{encoded}@{self.host}:{self.port}/{self.name}"

    @property
    def alembic_url(self) -> str:
        """
        Get database URL for Alembic migrations.

        Uses asyncpg driver - Alembic must be configured for async mode.
        See alembic.ini and env.py for async configuration.
        """
        password = self.password or os.getenv("APP_DATABASE_PASSWORD", "")
        encoded = quote_plus(password) if password else ""
        return f"postgresql+asyncpg://{self.user}:{encoded}@{self.host}:{self.port}/{self.name}"

    @property
    def raw_url(self) -> str:
        """
        Get raw PostgreSQL URL (without driver prefix).

        Useful for direct psql connections or third-party tools.
        """
        password = self.password or os.getenv("APP_DATABASE_PASSWORD", "")
        encoded = quote_plus(password) if password else ""
        return (
            f"postgresql://{self.user}:{encoded}@{self.host}:{self.port}/{self.name}"
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
    url = get_effective_database_url()

    if url:
        _validate_database_url_security(url)
        config = _parse_database_url(url)
    else:
        # Build from individual environment variables
        config = DatabaseConfig(
            driver=DatabaseDriver.POSTGRESQL,
            host=os.getenv("APP_DATABASE_HOST", "localhost"),
            port=int(os.getenv("APP_DATABASE_PORT", "5432")),
            name=os.getenv("APP_DATABASE_NAME", "resync"),
            user=os.getenv("APP_DATABASE_USER", "resync"),
            password=os.getenv("APP_DATABASE_PASSWORD", ""),
        )

    # Always override pool settings from specific environment variables if present.
    # We enforce a minimum of 2 for pool_size as requested.
    env_pool_size = os.getenv("APP_DATABASE_POOL_SIZE")
    if env_pool_size:
        config.pool_size = max(2, int(env_pool_size))
    elif config.pool_size < 2:
        config.pool_size = 2

    env_max_overflow = os.getenv("APP_DATABASE_MAX_OVERFLOW")
    if env_max_overflow:
        config.max_overflow = int(env_max_overflow)

    # v6.1.0: Standardized with APP_ prefix
    config.pool_timeout = int(os.getenv("APP_DATABASE_POOL_TIMEOUT", os.getenv("DATABASE_POOL_TIMEOUT", str(config.pool_timeout))))
    config.pool_recycle = int(os.getenv("APP_DATABASE_POOL_RECYCLE", os.getenv("DATABASE_POOL_RECYCLE", str(config.pool_recycle))))
    config.ssl_mode = os.getenv("DATABASE_SSL_MODE", config.ssl_mode)

    return config


def get_effective_database_url() -> str | None:
    """Return the single source of truth for database URL configuration."""
    return os.getenv("APP_DATABASE_URL")

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
    raw = os.getenv("APP_ENVIRONMENT") or ""
    v = raw.strip().lower()
    return v in {"prod", "production"}

def _validate_database_url_security(url: str) -> None:
    """
    Block obviously insecure APP_DATABASE_URL credentials in production.
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
    password: str | None = parsed.password

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
            "APP_DATABASE_URL contains an insecure database password; set a strong password/secret in production"
        )

# Singleton config instance
_config: DatabaseConfig | None = None

def get_config() -> DatabaseConfig:
    """Get singleton database config."""
    global _config
    if _config is None:
        _config = get_database_config()
    return _config
