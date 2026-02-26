"""Alembic async migration environment for Resync.

Usage:
    alembic upgrade head      # apply all migrations
    alembic downgrade -1      # rollback one
    alembic revision --autogenerate -m "description"
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Resolve DATABASE_URL from environment (preferred) or alembic.ini template
# ---------------------------------------------------------------------------
def _get_database_url() -> str:
    """Resolve the database URL from environment variables."""
    # APP_DATABASE_URL takes priority (matches application settings)
    url = os.getenv("APP_DATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        # Ensure asyncpg driver for async migrations
        return url.replace("postgresql://", "postgresql+asyncpg://")

    # Fall back to alembic.ini template substitution
    return config.get_main_option("sqlalchemy.url", "")  # type: ignore[return-value]


config.set_main_option("sqlalchemy.url", _get_database_url())

# ---------------------------------------------------------------------------
# Import models so Alembic can autogenerate migrations
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import all models to register them with metadata
from resync.core.database.models import Base, get_all_models  # noqa: E402
get_all_models()  # Ensure all model classes are imported and registered

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
