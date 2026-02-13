from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Alembic Config object, provides access to values in alembic.ini.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---- Project metadata ----
# Import Base and models so that autogenerate can detect schema changes.
from resync.core.database.engine import Base  # noqa: E402
# Ensure all model modules are imported so tables are registered on Base.metadata.
from resync.core.database import models as _models  # noqa: F401,E402

target_metadata = Base.metadata


def _get_database_url() -> str:
    # Prefer env var for enterprise deployments.
    url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("DATABASE_URL is not set and alembic.ini sqlalchemy.url is empty.")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Uses only a URL, doesn't require a DBAPI to be present.
    """
    url = _get_database_url()
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


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an AsyncEngine.

    Based on Alembic's official async template.
    """
    # Override URL from env if provided.
    config.set_main_option("sqlalchemy.url", _get_database_url())

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations() -> None:
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        import asyncio
        asyncio.run(run_migrations_online())


run_migrations()
