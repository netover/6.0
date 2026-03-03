from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from resync.core.database.models import Base, get_all_models

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

get_all_models()
target_metadata = Base.metadata

def get_url() -> str:
    url = os.getenv("APP_DATABASE_URL", "")
    if not url:
        raise RuntimeError("APP_DATABASE_URL is required for alembic migrations")
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True, include_schemas=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
