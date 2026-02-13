"""
Database Schema Creation for PostgreSQL.

Creates all schemas and tables required by Resync.
Run this migration to set up a fresh PostgreSQL database.
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import get_engine
from .models import Base, get_all_models

logger = logging.getLogger(__name__)


# SQL to create schemas
CREATE_SCHEMAS_SQL = """
-- Create schemas for organizing tables
CREATE SCHEMA IF NOT EXISTS tws;
CREATE SCHEMA IF NOT EXISTS context;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS learning;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Grant usage on schemas (adjust roles as needed)
-- GRANT USAGE ON SCHEMA tws TO your_app_role;
-- GRANT USAGE ON SCHEMA context TO your_app_role;
-- GRANT USAGE ON SCHEMA audit TO your_app_role;
-- GRANT USAGE ON SCHEMA analytics TO your_app_role;
-- GRANT USAGE ON SCHEMA learning TO your_app_role;
-- GRANT USAGE ON SCHEMA metrics TO your_app_role;
"""


async def create_schemas(engine: AsyncEngine | None = None) -> None:
    """
    Create all database schemas.

    Args:
        engine: Optional SQLAlchemy async engine. Uses default if not provided.
    """
    if engine is None:
        engine = get_engine()

    async with engine.begin() as conn:
        # Create schemas
        for statement in CREATE_SCHEMAS_SQL.strip().split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                try:
                    await conn.execute(text(statement))
                except Exception as e:
                    logger.warning("Schema creation warning: %s", e)

        logger.info("Database schemas created")


async def create_tables(engine: AsyncEngine | None = None) -> None:
    """
    Create all database tables.

    Args:
        engine: Optional SQLAlchemy async engine. Uses default if not provided.
    """
    if engine is None:
        engine = get_engine()

    # Import all models to ensure they're registered
    _ = get_all_models()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created")


async def drop_all_tables(engine: AsyncEngine | None = None, confirm: bool = False) -> None:
    """
    Drop all tables. USE WITH CAUTION!

    Args:
        engine: Optional SQLAlchemy async engine.
        confirm: Must be True to execute.
    """
    if not confirm:
        raise ValueError("Must pass confirm=True to drop tables")

    if engine is None:
        engine = get_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.warning("All database tables dropped!")


async def initialize_database(engine: AsyncEngine | None = None) -> None:
    """
    Initialize database with all schemas and tables.

    This is the main entry point for database setup.

    Args:
        engine: Optional SQLAlchemy async engine.
    """
    logger.info("Initializing database...")

    await create_schemas(engine)
    await create_tables(engine)

    logger.info("Database initialization complete")


async def check_database_connection(engine: AsyncEngine | None = None) -> bool:
    """
    Check if database connection is working.

    Args:
        engine: Optional SQLAlchemy async engine.

    Returns:
        True if connection successful, False otherwise.
    """
    if engine is None:
        engine = get_engine()

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        return False


def run_migrations() -> None:
    """
    Run migrations synchronously (for CLI usage).
    """
    asyncio.run(initialize_database())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
