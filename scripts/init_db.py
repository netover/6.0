#!/usr/bin/env python3
"""
Database initialization script for first production deployment.

Usage:
    # From project root:
    python scripts/init_db.py

    # Or via Docker:
    docker run --rm --env-file .env resync:prod python scripts/init_db.py

This script is idempotent — safe to run multiple times.
For schema changes AFTER the first deploy, use alembic migrations instead.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("init_db")


async def main() -> None:
    """Initialize the database schema."""
    logger.info("Starting database initialization...")

    try:
        from resync.core.database.schema import initialize_database, check_database_connection
        from resync.core.database.engine import get_engine

        engine = get_engine()

        logger.info("Checking database connectivity...")
        if not await check_database_connection(engine):
            logger.error("Cannot connect to database. Check DATABASE_URL.")
            sys.exit(1)

        logger.info("Database reachable. Creating schemas and tables...")
        await initialize_database(engine)
        logger.info("✓ Database initialization complete.")

    except Exception as exc:
        logger.error("Database initialization FAILED: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        # Ensure engine is disposed
        try:
            from resync.core.database.engine import _engine
            if _engine:
                await _engine.dispose()
        except Exception as e:
            pass  # best-effort cleanup on exit — engine may already be gone


if __name__ == "__main__":
    asyncio.run(main())


async def verify_db_init() -> None:
    """Best-effort verification that core schemas exist after initialization."""
    from sqlalchemy import text
    from resync.core.database.engine import async_engine

    async with async_engine.begin() as conn:
        # Verify schemas exist (learning, security, etc.); adjust list if schema list changes.
        schemas = ["tws", "context", "audit", "analytics", "learning", "metrics"]
        for s in schemas:
            q = text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :s")
            res = await conn.execute(q, {"s": s})
            if res.first() is None:
                print(f"WARNING: schema not found: {s}")

