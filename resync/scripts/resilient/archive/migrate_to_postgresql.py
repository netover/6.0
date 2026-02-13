#!/usr/bin/env python3
"""
Migration script from SQLite to PostgreSQL.

Usage:
    python scripts/migrate_to_postgresql.py

Environment Variables:
    - SQLITE_PATH: Path to SQLite database (default: resync_dev.db)
    - DATABASE_URL: PostgreSQL connection string
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_sqlite_to_postgresql():
    """Migrate data from SQLite to PostgreSQL."""

    # Check environment
    sqlite_path = os.getenv("SQLITE_PATH", "resync_dev.db")
    pg_url = os.getenv("DATABASE_URL")

    if not pg_url:
        logger.error("DATABASE_URL environment variable is required")
        logger.info("Example: postgresql://user:pass@localhost:5432/resync")
        return False

    if not Path(sqlite_path).exists():
        logger.error(f"SQLite database not found: {sqlite_path}")
        return False

    logger.info(f"Starting migration from {sqlite_path} to PostgreSQL")

    try:
        import sqlite3

        import asyncpg

        # Connect to SQLite
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]

        logger.info(f"Found {len(tables)} tables to migrate")

        # Connect to PostgreSQL
        pg_conn = await asyncpg.connect(pg_url)

        def _validate_identifier(name: str) -> str:
            """Validate and quote a SQL identifier to prevent injection."""
            if not name.isidentifier() and not all(c.isalnum() or c == '_' for c in name):
                raise ValueError(f"Invalid SQL identifier: {name!r}")
            # Double-quote to safely escape identifiers
            return f'"{name}"'

        for table in tables:
            if table.startswith("sqlite_"):
                continue

            logger.info(f"Migrating table: {table}")

            # Validate table name before use in SQL
            safe_table = _validate_identifier(table)

            # Get data from SQLite
            cursor.execute(f"SELECT * FROM {safe_table}")
            rows = cursor.fetchall()

            if not rows:
                logger.info(f"  No data in {table}")
                continue

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Insert into PostgreSQL with safe identifier quoting
            for row in rows:
                placeholders = ", ".join([f"${i + 1}" for i in range(len(columns))])
                safe_cols = ", ".join(_validate_identifier(c) for c in columns)

                query = f"INSERT INTO {safe_table} ({safe_cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

                try:
                    await pg_conn.execute(query, *list(row))
                except Exception as e:
                    logger.warning(f"  Row insert failed: {e}")

            logger.info(f"  Migrated {len(rows)} rows")

        # Cleanup
        await pg_conn.close()
        sqlite_conn.close()

        logger.info("Migration completed successfully!")
        return True

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install asyncpg")
        return False
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(migrate_sqlite_to_postgresql())
    sys.exit(0 if success else 1)
