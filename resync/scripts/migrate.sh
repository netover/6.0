#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db ./scripts/migrate.sh
#
# Alembic will read DATABASE_URL (preferred) and fall back to alembic.ini.
# See alembic/env.py for details.

if ! command -v alembic >/dev/null 2>&1; then
  echo "ERROR: alembic not found. Install deps first (pip install -r requirements.txt or pip install alembic)." >&2
  exit 1
fi

alembic upgrade head
echo "OK: alembic upgrade head"
