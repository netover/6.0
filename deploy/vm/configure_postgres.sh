#!/usr/bin/env bash
set -euo pipefail

# Configure PostgreSQL for Resync (creates role+db)
# Requires: sudo privileges, psql installed, postgres service running
#
# Env:
#   RESYNC_DB_NAME (default: resync)
#   RESYNC_DB_USER (default: resync)
#   RESYNC_DB_PASS (required)
#   RESYNC_DB_HOST (default: localhost)
#   RESYNC_DB_PORT (default: 5432)

RESYNC_DB_NAME="${RESYNC_DB_NAME:-resync}"
RESYNC_DB_USER="${RESYNC_DB_USER:-resync}"
RESYNC_DB_PASS="${RESYNC_DB_PASS:?RESYNC_DB_PASS is required}"
RESYNC_DB_HOST="${RESYNC_DB_HOST:-localhost}"
RESYNC_DB_PORT="${RESYNC_DB_PORT:-5432}"

echo "Creating role/database if missing..."
sudo -u postgres psql <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='${RESYNC_DB_USER}') THEN
    CREATE ROLE ${RESYNC_DB_USER} LOGIN PASSWORD '${RESYNC_DB_PASS}';
  END IF;
END
\$\$;

DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname='${RESYNC_DB_NAME}') THEN
    CREATE DATABASE ${RESYNC_DB_NAME} OWNER ${RESYNC_DB_USER};
  END IF;
END
\$\$;
SQL

echo "✅ Postgres configured."
echo "Suggested env:"
echo "  export APP_DATABASE_URL='postgresql+asyncpg://${RESYNC_DB_USER}:${RESYNC_DB_PASS}@${RESYNC_DB_HOST}:${RESYNC_DB_PORT}/${RESYNC_DB_NAME}'"
