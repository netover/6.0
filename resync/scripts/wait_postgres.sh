#!/usr/bin/env bash
set -euo pipefail

# Simple wait-for-Postgres script for CI/local use.
# Env:
#   PGHOST, PGPORT (default 5432), PGUSER, PGPASSWORD, PGDATABASE
# Requires: psql

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGDATABASE="${PGDATABASE:-postgres}"
TIMEOUT="${TIMEOUT:-30}"

if ! command -v psql >/dev/null 2>&1; then
  echo "ERROR: psql not found. Install postgresql-client." >&2
  exit 1
fi

echo "Waiting for Postgres at ${PGHOST}:${PGPORT} (timeout ${TIMEOUT}s)..."
start="$(date +%s)"
while true; do
  if PGPASSWORD="${PGPASSWORD:-}" psql "host=${PGHOST} port=${PGPORT} user=${PGUSER} dbname=${PGDATABASE}" -c "select 1" >/dev/null 2>&1; then
    echo "Postgres is ready."
    exit 0
  fi
  now="$(date +%s)"
  if (( now - start >= TIMEOUT )); then
    echo "ERROR: Postgres not ready after ${TIMEOUT}s." >&2
    exit 1
  fi
  sleep 1
done
