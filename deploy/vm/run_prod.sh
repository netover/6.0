#!/usr/bin/env bash
set -euo pipefail

# Production runner for VM deployments (idempotent-ish)
#
# What it does:
#  1) Ensures venv exists (runs install.sh if missing)
#  2) Runs compileall
#  3) Runs Alembic migrations (upgrade head)
#  4) Starts/restarts systemd unit if present
#  5) Runs smoke test against the configured base URL
#
# Required env:
#  - APP_DATABASE_URL
# Optional:
#  - APP_VALKEY_URL
#  - RESYNC_BASE_URL (default http://127.0.0.1:8000)
#  - RESYNC_WS_URL   (default ws://127.0.0.1:8000/ws)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${APP_DATABASE_URL:?APP_DATABASE_URL is required}"

if [[ ! -d ".venv" ]]; then
  echo "[run_prod] .venv missing — running deploy/vm/install.sh"
  ./deploy/vm/install.sh
fi

source .venv/bin/activate

echo "[run_prod] compileall"
python -m compileall -q resync

echo "[run_prod] alembic upgrade head"
python -m alembic upgrade head

if systemctl list-unit-files | grep -q '^resync\.service'; then
  echo "[run_prod] restarting systemd service resync (gunicorn)"
  sudo systemctl restart resync
  sudo systemctl --no-pager --full status resync || true
else
  echo "[run_prod] systemd unit not installed; starting uvicorn in foreground is up to you."
fi

echo "[run_prod] smoke test"
python deploy/vm/smoke.py

echo "✅ run_prod completed"
