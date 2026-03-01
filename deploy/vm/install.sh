#!/usr/bin/env bash
set -euo pipefail

# Resync VM installer (Ubuntu/Debian oriented)
# - Creates venv
# - Installs Python deps
# - Initializes DB schema/tables
# - Optionally configures systemd service

PYTHON_BIN="${PYTHON_BIN:-python3.14}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
DEV_REQ_FILE="${DEV_REQ_FILE:-requirements-dev.txt}"

echo "[1/6] Checking Python..."
$PYTHON_BIN -V

echo "[2/6] Creating virtualenv at $VENV_DIR"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/6] Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

echo "[4/6] Installing requirements"
python -m pip install -r "$REQ_FILE"
if [[ -f "$DEV_REQ_FILE" ]]; then
  python -m pip install -r "$DEV_REQ_FILE"
fi

echo "[5/6] Compileall sanity"
python -m compileall -q resync

echo "[6/6] Database init (requires APP_DATABASE_URL to point to an existing DB)"
python scripts/init_db.py

echo "✅ Install complete."
echo "Next:"
echo "  - Export APP_DATABASE_URL and APP_REDIS_URL (or configure .env)"
echo "  - Run: source $VENV_DIR/bin/activate && uvicorn resync.main:app --host 0.0.0.0 --port 8000"
