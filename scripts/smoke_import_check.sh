#!/usr/bin/env bash
set -euo pipefail

# Smoke checks for Resync (Python 3.14 target)
# - import-time wiring
# - compileall
# - basic HTTP/WS route checks via TestClient

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3.14}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "[smoke] using python: $PYTHON_BIN";

export PYTHONUNBUFFERED=1
export RESYNC_ENV="${RESYNC_ENV:-development}"

# 1) Syntax/import validation

echo "[smoke] compileall"
"$PYTHON_BIN" -m compileall -q resync

echo "[smoke] import app"
"$PYTHON_BIN" - <<'PY'
from resync.main import app
print("import ok:", type(app))
PY

# 2) Minimal runtime checks (no external services required)
# If your environment requires secrets/DB, ensure they are present before running.

echo "[smoke] TestClient basic endpoints"
"$PYTHON_BIN" - <<'PY'
import os
from fastapi.testclient import TestClient

from resync.main import app

client = TestClient(app)

# Basic health endpoints (should be cheap)
for path in ("/liveness", "/readiness"):
    r = client.get(path)
    assert r.status_code in (200, 503), (path, r.status_code, r.text)

# Prometheus endpoint (may be protected by token)
r = client.get("/metrics")
if r.status_code == 401:
    token = os.getenv("METRICS_TOKEN")
    if token:
        r = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
assert r.status_code in (200, 401, 404), ("/metrics", r.status_code, r.text)

# WebSocket smoke (best-effort): if endpoint is present and accepts, we connect then close.
# This will raise WebSocketDisconnect if not accepted.
ws_paths = [
    "/api/monitoring/ws",
    "/monitoring/ws",
]

for ws_path in ws_paths:
    try:
        with client.websocket_connect(ws_path) as ws:
            # Some servers immediately send data; don't require it.
            ws.close()
            print("ws ok:", ws_path)
            break
    except Exception as e:
        # Not fatal if route isn't enabled in this deployment.
        print("ws skip:", ws_path, "->", type(e).__name__)

print("smoke ok")
PY

echo "[smoke] done"
