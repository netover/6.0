#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m compileall -q resync
python deploy/vm/smoke.py
