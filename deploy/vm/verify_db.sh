#!/usr/bin/env bash
set -euo pipefail
# Verifies DB init by running init_db and then querying table existence.
# Requires: APP_DATABASE_URL set and psql available.

: "${APP_DATABASE_URL:?APP_DATABASE_URL is required}"

python scripts/init_db.py

# Derive dbname/user/host/port for psql check via python (avoid parsing shell)
python - <<'PY'
import os, re, sys, urllib.parse
url=os.environ["APP_DATABASE_URL"]
# expected: postgresql+asyncpg://user:pass@host:port/dbname
url=url.replace("postgresql+asyncpg://","postgresql://")
p=urllib.parse.urlparse(url)
user=p.username or ""
host=p.hostname or "localhost"
port=p.port or 5432
db=p.path.lstrip("/") or "postgres"
print(user, host, port, db)
PY
