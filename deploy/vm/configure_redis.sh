#!/usr/bin/env bash
set -euo pipefail

# Configure Redis for Resync (basic hardening)
# Requires: sudo privileges, redis-server installed and running
#
# Env:
#   RESYNC_REDIS_PASS (required)
#   RESYNC_REDIS_PORT (default 6379)

RESYNC_REDIS_PASS="${RESYNC_REDIS_PASS:?RESYNC_REDIS_PASS is required}"
RESYNC_REDIS_PORT="${RESYNC_REDIS_PORT:-6379}"

CONF="/etc/redis/redis.conf"
if [[ ! -f "$CONF" ]]; then
  echo "Redis config not found at $CONF"
  exit 1
fi

echo "Updating redis.conf (requirepass, protected-mode, bind)..."
sudo sed -i "s/^# *requirepass .*/requirepass ${RESYNC_REDIS_PASS}/" "$CONF" || true
if ! grep -q "^requirepass " "$CONF"; then
  echo "requirepass ${RESYNC_REDIS_PASS}" | sudo tee -a "$CONF" >/dev/null
fi

# ensure protected-mode yes
sudo sed -i "s/^protected-mode .*/protected-mode yes/" "$CONF" || true

# bind localhost by default (adjust as needed)
sudo sed -i "s/^bind .*/bind 127.0.0.1 ::1/" "$CONF" || true

# port
sudo sed -i "s/^port .*/port ${RESYNC_REDIS_PORT}/" "$CONF" || true

sudo systemctl restart redis-server

echo "✅ Redis configured."
echo "Suggested env:"
echo "  export APP_REDIS_URL='redis://:${RESYNC_REDIS_PASS}@127.0.0.1:${RESYNC_REDIS_PORT}/0'"
