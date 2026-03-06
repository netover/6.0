#!/usr/bin/env bash
set -euo pipefail

# Configure Valkey for Resync (basic hardening)
# Requires: sudo privileges, valkey-server installed and running
#
# Env:
#   RESYNC_VALKEY_PASS (required)
#   RESYNC_VALKEY_PORT (default 6379)

RESYNC_VALKEY_PASS="${RESYNC_VALKEY_PASS:?RESYNC_VALKEY_PASS is required}"
RESYNC_VALKEY_PORT="${RESYNC_VALKEY_PORT:-6379}"

CONF="/etc/valkey/valkey.conf"
if [[ ! -f "$CONF" ]]; then
  echo "Valkey config not found at $CONF"
  exit 1
fi

echo "Updating valkey.conf (requirepass, protected-mode, bind)..."
sudo sed -i "s/^# *requirepass .*/requirepass ${RESYNC_VALKEY_PASS}/" "$CONF" || true
if ! grep -q "^requirepass " "$CONF"; then
  echo "requirepass ${RESYNC_VALKEY_PASS}" | sudo tee -a "$CONF" >/dev/null
fi

# ensure protected-mode yes
sudo sed -i "s/^protected-mode .*/protected-mode yes/" "$CONF" || true

# bind localhost by default (adjust as needed)
sudo sed -i "s/^bind .*/bind 127.0.0.1 ::1/" "$CONF" || true

# port
sudo sed -i "s/^port .*/port ${RESYNC_VALKEY_PORT}/" "$CONF" || true

sudo systemctl restart valkey-server

echo "✅ Valkey configured."
echo "Suggested env:"
echo "  export APP_VALKEY_URL='valkey://:${RESYNC_VALKEY_PASS}@127.0.0.1:${RESYNC_VALKEY_PORT}/0'"
