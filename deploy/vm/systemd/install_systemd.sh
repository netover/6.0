#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/resync"
ENV_DIR="/etc/resync"
LOG_DIR="/var/log/resync"

sudo mkdir -p "$ENV_DIR" "$LOG_DIR"
sudo chown -R root:root "$ENV_DIR"
sudo chmod 750 "$ENV_DIR"

if ! id -u resync >/dev/null 2>&1; then
  sudo useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin resync
fi

sudo mkdir -p "$APP_DIR"
sudo chown -R resync:resync "$APP_DIR"
sudo chown -R resync:resync "$LOG_DIR"

sudo cp deploy/vm/systemd/resync.service /etc/systemd/system/resync.service
sudo systemctl daemon-reload
sudo systemctl enable resync.service

echo "✅ systemd unit installed"
echo "Next: create /etc/resync/resync.env and deploy code to /opt/resync"
