#!/usr/bin/env bash
# Install and enable the reachy-nova autostart service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="reachy-nova-autostart.service"

echo "Installing ${SERVICE_FILE}..."
sudo cp "${SCRIPT_DIR}/${SERVICE_FILE}" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_FILE}"

echo "Done. ${SERVICE_FILE} is enabled and will run on next boot."
echo "To start it now: sudo systemctl start ${SERVICE_FILE}"
echo "To check logs:   journalctl -u ${SERVICE_FILE}"
