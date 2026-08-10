#!/usr/bin/env bash
# Auto-start reachy_nova app via the reachy-mini daemon API.
# Waits for the daemon API, ensures the robot backend is running, then starts the app.

set -euo pipefail

DAEMON_URL="http://localhost:8000"
APP_NAME="reachy_nova"
TIMEOUT="${AUTOSTART_TIMEOUT:-120}"
POLL_INTERVAL=2

elapsed=0

echo "Waiting for reachy-mini daemon API (timeout: ${TIMEOUT}s)..."

# Phase 1: Wait for the daemon HTTP API to be reachable
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    if curl -sf "${DAEMON_URL}/api/apps/current-app-status" > /dev/null 2>&1; then
        echo "Daemon API is reachable after ${elapsed}s."
        break
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Timed out after ${TIMEOUT}s waiting for daemon API" >&2
    exit 1
fi

# Phase 2: Ensure the robot backend is initialized
daemon_state=$(curl -sf "${DAEMON_URL}/api/daemon/status" | python3 -c "import json,sys; print(json.load(sys.stdin).get('state',''))" 2>/dev/null || echo "")

if [ "$daemon_state" = "not_initialized" ]; then
    echo "Daemon backend not initialized. Starting it..."
    curl -sf -X POST "${DAEMON_URL}/api/daemon/start?wake_up=true" > /dev/null 2>&1 || true
fi

# Phase 3: Wait for backend to be running
echo "Waiting for daemon backend to be running..."
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    daemon_state=$(curl -sf "${DAEMON_URL}/api/daemon/status" | python3 -c "import json,sys; print(json.load(sys.stdin).get('state',''))" 2>/dev/null || echo "")
    if [ "$daemon_state" = "running" ]; then
        echo "Daemon backend is running after ${elapsed}s. Starting ${APP_NAME}..."
        response=$(curl -sf -X POST "${DAEMON_URL}/api/apps/start-app/${APP_NAME}" 2>&1) && {
            echo "Started ${APP_NAME}: ${response}"
            exit 0
        } || {
            echo "Failed to start ${APP_NAME}: ${response}" >&2
            exit 1
        }
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "Timed out after ${TIMEOUT}s waiting for daemon backend (last state: ${daemon_state})" >&2
exit 1
