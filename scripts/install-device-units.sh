#!/usr/bin/env bash
# install-device-units.sh — idempotent installer for the three on-device
# systemd --user units that encode boot exclusivity (task t3,
# docs/plans/2026-08-11-harness-round-2-alive-senses-resilient-start.md#t3),
# plus the journald persistence drop-in and legacy-unit mask that were
# hand-applied live on the Reachy Mini Wireless (pollen, 2026-08-11/12) and
# are reproduced here so they survive a reinstall.
#
# Units written:
#   - reachy-runtime.service       enabled  — the symbolic runtime presence
#                                              loop; Conflicts=reachy-demo-mode.service
#   - reachy-nova-harness.service  enabled  — the harness peripheral, via the
#                                              existing `python -m reachy_nova.harness
#                                              install-unit` CLI (unchanged)
#   - reachy-demo-mode.service     NOT enabled — manual-only (no [Install]);
#                                              Conflicts=reachy-runtime.service
#
# reachy-runtime.service and reachy-demo-mode.service are rendered from
# reachy_nova.harness.unit's pure text functions (runtime_unit_text /
# demo_mode_unit_text) — the single source of truth also covered by
# tests/test_harness_unit.py — so this script can never drift from what is
# tested.
#
# Also:
#   - masks the legacy system unit reachy-nova-autostart.service (sudo,
#     tolerates absence — a fresh device never had it, and masking an
#     absent/already-masked unit is itself idempotent)
#   - writes /etc/systemd/journald.conf.d/reachy-nova.conf with
#     Storage=persistent + SystemMaxUse=64M and restarts systemd-journald.
#     The device journald is volatile by default (journalctl finds no journal
#     files — debugging on 2026-08-12 had to fall back to the in-memory ring)
#     and the root disk is ~93% full, so persistence needs a hard cap, not
#     unbounded persistent storage.
#
# Idempotent: every step tolerates already-applied state. Safe to re-run.
#
# Usage:
#   scripts/install-device-units.sh [cli-venv-python] [nova-venv-python]
#
# Defaults:
#   cli-venv-python  = $HOME/reachy-mini-cli/.venv/bin/python
#   nova-venv-python = python3 (resolved from PATH; on the device this is the
#                      reachy_nova venv's interpreter)

set -euo pipefail

CLI_PYTHON="${1:-$HOME/reachy-mini-cli/.venv/bin/python}"
NOVA_PYTHON="${2:-python3}"

log() { printf '[install-device-units] %s\n' "$*"; }
warn() { printf '[install-device-units] WARN: %s\n' "$*" >&2; }

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

# --- 1. render + write the three units -----------------------------------
log "rendering reachy-runtime.service (cli python: $CLI_PYTHON)"
"$NOVA_PYTHON" -c "
from reachy_nova.harness import unit
print(unit.runtime_unit_text(python='$CLI_PYTHON'), end='')
" > "$UNIT_DIR/reachy-runtime.service"

log "rendering reachy-demo-mode.service (cli python: $CLI_PYTHON)"
"$NOVA_PYTHON" -c "
from reachy_nova.harness import unit
print(unit.demo_mode_unit_text(python='$CLI_PYTHON'), end='')
" > "$UNIT_DIR/reachy-demo-mode.service"

log "installing reachy-nova-harness.service via the existing install-unit CLI"
# --env-file is load-bearing on the device: the harness reads AWS credentials
# and model config from the repo's .env, not from environment.d drop-ins.
NOVA_ENV_FILE="${NOVA_ENV_FILE:-$HOME/git/reachy-nova/.env}"
if [ -f "$NOVA_ENV_FILE" ]; then
  "$NOVA_PYTHON" -m reachy_nova.harness install-unit --env-file "$NOVA_ENV_FILE"
else
  warn "no env file at $NOVA_ENV_FILE — installing without --env-file"
  "$NOVA_PYTHON" -m reachy_nova.harness install-unit
fi

log "systemctl --user daemon-reload"
systemctl --user daemon-reload

# --- 2. enable runtime + harness, never demo ------------------------------
# reachy-demo-mode.service is deliberately never named here: it has no
# [Install] section (unit.py's demo_mode_unit_text), so `enable` would fail
# on it anyway — but it is also never attempted, because a demo loop that
# could come up unattended at boot is the second presence this hardening
# exists to rule out.
log "enabling reachy-runtime.service and reachy-nova-harness.service (not demo-mode)"
systemctl --user enable reachy-runtime.service reachy-nova-harness.service

# --- 3. mask the legacy system autostart unit -----------------------------
# reachy-nova-autostart.service is the old ReachyMiniApp preset-enabled unit,
# superseded by the runtime+harness pair above. Masking it prevents a stale
# preset from waking it at boot and fighting the new pair for the same
# audio/motor resources. `systemctl mask` on an absent or already-masked unit
# is itself a no-op success, so this is idempotent without extra checks.
log "masking reachy-nova-autostart.service (sudo, tolerates absence)"
if sudo systemctl mask reachy-nova-autostart.service; then
    log "masked reachy-nova-autostart.service"
else
    warn "could not mask reachy-nova-autostart.service (continuing — may not exist on this device)"
fi

# --- 4. journald persistence drop-in --------------------------------------
log "writing /etc/systemd/journald.conf.d/reachy-nova.conf"
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\nSystemMaxUse=64M\n' \
    | sudo tee /etc/systemd/journald.conf.d/reachy-nova.conf >/dev/null

log "restarting systemd-journald"
sudo systemctl restart systemd-journald

log "install-device-units.sh completed successfully"
