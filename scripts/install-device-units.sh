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
#   - provisions the nova-writer Kiro agent config (config/kiro/nova-writer.json
#     -> ~/.kiro/agents/nova-writer.json), task t5. Kiro is optional on the
#     device, so this step is guarded on both sides (repo config present,
#     kiro-cli on PATH, ~/.kiro writable) and never fails the whole install —
#     see install_nova_writer_agent_config() below.
#   - installs the NetworkManager dispatcher hook for dual-network failover
#     (config/network/90-reachy-failover -> /etc/NetworkManager/dispatcher.d/,
#     task t3). Guarded, self-testing and SELF-REVERTING: if the hook's
#     dry-run self-test fails, the hook is removed again and the install
#     continues with a warning — a broken network hook is worse than no hook.
#     Skip it with --no-failover; see install_failover_hook() below.
#
# Idempotent: every step tolerates already-applied state. Safe to re-run.
#
# Usage:
#   scripts/install-device-units.sh [--no-failover] [cli-venv-python] [nova-venv-python]
#
# Flags:
#   --no-failover    skip installing the NetworkManager failover dispatcher
#                    hook (everything else is installed as usual)
#
# Defaults:
#   cli-venv-python  = $HOME/reachy-mini-cli/.venv/bin/python
#   nova-venv-python = python3 (resolved from PATH; on the device this is the
#                      reachy_nova venv's interpreter)

set -euo pipefail

INSTALL_FAILOVER=1
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --no-failover) INSTALL_FAILOVER=0 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

CLI_PYTHON="${POSITIONAL[0]:-$HOME/reachy-mini-cli/.venv/bin/python}"
NOVA_PYTHON="${POSITIONAL[1]:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '[install-device-units] %s\n' "$*"; }
warn() { printf '[install-device-units] WARN: %s\n' "$*" >&2; }

# --- 0. resolve what the failover hook will actually run ------------------
# The dispatcher hook (config/network/90-reachy-failover) carries the
# REFERENCE DEVICE's defaults compiled in. This install may not be that
# device: a different user, a different checkout, a different venv. So the
# installer resolves the three values the hook needs and renders them into
# /etc/default/reachy-failover, which the hook sources. Without this the
# installer would happily self-test a custom NOVA_PYTHON and then install a
# hook pointing at the hardcoded /home/pollen venv, which fails its own
# `[ -x ]` guard and silently skips every network event.
FAILOVER_DEFAULTS_FILE="${FAILOVER_DEFAULTS_FILE:-/etc/default/reachy-failover}"

resolve_python() {
    # An absolute path if we can find one: the hook guards on `[ -x ]`, which
    # a bare `python3` can never satisfy.
    local raw="$1" resolved=""
    resolved="$(command -v -- "$raw" 2>/dev/null || true)"
    [ -n "$resolved" ] || resolved="$raw"
    case "$resolved" in
        /*) ;;
        *) resolved="$(readlink -f -- "$resolved" 2>/dev/null || printf '%s' "$resolved")" ;;
    esac
    printf '%s\n' "$resolved"
}

NOVA_PYTHON_RESOLVED="$(resolve_python "$NOVA_PYTHON")"
# Same cascade as reachy_nova.netfailover.default_statedir().
FAILOVER_STATE_DIR="${REACHY_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/reachy}"
FAILOVER_USER="${REACHY_NOVA_USER:-$(id -un)}"

render_failover_defaults() {
    cat <<EOF
# /etc/default/reachy-failover — rendered by scripts/install-device-units.sh.
# Sourced by /etc/NetworkManager/dispatcher.d/90-reachy-failover so the hook
# runs the interpreter this install actually resolved. Re-rendered on every
# re-run of the installer; edit the installer, not this file.
REACHY_NOVA_PYTHON=$NOVA_PYTHON_RESOLVED
REACHY_STATE_DIR=$FAILOVER_STATE_DIR
REACHY_NOVA_USER=$FAILOVER_USER
EOF
}

# Testing seam (tests/test_failover_hook.py): print what the failover step
# would render, then stop — no units written, no sudo, no systemctl.
if [ "${INSTALL_DRY_RUN:-0}" = "1" ]; then
    log "dry run: would write $FAILOVER_DEFAULTS_FILE containing:"
    render_failover_defaults
    log "dry run: would self-test with $NOVA_PYTHON_RESOLVED"
    exit 0
fi

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

# --- 5. provision the nova-writer Kiro agent config (optional) -----------
# Copies config/kiro/nova-writer.json to ~/.kiro/agents/nova-writer.json so
# the on-device Kiro writer (task t5/t6, FORGE_WRITER=kiro) has its agent
# config in place before any ACP session starts it. Kiro itself is optional
# on the device: this step is guarded on both sides and never fails the
# whole install — it warns and continues (does not exit non-zero) if the
# repo's config is missing, if kiro-cli isn't on PATH, or if ~/.kiro can't
# be created. Plain `cp` of the same file twice is idempotent.
install_nova_writer_agent_config() {
    local repo_config="$REPO_ROOT/config/kiro/nova-writer.json"
    local kiro_agents_dir="$HOME/.kiro/agents"

    if [ ! -f "$repo_config" ]; then
        warn "no $repo_config — skipping nova-writer agent config provisioning"
        return 0
    fi

    if ! command -v kiro-cli >/dev/null 2>&1; then
        warn "kiro-cli not found on PATH — skipping nova-writer agent config provisioning"
        return 0
    fi

    if ! mkdir -p "$kiro_agents_dir" 2>/dev/null; then
        warn "could not create $kiro_agents_dir — skipping nova-writer agent config provisioning"
        return 0
    fi

    if ! cp "$repo_config" "$kiro_agents_dir/nova-writer.json" 2>/dev/null; then
        warn "could not copy $repo_config to $kiro_agents_dir/nova-writer.json — skipping nova-writer agent config provisioning"
        return 0
    fi
    log "provisioned nova-writer agent config to $kiro_agents_dir/nova-writer.json"
}

log "provisioning nova-writer Kiro agent config (optional — guarded, non-fatal if kiro is absent)"
install_nova_writer_agent_config

# --- 6. NetworkManager failover dispatcher hook (optional) ----------------
# Copies config/network/90-reachy-failover to
# /etc/NetworkManager/dispatcher.d/90-reachy-failover (root:root, 0755) so the
# dual-network failover policy (reachy_nova.netpolicy, driven by
# reachy_nova.netfailover) runs as root on wlan0 events — task t3, spec claims
# c22/h16/c26/h22.
#
# The step is guarded on both sides (repo script present, dispatcher.d
# present, sudo -n usable) and never fails the whole install.
#
# It is also SELF-REVERTING, which is the point of the self-test: after
# copying the hook we run `python -m reachy_nova.netfailover --self-test`,
# a DRY-RUN decision against the live scan that activates nothing and writes
# nothing. If that fails — a missing venv, an unimportable reachy_nova, no
# nmcli — the hook is REMOVED again and we warn. A dispatcher hook that
# errors on every network event is strictly worse than no hook at all, and
# this is the network we would otherwise be breaking while breaking it.
# Skipped entirely with --no-failover.
install_failover_hook() {
    local repo_hook="$REPO_ROOT/config/network/90-reachy-failover"
    local dispatcher_dir="/etc/NetworkManager/dispatcher.d"
    local installed="$dispatcher_dir/90-reachy-failover"

    if [ ! -f "$repo_hook" ]; then
        warn "no $repo_hook — skipping failover hook install"
        return 0
    fi

    if [ ! -d "$dispatcher_dir" ]; then
        warn "no $dispatcher_dir (NetworkManager dispatcher absent) — skipping failover hook install"
        return 0
    fi

    if ! sudo -n true 2>/dev/null; then
        warn "passwordless sudo unavailable — skipping failover hook install"
        return 0
    fi

    # The hook's own defaults are the reference device's. Render this
    # install's real values first, so the hook that lands one line below is
    # already pointing at the interpreter we are about to self-test.
    if render_failover_defaults | sudo -n tee "$FAILOVER_DEFAULTS_FILE" >/dev/null 2>&1; then
        log "wrote $FAILOVER_DEFAULTS_FILE (python=$NOVA_PYTHON_RESOLVED state=$FAILOVER_STATE_DIR user=$FAILOVER_USER)"
    else
        warn "could not write $FAILOVER_DEFAULTS_FILE — skipping failover hook install"
        warn "(the hook's compiled-in defaults are the reference device's and may not match this install)"
        return 0
    fi

    # `install` sets owner, group and mode in one atomic step, so the hook is
    # never briefly world-writable in the directory NM executes as root.
    if ! sudo -n install -o root -g root -m 0755 "$repo_hook" "$installed" 2>/dev/null; then
        warn "could not install $installed — skipping failover hook install"
        return 0
    fi
    log "installed $installed (root:root 0755)"

    # Self-test through EXACTLY the interpreter and state dir the installed
    # hook will use — a self-test that passes under a different interpreter
    # than the hook runs proves nothing about the hook.
    log "running failover self-test (dry run — activates nothing, writes nothing)"
    if REACHY_STATE_DIR="$FAILOVER_STATE_DIR" \
        "$NOVA_PYTHON_RESOLVED" -m reachy_nova.netfailover --self-test; then
        log "failover self-test passed — hook left in place"
        return 0
    fi

    warn "failover self-test FAILED — reverting: removing $installed"
    sudo -n rm -f "$FAILOVER_DEFAULTS_FILE" 2>/dev/null || true
    if sudo -n rm -f "$installed" 2>/dev/null; then
        warn "removed $installed (the robot is left exactly as it was before this step)"
    else
        warn "could NOT remove $installed — remove it by hand before the next network event"
    fi
    return 0
}

if [ "$INSTALL_FAILOVER" -eq 1 ]; then
    log "installing NetworkManager failover hook (guarded, self-testing, self-reverting)"
    install_failover_hook
else
    log "--no-failover given — skipping the NetworkManager failover hook"
fi

# --- 7. tailnet presence check (informational, never fatal) ---------------
# The robot is meant to be a Tailscale node (docs/setup.md "tailnet first",
# spec c25). Installing tailscale is a documented operator step, not this
# script's job — it only reports whether the node is up so a fresh device
# cannot silently ship without its primary reachability path.
if command -v tailscale >/dev/null 2>&1; then
    if tailscale status --json 2>/dev/null | grep -q '"BackendState": *"Running"'; then
        log "tailscale: node is up ($(tailscale ip -4 2>/dev/null | head -1))"
    else
        warn "tailscale is installed but not logged in — run: sudo tailscale up --hostname reachy-mini"
    fi
else
    warn "tailscale not installed — see docs/setup.md 'Reaching the robot: tailnet first'"
fi

log "install-device-units.sh completed successfully"
