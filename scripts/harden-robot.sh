#!/usr/bin/env bash
# harden-robot.sh — P0 security hardening checks for the on-robot Reachy Nova
# deployment (task t13, docs/plans/2026-08-09-...#t13).
#
# Idempotent: safe to re-run any number of times. Requires no sudo.
#
# Checks performed:
#   1. .env holding the AWS credentials is chmod 600 (not group/world
#      readable). Verified with `ls -l` after the chmod; fails loudly if the
#      mode is not exactly 600.
#   2. mosquitto's :1883 listener, if running, is bound to localhost only.
#      Verified with `ss -tlnp`. Bound to any non-loopback address is a
#      hard failure naming the finding; not running at all is a warning.
#   3. Prints a residual-exposure summary naming the LAN-open, unauthenticated
#      upstream daemons this harness does not own or attempt to close:
#      ReachyMiniOS daemon :8000 and Zenoh :7447.
#
# Usage:
#   scripts/harden-robot.sh [path-to-env-file]
#
# Defaults to ~/git/reachy-nova/.env when no path is given.

set -euo pipefail

ENV_FILE="${1:-$HOME/git/reachy-nova/.env}"

log() { printf '[harden-robot] %s\n' "$*"; }
fail() { printf '[harden-robot] FAIL: %s\n' "$*" >&2; exit 1; }
warn() { printf '[harden-robot] WARN: %s\n' "$*" >&2; }

# --- 1. .env permissions -------------------------------------------------
# The .env carries AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. It must not be
# readable by any user other than the owner.
if [[ -f "$ENV_FILE" ]]; then
    log "chmod 600 on $ENV_FILE"
    chmod 600 "$ENV_FILE"

    log "verifying: ls -l $ENV_FILE"
    ls_line="$(ls -l "$ENV_FILE")"
    log "$ls_line"

    mode="$(printf '%s' "$ls_line" | awk '{print $1}')"
    if [[ "$mode" != "-rw-------" ]]; then
        fail ".env at $ENV_FILE is mode '$mode' after chmod 600, expected '-rw-------' (600) — AWS credentials remain exposed to other local users"
    fi
    log "OK: $ENV_FILE is 600 (owner read/write only)"
else
    warn ".env not found at $ENV_FILE — skipping permission check (nothing to secure yet)"
fi

# --- 2. mosquitto binds localhost-only -----------------------------------
# The nervous-system MQTT broker must never be reachable from the LAN:
# allow_anonymous is true in config/mosquitto/mosquitto.conf, so any
# LAN-reachable listener on :1883 is an open, unauthenticated broker.
log "verifying: ss -tlnp (looking for :1883)"
mosquitto_line="$(ss -tlnp 2>/dev/null | grep ':1883' || true)"

if [[ -z "$mosquitto_line" ]]; then
    warn "mosquitto is not listening on :1883 right now — not running is a warning, not a hardening failure"
else
    log "$mosquitto_line"
    if printf '%s' "$mosquitto_line" | grep -qE '(^|[^0-9.:])(0\.0\.0\.0|\*|\[::\]):1883'; then
        fail "mosquitto :1883 is bound to all interfaces (0.0.0.0/*), not 127.0.0.1 — LAN clients can reach the broker unauthenticated (allow_anonymous is true)"
    fi
    if ! printf '%s' "$mosquitto_line" | grep -qE '(127\.0\.0\.1|\[::1\]):1883'; then
        fail "mosquitto :1883 listener found but not bound to 127.0.0.1/::1 — inspect the ss output above and rebind the listener to loopback"
    fi
    log "OK: mosquitto :1883 is bound to localhost only"
fi

# --- 3. residual exposure summary ----------------------------------------
cat <<'EOF'
[harden-robot]
[harden-robot] Residual exposure summary (accepted, upstream ReachyMiniOS reality):
[harden-robot]   - ReachyMiniOS daemon on :8000 — LAN-open, unauthenticated. Not owned by this
[harden-robot]     harness; upstream ships it this way today.
[harden-robot]   - Zenoh on :7447            — LAN-open, unauthenticated. Not owned by this
[harden-robot]     harness; upstream ships it this way today.
[harden-robot]   This harness itself opens no new network listeners beyond the
[harden-robot]   localhost-only mosquitto broker verified above. See docs/security.md.
EOF

log "harden-robot.sh completed successfully"
