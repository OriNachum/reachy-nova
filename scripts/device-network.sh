#!/usr/bin/env bash
# device-network.sh — device-side NetworkManager priority flip, task t1
# (docs/plans/2026-08-26-dual-network-never-downtime.md#t1,
# docs/specs/2026-08-26-dual-network-never-downtime.md).
#
# The Reachy Mini Wireless has one Wi-Fi radio (wlan0) and two NetworkManager
# connection profiles: the iPhone (5) hotspot (travelling network, decided by
# Ori 2026-08-26 to be PREFERRED) and bar-nachum (home Wi-Fi, the fallback).
# Both profiles keep autoconnect=yes; only the relative autoconnect-priority
# changes. The robot has no console, so every mutating step here schedules an
# automatic revert (spec boundary c19) that restores the previous priorities
# unless --commit is called within the window — a mistake can never strand
# the operator without SSH.
#
# Subcommands:
#   --check                 print both profiles' autoconnect /
#                            autoconnect-priority / timestamp; exit 0 if the
#                            order is correct (preferred priority > fallback
#                            priority, both autoconnect=yes), else exit 1.
#                            Never touches the system, never needs root.
#   --dry-run                print the exact nmcli commands --apply would
#                            run; changes nothing, never needs root.
#   --apply [--revert-after SECONDS]
#                            record the current priorities, then set the
#                            preferred/fallback priorities via
#                            `sudo -n nmcli connection modify`, then schedule
#                            an automatic revert after SECONDS (default 300)
#                            unless --commit cancels it first. Uses
#                            `systemd-run --on-active` when available, else a
#                            nohup background sleep+revert job tracked by a
#                            pidfile.
#   --commit                 cancel a pending scheduled revert (the applied
#                            state becomes permanent).
#   --revert                  restore the recorded priorities immediately
#                            (also what the scheduled revert job runs).
#
# Env overrides (also used by tests to stub nmcli / force code paths):
#   REACHY_NMCLI              nmcli binary to use (default: nmcli)
#   REACHY_NET_DRY=1           force --apply to behave like --dry-run
#   REACHY_NET_PREFERRED       preferred profile name (default: "iPhone (5)")
#   REACHY_NET_FALLBACK        fallback profile name (default: "bar-nachum")
#   REACHY_NET_PREFERRED_PRIORITY  target priority for the preferred profile
#                                   (default: 20)
#   REACHY_NET_FALLBACK_PRIORITY   target priority for the fallback profile
#                                   (default: 10)
#   REACHY_NET_NO_SYSTEMD_RUN=1  force the nohup fallback revert-scheduling
#                                path even when systemd-run is on PATH
#
# Idempotent: --check/--dry-run never mutate anything; --apply re-applying
# the same target priorities is a no-op modify; --commit/--revert tolerate
# no pending state (log and exit 0).

set -euo pipefail

NMCLI="${REACHY_NMCLI:-nmcli}"
PREFERRED="${REACHY_NET_PREFERRED:-iPhone (5)}"
FALLBACK="${REACHY_NET_FALLBACK:-bar-nachum}"
PREFERRED_PRIORITY_TARGET="${REACHY_NET_PREFERRED_PRIORITY:-20}"
FALLBACK_PRIORITY_TARGET="${REACHY_NET_FALLBACK_PRIORITY:-10}"

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/reachy"
REVERT_STATE_FILE="$STATE_DIR/network-revert-state.env"
REVERT_PID_FILE="$STATE_DIR/network-revert.pid"
REVERT_UNIT="reachy-network-revert"
DEFAULT_REVERT_AFTER=300

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

log() { printf '[device-network] %s\n' "$*"; }
warn() { printf '[device-network] WARN: %s\n' "$*" >&2; }

usage() {
    cat <<EOF
Usage: device-network.sh --check | --dry-run | --apply [--revert-after SECONDS] | --commit | --revert

  --check                    print both profiles' state; exit 0 iff the
                              preferred profile outranks the fallback and
                              both autoconnect=yes
  --dry-run                  print the nmcli commands --apply would run
  --apply [--revert-after S] apply the priority flip with an automatic
                              revert after S seconds (default $DEFAULT_REVERT_AFTER)
  --commit                   cancel a pending scheduled revert
  --revert                   restore the previously recorded priorities now
EOF
}

conn_field() {
    # conn_field <connection-name> <nmcli-field>
    local name="$1" field="$2"
    "$NMCLI" -t -f "$field" connection show "$name" 2>/dev/null \
        | sed -n "s/^${field}://p" \
        | head -n1
}

cmd_check() {
    local p_ac p_prio p_ts f_ac f_prio f_ts

    p_ac="$(conn_field "$PREFERRED" connection.autoconnect)"
    p_prio="$(conn_field "$PREFERRED" connection.autoconnect-priority)"
    p_ts="$(conn_field "$PREFERRED" connection.timestamp)"
    f_ac="$(conn_field "$FALLBACK" connection.autoconnect)"
    f_prio="$(conn_field "$FALLBACK" connection.autoconnect-priority)"
    f_ts="$(conn_field "$FALLBACK" connection.timestamp)"

    printf 'profile=%s autoconnect=%s autoconnect-priority=%s timestamp=%s\n' \
        "$PREFERRED" "${p_ac:-?}" "${p_prio:-?}" "${p_ts:-?}"
    printf 'profile=%s autoconnect=%s autoconnect-priority=%s timestamp=%s\n' \
        "$FALLBACK" "${f_ac:-?}" "${f_prio:-?}" "${f_ts:-?}"

    if [[ "$p_ac" == "yes" && "$f_ac" == "yes" \
        && "${p_prio:-0}" -gt "${f_prio:-0}" ]] 2>/dev/null; then
        log "order OK: '$PREFERRED' ($p_prio) outranks '$FALLBACK' ($f_prio), both autoconnect=yes"
        return 0
    fi

    warn "order NOT OK: want '$PREFERRED' priority > '$FALLBACK' priority, both autoconnect=yes"
    return 1
}

print_apply_commands() {
    printf '  sudo -n %s connection modify %s connection.autoconnect yes connection.autoconnect-priority %s\n' \
        "$NMCLI" "$PREFERRED" "$PREFERRED_PRIORITY_TARGET"
    printf '  sudo -n %s connection modify %s connection.autoconnect yes connection.autoconnect-priority %s\n' \
        "$NMCLI" "$FALLBACK" "$FALLBACK_PRIORITY_TARGET"
}

cmd_dry_run() {
    log "dry run — would apply:"
    print_apply_commands
}

cancel_pending_revert() {
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user stop "${REVERT_UNIT}.timer" >/dev/null 2>&1 || true
        systemctl --user stop "${REVERT_UNIT}.service" >/dev/null 2>&1 || true
    fi

    if [[ -f "$REVERT_PID_FILE" ]]; then
        local pid
        pid="$(cat "$REVERT_PID_FILE" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$REVERT_PID_FILE"
    fi
}

schedule_revert() {
    local seconds="$1"

    cancel_pending_revert

    if [[ "${REACHY_NET_NO_SYSTEMD_RUN:-0}" != "1" ]] && command -v systemd-run >/dev/null 2>&1; then
        log "scheduling revert via systemd-run in ${seconds}s (unit=$REVERT_UNIT)"
        if systemd-run --user --unit="$REVERT_UNIT" --on-active="${seconds}s" \
            -- "$SELF" --revert; then
            return 0
        fi
        warn "systemd-run failed to schedule the revert — falling back to a background job"
    fi

    log "scheduling revert via background job in ${seconds}s (no systemd-run)"
    nohup bash -c "sleep '$seconds'; exec '$SELF' --revert" >/dev/null 2>&1 &
    disown
    echo "$!" > "$REVERT_PID_FILE"
}

# Restores the previously recorded state (priority + autoconnect) for both
# profiles and clears the pending-revert bookkeeping. Used both by an
# explicit/scheduled `--revert` and by `--apply`'s own failure path — assumes
# the caller already confirmed $REVERT_STATE_FILE exists.
perform_revert() {
    local PREFERRED_NAME FALLBACK_NAME
    local PREFERRED_PRIORITY FALLBACK_PRIORITY
    local PREFERRED_AUTOCONNECT FALLBACK_AUTOCONNECT
    # shellcheck source=/dev/null
    source "$REVERT_STATE_FILE"

    log "reverting: '$PREFERRED_NAME' -> autoconnect $PREFERRED_AUTOCONNECT priority $PREFERRED_PRIORITY, '$FALLBACK_NAME' -> autoconnect $FALLBACK_AUTOCONNECT priority $FALLBACK_PRIORITY"
    sudo -n "$NMCLI" connection modify "$PREFERRED_NAME" \
        connection.autoconnect "$PREFERRED_AUTOCONNECT" connection.autoconnect-priority "$PREFERRED_PRIORITY"
    sudo -n "$NMCLI" connection modify "$FALLBACK_NAME" \
        connection.autoconnect "$FALLBACK_AUTOCONNECT" connection.autoconnect-priority "$FALLBACK_PRIORITY"

    rm -f "$REVERT_STATE_FILE" "$REVERT_PID_FILE"
}

cmd_apply() {
    local revert_after="$1"

    mkdir -p "$STATE_DIR"

    if [[ "${REACHY_NET_DRY:-0}" == "1" ]]; then
        cmd_dry_run
        return 0
    fi

    local cur_p_prio cur_f_prio cur_p_ac cur_f_ac
    cur_p_prio="$(conn_field "$PREFERRED" connection.autoconnect-priority)"
    cur_f_prio="$(conn_field "$FALLBACK" connection.autoconnect-priority)"
    cur_p_ac="$(conn_field "$PREFERRED" connection.autoconnect)"
    cur_f_ac="$(conn_field "$FALLBACK" connection.autoconnect)"
    cur_p_prio="${cur_p_prio:-0}"
    cur_f_prio="${cur_f_prio:-0}"
    cur_p_ac="${cur_p_ac:-yes}"
    cur_f_ac="${cur_f_ac:-yes}"

    # ARM the rollback FIRST — record the originals and schedule the revert —
    # before touching NetworkManager at all, so a failure partway through the
    # mutations (or an unreachable systemd-run) can never leave a stray,
    # unrevertable change behind.
    {
        printf 'PREFERRED_NAME=%q\n' "$PREFERRED"
        printf 'FALLBACK_NAME=%q\n' "$FALLBACK"
        printf 'PREFERRED_PRIORITY=%q\n' "$cur_p_prio"
        printf 'FALLBACK_PRIORITY=%q\n' "$cur_f_prio"
        printf 'PREFERRED_AUTOCONNECT=%q\n' "$cur_p_ac"
        printf 'FALLBACK_AUTOCONNECT=%q\n' "$cur_f_ac"
    } > "$REVERT_STATE_FILE"

    schedule_revert "$revert_after"

    log "applying: '$PREFERRED' priority $cur_p_prio -> $PREFERRED_PRIORITY_TARGET, '$FALLBACK' priority $cur_f_prio -> $FALLBACK_PRIORITY_TARGET"

    if ! sudo -n "$NMCLI" connection modify "$PREFERRED" \
        connection.autoconnect yes connection.autoconnect-priority "$PREFERRED_PRIORITY_TARGET"; then
        warn "failed to modify '$PREFERRED' — rolling back the armed revert immediately"
        cancel_pending_revert
        perform_revert
        exit 1
    fi

    if ! sudo -n "$NMCLI" connection modify "$FALLBACK" \
        connection.autoconnect yes connection.autoconnect-priority "$FALLBACK_PRIORITY_TARGET"; then
        warn "failed to modify '$FALLBACK' — rolling back the armed revert immediately"
        cancel_pending_revert
        perform_revert
        exit 1
    fi

    log "revert scheduled in ${revert_after}s — run '$SELF --commit' to keep this change"
}

cmd_commit() {
    cancel_pending_revert
    rm -f "$REVERT_STATE_FILE"
    log "committed — pending revert cancelled"
}

cmd_revert() {
    if [[ ! -f "$REVERT_STATE_FILE" ]]; then
        warn "no pending revert state found ($REVERT_STATE_FILE) — nothing to revert"
        return 0
    fi

    perform_revert
}

main() {
    local action="" revert_after="$DEFAULT_REVERT_AFTER"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check)
                action="check"
                shift
                ;;
            --dry-run)
                action="dry-run"
                shift
                ;;
            --apply)
                action="apply"
                shift
                ;;
            --revert-after)
                revert_after="${2:?--revert-after requires SECONDS}"
                shift 2
                ;;
            --commit)
                action="commit"
                shift
                ;;
            --revert)
                action="revert"
                shift
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                warn "unknown argument: $1"
                usage
                exit 2
                ;;
        esac
    done

    if [[ "${REACHY_NET_DRY:-0}" == "1" && "$action" == "apply" ]]; then
        action="dry-run"
    fi

    case "$action" in
        check) cmd_check ;;
        dry-run) cmd_dry_run ;;
        apply) cmd_apply "$revert_after" ;;
        commit) cmd_commit ;;
        revert) cmd_revert ;;
        "")
            usage
            exit 2
            ;;
    esac
}

main "$@"
