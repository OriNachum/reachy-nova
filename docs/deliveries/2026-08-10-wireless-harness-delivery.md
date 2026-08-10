# Expected Delivery — Wireless harness over reachy-mini-cli (waves 1–3 + device bring-up)

Date opened: 2026-08-10 · branch: `wireless-harness` · plan:
`docs/plans/2026-08-09-reachy-nova-ships-as-an-on-device-ai-harness-over.md`
· spec: `docs/specs/2026-08-09-…-harness-over.md`

This document is the delivery contract for the PR that closes this arc. It is
written **before** implementation fan-out; the PR is judged against it.

## What ships in the PR (repo deliverables)

A new `reachy_nova/harness/` package — an on-device AI peripheral that attaches
to reachy-mini-cli's symbolic runtime **only via its documented seams**, never
via a `reachy_mini` SDK client:

| Piece | Plan task | Contract it implements |
|---|---|---|
| `reachy_nova/config.py` | t1 (done) | All model IDs + region env-configurable, old literals as defaults |
| `reachy_nova/harness/statedir.py` | t1 (done) | State-dir map; engine heartbeat liveness (`state.json` `updated`, TTL 2s/skew 1s); embody PID+argv exclusivity check |
| `reachy_nova/nova_omni.py` | t2 | Request-response clip+image+text understanding; Nova 2 Lite fallback exercised by a forced-failure test (no network) |
| Packaging: `pyproject` name `reachy-nova`, console script `reachy-nova-harness`, `.github/workflows/publish.yml` | t3 | Publish pipeline mirroring reachy-mini-cli (pytest gate, TestPyPI devN on PRs, PyPI trusted publishing on main) |
| `tests/test_harness_boundary.py` | t4 | AST walk over `reachy_nova/harness/` forbidding `reachy_mini` imports and `set_target` — in the merge gate |
| `reachy_nova/harness/bus.py` + localhost mosquitto conf | t6 | paho subscriber on `reachy/events/#` + `reachy/state/#` (QoS 0, `REACHY_MQTT_URL`, default `localhost:1883`); nervous-system rules.yaml prioritization routed to `sonic.inject_text`; retained harness state topic + Last Will |
| `reachy_nova/harness/cognition_feed.py` | t10 | thinking/message/emotion NDJSON per reachy-mini-cli `docs/export-schema.md` — reterminal bridge parses it unmodified |
| `reachy_nova/harness/supervisor.py` + unit render/install | t11 | PID supervisor; refuses start while `agent embody` is live; systemd `--user` unit `reachy-nova-harness.service` `After=reachy-runtime.service`, NOT a presence unit; `[SENSE stage=… source=nova …]` journald lines |
| `scripts/harden-robot.sh` + `docs/security.md` | t13 | chmod 600 `.env`, IAM policy scoped to invoked model ARNs, mosquitto bound 127.0.0.1, documented residual LAN exposure |
| `reachy_nova/harness/hearing.py` | t7 | Audio-tee client: `AF_UNIX` stream, one JSON header line (`f32le` mono, samplerate may be `null` — handled, never guessed), partial-sample buffering, `np.interp` resample to 16k, reconnect-with-backoff, **half-duplex echo gate** (feed suppressed while the speaker is playing) |
| `reachy_nova/harness/speaking.py` | t8 (adapted) | **Deviation from plan, verified 2026-08-10**: the #162 streaming speaker feed does not exist in reachy-mini-cli 0.48.0. Speak path = daemon HTTP route (`POST /api/media/sounds/upload` multipart WAV → `POST /api/media/play_sound`), the proven `agent embody` route. Utterance-buffered playback; playback failure/preemption routes to Sonic's interruption path — no stuck `_speaking` state. Adapts to #162 when it ships. |
| `reachy_nova/harness/tools.py` | t9 | Sonic `toolConfiguration` built over the intents spool: `run_behavior`, `declare_goal`, `set_mode`, `set_inhibition`, `goto`, plus `create_rule` (managed `nova-` block in `rules.toml` + reload spool). Every tool returns the `await_result` payload or the degraded submitted-only note — no call silently vanishes |
| Memory leg | t14 | qq context injection routed through the bus/nervous-system path (throttle preserved), no direct `inject_text` bypass |
| Nova Act flag | t15 | `NOVA_ACT_ENABLED=0` default; flag off ⇒ zero Playwright import |
| `tests/chaos/` | t12 | Local-fake chaos suite: harness kill, wifi down/up, bad AWS creds, runtime-restart-under-live-harness — each case asserts a named `[SENSE]` drop and clean reconnect; on-robot equivalents listed in the checklist |

Test gate: `uv run pytest` green (baseline 186 tests + everything above).

## What ships on the device (192.168.1.162, not in the PR diff)

Recorded in the scope doc's verification section with command output:

1. reachy-mini-cli checkout pulled to v0.48.0 and editable-installed into its
   venv (was: stale 0.29.0 install).
2. `reachy-runtime.service` hand-authored (upstream text minus
   `Requires=reachy-daemon.service` — ReachyMiniOS owns the SDK daemon as a
   system service; deviation to be filed upstream), enabled; demo-mode
   disabled. Rollback = `service enable demo` (< 5 min drill).
3. mosquitto installed, bound to 127.0.0.1 only.
4. `reachy-nova` checkout on the harness branch, venv synced,
   `reachy-nova-harness.service` installed `After=reachy-runtime.service`;
   linger already on.
5. `.env` chmod 600.

## Acceptance evidence (the "Survive restart" ask)

The PR description links recorded results for:

- **Voice roundtrip**: tee → Sonic → HTTP playback on the robot; transcript +
  `[SENSE]` lines as evidence. Echo test: speaker playing Sonic audio in a
  silent room produces no self-transcription.
- **Runtime restart under live harness**: `systemctl --user restart
  reachy-runtime` mid-session → harness logs named drops, re-attaches within
  10s without its own restart.
- **Harness restart**: `systemctl --user restart reachy-nova-harness` →
  fresh Sonic session, seams re-attach.
- **Full reboot**: power-cycle → both units come back by themselves; runtime
  owns the single SDK/media session; harness attaches; conversation works.
- **Wifi pull**: runtime presence (rules, breathing) unaffected; harness
  reconnects with named drops when wifi returns.

## Explicitly deferred (not in this PR)

- t5 — behavior lock-in LibraryEntry PRs upstream to reachy-mini-cli
  (cross-repo; separate arc).
- t16 sign-off items beyond the evidence above (full 3-scenario acceptance +
  ReachyMiniApp retirement PR) — retirement stays gated on the acceptance run.
- Nova 2 Omni live verification (preview enablement pending) — Lite-2 fallback
  is the tested path.
- Slack/feedback legs (v2 per spec decision).
