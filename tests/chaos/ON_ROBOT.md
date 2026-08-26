# On-robot chaos checklist

Each local chaos test in this directory has a real-hardware equivalent. Run
these on the Reachy Mini Wireless (device: `192.168.1.162`; mDNS unreliable)
with the harness running under systemd (`reachy-nova-harness.service`) and the
runtime under `reachy-runtime`. Watch the harness log with:

```bash
journalctl --user -fu reachy-nova-harness | grep -F '[SENSE'
```

Every case below must show BOTH the named drop line and the recovery line,
with **no harness restart** in between (check `systemctl --user show
reachy-nova-harness -p NRestarts` before and after — it must not change,
except in case 3 where the harness itself is the thing being killed).

## 1. Runtime restart — "the body dies under the mind"

Local: `test_chaos_runtime_restart.py`

1. Confirm audio is flowing (speak near the robot; `stage=hear` lines with
   `event=header` / chunks being fed).
2. Restart the runtime:

   ```bash
   systemctl --user restart reachy-runtime
   ```

3. Grep for the named drop (either wording is a pass):

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -E 'stage=hear.*event=disconnect.*(tee-closed|tee-read-failed)'
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -F 'tee-unavailable'
   ```

4. Within 10 s of the runtime being back (`systemctl --user status
   reachy-runtime`), grep for the recovery:

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -E 'stage=hear.*event=connect.*reconnected'
   ```

5. Spool survival: while the runtime is DOWN (between `systemctl --user stop
   reachy-runtime` and `start`), ask Nova to move ("nod for me"). Expect a
   `stage=act ... degraded ... reason=engine did not confirm in time` line, a
   surviving file under
   `~/.local/state/reachy/behavior/intents/commands/`, and — once the engine
   is started again — the motion actually happening (the late engine drains
   the spool).

## 2. AWS / cloud loss — "bad creds mid-conversation"

Local: `test_chaos_aws_loss.py`

1. Mid-conversation, break the mouth. Two equivalent injections:
   - **Bad creds**: edit the env file the unit bakes in (`.env`), set
     `AWS_ACCESS_KEY_ID=AKIAINVALIDINVALID`, and `systemctl --user restart
     reachy-nova-harness` — Sonic's stream dies; or, without any restart,
   - **Dead media route**: `systemctl --user stop reachy-runtime` right as
     the robot starts a sentence (the playback POST gets connection refused).
2. Grep the named drop and the purge:

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -F 'dropped reason=playback-http-failed'
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -F 'dropped reason=preempted-after-failure'
   ```

3. Heal (restore the real key / `systemctl --user start reachy-runtime`),
   then talk to the robot again. Recovery is a fresh
   `stage=speak ... played duration=` line with `NRestarts` unchanged in the
   dead-media variant.
4. Confirm the mic never stayed gated: no `echo gate armed` line without a
   matching `echo gate cleared` after the failure.

## 3. Harness kill — "harness dies, robot unaffected + clean rebirth"

Local: `test_chaos_harness_kill.py`

1. Note the harness PID: `cat ~/.local/state/reachy/nova-harness.pid`.
2. Kill it the ugly way (no SIGTERM, no cleanup):

   ```bash
   kill -9 "$(cat ~/.local/state/reachy/nova-harness.pid)"
   ```

3. Robot unaffected: the runtime keeps breathing/idling
   (`systemctl --user status reachy-runtime` still active; the body still
   moves). The stale PID file is still on disk — that is expected.
4. Rebirth: systemd restarts the harness (or start it by hand:
   `systemctl --user start reachy-nova-harness`). Grep the reclaim and the
   engine pickup:

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -F 'reclaimed stale pid='
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -F 'engine live'
   ```

5. Clean SIGTERM path: `systemctl --user stop reachy-nova-harness` must show
   `harness down` (and component stops) in the log, exit within the unit's
   stop timeout, and remove `nova-harness.pid`.
6. Engine-loss visibility while the harness lives: stop the runtime for ~10 s
   and grep `dropped reason=engine-heartbeat-lost`, then start it and grep
   the second `engine live`.

## 4. Wifi flap — "network down/up under a live session"

Local: `test_chaos_wifi_flap.py`

> On the stock robot the broker is localhost, so a wifi flap does not sever
> the MQTT session — run this case with the mesh/remote broker configured
> (`REACHY_MQTT_URL` pointing off-box), or emulate it locally with
> `systemctl restart mosquitto` (same session loss, same code path).

1. Flap the radio:

   ```bash
   nmcli radio wifi off
   sleep 15
   nmcli radio wifi on
   ```

2. Grep the named drop while the network is down:

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -E 'stage=bus.*event=disconnect.*paho will retry'
   ```

3. After the wifi returns, grep the recovery — a reconnect (either label,
   depending on whether paho noticed the drop first), the runtime's Last
   Will flip, and its return:

   ```bash
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -E 'stage=bus.*event=(connect|reconnect)'
   journalctl --user -u reachy-nova-harness --since -2min \
     | grep -E 'event=runtime-(offline|online)'
   ```

4. Replay hygiene: right after the reconnect there must be NO inject caused
   by retained state — any retained replay shows up only as
   `dropped reason=not-an-event-topic` under `stage=route`, never as a
   `stage=inject` line without a fresh runtime event behind it.
5. Recovery proof: trigger a rule fire (speak to the robot) and grep a fresh
   `stage=inject ... injecting priority=` line.

## 5. Kiro degraded start — "the writer dies at cold boot and must retry itself"

Local: `tests/test_kiro_session.py` (section 7, "Degraded start", and section 8,
`request_restart`)

> Live finding, 2026-08-26: the harness started before Wi-Fi associated,
> `kiro-cli` exited on spawn, `start()` raised, and the supervisor logged
> `start failed name=kiro_session detail=kiro-cli process exited` — nothing
> retried, and the writer (`FORGE_WRITER=kiro`) stayed absent for hours until
> a human restarted the unit. `reachy_nova/harness/kiro_session.py:27-39` is
> the fix: the initial spawn is treated exactly like every later one, so a
> failure comes up **degraded** (watchdog armed) instead of propagating.

Preconditions: `reachy-nova-harness.service` enabled, `FORGE_WRITER=kiro`.
The unit's `.env` is loaded once at process start via `load_dotenv(path,
override=False)` (`reachy_nova/harness/supervisor.py:357-364`, called from the
`run` command at `supervisor.py:388`); on the robot this is
`$HOME/git/reachy-nova/.env` (`scripts/install-device-units.sh:148`,
`NOVA_ENV_FILE`).

### Step A — reproduce the start()-raises path without a reboot

Edit the harness `.env` to point `KIRO_CLI_BIN` at a binary that cannot
possibly exist, then restart the harness once (this restart is the fault
injection, the same shape as case 2's bad-creds restart — it is not the
"no restart between drop and recovery" restart the checklist rules out):

```bash
sed -i '/^KIRO_CLI_BIN=/d' "$HOME/git/reachy-nova/.env"
echo 'KIRO_CLI_BIN=/nonexistent/kiro-cli' >> "$HOME/git/reachy-nova/.env"
systemctl --user restart reachy-nova-harness
```

Grep for the degraded start (present) and the old failure (must be absent):

```bash
journalctl --user -u reachy-nova-harness --since -2min \
  | grep -F 'started degraded (initial spawn failed'
journalctl --user -u reachy-nova-harness --since -2min \
  | grep -F 'started name=kiro_session'
journalctl --user -u reachy-nova-harness --since -2min \
  | grep -F 'start failed name=kiro_session'   # must return NOTHING
```

The watchdog keeps retrying under capped exponential backoff
(`kiro_session.py:524-565`) rather than giving up; confirm it is alive with:

```bash
journalctl --user -u reachy-nova-harness --since -2min \
  | grep -F 'restart failed:'
```

### Step B — recover without a manual restart of the harness

**Restoring `KIRO_CLI_BIN` in `.env` alone is NOT enough**, and this is the
honest mechanism, verified by reading the code rather than assumed:

- `KiroAcpSession.__init__` reads the binary path from `os.environ` at
  construction time (`reachy_nova/kiro_acp.py:175-177`:
  `source = os.environ if env is None else env; self._binary = binary if
  binary is not None else source.get(BINARY_ENV, DEFAULT_BINARY)`).
- A new `KiroAcpSession` is constructed on *every* spawn attempt — the
  watchdog's `_spawn_session()` calls `self._session_factory()` fresh each
  time (`kiro_session.py:394-424`) — so the env lookup itself is not cached
  across retries.
- But `os.environ` is the *process's* environment, populated once at harness
  start by `load_dotenv(path, override=False)` (`supervisor.py:364`).
  Rewriting `.env` on disk does not touch a running process's `os.environ` —
  there is no re-read anywhere in `kiro_session.py` / `kiro_acp.py` /
  `supervisor.py`. So every subsequent watchdog respawn — including one
  forced early via `request_restart` — retries the exact same
  `/nonexistent/kiro-cli` path and fails again. Confirm this empirically
  before moving on:

  ```bash
  sed -i 's#^KIRO_CLI_BIN=.*#KIRO_CLI_BIN=kiro-cli#' "$HOME/git/reachy-nova/.env"
  # no harness restart yet — this must still fail:
  journalctl --user -fu reachy-nova-harness | grep -F 'restart failed:'
  ```

  This is the honest conclusion for *this specific fault class*: an
  `os.environ`-only value baked in at process exec genuinely needs a harness
  restart to change. That restart is out of scope for "recovery without a
  restart" and is deferred to h12's own remaining-work item (a real
  power-cycle drill).

- What *does* re-resolve on every spawn, from disk, with no env or restart
  involved, is the `~/.local/bin` fallback path in
  `KiroAcpSession.start()` (`reachy_nova/kiro_acp.py:269-277`): for a bare
  (no `os.sep`) binary name, `shutil.which(argv[0])` and
  `candidate.exists()` are both live filesystem checks made fresh at spawn
  time. So the recovery this case actually demonstrates without touching the
  harness process is: the binary becoming resolvable again on disk, picked
  up by the *next* spawn attempt — accelerated via the network-change seam
  instead of waiting out the backoff. Reproduce that variant of Step A
  instead of the `.env` edit:

  ```bash
  mv "$HOME/.local/bin/kiro-cli" "$HOME/.local/bin/kiro-cli.bak"
  systemctl --user restart reachy-nova-harness   # fault injection, as in Step A
  ```

  Then recover with no further harness restart, by putting the binary back
  and nudging the watchdog immediately via the network-change seam
  (`NetworkUnit` polls `<statedir>/network-change`,
  `reachy_nova/harness/network.py:167`; `NetworkReactor.on_network_change`
  calls `kiro_unit.request_restart(reason)` on a `joined`/`moved` transition,
  `reachy_nova/harness/app.py:265`, `KiroSessionUnit.request_restart` at
  `kiro_session.py:277-304`):

  ```bash
  mv "$HOME/.local/bin/kiro-cli.bak" "$HOME/.local/bin/kiro-cli"
  python3 -c "import json, time; from reachy_nova.harness import statedir; \
  statedir.network_change_path().write_text(json.dumps({'ssid': 'bar-nachum', 'ip': '192.168.1.162', 'ts': time.time()}))"
  ```

Grep the recovery line:

```bash
journalctl --user -fu reachy-nova-harness | grep -F 'kiro session unit recovered'
```

`NRestarts` (`systemctl --user show reachy-nova-harness -p NRestarts`) is
unchanged between the binary-file restore and the `recovered` line — no
harness restart happened in that window.

### Evidence checklist

Paste into the delivery record:

- `started degraded (initial spawn failed` line (Step A)
- `started name=kiro_session` present, `start failed name=kiro_session`
  absent (Step A)
- at least one `restart failed:` line while the fault is live (Step A)
- `kiro session unit recovered` line (Step B)
- `NRestarts` before/after Step B, unchanged

**This drill proves the degraded-start path (h12's watchdog-retry /
never-give-up behavior), NOT the boot-ordering race** (`network-online.target`
reached before `wlan0` associates, `reachy_nova/harness/unit.py`'s
`NETWORK_ORDERING_COMMENT`). The boot-ordering race needs a real power-cycle
with the AP off, which is deferred — see the h12 remaining-work line in
`docs/deliveries/2026-08-26-dual-network-never-downtime.md`.
