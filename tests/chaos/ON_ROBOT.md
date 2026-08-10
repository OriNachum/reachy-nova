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
