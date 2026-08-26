# Delivery Summary — dual-network-never-downtime

plan: `dual-network-never-downtime` · run: `complete` · date: `2026-08-26`
baseline: `devague summary skeleton`

## Intent

Execute the converged plan seeded from the frame "The Reachy Mini Wireless
stays reachable and its mind stays online across both of Ori's networks —
home Wi-Fi (bar-nachum) and the iPhone (5) hotspot — with never any
downtime". Decisions that shaped it: failover on the single radio (q1),
hotspot-first ordering with the robot on Ori's Tailscale tailnet (c11/q3),
cellular data cost accepted (q2), uplink loss out of scope (c27). Run via
/assign-to-workforce: waves [t1,t2,t4,t7] → [t3,t5] → t6 → t8, TDD-gated
`--no-ff` merges on branch `spec/dual-network-never-downtime`. All live
evidence produced on the CM4 on 2026-08-26 by the main agent, with Ori
driving the phone. No addresses appear in this document by Ori's rule —
placeholders such as `<home-lan-ip>` stand in.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Device network profiles: invert autoconnect priorities (iPhone (5) above bar-nachum), keep both autoconnect=yes, applied by a scripted change with an automatic revert timer (scripts/device-network.sh, idempotent, dry-run flag)
- `t2` — Failover policy as a pure, testable module: reachy_nova/netpolicy.py decides, from a scan (visible SSIDs+signal), the active connection, and time since last change, which profile to activate — bounded ≤30 s after link loss, re-evaluate every ≤30 s while disconnected, every ≤60 s while on the fallback, prefer the hotspot when visible, never storm (≤1 attempt per 30 s), and emit a structured decision record
- `t3` — NetworkManager dispatcher hook that runs the failover policy as root on down/up/connectivity-change events and on a ≤30 s timer while disconnected, activates the chosen profile via nmcli, and notifies the harness of a route change by touching <state>/network-change (atomic write of {ssid, ip, ts}); installed and enabled by scripts/install-device-units.sh with an automatic revert if the hook fails its self-test
- `t4` — Harness network awareness: reachy_nova/harness/network.py NetworkUnit polls the default route + wlan0 address every 2 s and watches <state>/network-change; on a change it logs one latched '[SENSE stage=supervise source=nova event=network] joined=<ssid> ip=<addr>' or 'dropped reason=no-route' line and fires on_change callbacks that restart the Sonic stream and the Kiro session immediately (not at the 180 s liveness deadline)
- `t5` — Mind survives a network-less start and a network change: the Kiro session's INITIAL spawn failure is retried under the existing watchdog/backoff instead of 'start failed' once; the harness starts fully with Wi-Fi down (named absences for Sonic/Kiro, pid claimed, engine live); NetworkUnit wired into build_app() with its callbacks; unit file no longer depends on network-online.target being meaningful
- `t6` — Robot on the tailnet: install tailscale on the CM4 (apt, measured against the 1.3 G free root disk), enabled at boot after network, node key expiry disabled in the admin console (documented step), hostname 'reachy-mini'; docs/setup.md documents tailnet-first reachability with 'reachy wireless find' and the /etc/hosts pin as fallbacks; scripts/install-device-units.sh gains a guarded, non-fatal tailscale presence check
- `t7` — Sonic stream-death restart gains exponential backoff with jitter (3 s → 60 s cap, reset on a healthy minute) so an offline robot does not reopen a Bedrock stream every 6 s; the network-change callback from t4 short-circuits the backoff to an immediate restart
- `t8` — Live drill + delivery record: with the harness running, Ori removes bar-nachum (AP off or out of range) with the hotspot up → robot on the hotspot, answers speech, kiro_session started, within 60 s; restore bar-nachum → same; body unaffected throughout (runtime never restarts, heartbeat never lapses, pat/face reactions fire); zero SSH; transitions logged; docs/deliveries/2026-08-26-dual-network-never-downtime.md records evidence per honesty condition

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `scripts/device-network.sh` (`--check/--dry-run/--apply --revert-after/--commit/--revert`, systemd-run timer with nohup fallback) + 19 tests + `docs/setup.md` "Networks". Applied on the robot 10:15 BST with a 600 s revert, `--check` OK, committed: iPhone (5)=20 > bar-nachum=10, both autoconnect. |
| `t2` | delivered | `reachy_nova/netpolicy.py` (pure, stdlib-only, AST-checked) + 25 tests; every rule in the task text covered. |
| `t3` | delivered | `reachy_nova/netfailover.py` (nmcli driver, `--once/--loop/--self-test/--dry-run`, atomic `netfailover.json` + `network-change`), `config/network/90-reachy-failover` dispatcher hook, guarded installer step with self-test + self-revert, `--no-failover`; 66 tests. Review round (qodo, PR #12): rounds serialised under an inter-process lock, hook interpreter/state dir rendered into `/etc/default/reachy-failover` by the installer, `network-change` synced on every online round (not only after this driver's own activation), loop waits on the stop event. Installed on the robot 10:15 BST; self-test passed (`on-fallback-hotspot-absent`). Judgment call: the hook detaches work into transient systemd units rather than running inline (NM waits on dispatcher scripts; an inline 45 s join would block the radio's own event queue). |
| `t4` | delivered | `reachy_nova/harness/network.py` `NetworkUnit` (+ `statedir.network_change_path`) + 18 tests; lines `joined=… ip=…`, `dropped reason=no-route`, `moved …`, `initial=true` on the baseline. |
| `t5` | delivered | Degraded Kiro start + watchdog retry (`KiroSessionUnit.request_restart`, `status()["degraded"]`), Sonic `start()` no longer raises, `NetworkReactor` wiring in `build_app()` (baseline observation never restarts), unit-text comment; 37 tests. Live: harness started with the radio off came up with 9 components + `engine live`; Wi-Fi back → Sonic restart requested 1 s after the route, listening 15 s after; Kiro recovered in 5 s. |
| `t6` | delivered | Tailscale 1.102.3 installed from the vendor repo, `tailscaled` enabled, node `reachy-mini` logged in by Ori; key expiry **disabled by Ori in the admin console (reported done; not re-verified by machine — see claims)**; SSH from spark over the tailnet verified while the robot was on the hotspot; root disk 1.3 G → 1.2 G free; docs/setup.md "tailnet first" + informational installer check. |
| `t7` | delivered | Exponential backoff 3→60 s with jitter, healthy-period reset, `request_immediate_restart(reason)`, env `NOVA_SONIC_RESTART_{BASE,MAX}_S`; 25 tests. Observed live offline: `restarting session in 24s (attempt 4)`. |
| `t8` | delivered | Drill at home, phone only, zero SSH: A) hotspot screen opened → loop decided `activate iPhone (5)` 10:35:14 → switched 10:35:16 → `joined` 10:35:20 → Sonic listening + Kiro recovered 10:35:25 (≈9 s mind outage) → Ori spoke and heard it. B) hotspot off 10:37:57 → `bar-nachum` activated 10:38:06 → `joined` 10:38:07 → Sonic listening 10:38:13, Kiro recovered 10:38:14 → Ori spoke and heard it. Runtime never restarted. |

## Mid-work Decisions

- No `/deviate` records were needed: no task departed from its confirmed
  contract. Decisions taken inside task scope:
- t3: work detached into `reachy-netfailover-once-*` / `reachy-netfailover-loop`
  transient units instead of inline in the dispatcher (deadlock avoidance);
  `pre-down` handled but not symlinked into `pre-down.d/` (`down` and
  `connectivity-change` are the load-bearing events).
- t5: the first (baseline) network observation is logged but never restarts
  the legs — added after the agent flagged a needless restart ~2 s after every
  boot.
- t6 (device): Tailscale installed by the main agent, login and key-expiry
  performed by Ori; the tailnet address is deliberately absent from every
  committed file (Ori's rule, 2026-08-26); a first version of docs/setup.md
  that contained it was replaced and the branch's spec/scope records were
  scrubbed in a follow-up commit (`686d329`) — earlier branch commits still
  carry the addresses until the squash-merge drops them.
- Speaker volume raised 62 % → 100 % (daemon `/api/volume/set`) at Ori's
  request during the drill — device state, not repo code.
- Volume of pre-existing private LAN/hotspot addresses in files this branch
  ADDS was replaced with placeholders / RFC 5737 documentation addresses
  before the PR; files that already existed on `main` were not rewritten.

## Drift From Plan

No drift: every task is `delivered` against its confirmed acceptance
criteria (table above). Two acceptance items are evidenced weaker than
written and are called out as such in Delivery Claims: h12 (the boot
`start failed` line did not reproduce live) and h9's "pat/face reactions
fire while Wi-Fi is down" (only Ori's observation during the morning's
forced-failover probe, not a journal line).

## Evidence

- tests: `uv run pytest -q` on `686d329`+scrub — **1044 passed** (856 baseline → +188 across t1–t7)
- tests: `tests/test_netpolicy.py`, `tests/test_netfailover.py`, `tests/test_failover_hook.py`, `tests/test_harness_network.py`, `tests/test_harness_network_wiring.py`, `tests/test_kiro_session.py`, `tests/test_nova_sonic_backoff.py`, `tests/test_device_network_script.py` — all pass
- lint: `bash -n scripts/install-device-units.sh` — ok; shellcheck run by the t1/t3 agents — clean
- commits: `8756d88..HEAD` on `spec/dual-network-never-downtime` (merges `ae6dd5f` t2, `550c504` t4, `4150ab8` t1, `8a4bcc4` t7, `dccc104` t3, `b45ccd8` t5, `95d9eef` t6, `686d329` scrub)
- robot journal (user unit `reachy-nova-harness`, `NetworkManager`, `reachy-failover`, `reachy-netfailover-loop`), 2026-08-26 10:14–10:39 BST — quoted timestamps above
- on-robot logs: `/tmp/netwatch.log`, `/tmp/wifidown-test.log`, `/tmp/force-failover.log`, `/tmp/hotspot-test.log` (ephemeral)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| c22/h16: the visible fallback is joined within 30 s of losing the current network | high | drill B: `link timed out` 10:37:57 → `bar-nachum` activated 10:38:06 (9 s) |
| c26/h22: on the fallback, an appearing hotspot is joined within 60 s | high | drill A: loop `activate iPhone (5)` 10:35:14, switched 10:35:16, ≈20 s after the SSID appeared |
| c17/h7: Sonic restart is triggered within 10 s of the new default route | high | `Immediate restart requested` 10:35:20 (route 10:35:18) and 10:38:07 (route 10:38:06) |
| c12/h5: Kiro comes back after Wi-Fi returns with no manual restart | high | `kiro session unit recovered` 10:17:41, 10:35:25, 10:38:14 |
| c18/h14: harness starts fully with Wi-Fi off | high | `harness up pid=33266 components=9` + `engine live` 10:17:08 with `dropped reason=no-route initial=true` |
| c13/h13: one joined + one dropped line per transition | high | journal 10:35:16/10:35:20 and 10:37:59/10:38:07, no repeats |
| c7/h11, c9/h2, c6/h10: both directions, speech answered, phone only, zero SSH | high | drill timestamps + Ori: "I talked to it, I heard it" (A), "I can hear and we speak" (B) |
| c4/h9: body unaffected — runtime never restarted, heartbeat held during the drill | medium | `reachy-runtime` 0 restarts today; one `engine-heartbeat-lost` at 10:16:36 (outside any network event — the known upstream tick-overrun flap); pat/face-while-offline rests on Ori's observation during the 09:23 probe, no journal line |
| c8/h12: the boot-time `start failed name=kiro_session` is reproduced then fixed | low | the failure did not reproduce with the radio off (Kiro spawned offline in 14 s); the degraded-start path is unit-tested (`tests/test_kiro_session.py`) but not exercised live |
| c25/h20: robot on the tailnet, key expiry disabled, SSH over the tailnet on either network | medium | SSH over tailnet verified from spark while the robot was on the hotspot (10:35) and on the LAN; key expiry disable reported done by Ori, last machine read (10:25) still showed an expiry — re-check with `tailscale status --json` |
| c5/h1/h21: spark reaches the robot on either network | high | tailnet SSH both networks; `reachy wireless find` returned the robot on the shared LAN (2.2 s sweep) |
| c2/h19: two profiles, hotspot outranks home, both autoconnect | high | `scripts/device-network.sh --check` OK on the robot 10:15 |
| c19/h15: every device-side change carried a revert timer | high | `--apply --revert-after 600` (t1), installer self-revert (t3), radio-off test with a 120 s `systemd-run` safety, all four probe scripts |
| Sonic no longer hammers Bedrock offline | high | `restarting session in 24s (attempt 4)` 10:17:25; `tests/test_nova_sonic_backoff.py` |

## Review round (PR #12)

8 qodo findings, all accepted and fixed in `0ac0dce` (device-network: rollback armed before mutations, autoconnect restored), `6107e66` (Sonic: backoff waits interruptible by stop/immediate-restart; vocalize master gain `NOVA_VOCALIZE_GAIN`=0.35 because chirps were louder than speech), `b98dcfd` (netfailover: round lock, defaults file, network-change sync, loop stop wait). Suite 1044 → 1077. Redeployed to the robot 11:06 BST: installer rendered the defaults file, self-test passed, harness baseline now logs the real SSID.

## Remaining Work / Follow-up

- Verify key expiry is disabled on the `reachy-mini` node (`tailscale status --json` → `KeyExpiry` absent) — Ori reported it done after the last machine read.
- h12 stays `low`: reproduce the cold-boot Kiro failure deliberately (power-cycle with Wi-Fi unavailable) to exercise the degraded-start path live.
- h12: case 5 now exists in `tests/chaos/ON_ROBOT.md` for the Kiro degraded-start path (watchdog-retry proof); it is not the boot-ordering race above, which still needs the power-cycle drill.
- The failover loop re-scans every 60 s while on the fallback (by spec); watch for Wi-Fi latency blips on `bar-nachum` from the periodic rescans and relax to 120 s if seen.
- `pre-down.d/` symlink for the hook is not installed (not needed by the drill); add if a pre-down decision ever matters.
- Frame parks v1 (Sonic conversation resume vs restart — restart is what ships), v3 (mDNS over the hotspot), v5 (Tailscale DERP vs direct over the hotspot NAT) remain open, nonblocking.
- Earlier branch commits contain the addresses later scrubbed; squash-merge the PR (repo convention) so they do not enter `main`'s history.
- Robot checkout is on `spec/dual-network-never-downtime`; switch it back to `main` after the merge.
