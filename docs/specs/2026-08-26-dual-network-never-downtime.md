# dual-network-never-downtime

> The Reachy Mini Wireless stays reachable and its mind stays online across both of Ori's networks — home Wi-Fi (bar-nachum) and the iPhone (5) hotspot — with never any downtime: the robot joins whichever is available, moves between them without a restart, and operators on spark can always find it
> instruction: run the c9 drill; announcement holds only if both directions pass with zero SSH

## Audience

- Ori as the operator (on spark, or carrying the robot with only the phone) and the robot's own mind (the harness), which must keep its cloud tiers and writer alive without anyone SSHing in

## Before → After

- Before: today the robot has one radio (wlan0) with two NetworkManager profiles: bar-nachum (autoconnect prio 10, home) and iPhone (5) (prio 5, PSK stored, timestamp 0 = never joined until 2026-08-26); it only ever holds one at a time and re-associates on loss
- After: the robot joins whichever of bar-nachum / iPhone (5) is available and moves between them with no reboot and no SSH; within 60 s of a switch the mind is fully back (Sonic answers speech, `kiro_session` alive, MQTT/tee untouched because they are local); spark rediscovers it on the shared network with one command

## Why it matters

- taking the robot out of the house today gives a body that is alive but a mind that may be silently half-dead: the writer failed at cold boot before Wi-Fi came up and was never retried (journal 2026-08-26), and spark cannot find the robot on the hotspot subnet at all

## Requirements

- operators on spark can find the robot on either network: today the registry pins `last_ip` 192.168.1.162 and reachy-mini.local mDNS does not resolve; on the hotspot the robot gets a 172.20.10.x address that spark only sees if spark is on the hotspot too
  - honesty: from spark on the hotspot, 'reachy wireless find' returns the robot's 172.20.10.x address within 30 s of the robot joining, and the registry/pinned alias are refreshed rather than stale
- the mind survives a network change: the harness (or its unit) retries the kiro writer's initial spawn under the same watchdog/backoff it already uses for later restarts, and Sonic's stream is re-established on address loss instead of waiting for the liveness watchdog; unit ordering no longer relies on network-online.target alone
  - honesty: with Wi-Fi deliberately down at harness start, `kiro_session` is absent at first and then shows 'started' in the journal after Wi-Fi returns, with no manual restart
- network transitions are visible by name in the journal like every other degradation (\[SENSE stage=supervise ... event=network\] joined=<ssid> ip=<addr> / dropped reason=no-route) so a silent half-dead mind cannot recur
  - honesty: grep '\[SENSE stage=supervise' for event=network yields one joined and one dropped line per transition of the drill, no more
- an address change is an explicit trigger, not a timeout: the mind reacts to the default route changing (NM dispatcher hook or a harness route poll) by restarting the Sonic stream and the Kiro session immediately — because Sonic's liveness watchdog defaults to 180 s (`NOVA_SONIC_LIVENESS_S`, `nova_sonic.py` `DEFAULT_LIVENESS_S`), which alone cannot meet the 60 s success signal
  - honesty: after a switch, the journal shows the Sonic restart within 10 s of the new default route, not at the liveness deadline
- failover must be bounded in time AND keep retrying: on link loss NM re-tries the higher-priority profile with `wpa_supplicant` temp-disable backoffs (10 s, 20 s, …) and only falls through after autoconnect-retries are exhausted (48 s with no attempt in the 08:58 test); and after ONE failed fallback attempt (ssid-not-found at 09:23:33) NM stopped trying for 3.5 min. The spec sets a hard bound — the visible fallback is joined within 30 s of losing the current network — and re-evaluates every ≤30 s while disconnected (dispatcher hook or a harness route poll; autoconnect-retries=1 on the preferred profile is not sufficient alone)
  - honesty: walking out of bar-nachum range with the hotspot visible, nmcli on the robot shows iPhone (5) activated within 30 s of the 'link timed out' NetworkManager line

## Honesty conditions

- the drill in c9 passes in both directions on the real robot, with the transitions logged, before the announcement is made
- nmcli on the robot shows exactly two Wi-Fi profiles on wlan0 with autoconnect=yes, bar-nachum at higher priority than iPhone (5), and both with a non-zero timestamp
- during the c9 drill the runtime never restarts, the engine heartbeat never lapses, and pat/face reactions still fire while Wi-Fi is down (journal + state.json)
- the drill is executed by Ori with the phone only — no laptop, no SSH — and the robot answers speech afterwards
- each direction of the switch shows Sonic answering speech and `kiro_session` 'started' within 60 s of the new default route, and 'reachy wireless find' (or the tailnet address) reaches the robot
- the 2026-08-26 boot journal line 'start failed name=`kiro_session` … kiro-cli process exited' is reproduced with Wi-Fi down at start, then disappears after the fix
- drill: kill bar-nachum with the harness running and the hotspot up; journal shows `kiro_session` started (not 'start failed') and Sonic answers speech within 60 s, with no restart command issued
- a network switch produces exactly one joined line and one dropped line per transition (latched), grep-able with the existing \[SENSE stage=...\] convention
- with both networks visible the robot sits on iPhone (5); when the phone leaves, it is on bar-nachum within the c22 bound; when the phone returns, it moves back to the hotspot (NM does not roam to a higher-priority network on its own while connected — needs a periodic re-evaluation or dispatcher logic)
- with the home AP powered but its WAN unplugged, either the robot switches to the hotspot within 60 s, or the spec explicitly declares uplink loss out of scope
- starting the harness with Wi-Fi disabled yields 'component absent' lines for Sonic/Kiro within 30 s and a running harness (pid claimed, engine live), not a hung start
- every network-changing command in the plan's runbook carries an explicit revert timer, and the drill confirms SSH returns on bar-nachum without intervention
- robot attached to iPhone (5), Hotspot screen closed, phone in pocket for 10 min: robot still has its 172.20.10.x address and Sonic answers speech

## Success signals

- live drill, both directions, zero SSH: with the harness running, remove bar-nachum (power the AP off or walk out of range) while the hotspot broadcasts -> robot on 172.20.10.x, answers a spoken question and shows `kiro_session` started in the journal within 60 s; restore bar-nachum -> robot back on 192.168.1.162 with the same; 'reachy wireless find' from spark on the same network returns it each time

## Scope / boundaries

- the body is unaffected by any network change: the runtime, rules, senses, tee, spool and local MQTT broker are all on-device; 'downtime' can only ever mean the mind's cloud tiers (Sonic/Lite/Act) and the writer's auth
- the harness must still start and run with NO network (home offline, robot in a bag): any 'wait for a route' logic is bounded and non-fatal — the body-first rule holds; a network-less start yields named absences (Sonic, Kiro) and later self-heal, never a stuck unit
- every device-side network change (profile edits, priorities, dispatcher hooks, tailscale) is applied with a scheduled automatic revert — the pattern used for the 2026-08-26 hotspot test — so a mistake can never cut the operator's only SSH path to a robot with no console

## Non-goals

- no change to the body: the runtime, rules, senses, tee, spool and local broker are not touched; no new radio or tethering hardware in this round unless q1 decides otherwise
- spark's own network switching is manual and out of scope: the spec never automates moving spark between bar-nachum and the hotspot (it is a server with docker bridges and tailscale); spark either joins the hotspot by hand or reaches the robot over the tailnet if q3 says yes

## Assumptions

- a live join of iPhone (5) works: 2026-08-26 07:33 the robot activated the hotspot profile (172.20.10.2/28, default route via 172.20.10.1, internet 36 ms) and returned to bar-nachum in ~20 s total; the hotspot must be broadcasting (Ori opens the Personal Hotspot screen) for the SSID to be visible
- NetworkManager fails over only on association loss: with bar-nachum up but its uplink dead, the robot stays on home with no internet and the hotspot is never tried (nmcli connectivity check is 'full' but no policy acts on it) — 'never downtime' as specced covers AP loss, not ISP loss
- the iPhone hotspot SSID stays visible to the robot unattended: during the away-test the robot saw 'iPhone (5)' continuously for ~2 min while Ori carried the phone — pending Ori's confirmation that the Hotspot screen was closed at the time
- while the robot is already attached, iOS keeps the Personal Hotspot up with the screen closed — so hotspot-first ordering keeps the robot connected through the day; only a drop-and-rejoin needs the screen opened (untested; test = attach, close screen, wait 10 min, verify still attached)

## Scope exploration

- `s1` — `device nmcli: connection profiles + device wifi list (2026-08-26 07:30)`: single wlan0 interface; profiles bar-nachum prio 10 and iPhone (5) prio 5 both autoconnect; no second radio, no ethernet in use — 'both networks' can only mean failover/roaming on one radio unless hardware is added
  - seeds: `c2`
- `s2` — `/tmp/hotspot-test.log on device (nmcli connection up round-trip)`: join + DHCP + route ≈ 6 s each way; ~10-20 s of no-network per switch — 'never downtime' has to be measured against the body (unaffected) vs the mind (Sonic stream, MQTT local, Kiro) separately
  - seeds: `c3`
- `s3` — `docs/architecture.md §8 + harness units (supervisor, hearing, kiro_session) + journal 2026-08-26 boot`: harness reconnects seams with backoff and Sonic has clock-step/liveness watchdogs, but the kiro session failed at cold boot before Wi-Fi associated (After=network-online.target is satisfied too early) and was never retried until a manual restart — a network switch today = a possible silent loss of the writer
  - seeds: `c4`
- `s4` — `~/.local/state/reachy/units.json registry, reachy wireless {find,list,pin}, spark nmcli (two iPhone (5) profiles, on bar-nachum at 192.168.1.118)`: the two machines must be on the SAME network to talk (hotspot is 172.20.10.0/28, home is 192.168.1.0/24, no routing between); discovery must re-run after a switch (reachy wireless find) and the pinned /etc/hosts alias goes stale
  - seeds: `c5`
- `s5` — `challenge pass / adjacent-systems lens: reachy wireless find --json + overview, ~/.local/state/reachy/units.json`: discovery is an IPv4 HTTP sweep of spark's local /24-or-narrower subnets (268 hosts in 2.2 s), registry keyed by `hardware_id` with `last_ip`; it cannot see across subnets, so on the hotspot spark must be on the hotspot too, and 'wireless pin' aliases go stale on every switch
  - seeds: `c5`
- `s6` — `challenge pass / adjacent-systems lens: device 'which tailscale', systemctl tailscaled`: no tailscale on the robot (spark has 100.127.105.72); overlay networking is an unexamined alternative to subnet-bound discovery — raised as a user decision, not assumed
- `s7` — `challenge pass / assumptions lens: nmcli general status (CONNECTIVITY full), connection profiles hidden=no, autoconnect-retries default`: NM has connectivity checking but no failover-on-uplink-loss policy; hidden=no means NM waits for a beacon rather than probing the hotspot SSID
  - seeds: `c16`
- `s8` — `challenge pass / failure-mode lens: reachy_nova/nova_sonic.py (_liveness_window, clock-step), harness/unit.py, grep for route/network in harness`: the harness has zero network awareness today — only unit ordering (After=network-online.target) and Sonic's 180 s liveness + clock-step watchdogs; nothing reacts to an IP change
  - seeds: `c17`
- `s9` — `challenge pass / lifecycle lens: /etc/NetworkManager/dispatcher.d ({no-wait,pre-up,pre-down}.d present), systemd --user units`: a dispatcher hook is available as the route-change trigger mechanism on ReachyMiniOS; the harness unit runs as pollen so the hook must signal it, not restart it as root
  - seeds: `c18`
- `s10` — `challenge pass / security lens: iPhone (5) profile psk-flags 0 (system-stored, root-only), SSH as pollen, docs/security.md Kiro trust`: clean pass: no new secret surfaces; on the hotspot the robot's SSH is reachable by hotspot peers exactly as on the home LAN; the Kiro full-shell boundary is unchanged by networking
- `s11` — `challenge pass / reversibility lens: /tmp/hotspot-test.sh pattern (nmcli up, verify, nmcli up bar-nachum)`: the robot has no console; the revert-timer pattern is the containment for all network edits
  - seeds: `c19`
- `s12` — `challenge probe / failure-mode lens: /tmp/netwatch.log + journalctl NetworkManager,wpa_supplicant 08:56–09:00 on the robot`: 08:58:12 link timed out (supplicant-timeout) → NM auto-activated bar-nachum again, ASSOC-REJECT status 16 twice, SSID temp-disabled 10 s then 20 s, reconnected 08:59:01 when back in range; `iphone_visible`=1 from 08:57:57 throughout; iPhone (5) never attempted. Sonic: stream died 08:57:02 (link degrading) and self-restarted in 3 s — stream-death recovery already exists, the gap is the dead-link window
  - seeds: `c22`
- `s13` — `challenge probe / assumptions lens: netwatch iphone_visible column 08:57:57–08:59:47`: SSID beaconed continuously while the phone was carried; resolves park v2 if the Hotspot screen was closed
  - seeds: `c23`
- `s14` — `challenge pass / assumptions lens: NetworkManager autoconnect semantics (priority applies at activation time only; no roaming to a better profile while connected)`: hotspot-first ordering alone gives 'prefer at (re)connect time', not 'always on the hotspot when present' — the return-to-hotspot leg needs explicit logic, and the data-cost question is open
  - seeds: `c11`
- `s15` — `challenge probe / lifecycle lens: forced failover 09:23–09:27 on the robot (/tmp/force-failover.log, journalctl NetworkManager + wpa_supplicant)`: with bar-nachum made unavailable, NM auto-activated iPhone (5) after 2 s — fallback SELECTION works — but the join failed 25 s later with reason ssid-not-found: with the Hotspot screen closed and no client attached, the iPhone stopped beaconing (it had been visible in the scan cache until then). NM made no further attempt for the remaining 3.5 min. Sonic hammered Bedrock every ~6 s with a fixed 3 s restart delay (`AWS_IO_DNS_QUERY_FAILED`) while offline; recovered on its own 7 s after bar-nachum returned

## Decisions

- decided by Ori 2026-08-26: with the robot on Tailscale, the iPhone (5) hotspot becomes the PREFERRED network (autoconnect-priority above bar-nachum) and bar-nachum the fallback — the robot rides the phone wherever Ori is, and falls back to home Wi-Fi when the phone leaves; both profiles keep autoconnect=yes
- q1 decided by Ori 2026-08-26: 'both networks' = failover on the single wlan0 radio; no second interface is available, so simultaneous membership is out. Order (per c11, revised): hotspot first, bar-nachum second
- q2 decided by Ori 2026-08-26: 'never downtime' covers all three — body (already true, verified), the voice/mind (Sonic and the Kiro writer must self-heal across a network change, c12) and spark reaching the robot (discovery/alias refresh, c5)
- q3 decided by Ori 2026-08-26: the robot joins the Tailscale tailnet (spark is already on it) — spark reaches the robot by its tailnet address on either network; subnet discovery ('reachy wireless find') stays as the fallback when the tailnet is down

## Hard questions

- is the kiro-cli exit at boot really network ordering (auth refresh needs a route) or something else — confirm by reproducing with Wi-Fi down before changing units
- cellular data: Nova Sonic streams the mic continuously at 16 kHz/16-bit (~32 kB/s ≈ 115 MB per hour of awake time, plus vision clips to Lite and Sonic's 24 kHz audio back) — on the hotspot at home this all rides the phone's cellular plan and battery; is that acceptable, or should hotspot-first apply only when bar-nachum is absent (which is what priority ordering cannot express by itself)?

## Open parks

- [unknown_nonblocking] whether an in-flight Sonic conversation can be resumed across an IP change, or only restarted with context re-injected — untested
- [unknown_nonblocking] whether mDNS works across the phone hotspot (avahi is active on the robot; reachy-mini.local never resolved on bar-nachum) — would give spark a stable name on the hotspot; nonblocking, discovery sweep works regardless
- [follow_up] Sonic's stream-death restart uses a fixed 3 s delay with no backoff: offline it re-opened a Bedrock stream every ~6 s for minutes (DNS failures) — harmless today but noisy and battery/CPU-costly on the hotspot; belongs plan-side as a risk/task once the plan exists

## Resolved vagueness

- [unknown_blocking] iPhone Personal Hotspot visibility to non-Apple clients: the SSID was visible only while Ori had the hotspot screen open; whether the robot can REJOIN unattended (phone in pocket) is untested — probe: close the hotspot screen, drop bar-nachum, watch nmcli on the robot; mitigation candidate: 802-11-wireless.hidden=yes so NM probes actively (currently 'no') — resolved: unattended (re)join of the iPhone hotspot does NOT work when the phone's Hotspot screen is closed and no client is attached: iOS stops advertising the SSID (forced-failover probe 09:23:33 ssid-not-found). It is only joinable while Ori has the screen open — or, presumably, while the robot is already attached (hotspot-first keeps it attached). Consequence: a robot that drops off the hotspot needs Ori to open the screen to get back; there is no unattended path
