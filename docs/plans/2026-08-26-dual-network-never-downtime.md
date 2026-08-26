# Build Plan — dual-network-never-downtime

slug: `dual-network-never-downtime` · status: `exported` · from frame: `dual-network-never-downtime`

> The Reachy Mini Wireless stays reachable and its mind stays online across both of Ori's networks — home Wi-Fi (bar-nachum) and the iPhone (5) hotspot — with never any downtime: the robot joins whichever is available, moves between them without a restart, and operators on spark can always find it

## Tasks

### t1 — Device network profiles: invert autoconnect priorities (iPhone (5) above bar-nachum), keep both autoconnect=yes, applied by a scripted change with an automatic revert timer (scripts/device-network.sh, idempotent, dry-run flag)

- covers: c2, h19, c19, h15
- acceptance:
  - scripts/device-network.sh --apply sets iPhone (5) autoconnect-priority > bar-nachum and exits 0; --check prints both profiles' priority/autoconnect/timestamp and exits non-zero if the order is wrong
  - every mutating step is wrapped in a revert timer (systemd-run --on-active or a background sleep+revert) that restores the previous priorities unless --commit is passed within the window; tests cover the script's argument/plan logic via a shell-out stub
  - docs/setup.md gains a 'Networks' section listing the two profiles, the order, and the revert-timer rule

### t2 — Failover policy as a pure, testable module: `reachy_nova`/netpolicy.py decides, from a scan (visible SSIDs+signal), the active connection, and time since last change, which profile to activate — bounded ≤30 s after link loss, re-evaluate every ≤30 s while disconnected, every ≤60 s while on the fallback, prefer the hotspot when visible, never storm (≤1 attempt per 30 s), and emit a structured decision record

- covers: c22, c26
- acceptance:
  - unit tests: link lost + hotspot visible → activate hotspot within one tick; hotspot gone + home visible → activate home; on home + hotspot appears → move to hotspot; on home + hotspot absent → no attempt and no log line beyond one latched 'waiting' per minute
  - module imports only stdlib and never calls nmcli itself (nmcli is injected as a callable) — AST-checked like the harness boundary

### t3 — NetworkManager dispatcher hook that runs the failover policy as root on down/up/connectivity-change events and on a ≤30 s timer while disconnected, activates the chosen profile via nmcli, and notifies the harness of a route change by touching <state>/network-change (atomic write of {ssid, ip, ts}); installed and enabled by scripts/install-device-units.sh with an automatic revert if the hook fails its self-test

- depends on: t2
- covers: c22, h16, c26, h22
- acceptance:
  - install script places config/network/90-reachy-failover in /etc/NetworkManager/dispatcher.d with a self-test (dry-run against the current scan) and reverts the install on failure
  - walking out of bar-nachum range with the hotspot visible: nmcli shows iPhone (5) activated within 30 s of NetworkManager's 'link timed out' line (h16)
  - robot on bar-nachum, Ori opens the Hotspot screen for 30 s → iPhone (5) activated within 60 s; screen kept closed → stays on bar-nachum with ≤1 network line per minute (h22)
  - the hook writes <state>/network-change on every activation with ssid/ip/ts, readable by the harness as pollen

### t4 — Harness network awareness: `reachy_nova`/harness/network.py NetworkUnit polls the default route + wlan0 address every 2 s and watches <state>/network-change; on a change it logs one latched '\[SENSE stage=supervise source=nova event=network\] joined=<ssid> ip=<addr>' or 'dropped reason=no-route' line and fires `on_change` callbacks that restart the Sonic stream and the Kiro session immediately (not at the 180 s liveness deadline)

- covers: c13, h13, c17, h7
- acceptance:
  - unit tests with a fake route reader: one transition → exactly one joined line and one dropped line, no repeats while the state is stable
  - `on_change` fires within one poll tick of the route change; the Sonic restart callback is invoked once per transition (test with a recording stub)
  - AST boundary test still passes (no `reachy_mini` import, no `set_target`)

### t5 — Mind survives a network-less start and a network change: the Kiro session's INITIAL spawn failure is retried under the existing watchdog/backoff instead of 'start failed' once; the harness starts fully with Wi-Fi down (named absences for Sonic/Kiro, pid claimed, engine live); NetworkUnit wired into `build_app`() with its callbacks; unit file no longer depends on network-online.target being meaningful

- depends on: t4
- covers: c12, h5, c18, h14, c8, h12
- acceptance:
  - test: KiroSessionUnit whose first spawn raises → unit is alive with restarts>0 after the monitor tick, no exception to the supervisor; supervisor lists it as started-degraded, not failed
  - on the robot: harness started with Wi-Fi disabled shows 'component absent' lines for Sonic/Kiro within 30 s and 'harness up' + 'engine live'; after Wi-Fi returns the journal shows `kiro_session` 'started' with no manual restart (h5, h14)
  - the 2026-08-26 'start failed name=`kiro_session` … kiro-cli process exited' line is reproduced before the fix and absent after (h12)
  - `build_app`() constructs NetworkUnit and registers Sonic-restart + Kiro-restart callbacks; existing harness tests unchanged

### t6 — Robot on the tailnet: install tailscale on the CM4 (apt, measured against the 1.3 G free root disk), enabled at boot after network, node key expiry disabled in the admin console (documented step), hostname 'reachy-mini'; docs/setup.md documents tailnet-first reachability with 'reachy wireless find' and the /etc/hosts pin as fallbacks; scripts/install-device-units.sh gains a guarded, non-fatal tailscale presence check

- depends on: t1, t3
- covers: c25, h20, c5, h1, h21
- acceptance:
  - from spark on bar-nachum, ssh pollen@reachy-mini (tailnet) succeeds while the robot is on the hotspot, and vice versa (h20)
  - 'tailscale status --json' on the robot shows KeyExpiry disabled; root disk free space before/after recorded in the delivery doc
  - spark reaches the robot via the tailnet within 30 s of the robot joining either network; 'reachy wireless find' still returns it when both are on the same subnet (h1, h21)

### t7 — Sonic stream-death restart gains exponential backoff with jitter (3 s → 60 s cap, reset on a healthy minute) so an offline robot does not reopen a Bedrock stream every 6 s; the network-change callback from t4 short-circuits the backoff to an immediate restart

- covers: c17
- acceptance:
  - unit test: five consecutive stream deaths schedule restarts at ≥3, ≥6, ≥12, ≥24, ≥48 s; a network-change signal resets the delay to 0
  - journal during a 4-minute offline window shows ≤8 restart attempts instead of ~40

### t8 — Live drill + delivery record: with the harness running, Ori removes bar-nachum (AP off or out of range) with the hotspot up → robot on the hotspot, answers speech, `kiro_session` started, within 60 s; restore bar-nachum → same; body unaffected throughout (runtime never restarts, heartbeat never lapses, pat/face reactions fire); zero SSH; transitions logged; docs/deliveries/2026-08-26-dual-network-never-downtime.md records evidence per honesty condition

- depends on: t1, t3, t5, t6, t7
- covers: c1, h4, c4, h9, c6, h10, c7, h11, c9, h2
- acceptance:
  - both directions pass with journal timestamps quoted: Sonic answer + `kiro_session` started within 60 s of 'policy: set … default for IPv4 routing' (h2, h11)
  - runtime unit shows no restart and state.json heartbeat has no gap > 2 s during the drill; a pat produces pet-reaction while Wi-Fi is down (h9)
  - the drill is executed by Ori with the phone only (h10) and the announcement (c1) is marked held only after both directions pass (h4)

## Risks

- [unknown_nonblocking] tailscale on a 91 %-full root disk: the install (~50 MB) is fine but the journald cap and skills-forged growth could leave no headroom — measure before/after in t6 and clean if < 500 MB free (task t6)
- [unknown_nonblocking] every device-side task (t1, t3, t5, t6) needs the robot physically present and Ori with the phone for the hotspot; the revert-timer rule (c19) is the safety net but the schedule is gated on a human
- [unknown_nonblocking] whether an in-flight Sonic conversation can be resumed across an IP change or only restarted with context re-injected — t4/t7 assume restart (frame park v1) (task t4)
- [unknown_nonblocking] Tailscale path over the iPhone hotspot NAT may relay via DERP — affects deploy latency, not reachability (frame park v5) (task t6)
