# Setup Guide

Follow this guide to get Reachy Nova running on your Reachy Mini robot or development machine.

## Prerequisites

-   **Operating System**: Linux (Ubuntu 22.04+ recommended) or macOS.
-   **Python**: Version 3.12 or newer.
-   **UV**: The `uv` package manager (recommended) or robust `pip`/`venv` setup.
-   **AWS Account**: Access to Amazon Bedrock with the following models enabled:
    -   `amazon.nova-2-sonic-v1:0` (us-east-1)
    -   `us.amazon.nova-2-lite-v1:0` (us-east-1)
    -   Access to Nova Act execution role if running browser automation.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/pollen-robotics/reachy_nova.git
    cd reachy_nova
    ```

2.  **Install Dependencies**:
    We recommend using `uv` for fast dependency resolution.
    ```bash
    uv sync
    ```
    This will install `reachy-mini`, `boto3`, `nova-act`, `opencv-python`, and other required packages.

3.  **Configure Environment**:
    Create a `.env` file from the sample:
    ```bash
    cp .env.sample .env
    ```

    Edit `.env` with your AWS credentials:
    ```ini
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_DEFAULT_REGION=us-east-1
    ```

## Running the Application

### Development Mode

Run the application directly:
```bash
uv run python -m reachy_nova.main
```

The application will:
1.  Connect to Reachy Mini (or start a mock if no robot is found).
2.  Start the internal web server at `http://localhost:8042`.
3.  Begin listening for voice commands.

### Hardware Setup (Reachy Mini)

Ensure your Reachy Mini is connected via USB or properly networked. The `reachy-mini` library handles device discovery.

-   **Camera**: Ensure the camera is accessible (check `/dev/video*` permissions on Linux).
-   **Microphone**: Ensure the microphone is set as the default input or properly selected by `pyaudio`.
-   **Speakers**: Ensure audio output is configured.

## Networks

On the Reachy Mini Wireless (CM4, ReachyMiniOS, NetworkManager, single
`wlan0` radio) there are two Wi-Fi connection profiles the robot can hold:

-   **`iPhone (5)`** — Ori's phone hotspot, the *travelling* network. Decided
    2026-08-26 to be the **preferred** profile (higher
    `autoconnect-priority`), so the robot rides the phone wherever Ori goes
    and falls back to home Wi-Fi only when the phone leaves.
-   **`bar-nachum`** — the home Wi-Fi network, the **fallback**.

Both profiles keep `autoconnect=yes`; only the relative
`autoconnect-priority` changes to decide which one NetworkManager prefers at
(re)connect time. Note the iPhone hotspot is only *joinable* while Ori has
the Personal Hotspot screen open, or while a client (including the robot
itself) is already attached — iOS stops beaconing the SSID otherwise, so a
robot that has fallen back to home Wi-Fi needs Ori to open the Hotspot
screen once to get picked back up; there is no unattended return.

### Why hotspot-first

With hotspot-first ordering, the robot stays on the phone as long as it's
reachable, and only drops back to `bar-nachum` when the phone is out of
range or its hotspot is off. This matches how Ori actually uses the robot
(carried around, not fixed at home).

### The revert-timer rule

The robot has **no console**. Every device-side network change (profile
priority edits, dispatcher hooks, Tailscale install, …) must be applied with
a scheduled automatic revert, so a mistake in a change can never strand an
operator without an SSH path back in. This is why
`scripts/device-network.sh --apply` never applies a change permanently on
its own — it always schedules a revert (default 300s) that undoes the
change unless `--commit` is called within the window.

### `scripts/device-network.sh`

Idempotent script to inspect and flip the two profiles' relative priority:

```bash
# print both profiles' autoconnect / autoconnect-priority / timestamp;
# exits 0 iff iPhone (5) outranks bar-nachum and both autoconnect=yes
scripts/device-network.sh --check

# print (without running) the nmcli commands --apply would issue
scripts/device-network.sh --dry-run

# apply the priority flip (sudo -n nmcli connection modify), with an
# automatic revert after 300s (override with --revert-after SECONDS)
# unless --commit is run first
scripts/device-network.sh --apply --revert-after 300

# cancel the pending revert once you've confirmed SSH still works
scripts/device-network.sh --commit

# or restore the previous priorities immediately
scripts/device-network.sh --revert
```

`--check` and `--dry-run` never require root and never mutate anything;
`--apply`/`--revert` use `sudo -n nmcli connection modify` (the `pollen`
user has passwordless `sudo -n nmcli` on the device). The revert timer uses
`systemd-run --on-active` when available, falling back to a background
`sleep` + revert job tracked by a pidfile under
`${XDG_STATE_HOME:-$HOME/.local/state}/reachy/`.

Profile names, target priorities, and the nmcli binary are overridable via
`REACHY_NET_PREFERRED`, `REACHY_NET_FALLBACK`,
`REACHY_NET_PREFERRED_PRIORITY`, `REACHY_NET_FALLBACK_PRIORITY`, and
`REACHY_NMCLI` — mainly used by the test suite
(`tests/test_device_network_script.py`) to stub `nmcli` out.

### Reaching the robot: tailnet first

The robot is a node on Ori's Tailscale tailnet (`reachy-mini`, installed from
`pkgs.tailscale.com`, `tailscaled` enabled at boot). Reach it by tailnet name
or address from any tailnet device **regardless of which Wi‑Fi the robot is
on** — including the phone hotspot, whose subnet spark can never see:

```bash
ssh pollen@reachy-mini            # MagicDNS
ssh pollen@"$(tailscale ip -4 reachy-mini)"   # tailnet address, resolved at use
tailscale ip -4 reachy-mini       # from any tailnet member
```

Two operator rules:

- **Disable key expiry** for the `reachy-mini` node in the Tailscale admin
  console (Machines → reachy-mini → … → Disable key expiry). With the default
  180‑day expiry the robot would silently drop off the tailnet one day.
- The subnet sweep (`reachy wireless find`) and the `/etc/hosts` pin
  (`reachy wireless pin`) are **fallbacks** for when the tailnet is down;
  they only work when spark and the robot share a subnet, and the pin goes
  stale on every network switch — refresh it when you use it.

Disk note: the install cost ~50 MB on the CM4's root filesystem (1.3 G → 1.2 G
free on 2026‑08‑26); keep an eye on it alongside the journald cap.

## Troubleshooting

### Voice Not Working
-   Check AWS credentials in `.env`.
-   Verify `amazon.nova-2-sonic-v1:0` access in AWS Bedrock console.
-   Check microphone permissions.

### Vision Not Working
-   Verify camera access (`cv2.VideoCapture`).
-   Check `us.amazon.nova-2-lite-v1:0` access in AWS Bedrock.

### Browser Automation Fails
-   Ensure `nova-act` is installed correctly.
-   Check if Playwright browsers are installed:
    ```bash
    playwright install chromium
    ```

### Tracking Issues
-   If face tracking is slow, ensure you are running on a machine with decent CPU or GPU.
-   YOLOv8n is used by default for speed.
