# Setup Guide — NVIDIA DGX Spark (Ubuntu 24.04, aarch64)

Specific steps required to run Reachy Nova on the DGX Spark with a
**wireless** Reachy Mini (WebRTC audio/video over LAN).

---

## 1. Python & UV

```bash
# uv is already available; verify Python 3.12 is active
python3 --version   # should be 3.12.x
uv --version
```

---

## 2. Clone and install Python dependencies

```bash
cd ~/git/reachy_nova
uv sync
```

---

## 3. GStreamer system libraries

The wireless backend streams audio/video over WebRTC using GStreamer.
Several system libraries must be installed before the Python packages can work.

### 3a. Core GStreamer + dev headers

```bash
sudo apt update && sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gst-plugins-bad-1.0
```

### 3b. Cairo + GObject (required by `gst-signalling` Python package)

```bash
sudo apt install -y libcairo2-dev libgirepository1.0-dev
```

### 3c. GStreamer Rust WebRTC plugin (`webrtcsrc` element)

The `webrtcsrc` element is **not** in the Ubuntu package repos — it must be
built from source using Rust.  This takes ~10–20 minutes on first run.

```bash
# Install Rust toolchain if not present
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

# Build gst-plugin-webrtc
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git /tmp/gst-plugins-rs
cd /tmp/gst-plugins-rs
cargo build --release --package gst-plugin-webrtc --locked

# Install the shared library to the GStreamer plugin path
sudo cp target/release/*.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/

# Clean up
cd ~ && rm -rf /tmp/gst-plugins-rs
```

Verify:
```bash
gst-inspect-1.0 webrtcsrc   # should print element details, not "no such element"
```

---

## 4. Install `gst-signalling` Python package

```bash
uv add gst-signalling
```

---

## 5. Configure environment

```bash
cp .env.sample .env
# Edit .env with AWS credentials (see .env.sample for all keys)
```

---

## 6. Run

```bash
uv run python -m reachy_nova.main
```

The app starts in sleep mode and wakes when it hears "hey reachy" (parakeet
backend by default) or detects a snap/clap.

---

## Known issues on Spark

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'gst_signalling'` | Package not installed | `uv add gst-signalling` |
| `ValueError: Namespace GstWebRTC not available` | GStreamer bad plugins missing | Step 3a above |
| `RuntimeError: Failed to create webrtcsrc element` | Rust WebRTC plugin not installed | Step 3c above |
| `pycairo build fails` — "Dependency cairo not found" | Missing system lib | `sudo apt install libcairo2-dev libgirepository1.0-dev` |
| `cargo build` fails — "gstreamer-1.0.pc not found" | GStreamer dev headers missing | Step 3a above |
| uv resolution error "reachy-mini-app not found" | Stale `keywords` extra in pyproject.toml | Remove `keywords = ["reachy-mini-app"]` from `[project.optional-dependencies]` |
