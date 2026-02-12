#!/usr/bin/env python3
"""Standalone Nova Sonic demo — mic in, speaker out, no robot needed.

Usage:
    uv run sonic-demo
    uv run sonic-demo --voice tiffany --system "You are a pirate."

Loads .env from the project root for AWS credentials.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from reachy_nova.nova_sonic import NovaSonic, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, INPUT_CHUNK_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sonic_demo")

# ── Audio playback buffer ──────────────────────────────────────────
output_lock = threading.Lock()
output_buffer = bytearray()


def on_audio_output(samples: np.ndarray) -> None:
    """Append synthesised audio to the playback buffer."""
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with output_lock:
        output_buffer.extend(pcm)


def on_transcript(role: str, text: str) -> None:
    tag = "YOU" if role == "USER" else "NOVA"
    print(f"  [{tag}] {text}")


def on_state_change(state: str) -> None:
    logger.info(f"state → {state}")


# ── PyAudio callbacks ──────────────────────────────────────────────
def speaker_callback(in_data, frame_count, time_info, status):
    """Pull audio from the output buffer for the speaker."""
    n_bytes = frame_count * 2  # 16-bit mono
    with output_lock:
        chunk = bytes(output_buffer[:n_bytes])
        del output_buffer[:n_bytes]
    if len(chunk) < n_bytes:
        chunk += b"\x00" * (n_bytes - len(chunk))
    return (chunk, pyaudio.paContinue)


def main():
    parser = argparse.ArgumentParser(description="Nova Sonic standalone demo")
    parser.add_argument("--voice", default="matthew", help="Voice ID (default: matthew)")
    parser.add_argument("--system", default=None, help="System prompt override")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    args = parser.parse_args()

    system_prompt = args.system or (
        "You are Nova, a friendly voice assistant. "
        "Keep responses short and conversational."
    )

    # ── Create Nova Sonic ──────────────────────────────────────────
    sonic = NovaSonic(
        region=args.region,
        voice_id=args.voice,
        system_prompt=system_prompt,
        on_transcript=on_transcript,
        on_audio_output=on_audio_output,
        on_state_change=on_state_change,
    )

    stop_event = threading.Event()

    # Graceful shutdown
    def handle_signal(sig, frame):
        print("\nShutting down...")
        stop_event.set()
    signal.signal(signal.SIGINT, handle_signal)

    # ── Start Nova Sonic ───────────────────────────────────────────
    logger.info("Starting Nova Sonic...")
    sonic.start(stop_event)
    time.sleep(1)  # Let the session establish

    # ── Start PyAudio ──────────────────────────────────────────────
    pa = pyaudio.PyAudio()

    # Speaker output stream (24kHz)
    speaker_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=OUTPUT_SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024,
        stream_callback=speaker_callback,
    )

    # Mic input stream (16kHz)
    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_SAMPLE_RATE,
        input=True,
        frames_per_buffer=INPUT_CHUNK_SIZE,
    )

    logger.info("Listening — speak into your mic (Ctrl+C to quit)")
    speaker_stream.start_stream()

    # ── Main loop: read mic → feed sonic ───────────────────────────
    try:
        while not stop_event.is_set():
            try:
                pcm_data = mic_stream.read(INPUT_CHUNK_SIZE, exception_on_overflow=False)
                samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                sonic.feed_audio(samples)
            except OSError:
                pass
    except KeyboardInterrupt:
        stop_event.set()

    # ── Cleanup ────────────────────────────────────────────────────
    logger.info("Cleaning up...")
    mic_stream.stop_stream()
    mic_stream.close()
    speaker_stream.stop_stream()
    speaker_stream.close()
    pa.terminate()
    time.sleep(0.5)
    logger.info("Done.")


if __name__ == "__main__":
    main()
