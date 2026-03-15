"""Audio format conversion and resampling for mic→Sonic and Sonic→speaker paths."""

import numpy as np


def preprocess_mic_audio(audio, mic_sr: int, target_sr: int = 16000, aec_channel: int | None = 0) -> np.ndarray:
    """Convert raw mic audio to float32 mono at target sample rate.

    Handles bytes (int16), non-float32 numpy arrays, multi-channel selection,
    and resampling via linear interpolation.

    Args:
        aec_channel: Channel index to select from multi-channel audio.
            0 = AEC-processed (default), None = average all channels.
    """
    if isinstance(audio, bytes):
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        if aec_channel is not None and aec_channel < audio.shape[1]:
            audio = audio[:, aec_channel]
        else:
            audio = audio.mean(axis=1)

    if mic_sr != target_sr:
        ratio = target_sr / mic_sr
        n_out = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


def resample_output(
    chunk: np.ndarray,
    source_sr: int,
    target_sr: int,
    speed_factor: float = 1.0,
    volume_gain: float = 1.0,
) -> np.ndarray:
    """Resample speaker output with optional speed adjustment and volume gain."""
    effective_source_rate = source_sr * speed_factor
    ratio = target_sr / effective_source_rate
    n_out = int(len(chunk) * ratio)
    indices = np.linspace(0, len(chunk) - 1, n_out)
    chunk = np.interp(indices, np.arange(len(chunk)), chunk).astype(np.float32)

    if volume_gain != 1.0:
        chunk = np.clip(chunk * volume_gain, -1.0, 1.0).astype(np.float32)

    return chunk
