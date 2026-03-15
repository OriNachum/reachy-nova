"""Verify and configure XMOS XVF3800 AEC settings at startup."""

import logging
import struct

logger = logging.getLogger(__name__)


def _decode_int32(result: list) -> int | None:
    """Decode an int32 value from ReSpeaker read() result.

    ReSpeaker.read() returns raw byte lists [status, b0, b1, b2, b3]
    for int32 parameters.
    """
    if result is None or len(result) < 5:
        return None
    return struct.unpack("<i", bytes(result[1:5]))[0]


def verify_xmos_aec(respeaker) -> None:
    """Check and enable XMOS AEC settings for echo cancellation.

    Args:
        respeaker: A reachy_mini ReSpeaker device instance.
    """
    if respeaker is None:
        logger.warning("[AEC] No ReSpeaker device — skipping AEC verification")
        return

    # 1. Echo suppression (PP_ECHOONOFF)
    try:
        result = respeaker.read("PP_ECHOONOFF")
        value = _decode_int32(result)
        if value is not None:
            if value == 0:
                logger.warning("[AEC] Echo suppression is OFF — enabling")
                respeaker.write("PP_ECHOONOFF", [1])
                logger.info("[AEC] Echo suppression enabled")
            else:
                logger.info(f"[AEC] Echo suppression already enabled (value={value})")
    except Exception as e:
        logger.error(f"[AEC] Failed to read/write PP_ECHOONOFF: {e}")

    # 2. Non-linear attenuation (PP_NLATTENONOFF)
    try:
        result = respeaker.read("PP_NLATTENONOFF")
        value = _decode_int32(result)
        if value is not None:
            if value == 0:
                logger.warning("[AEC] Non-linear attenuation is OFF — enabling")
                respeaker.write("PP_NLATTENONOFF", [1])
                logger.info("[AEC] Non-linear attenuation enabled")
            else:
                logger.info(f"[AEC] Non-linear attenuation already enabled (value={value})")
    except Exception as e:
        logger.error(f"[AEC] Failed to read/write PP_NLATTENONOFF: {e}")

    # 3. AEC convergence status (read-only diagnostic)
    try:
        result = respeaker.read("AEC_AECCONVERGED")
        value = _decode_int32(result)
        if value is not None:
            if value:
                logger.info("[AEC] AEC filter has converged")
            else:
                logger.info("[AEC] AEC filter has NOT yet converged (will converge during playback)")
    except Exception as e:
        logger.error(f"[AEC] Failed to read AEC_AECCONVERGED: {e}")

    # 4. Channel routing diagnostic (AUDIO_MGR_OP_L, uint8 x2)
    try:
        result = respeaker.read("AUDIO_MGR_OP_L")
        if result is not None:
            # uint8 result includes status byte: [status, val0, val1]
            logger.info(f"[AEC] Channel routing AUDIO_MGR_OP_L: {result}")
    except Exception as e:
        logger.error(f"[AEC] Failed to read AUDIO_MGR_OP_L: {e}")
