"""Audio conversion utilities for telephony integration.

Provides PCMU (G.711 mu-law) decoding and sample rate conversion
for converting Telnyx media stream audio to Deepgram-compatible format.

Telnyx provides: PCMU, 8kHz, mono
Deepgram expects: linear16 (PCM), 16kHz, mono

Conversion pipeline:
    PCMU bytes (8-bit compressed)
        → ulaw_to_linear() → PCM16 samples (16-bit signed)
        → pcm8k_to_pcm16k() → PCM16 @ 16kHz
        → Deepgram

Also provides ``PCM16Resampler`` — a stateful streaming resampler used by
the MAI-Transcribe-2-Streaming adapter (browser native rate → 16 kHz).
"""

import struct

# ITU-T G.711 mu-law decoding table
# Pre-computed for performance (avoids per-sample computation)
_ULAW_DECODE_TABLE: list[int] = []


def _build_ulaw_decode_table() -> None:
    """Build the mu-law decoding lookup table (called once at import)."""
    for ulaw_byte in range(256):
        # Mu-law is stored complemented
        ulaw_byte = ~ulaw_byte & 0xFF

        # Extract sign, exponent, mantissa
        sign = ulaw_byte & 0x80
        exponent = (ulaw_byte >> 4) & 0x07
        mantissa = ulaw_byte & 0x0F

        # Reconstruct linear value
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84  # Subtract bias

        # Apply sign
        if sign:
            sample = -sample

        _ULAW_DECODE_TABLE.append(sample)


# Build table at import time
_build_ulaw_decode_table()


def ulaw_to_linear(ulaw_bytes: bytes) -> bytes:
    """Convert mu-law (PCMU) encoded bytes to 16-bit signed linear PCM.

    Args:
        ulaw_bytes: Raw mu-law encoded audio (1 byte per sample).

    Returns:
        16-bit signed PCM data (2 bytes per sample, little-endian).

    Example:
        >>> ulaw_data = bytes([0xFF, 0x00])  # 2 mu-law samples
        >>> pcm_data = ulaw_to_linear(ulaw_data)
        >>> len(pcm_data) == 4  # 2 samples × 2 bytes
        True
    """
    if not ulaw_bytes:
        return b""

    # Use struct for efficient packing of 16-bit samples
    samples = [_ULAW_DECODE_TABLE[b] for b in ulaw_bytes]
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm8k_to_pcm16k(pcm_data: bytes) -> bytes:
    """Upsample 16-bit PCM from 8kHz to 16kHz (2x interpolation).

    Uses linear interpolation between samples. For each input sample,
    produces two output samples: the original and the average of
    adjacent samples.

    Args:
        pcm_data: 16-bit signed PCM data at 8kHz (little-endian).

    Returns:
        16-bit signed PCM data at 16kHz (little-endian).

    Example:
        >>> # 4 samples at 8kHz = 8 bytes
        >>> pcm_8k = struct.pack("<4h", 0, 1000, 2000, 3000)
        >>> pcm_16k = pcm8k_to_pcm16k(pcm_8k)
        >>> # 8 samples at 16kHz = 16 bytes
        >>> len(pcm_16k) == 16
        True
    """
    if not pcm_data:
        return b""

    # Unpack input samples
    num_samples = len(pcm_data) // 2
    if num_samples == 0:
        return b""

    samples = struct.unpack(f"<{num_samples}h", pcm_data)

    # Build upsampled output
    output: list[int] = []
    for i in range(num_samples):
        # Original sample
        output.append(samples[i])

        # Interpolated sample (average of current and next)
        if i < num_samples - 1:
            interpolated = (samples[i] + samples[i + 1]) // 2
        else:
            # Last sample: repeat
            interpolated = samples[i]
        output.append(interpolated)

    return struct.pack(f"<{len(output)}h", *output)


def pcmu_8k_to_pcm_16k(ulaw_bytes: bytes) -> bytes:
    """Convert PCMU 8kHz audio to linear PCM 16kHz.

    This is the main entry point for telephony audio conversion.
    Combines mu-law decoding with 2x upsampling.

    Args:
        ulaw_bytes: Raw PCMU audio (mu-law, 8kHz, mono).

    Returns:
        Linear PCM audio (16-bit signed, 16kHz, mono).

    Example:
        >>> # 160 PCMU bytes = 20ms @ 8kHz
        >>> ulaw_data = bytes(160)
        >>> pcm_data = pcmu_8k_to_pcm_16k(ulaw_data)
        >>> # 320 samples × 2 bytes = 640 bytes @ 16kHz
        >>> len(pcm_data) == 640
        True
    """
    # Step 1: Decode mu-law → linear PCM (8kHz)
    pcm_8k = ulaw_to_linear(ulaw_bytes)

    # Step 2: Upsample 8kHz → 16kHz
    pcm_16k = pcm8k_to_pcm16k(pcm_8k)

    return pcm_16k


def mp3_to_pcmu_8k(mp3_data: bytes) -> bytes:
    """Convert MP3 audio to PCMU 8kHz mono using ffmpeg subprocess.

    Args:
        mp3_data: Raw MP3 audio bytes.

    Returns:
        PCMU (mu-law) encoded audio at 8kHz mono.
    """
    import subprocess

    if not mp3_data:
        return b""

    # Decode MP3 → PCM16 8kHz mono via ffmpeg
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", "8000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ],
        input=mp3_data,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg MP3 decode failed (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:200]}"
        )

    pcm_8k = proc.stdout
    if not pcm_8k:
        return b""

    # Convert PCM16 8kHz → PCMU 8kHz
    import struct

    num_samples = len(pcm_8k) // 2
    samples = struct.unpack(f"<{num_samples}h", pcm_8k)

    # Reuse mu-law encoding from telnyx_media
    from app.services.telnyx_media import _linear_to_ulaw

    return bytes(_linear_to_ulaw(s) for s in samples)


class PCM16Resampler:
    """Stateful streaming resampler for PCM16 mono little-endian audio.

    Converts a continuous stream of 16-bit PCM chunks from ``src_rate`` to
    ``dst_rate`` using linear interpolation. Continuity is preserved across
    chunk boundaries: the fractional read position carries into the next
    ``process()`` call, so a stream fed one ~50 ms frame at a time produces
    the same output as if it had been resampled in one pass. Both integer
    ratios (e.g. 48 kHz → 16 kHz) and fractional ratios (44.1 kHz → 16 kHz)
    are handled correctly.

    Intended for the MAI-Transcribe-2-Streaming adapter, which must forward
    PCM16 mono at 16 kHz while the browser captures at its native rate.
    The telephony conversion helpers above are unrelated and unchanged.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        # Input samples advanced per output sample.
        self._step: float = src_rate / dst_rate
        # Input samples not yet fully consumed by interpolation.
        self._buffer: list[int] = []
        # Fractional read position within ``self._buffer``.
        self._position: float = 0.0

    def process(self, chunk: bytes) -> bytes:
        """Resample one PCM16 chunk; returns the resampled PCM16 bytes.

        Args:
            chunk: 16-bit signed little-endian samples at the source rate.

        Returns:
            Resampled 16-bit signed little-endian samples at the target
            rate. Output is held back until both interpolation neighbours
            of a sample are known, so the stream lags by at most one input
            sample — never by a full chunk.
        """
        num_samples = len(chunk) // 2
        if num_samples == 0:
            return b""
        self._buffer.extend(struct.unpack(f"<{num_samples}h", chunk[: num_samples * 2]))

        buffer = self._buffer
        output: list[int] = []
        while int(self._position) < len(buffer) - 1:
            index = int(self._position)
            fraction = self._position - index
            value = buffer[index] + (buffer[index + 1] - buffer[index]) * fraction
            output.append(int(round(value)))
            self._position += self._step

        # Drop the fully consumed prefix; keep the interpolation neighbour.
        consumed = int(self._position)
        if consumed:
            del buffer[:consumed]
            self._position -= consumed

        if not output:
            return b""
        return struct.pack(f"<{len(output)}h", *output)
