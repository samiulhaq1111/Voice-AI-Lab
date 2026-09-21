"""Telnyx bidirectional media stream handler.

Handles WebSocket messages from Telnyx media streaming and sends
deterministic test audio back to the caller. No AI integration yet.

Audio format: PCMU (G.711 mu-law), 8 kHz, mono, 20ms RTP chunks.
"""

import base64
import json
import math

from app.core.logging import logger

# PCMU RTP chunk: 20ms at 8kHz = 160 samples = 160 bytes
_PCMU_CHUNK_SAMPLES = 160
_PCMU_SAMPLE_RATE = 8000
_PCMU_CHUNK_DURATION_MS = 20


def generate_pcmu_sine_wave(
    frequency: float = 440.0,
    duration_ms: int = 1000,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a PCMU-encoded sine wave.

    Args:
        frequency: Sine wave frequency in Hz.
        duration_ms: Duration in milliseconds.
        amplitude: Amplitude 0.0-1.0.

    Returns:
        Raw PCMU bytes (mu-law encoded).
    """
    num_samples = int(_PCMU_SAMPLE_RATE * duration_ms / 1000)
    pcmu_bytes = bytearray()
    max_val = 32767.0

    for i in range(num_samples):
        t = i / _PCMU_SAMPLE_RATE
        sample = amplitude * max_val * math.sin(2.0 * math.pi * frequency * t)
        # Linear to mu-law encoding (simplified)
        pcmu_bytes.append(_linear_to_ulaw(int(sample)))

    return bytes(pcmu_bytes)


def _linear_to_ulaw(sample: int) -> int:
    """Convert a 16-bit linear PCM sample to mu-law (PCMU).

    Uses the ITU-T G.711 mu-law companding algorithm.
    """
    # Constants for mu-law encoding
    bias = 0x84
    clip = 32635
    sign = 0

    if sample < 0:
        sign = 0x80
        sample = -sample

    if sample > clip:
        sample = clip

    sample += bias
    # Find the segment
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1

    # Extract the mantissa
    mantissa = (sample >> (exponent + 4)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa)
    return ulaw_byte & 0xFF


def pcmu_chunks_from_bytes(
    pcmu_data: bytes, chunk_size: int = _PCMU_CHUNK_SAMPLES
) -> list[bytes]:
    """Split raw PCMU data into fixed-size chunks."""
    chunks = []
    for i in range(0, len(pcmu_data), chunk_size):
        chunk = pcmu_data[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


class MediaStreamSession:
    """Tracks state for a single Telnyx media stream WebSocket connection."""

    def __init__(self, call_control_id: str = "") -> None:
        self.call_control_id = call_control_id
        self.stream_id: str = ""
        self.started: bool = False
        self.stopped: bool = False
        self.media_packets_in: int = 0
        self.media_packets_out: int = 0
        self.encoding: str = ""
        self.sample_rate: int = 0
        self._test_chunks: list[str] = []
        self._chunk_index: int = 0

    def _ensure_test_audio(self) -> None:
        """Generate test audio chunks lazily (440Hz sine, 1 second)."""
        if not self._test_chunks:
            pcmu_data = generate_pcmu_sine_wave(
                frequency=440.0, duration_ms=1000
            )
            raw_chunks = pcmu_chunks_from_bytes(pcmu_data)
            self._test_chunks = [
                base64.b64encode(c).decode("ascii") for c in raw_chunks
            ]
            logger.debug(
                "[VOICE:TELNYX:MEDIA] test audio generated chunks=%d",
                len(self._test_chunks),
            )

    def next_outbound_media(self) -> dict | None:
        """Return the next outbound media message, or None if stopped."""
        if self.stopped:
            return None
        self._ensure_test_audio()
        if not self._test_chunks:
            return None
        payload = self._test_chunks[self._chunk_index % len(self._test_chunks)]
        self._chunk_index += 1
        self.media_packets_out += 1
        return {"event": "media", "media": {"payload": payload}}

    def handle_message(self, raw: str | bytes) -> dict | None:
        """Process an inbound WebSocket message from Telnyx.

        Returns an outbound message dict to send back, or None.
        """
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                "[VOICE:TELNYX:MEDIA] malformed message error=%s", e
            )
            return None

        if not isinstance(msg, dict):
            logger.warning("[VOICE:TELNYX:MEDIA] non-object message ignored")
            return None

        event = msg.get("event", "")

        if event == "connected":
            version = msg.get("version", "")
            logger.info("[VOICE:TELNYX:MEDIA] stream.connected version=%s", version)
            return None

        if event == "start":
            start_data = msg.get("start", {})
            self.stream_id = msg.get("stream_id", "")
            self.call_control_id = start_data.get(
                "call_control_id", self.call_control_id
            )
            media_format = start_data.get("media_format", {})
            self.encoding = media_format.get("encoding", "")
            self.sample_rate = media_format.get("sample_rate", 0)
            self.started = True
            logger.debug(
                "[VOICE:TELNYX:MEDIA] start stream_id=%s "
                "call_control_id=%s encoding=%s sample_rate=%d",
                self.stream_id,
                self.call_control_id,
                self.encoding,
                self.sample_rate,
            )
            return None

        if event == "media":
            self.media_packets_in += 1
            return self.next_outbound_media()

        if event == "stop":
            stop_data = msg.get("stop", {})
            self.stream_id = msg.get("stream_id", self.stream_id)
            self.call_control_id = stop_data.get(
                "call_control_id", self.call_control_id
            )
            self.stopped = True
            logger.debug(
                "[VOICE:TELNYX:MEDIA] stop stream_id=%s "
                "call_control_id=%s total_in=%d total_out=%d",
                self.stream_id,
                self.call_control_id,
                self.media_packets_in,
                self.media_packets_out,
            )
            return None

        logger.debug(
            "[VOICE:TELNYX:MEDIA] unknown event=%s ignored", event
        )
        return None
