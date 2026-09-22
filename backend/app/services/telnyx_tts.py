"""Telephone TTS service (Phase 6C — telephony TTS milestone).

Converts agent text responses to telephone-compatible audio:

    AgentRuntime text response
        ↓
    ElevenLabs TTS (MP3 44.1kHz)
        ↓
    ffmpeg → PCM16 8kHz mono
        ↓
    _linear_to_ulaw() → PCMU 8kHz
        ↓
    160-byte chunks (20ms) → base64
        ↓
    Telnyx bidirectional media WebSocket

Uses the existing TTS provider abstraction (get_tts_provider).
No ElevenLabs-specific logic at the application level.
"""

import asyncio
import base64
import json
import time

from fastapi import WebSocket

from app.core.logging import logger
from app.providers.factory import get_tts_provider
from app.providers.tts.interface import TTSInterface
from app.services.telephony_events import broadcast_telephony_event
from app.services.telnyx_media import pcmu_chunks_from_bytes
from app.utils.audio import mp3_to_pcmu_8k


class TelnyxTTSService:
    """Handles TTS synthesis and audio delivery for telephone calls.

    Lifecycle:
        - Created once per telephony session
        - TTS provider is session-level (reused across turns)
        - Closed during session cleanup
    """

    def __init__(self) -> None:
        self._tts: TTSInterface | None = None

    async def start(self) -> None:
        """Create the session-level TTS provider."""
        try:
            self._tts = get_tts_provider()
            logger.debug(
                "[VOICE:TELNYX:TTS] provider created name=%s",
                self._tts.provider_name,
            )
        except Exception as e:
            logger.warning(
                "[VOICE:TELNYX:TTS] provider creation failed error=%s", e
            )
            self._tts = None

    async def stop(self) -> None:
        """Close the session-level TTS provider."""
        if self._tts is not None:
            try:
                await self._tts.close()
            except Exception as e:
                logger.warning("[VOICE:TELNYX:TTS] close error=%s", e)
            self._tts = None

    async def synthesize_and_send(
        self,
        text: str,
        turn: int,
        websocket: WebSocket,
        *,
        call_id: str = "",
    ) -> bool:
        """Synthesize text to speech and send audio to Telnyx.

        Args:
            text: The agent response text to speak.
            turn: Current turn number for logging.
            websocket: The Telnyx media WebSocket.
            call_id: Telnyx call_control_id for observability events.

        Returns:
            True if audio was sent successfully, False on failure.
        """
        if not text or self._tts is None:
            if self._tts is None:
                logger.warning(
                    "[VOICE:TELNYX:TTS] turn=%d skipped — no TTS provider",
                    turn,
                )
            return False

        tts_start = time.monotonic()
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d tts_start text_length=%d",
            turn,
            len(text),
        )
        await broadcast_telephony_event(
            "tts_processing",
            call_id,
            "Generating voice with ElevenLabs",
            turn=turn,
            metadata={"provider": "elevenlabs", "text_length": len(text)},
        )

        try:
            # Run TTS synthesis in background thread to avoid blocking
            tts_result = await self._tts.synthesize(text=text)
            tts_ms = (time.monotonic() - tts_start) * 1000
            logger.info(
                "[VOICE:TELNYX:TTS] turn=%d tts_done bytes=%d tts_ms=%.0f",
                turn,
                len(tts_result.audio_data),
                tts_ms,
            )
            await broadcast_telephony_event(
                "tts_completed",
                call_id,
                "Voice generated",
                turn=turn,
                metadata={
                    "duration_ms": round(tts_ms),
                    "bytes": len(tts_result.audio_data),
                },
            )
        except Exception as e:
            tts_ms = (time.monotonic() - tts_start) * 1000
            logger.error(
                "[VOICE:TELNYX:TTS] turn=%d tts_error "
                "error_type=%s error=%s tts_ms=%.0f",
                turn,
                type(e).__name__,
                str(e),
                tts_ms,
            )
            return False

        # Convert MP3 → PCMU 8kHz
        try:
            pcmu_data = await asyncio.to_thread(mp3_to_pcmu_8k, tts_result.audio_data)
        except Exception as e:
            logger.error(
                "[VOICE:TELNYX:TTS] turn=%d audio_conversion_error "
                "error=%s",
                turn,
                str(e),
            )
            return False

        if not pcmu_data:
            logger.warning(
                "[VOICE:TELNYX:TTS] turn=%d conversion produced empty audio",
                turn,
            )
            return False

        # Split into 20ms chunks and send via WebSocket with real-time pacing.
        # Telnyx expects media at ~real-time pace (1 chunk per 20ms).
        chunks = pcmu_chunks_from_bytes(pcmu_data)
        total_bytes = 0
        for chunk in chunks:
            payload = base64.b64encode(chunk).decode("ascii")
            try:
                await websocket.send_text(
                    json.dumps({"event": "media", "media": {"payload": payload}})
                )
                total_bytes += len(chunk)
            except Exception as e:
                logger.error(
                    "[VOICE:TELNYX:TTS] turn=%d websocket_send_error "
                    "error=%s sent_bytes=%d",
                    turn,
                    str(e),
                    total_bytes,
                )
                return False
            # Pace at real-time: 20ms per 160-byte PCMU chunk
            await asyncio.sleep(0.020)

        send_ms = (time.monotonic() - tts_start) * 1000
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d audio_sent chunks=%d "
            "bytes=%d total_ms=%.0f",
            turn,
            len(chunks),
            total_bytes,
            send_ms,
        )
        await broadcast_telephony_event(
            "audio_streaming",
            call_id,
            "Streaming AI voice to caller",
            turn=turn,
            metadata={
                "chunks": len(chunks),
                "bytes": total_bytes,
            },
        )
        await broadcast_telephony_event(
            "turn_completed",
            call_id,
            f"Turn {turn} completed",
            turn=turn,
            metadata={"total_ms": round(send_ms)},
        )
        return True
