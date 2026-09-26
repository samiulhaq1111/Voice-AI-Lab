"""Telephone TTS service (Phase 6C — telephony TTS, Phase 8B — streaming).

Preferred streaming path (no transcoding, lowest time-to-first-audio):

    AgentRuntime text response
        ↓
    ElevenLabs Flash v2.5 /stream (output_format=ulaw_8000)
        ↓  raw mu-law 8kHz bytes, progressively over chunked HTTP
    160-byte PCMU frames (20ms) → base64
        ↓
    Telnyx bidirectional media WebSocket (real-time paced)

Fallback path (used only when the provider cannot stream mu-law):

    ElevenLabs TTS (MP3 44.1kHz, buffered)
        ↓
    ffmpeg → PCM16 8kHz mono → _linear_to_ulaw() → PCMU 8kHz
        ↓
    160-byte chunks (20ms) → base64 → Telnyx media WebSocket

Uses the existing TTS provider abstraction (get_tts_provider).
No ElevenLabs-specific logic at the application level.
"""

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket

from app.core.logging import logger
from app.providers.factory import get_tts_provider
from app.providers.tts.interface import TTSInterface
from app.services.telephony_events import broadcast_telephony_event
from app.services.telnyx_media import pcmu_chunks_from_bytes
from app.utils.audio import mp3_to_pcmu_8k

# ElevenLabs native telephony format: raw mu-law (PCMU) 8kHz mono.
# Telnyx media streaming already expects PCMU/8000, so streamed chunks
# are forwarded verbatim — no MP3 decode, no resampling, no ffmpeg.
_TELEPHONY_OUTPUT_FORMAT = "ulaw_8000"

# One 20ms PCMU frame at 8kHz = 160 samples = 160 bytes.
_PCMU_FRAME_BYTES = 160
_CHUNK_PACE_SECONDS = 0.020
_ULAW_BYTES_PER_SECOND = 8000


class _StreamingUnavailableError(Exception):
    """Raised internally when the streaming path cannot serve this turn.

    Only raised *before* any audio has reached Telnyx, so the caller can
    safely retry the turn through the buffered MP3 fallback path.
    """


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
    ) -> dict[str, Any]:
        """Speak `text` to the caller as early as possible.

        Streams ElevenLabs mu-law 8kHz audio straight into the Telnyx media
        WebSocket, so the caller starts hearing the response while the rest
        is still being generated. Falls back to buffered MP3 + ffmpeg when
        streaming is unavailable.

        Args:
            text: The agent response text to speak.
            turn: Current turn number for logging.
            websocket: The Telnyx media WebSocket.
            call_id: Telnyx call_control_id for observability events.

        Returns:
            Dict with timing metrics:
                - success: bool
                - tts_start: monotonic timestamp
                - first_audio_sent_at: monotonic timestamp or None
                - tts_complete_at: monotonic timestamp
        """
        if not text or self._tts is None:
            if self._tts is None:
                logger.warning(
                    "[VOICE:TELNYX:TTS] turn=%d skipped — no TTS provider",
                    turn,
                )
            return {
                "success": False,
                "first_audio_sent_at": None,
                "tts_start": None,
                "tts_complete_at": None,
            }

        tts_start = time.monotonic()
        first_audio_sent_at: float | None = None
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d tts_start text_length=%d "
            "mode=stream format=%s",
            turn,
            len(text),
            _TELEPHONY_OUTPUT_FORMAT,
        )
        await broadcast_telephony_event(
            "tts_processing",
            call_id,
            "Generating voice with ElevenLabs",
            turn=turn,
            metadata={
                "provider": "elevenlabs",
                "text_length": len(text),
                "mode": "stream",
            },
        )

        try:
            stream_result = await self._stream_to_telnyx(
                text, turn, websocket, call_id, tts_start
            )
            ok = stream_result["success"]
            first_audio_sent_at = stream_result.get("first_audio_sent_at")
        except _StreamingUnavailableError as e:
            logger.info(
                "[VOICE:TELNYX:TTS] turn=%d streaming_unavailable reason=%s "
                "— using buffered MP3 fallback",
                turn,
                e,
            )
            buffered_result = await self._send_buffered(
                text, turn, websocket, call_id, tts_start
            )
            ok = buffered_result["success"]
            first_audio_sent_at = buffered_result.get("first_audio_sent_at")

        return {
            "success": ok,
            "tts_start": tts_start,
            "first_audio_sent_at": first_audio_sent_at,
            "tts_complete_at": time.monotonic(),
        }

    async def stream_sentences(
        self,
        sentences: AsyncIterator[str],
        turn: int,
        websocket: WebSocket,
        *,
        call_id: str = "",
    ) -> dict[str, Any]:
        """Stream sentences to TTS as they become available.

        Processes sentences sequentially — each sentence gets its own
        ElevenLabs stream, and audio is forwarded to Telnyx as it arrives.
        This enables incremental TTS during LLM streaming.

        Args:
            sentences: Async iterator of complete sentences from the
                sentence buffer.
            turn: Current turn number for logging.
            websocket: The Telnyx media WebSocket.
            call_id: Telnyx call_control_id for observability events.

        Returns:
            Dict with timing metrics:
                - tts_start: monotonic timestamp when TTS started
                - first_audio_ms: time from tts_start to first audio frame
                - first_audio_sent_at: absolute monotonic timestamp of first audio
                - total_sentences: number of sentences processed
                - total_frames: total PCMU frames sent
                - total_bytes: total bytes sent
                - success: True if all sentences processed
        """
        if self._tts is None:
            logger.warning(
                "[VOICE:TELNYX:TTS] turn=%d stream_sentences skipped "
                "— no TTS provider",
                turn,
            )
            return {"success": False, "total_sentences": 0}

        tts_start = time.monotonic()
        first_audio_ms: float | None = None
        first_audio_sent_at: float | None = None
        total_sentences = 0
        total_frames = 0
        total_bytes = 0
        success = True

        await broadcast_telephony_event(
            "tts_processing",
            call_id,
            "Streaming sentences to ElevenLabs",
            turn=turn,
            metadata={"mode": "sentence_stream"},
        )

        async for sentence in sentences:
            if not sentence.strip():
                continue

            total_sentences += 1
            sentence_start = time.monotonic()

            logger.debug(
                "[VOICE:TELNYX:TTS] turn=%d sentence=%d text='%s'",
                turn,
                total_sentences,
                sentence[:80],
            )

            try:
                # Stream this sentence through TTS
                stream = self._open_stream(sentence)
                pending = bytearray()
                sentence_frames = 0

                async for piece in stream:
                    if not piece:
                        continue
                    pending.extend(piece)

                    # Send every complete PCMU frame immediately
                    while len(pending) >= _PCMU_FRAME_BYTES:
                        frame = bytes(pending[:_PCMU_FRAME_BYTES])
                        del pending[:_PCMU_FRAME_BYTES]

                        if not await self._send_frame(frame, turn, websocket):
                            success = False
                            break

                        sentence_frames += 1
                        total_frames += 1
                        total_bytes += len(frame)

                        if first_audio_ms is None:
                            first_audio_ms = (
                                time.monotonic() - tts_start
                            ) * 1000
                            first_audio_sent_at = time.monotonic()
                            await self._announce_first_audio(
                                turn,
                                call_id,
                                first_audio_ms,
                                _TELEPHONY_OUTPUT_FORMAT,
                            )

                        # Pace at real-time
                        await asyncio.sleep(_CHUNK_PACE_SECONDS)

                    if not success:
                        break

                await self._close_stream(stream)

                sentence_ms = (time.monotonic() - sentence_start) * 1000
                logger.debug(
                    "[VOICE:TELNYX:TTS] turn=%d sentence=%d completed "
                    "frames=%d ms=%.0f",
                    turn,
                    total_sentences,
                    sentence_frames,
                    sentence_ms,
                )

            except Exception as e:
                logger.error(
                    "[VOICE:TELNYX:TTS] turn=%d sentence=%d error=%s",
                    turn,
                    total_sentences,
                    e,
                )
                success = False
                break

        # Emit turn summary
        total_ms = (time.monotonic() - tts_start) * 1000
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d sentence_stream completed "
            "sentences=%d frames=%d bytes=%d ttfa_ms=%s total_ms=%.0f",
            turn,
            total_sentences,
            total_frames,
            total_bytes,
            f"{first_audio_ms:.0f}" if first_audio_ms is not None else "n/a",
            total_ms,
        )

        if total_frames > 0:
            await self._emit_turn_summary(
                turn=turn,
                call_id=call_id,
                tts_start=tts_start,
                provider_done_ms=total_ms,
                first_audio_ms=first_audio_ms,
                frames_sent=total_frames,
                sent_bytes=total_bytes,
                audio_format=_TELEPHONY_OUTPUT_FORMAT,
            )

        return {
            "tts_start": tts_start,
            "first_audio_ms": first_audio_ms,
            "first_audio_sent_at": first_audio_sent_at,
            "total_sentences": total_sentences,
            "total_frames": total_frames,
            "total_bytes": total_bytes,
            "success": success,
        }

    # --- streaming path (preferred) ------------------------------------

    def _open_stream(self, text: str) -> AsyncIterator[bytes]:
        """Start provider streaming, or signal that it is unavailable."""
        stream_fn = getattr(self._tts, "stream_synthesize", None)
        if stream_fn is None:
            raise _StreamingUnavailableError("provider has no stream_synthesize")

        try:
            stream = stream_fn(
                text=text, output_format=_TELEPHONY_OUTPUT_FORMAT
            )
        except TypeError as e:
            # Provider does not accept an output_format override.
            raise _StreamingUnavailableError(
                f"stream_synthesize signature unsupported: {e}"
            ) from e

        if not hasattr(stream, "__aiter__"):
            if asyncio.iscoroutine(stream):
                stream.close()
            raise _StreamingUnavailableError("provider stream is not async-iterable")
        return stream

    async def _stream_to_telnyx(
        self,
        text: str,
        turn: int,
        websocket: WebSocket,
        call_id: str,
        tts_start: float,
    ) -> dict[str, Any]:
        """Forward provider mu-law chunks to Telnyx as they arrive.

        Returns dict with success, first_audio_sent_at, frames_sent.
        """
        stream = self._open_stream(text)

        pending = bytearray()
        provider_bytes = 0
        sent_bytes = 0
        frames_sent = 0
        first_audio_ms: float | None = None
        first_audio_sent_at: float | None = None
        provider_done_ms: float | None = None
        stream_error: str | None = None

        try:
            async for piece in stream:
                if not piece:
                    continue
                provider_bytes += len(piece)
                provider_done_ms = (time.monotonic() - tts_start) * 1000
                pending.extend(piece)

                # Ship every complete 20ms PCMU frame immediately — never
                # wait for the whole response to be generated.
                while len(pending) >= _PCMU_FRAME_BYTES:
                    frame = bytes(pending[:_PCMU_FRAME_BYTES])
                    del pending[:_PCMU_FRAME_BYTES]

                    if not await self._send_frame(frame, turn, websocket):
                        return {
                            "success": False,
                            "first_audio_sent_at": first_audio_sent_at,
                            "frames_sent": frames_sent,
                        }

                    frames_sent += 1
                    sent_bytes += len(frame)
                    if first_audio_ms is None:
                        first_audio_ms = (time.monotonic() - tts_start) * 1000
                        first_audio_sent_at = time.monotonic()
                        await self._announce_first_audio(
                            turn, call_id, first_audio_ms, _TELEPHONY_OUTPUT_FORMAT
                        )
                    # Pace at real-time: 20ms per 160-byte PCMU frame
                    await asyncio.sleep(_CHUNK_PACE_SECONDS)
        except Exception as e:
            stream_error = f"{type(e).__name__}: {e}"
            logger.error(
                "[VOICE:TELNYX:TTS] turn=%d stream_error error=%s frames_sent=%d",
                turn,
                stream_error,
                frames_sent,
            )
        finally:
            await self._close_stream(stream)

        if frames_sent == 0:
            # Nothing reached the caller yet → safe to try the fallback.
            raise _StreamingUnavailableError(
                stream_error
                or f"stream produced no playable audio (provider_bytes={provider_bytes})"
            )

        if stream_error is not None:
            await broadcast_telephony_event(
                "error",
                call_id,
                "AI voice stream ended early",
                turn=turn,
                metadata={"stage": "tts_stream", "partial": True},
            )

        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d stream_sent frames=%d bytes=%d "
            "provider_bytes=%d ttfa_ms=%s provider_ms=%s",
            turn,
            frames_sent,
            sent_bytes,
            provider_bytes,
            f"{first_audio_ms:.0f}" if first_audio_ms is not None else "n/a",
            f"{provider_done_ms:.0f}" if provider_done_ms is not None else "n/a",
        )
        await self._emit_turn_summary(
            turn=turn,
            call_id=call_id,
            tts_start=tts_start,
            provider_done_ms=provider_done_ms,
            first_audio_ms=first_audio_ms,
            frames_sent=frames_sent,
            sent_bytes=sent_bytes,
            audio_format=_TELEPHONY_OUTPUT_FORMAT,
        )
        return {
            "success": True,
            "first_audio_sent_at": first_audio_sent_at,
            "frames_sent": frames_sent,
        }

    # --- buffered fallback path ----------------------------------------

    async def _send_buffered(
        self,
        text: str,
        turn: int,
        websocket: WebSocket,
        call_id: str,
        tts_start: float,
    ) -> dict[str, Any]:
        """Legacy path: full MP3 first, then ffmpeg → PCMU, then paced send.

        Returns dict with success, first_audio_sent_at, frames_sent.
        """
        try:
            tts_result = await self._tts.synthesize(text=text)
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
            return {"success": False, "first_audio_sent_at": None, "frames_sent": 0}

        provider_done_ms = (time.monotonic() - tts_start) * 1000
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d tts_done mode=buffered bytes=%d tts_ms=%.0f",
            turn,
            len(tts_result.audio_data),
            provider_done_ms,
        )

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
            return {"success": False, "first_audio_sent_at": None, "frames_sent": 0}

        if not pcmu_data:
            logger.warning(
                "[VOICE:TELNYX:TTS] turn=%d conversion produced empty audio",
                turn,
            )
            return {"success": False, "first_audio_sent_at": None, "frames_sent": 0}

        # Split into 20ms chunks and send via WebSocket with real-time pacing.
        # Telnyx expects media at ~real-time pace (1 chunk per 20ms).
        chunks = pcmu_chunks_from_bytes(pcmu_data)
        total_bytes = 0
        first_audio_ms: float | None = None
        first_audio_sent_at: float | None = None
        for chunk in chunks:
            if not await self._send_frame(chunk, turn, websocket):
                return {
                    "success": False,
                    "first_audio_sent_at": first_audio_sent_at,
                    "frames_sent": total_bytes // _PCMU_FRAME_BYTES,
                }
            total_bytes += len(chunk)
            if first_audio_ms is None:
                first_audio_ms = (time.monotonic() - tts_start) * 1000
                first_audio_sent_at = time.monotonic()
                await self._announce_first_audio(
                    turn, call_id, first_audio_ms, "pcmu_8000_buffered"
                )
            # Pace at real-time: 20ms per 160-byte PCMU chunk
            await asyncio.sleep(_CHUNK_PACE_SECONDS)

        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d audio_sent mode=buffered chunks=%d "
            "bytes=%d ttfa_ms=%s",
            turn,
            len(chunks),
            total_bytes,
            f"{first_audio_ms:.0f}" if first_audio_ms is not None else "n/a",
        )
        await self._emit_turn_summary(
            turn=turn,
            call_id=call_id,
            tts_start=tts_start,
            provider_done_ms=provider_done_ms,
            first_audio_ms=first_audio_ms,
            frames_sent=len(chunks),
            sent_bytes=total_bytes,
            audio_format="pcmu_8000_buffered",
        )
        return {
            "success": True,
            "first_audio_sent_at": first_audio_sent_at,
            "frames_sent": len(chunks),
        }

    # --- shared helpers --------------------------------------------------

    @staticmethod
    async def _send_frame(frame: bytes, turn: int, websocket: WebSocket) -> bool:
        """Send one base64 PCMU frame to Telnyx as a media event."""
        payload = base64.b64encode(frame).decode("ascii")
        try:
            await websocket.send_text(
                json.dumps({"event": "media", "media": {"payload": payload}})
            )
        except Exception as e:
            logger.error(
                "[VOICE:TELNYX:TTS] turn=%d websocket_send_error error=%s",
                turn,
                str(e),
            )
            return False
        return True

    @staticmethod
    async def _close_stream(stream: AsyncIterator[bytes]) -> None:
        """Release the provider stream without masking the original error."""
        aclose = getattr(stream, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception:
            pass

    async def _announce_first_audio(
        self,
        turn: int,
        call_id: str,
        ttfa_ms: float,
        audio_format: str,
    ) -> None:
        """Tell observers the caller is now hearing AI audio.

        Emitted the instant the FIRST PCMU frame is handed to Telnyx — not
        after the whole utterance has been synthesized.
        """
        await broadcast_telephony_event(
            "tts_first_audio",
            call_id,
            f"AI voice reaching the caller ({ttfa_ms / 1000:.2f}s)",
            turn=turn,
            metadata={"ttfa_ms": round(ttfa_ms), "format": audio_format},
        )
        await broadcast_telephony_event(
            "audio_streaming",
            call_id,
            "Streaming AI voice to caller",
            turn=turn,
            metadata={
                "format": audio_format,
                "frame_bytes": _PCMU_FRAME_BYTES,
            },
        )

    async def _emit_turn_summary(
        self,
        *,
        turn: int,
        call_id: str,
        tts_start: float,
        provider_done_ms: float | None,
        first_audio_ms: float | None,
        frames_sent: int,
        sent_bytes: int,
        audio_format: str,
    ) -> None:
        """Emit the terminal tts_completed event.

        Note: turn_completed is emitted by _agent_worker() with all
        end-to-end metrics.
        """
        total_ms = (time.monotonic() - tts_start) * 1000
        playback_ms = sent_bytes / _ULAW_BYTES_PER_SECOND * 1000

        await broadcast_telephony_event(
            "tts_completed",
            call_id,
            "Voice generated",
            turn=turn,
            metadata={
                "tts_total_ms": round(provider_done_ms or total_ms),
                "bytes": sent_bytes,
                "chunks": frames_sent,
                "tts_ttfa_ms": round(first_audio_ms) if first_audio_ms else None,
                "playback_ms": round(playback_ms),
                "format": audio_format,
            },
        )
        logger.info(
            "[VOICE:TELNYX:TTS] turn=%d completed ttfa_ms=%s total_ms=%.0f "
            "playback_ms=%.0f frames=%d bytes=%d",
            turn,
            f"{first_audio_ms:.0f}" if first_audio_ms is not None else "n/a",
            total_ms,
            playback_ms,
            frames_sent,
            sent_bytes,
        )
