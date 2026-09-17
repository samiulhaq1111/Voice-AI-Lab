"""Deepgram realtime streaming STT adapter (Phase 6A).

Implements ``StreamingSTTSession`` using Deepgram's streaming WebSocket API
(``wss://api.deepgram.com/v1/listen``) via the lightweight ``websockets``
client library — consistent with the existing adapter style that uses raw
``httpx`` for the batch REST API.

The existing batch ``DeepgramAdapter.transcribe()`` is untouched.

Protocol (client -> Deepgram):
    - binary frames: raw PCM audio (linear16)
    - JSON {"type": "CloseStream"} to finish gracefully

Protocol (Deepgram -> client), relevant messages:
    - {"type": "Results", "is_final": bool, "channel": {...}}
    - {"type": "UtteranceEnd", ...}
    - {"type": "Metadata", ...}            (ignored, logged)
    - {"type": "CloseStream"}              (stream finished)
    - {"type": "Error", ...}

API keys are never logged.
"""

import asyncio
import json
import time
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from app.core.logging import logger
from app.providers.stt.streaming import (
    StreamConfig,
    StreamEvent,
    StreamingSTTError,
    StreamingSTTSession,
)

_DEEPGRAM_STREAM_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramStreamingSession(StreamingSTTSession):
    """One realtime streaming session against the Deepgram listen WebSocket."""

    def __init__(self, config: StreamConfig) -> None:
        from app.core.config import settings

        self._config = config
        self._api_key = settings.deepgram_api_key
        self._open_timeout = min(settings.stt_timeout, 10.0)
        self._ws: Any = None
        self._closed = False
        self._finished = False
        self._chunks_sent = 0
        self._bytes_sent = 0

    @property
    def provider_name(self) -> str:
        return "deepgram"

    def _build_url(self) -> str:
        """Build the Deepgram listen WebSocket URL with query parameters."""
        cfg = self._config
        params = {
            "model": cfg.model or "nova-3",
            "language": cfg.language,
            "encoding": cfg.encoding,
            "sample_rate": str(cfg.sample_rate),
            "channels": str(cfg.channels),
            "interim_results": "true" if cfg.interim_results else "false",
            "endpointing": str(cfg.endpointing_ms),
            "utterance_end_ms": str(cfg.utterance_end_ms),
            "smart_format": "true",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{_DEEPGRAM_STREAM_URL}?{query}"

    async def start(self) -> None:
        """Open the Deepgram streaming WebSocket connection."""
        if self._ws is not None:
            raise StreamingSTTError("Streaming session already started")

        url = self._build_url()
        logger.info(
            "[VOICE:REALTIME] deepgram_connecting model=%s "
            "sample_rate=%d encoding=%s language=%s endpointing=%dms utterance_end_ms=%d",
            self._config.model,
            self._config.sample_rate,
            self._config.encoding,
            self._config.language,
            self._config.endpointing_ms,
            self._config.utterance_end_ms,
        )
        try:
            # Authorization header only — never logged, never in URL.
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=self._open_timeout,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=None,
            )
        except (TimeoutError, OSError, Exception) as e:
            logger.error(
                "[VOICE:REALTIME] Deepgram connection failed error_type=%s",
                type(e).__name__,
            )
            raise StreamingSTTError(
                f"Deepgram streaming connection failed: {type(e).__name__}"
            ) from e

        logger.info(
            "[VOICE:REALTIME] deepgram_connected model=%s", self._config.model
        )

    async def send_audio(self, chunk: bytes) -> None:
        """Forward one binary PCM chunk to Deepgram."""
        if self._ws is None:
            raise StreamingSTTError("Streaming session not started")
        if self._finished or self._closed:
            return
        try:
            await self._ws.send(chunk)
            self._chunks_sent += 1
            self._bytes_sent += len(chunk)
            # Periodic send log (never per-frame — too spammy)
            if self._chunks_sent == 1 or self._chunks_sent % 100 == 0:
                logger.info(
                    "[VOICE:REALTIME] deepgram_audio_sent chunk=%d bytes=%d "
                    "total_bytes=%d",
                    self._chunks_sent,
                    len(chunk),
                    self._bytes_sent,
                )
        except ConnectionClosed as e:
            logger.warning(
                "[VOICE:REALTIME] Deepgram closed during send code=%s",
                e.rcvd.code if e.rcvd else "n/a",
            )
            raise StreamingSTTError("Deepgram stream closed while sending") from e

    async def receive(self) -> StreamEvent | None:
        """Await the next *meaningful* provider event.

        Returns None ONLY when the provider stream has ended (connection
        closed). Ignorable messages — Metadata, empty-transcript silence
        ticks, stray binary frames, unknown types — are consumed internally
        without ending the stream.

        This contract is critical: the gateway's event pump treats None as
        stream end. A None for an ignorable message would prematurely kill
        transcript forwarding (the Phase 6A "no transcripts" bug).
        """
        if self._ws is None or self._closed:
            return None
        while True:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as e:
                logger.info(
                    "[VOICE:REALTIME] Deepgram disconnected code=%s",
                    e.rcvd.code if e.rcvd else "n/a",
                )
                return None
            except asyncio.CancelledError:
                raise

            if isinstance(raw, bytes):
                # Deepgram only sends JSON text on this endpoint; ignore
                # stray binary frames without ending the stream.
                logger.debug(
                    "[VOICE:REALTIME] deepgram_binary_frame_ignored bytes=%d",
                    len(raw),
                )
                continue

            logger.info(
                "[VOICE:REALTIME] deepgram_message_received bytes=%d", len(raw)
            )
            event = self._translate_message(raw)
            if event is not None:
                return event
            # Ignorable message (Metadata / silence tick / unknown type) —
            # keep waiting for the next meaningful event.
            continue

    def _translate_message(self, raw: str) -> StreamEvent | None:
        """Translate one Deepgram JSON message into a StreamEvent (or None)."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[VOICE:REALTIME] Non-JSON Deepgram message ignored")
            return None

        msg_type = data.get("type", "")

        if msg_type == "Results":
            return self._translate_results(data)
        if msg_type == "UtteranceEnd":
            logger.info("[VOICE:REALTIME] deepgram_utterance_end")
            return StreamEvent(type="utterance_end")
        if msg_type == "Metadata":
            logger.debug("[VOICE:REALTIME] deepgram_event type=Metadata")
            return None
        if msg_type == "CloseStream":
            logger.info("[VOICE:REALTIME] deepgram_event type=CloseStream")
            return None
        if msg_type == "Error":
            # Deepgram error messages carry a description, not credentials
            description = data.get("description", "") or data.get("message", "")
            logger.error(
                "[VOICE:REALTIME] deepgram_error code=%s message=%s",
                data.get("code", "n/a"),
                str(description)[:200],
            )
            return StreamEvent(
                type="error",
                text=description[:200] or "Deepgram stream error",
            )

        logger.info("[VOICE:REALTIME] deepgram_event type=%s", msg_type)
        return None

    def _translate_results(self, data: dict[str, Any]) -> StreamEvent | None:
        """Translate a Deepgram 'Results' message into partial/final events."""
        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return None

        best = alternatives[0]
        text = best.get("transcript", "")
        confidence = float(best.get("confidence", 0.0))
        is_final = bool(data.get("is_final", False))
        speech_final = bool(data.get("speech_final", False))

        logger.info(
            "[VOICE:REALTIME] deepgram_result is_final=%s speech_final=%s "
            "transcript_length=%d",
            is_final,
            speech_final,
            len(text),
        )

        if not text:
            # Empty transcript (e.g. silence tick) — nothing to report
            return None

        if is_final:
            logger.info('[VOICE:REALTIME] Transcript final="%s"', text[:120])
            return StreamEvent(
                type="final",
                text=text,
                confidence=confidence,
                metadata={"speech_final": speech_final},
            )

        logger.info('[VOICE:REALTIME] Transcript partial="%s"', text[:120])
        return StreamEvent(type="partial", text=text, confidence=confidence)

    async def finish(self) -> None:
        """Signal end-of-audio via Deepgram's CloseStream control message."""
        if self._finished or self._ws is None or self._closed:
            return
        self._finished = True
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
            logger.info("[VOICE:REALTIME] CloseStream sent to Deepgram")
        except ConnectionClosed:
            logger.info("[VOICE:REALTIME] Deepgram already closed on finish")

    async def close(self) -> None:
        """Close the WebSocket connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:REALTIME] Deepgram close error error_type=%s",
                    type(e).__name__,
                )
            finally:
                self._ws = None
        logger.info("[VOICE:REALTIME] deepgram_closed")

    @staticmethod
    def monotonic_ms() -> float:
        """Current monotonic clock in milliseconds (timing helper)."""
        return time.monotonic() * 1000
