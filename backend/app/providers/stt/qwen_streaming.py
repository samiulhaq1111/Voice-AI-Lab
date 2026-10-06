"""Qwen ASR realtime streaming STT adapter (Kaggle WebSocket server).

Implements ``StreamingSTTSession`` against the Qwen ASR WebSocket server
(typically a Kaggle tunnel), speaking its native wire protocol. Selected
via ``STT_PROVIDER=qwen``; Deepgram remains the default provider.

One session = one persistent WebSocket connection for the whole realtime
voice session (never one connection per audio chunk).

Protocol (client -> server):
    {"type": "start", "encoding": "linear16", "sample_rate": 16000,
     "channels": 1, "endpointing_ms": 300, "utterance_end_ms": 1000,
     "vad_threshold": 0.012, "inference_step_ms": 500, "language": "en"}
    - binary frames: raw PCM16 little-endian, mono (16 kHz preferred —
      the server resamples if the client sends another rate)
    - {"type": "CloseStream"} to finish gracefully

Protocol (server -> client), relevant messages:
    {"type": "Metadata", "model": "Qwen/Qwen3-ASR-0.6B", ...}
    {"type": "Results", "is_final": bool, "speech_final": bool,
     "text": str, "confidence": float, "language": str}
    {"type": "UtteranceEnd", "last_word_end": float}

Events are mapped onto the provider-neutral ``StreamEvent`` contract used
by the realtime gateway (partial / final / utterance_end / error), so the
existing pump, turn release and agent worker run unchanged.

Audio contents and credentials are never logged.
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

# Start-message defaults expected by the working Kaggle ASR server.
_QWEN_DEFAULT_VAD_THRESHOLD = 0.012
_QWEN_DEFAULT_INFERENCE_STEP_MS = 500

# Bound the WebSocket handshake so a dead Kaggle tunnel fails fast.
_OPEN_TIMEOUT_CAP_S = 10.0
_PING_INTERVAL_S = 20.0
_PING_TIMEOUT_S = 20.0


class QwenKaggleStreamingSession(StreamingSTTSession):
    """One realtime streaming session against the Qwen ASR WebSocket server."""

    def __init__(self, config: StreamConfig) -> None:
        from app.core.config import settings

        self._config = config
        self._ws_url = settings.qwen_asr_ws_url
        if not self._ws_url:
            raise ValueError("Qwen ASR WebSocket URL not configured. Set QWEN_ASR_WS_URL.")
        # Optional: send a standard bearer header only when a key is
        # configured (the local Kaggle server requires one).
        self._api_key = settings.qwen_asr_api_key
        self._open_timeout = min(settings.stt_timeout, _OPEN_TIMEOUT_CAP_S)
        self._ws: Any = None
        self._closed = False
        self._finished = False
        self._chunks_sent = 0
        self._bytes_sent = 0
        # Minimal latency instrumentation (existing monotonic-mark convention).
        # All qwen_*_ms values are anchored at "session ready" (start message
        # sent) so they measure ASR latency, not connect overhead.
        self._ready_at: float | None = None
        self._first_partial_at: float | None = None
        self._first_final_at: float | None = None

    @property
    def provider_name(self) -> str:
        return "qwen"

    def _build_start_message(self) -> dict[str, Any]:
        """Build the Qwen start message from the shared StreamConfig."""
        cfg = self._config
        return {
            "type": "start",
            "encoding": cfg.encoding,
            "sample_rate": cfg.sample_rate,
            "channels": cfg.channels,
            "endpointing_ms": cfg.endpointing_ms,
            "utterance_end_ms": cfg.utterance_end_ms,
            "vad_threshold": _QWEN_DEFAULT_VAD_THRESHOLD,
            "inference_step_ms": _QWEN_DEFAULT_INFERENCE_STEP_MS,
            "language": cfg.language,
        }

    async def start(self) -> None:
        """Open the persistent Qwen WebSocket connection and send ``start``."""
        if self._ws is not None:
            raise StreamingSTTError("Streaming session already started")

        started_at = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "open_timeout": self._open_timeout,
                "ping_interval": _PING_INTERVAL_S,
                "ping_timeout": _PING_TIMEOUT_S,
                "close_timeout": 5,
                "max_size": None,
            }
            if self._api_key:
                kwargs["additional_headers"] = {
                    "Authorization": f"Bearer {self._api_key}"
                }
            self._ws = await websockets.connect(self._ws_url, **kwargs)
        except (TimeoutError, OSError, Exception) as e:
            logger.error(
                "[QWEN:STT] connect_failed url=%s error_type=%s",
                self._ws_url,
                type(e).__name__,
            )
            raise StreamingSTTError(
                f"Qwen ASR streaming connection failed: {type(e).__name__}"
            ) from e

        try:
            await self._ws.send(json.dumps(self._build_start_message()))
        except ConnectionClosed as e:
            raise StreamingSTTError(
                "Qwen ASR stream closed while sending start"
            ) from e

        self._ready_at = time.monotonic()
        logger.info(
            "[QWEN:STT] connected url=%s connect_ms=%.1f sample_rate=%d "
            "encoding=%s language=%s endpointing_ms=%d utterance_end_ms=%d",
            self._ws_url,
            (self._ready_at - started_at) * 1000,
            self._config.sample_rate,
            self._config.encoding,
            self._config.language,
            self._config.endpointing_ms,
            self._config.utterance_end_ms,
        )

    async def send_audio(self, chunk: bytes) -> None:
        """Forward one binary PCM16 chunk to the Qwen server."""
        if self._ws is None:
            raise StreamingSTTError("Streaming session not started")
        if self._finished or self._closed:
            return
        try:
            await self._ws.send(chunk)
        except ConnectionClosed as e:
            logger.warning(
                "[QWEN:STT] closed_during_send code=%s",
                e.rcvd.code if e.rcvd else "n/a",
            )
            raise StreamingSTTError("Qwen ASR stream closed while sending") from e
        self._chunks_sent += 1
        self._bytes_sent += len(chunk)

    async def receive(self) -> StreamEvent | None:
        """Await the next *meaningful* provider event.

        Returns None ONLY when the provider stream has ended (or the session
        was closed) — ignorable frames (binary, unknown types, empty
        transcripts) are consumed internally, matching the contract the
        gateway pump relies on.
        """
        if self._ws is None or self._closed:
            return None
        while True:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as e:
                logger.info(
                    "[QWEN:STT] disconnected code=%s",
                    e.rcvd.code if e.rcvd else "n/a",
                )
                return None
            except asyncio.CancelledError:
                raise

            if isinstance(raw, bytes):
                # The server only sends JSON text frames; ignore stray
                # binary frames without ending the stream.
                logger.debug("[QWEN:STT] binary_frame_ignored bytes=%d", len(raw))
                continue

            event = self._translate_message(raw)
            if event is not None:
                return event
            # Ignorable frame — keep waiting for the next meaningful event.
            continue

    def _translate_message(self, raw: str) -> StreamEvent | None:
        """Translate one Qwen JSON frame into a StreamEvent (or None)."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[QWEN:STT] non_json_frame_ignored")
            return None
        if not isinstance(data, dict):
            return None

        msg_type = data.get("type", "")

        if msg_type == "Results":
            return self._translate_results(data)
        if msg_type == "Metadata":
            # Initial session metadata (model, sample rates, session id) —
            # informational only.
            logger.debug(
                "[QWEN:STT] metadata model=%s session_id=%s",
                data.get("model"),
                data.get("session_id"),
            )
            return None
        if msg_type == "UtteranceEnd":
            last_word_end = data.get("last_word_end")
            logger.info("[QWEN:STT] utterance_end last_word_end=%s", last_word_end)
            return StreamEvent(
                type="utterance_end",
                metadata={"last_word_end": last_word_end},
            )
        if msg_type == "error":
            detail = str(
                data.get("message") or data.get("error") or "Qwen ASR stream error"
            )
            logger.error("[QWEN:STT] server_error detail=%s", detail[:200])
            return StreamEvent(type="error", text=detail[:200])

        logger.debug("[QWEN:STT] unhandled_frame type=%s", msg_type)
        return None

    def _translate_results(self, data: dict[str, Any]) -> StreamEvent | None:
        """Translate a Qwen ``Results`` message into a partial/final event."""
        text = str(data.get("text") or "")
        if not text:
            # Empty transcript (silence tick) — nothing to report.
            return None

        is_final = bool(data.get("is_final", False))
        speech_final = bool(data.get("speech_final", False))
        try:
            confidence = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        now = time.monotonic()
        if is_final:
            if self._first_final_at is None:
                self._first_final_at = now
                logger.info(
                    '[QWEN:STT] final final_ms=%s speech_final=%s text="%s"',
                    self._final_ms(),
                    speech_final,
                    text[:120],
                )
            else:
                logger.info(
                    '[QWEN:STT] final speech_final=%s text="%s"',
                    speech_final,
                    text[:120],
                )
            return StreamEvent(
                type="final",
                text=text,
                confidence=confidence,
                metadata={"speech_final": speech_final},
            )

        if self._first_partial_at is None:
            self._first_partial_at = now
            logger.info(
                '[QWEN:STT] first_partial first_partial_ms=%s text="%s"',
                self._partial_ms(),
                text[:120],
            )
        else:
            logger.info('[QWEN:STT] partial text="%s"', text[:120])
        return StreamEvent(type="partial", text=text, confidence=confidence)

    def _partial_ms(self) -> str:
        """First-partial latency (ms) since the session became ready."""
        if self._ready_at is None or self._first_partial_at is None:
            return "n/a"
        return f"{(self._first_partial_at - self._ready_at) * 1000:.1f}"

    def _final_ms(self) -> str:
        """First-final latency (ms) since the session became ready."""
        if self._ready_at is None or self._first_final_at is None:
            return "n/a"
        return f"{(self._first_final_at - self._ready_at) * 1000:.1f}"

    async def finish(self) -> None:
        """Signal end-of-audio via the Qwen ``CloseStream`` control message."""
        if self._finished or self._ws is None or self._closed:
            return
        self._finished = True
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
            logger.info("[QWEN:STT] finish_sent type=CloseStream")
        except ConnectionClosed:
            logger.info("[QWEN:STT] already closed on finish")

    async def close(self) -> None:
        """Close the WebSocket connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        ws, self._ws = self._ws, None
        # Detached first so a racing send_audio()/finish() can no longer use
        # the connection while the close handshake is in flight.
        if ws is not None:
            try:
                await ws.close()
            except Exception as e:
                logger.warning(
                    "[QWEN:STT] close_error error_type=%s", type(e).__name__
                )
        logger.info("[QWEN:STT] closed")
