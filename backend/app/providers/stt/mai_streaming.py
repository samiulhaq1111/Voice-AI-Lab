"""Microsoft MAI-Transcribe-2-Streaming realtime STT adapter (Azure AI).

Implements ``StreamingSTTSession`` against the MAI Realtime API
(``wss://<resource>.services.ai.azure.com/mai/v1/realtime?intent=transcription``).
Selected via ``STT_PROVIDER=mai`` or the per-session ``stt_provider`` START
field; Deepgram remains the default provider.

One session = one persistent WebSocket connection for the whole realtime
voice session (never one connection per audio chunk). Authentication uses
the server-side ``api-key`` header — the key never reaches the browser and
is never logged.

Protocol (client -> server), per the documented OpenAI-Realtime-like
transcription protocol:
    {"type": "session.update", "event_id": "configure-1", "session": {
       "type": "transcription",
       "audio": {"input": {
         "format": {"type": "audio/pcm", "rate": 16000},
         "transcription": {"model": "<deployment>", "language": ...},
         "turn_detection": null, "noise_reduction": null}}}}
       (sent once; settings cannot change after the first audio append)
    {"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
       - signed little-endian PCM16, mono, 16 kHz, no WAV header
    {"type": "input_audio_buffer.commit"}
       - finalizes all audio sent so far; no automatic VAD and no automatic
         commits exist (turn_detection is null)

Protocol (server -> client), relevant messages:
    {"type": "session.created", ...}
    {"type": "session.updated", ...}                       (handshake ack)
    {"type": "conversation.item.input_audio_transcription.delta",
     "item_id": ..., "delta": "<newly finalized text>"}
    {"type": "conversation.item.input_audio_transcription.intermediate",
     "item_id": ..., "intermediate": "<provisional suffix>"}
       - the provisional suffix is relative to the latest delta event and
         replaces any previously received suffix; the current best
         hypothesis is  "".join(deltas) + intermediate
    {"type": "input_audio_buffer.committed", ...}
       - commit acknowledgement; finalization continues afterwards
    {"type": "conversation.item.input_audio_transcription.completed",
     "transcript": "<full final text for the committed interval>"}
    {"type": "error", ...}

Event mapping onto the provider-neutral ``StreamEvent`` contract:
    delta / intermediate -> "partial" (display hypothesis, never releases
                            a turn — the gateway cancels pending releases
                            on any new partial evidence)
    completed            -> "final" with ``metadata["speech_final"] = True``
                            so the gateway's existing early-release path
                            (600 ms settle, accumulation across segments)
                            runs unchanged; the full ``transcript`` field is
                            authoritative and never re-derived from deltas
    error                -> "error"
    input_audio_buffer.committed / session frames -> consumed internally
    (None), never a stream termination for the gateway pump

Endpointing: the protocol has no server-side speech detection, so the
documentation expects the client to commit "at a natural pause in speech".
This adapter therefore runs a minimal client-side endpoint check on the
16 kHz PCM it forwards (per-frame RMS against a fixed threshold, with a
trailing-silence window of ``StreamConfig.endpointing_ms``) and commits
when speech is followed by that much silence — the same endpointing
semantics Deepgram/Qwen provide server-side. A safety cap also commits
audio that stays uncommitted for too long. Sessions are limited to one
hour by the service; this adapter does not add reconnection.

Audio contents and credentials are never logged.
"""

import asyncio
import base64
import json
import math
import struct
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
from app.utils.audio import PCM16Resampler

# The API accepts only signed little-endian PCM16, mono, at this rate.
_MAI_SAMPLE_RATE = 16000
_MAI_WS_PATH = "/mai/v1/realtime?intent=transcription"
_MAI_DEFAULT_DEPLOYMENT = "MAI-Transcribe-2-Streaming"

# Bound the WebSocket handshake and the session.update -> session.updated
# exchange so a dead endpoint fails fast.
_OPEN_TIMEOUT_CAP_S = 10.0
_SESSION_UPDATE_TIMEOUT_S = 10.0
_PING_INTERVAL_S = 20.0
_PING_TIMEOUT_S = 20.0

# Client-side endpoint check (see module docstring): a forwarded frame whose
# PCM16 RMS is below this value counts as silence (~-42 dBFS). The
# trailing-silence window is StreamConfig.endpointing_ms (default 300 ms,
# mirroring the endpointing the other providers apply server-side).
_SPEECH_RMS_THRESHOLD = 250.0

# Safety cap: commit uncommitted audio at this age even without a silence
# gap, so continuous speech still produces finals within a bounded time.
_MAX_UNCOMMITTED_MS = 15000.0

# MAI realtime transcription wire contract (verified against the Azure
# Speech Service "MAI-Transcribe-2-Streaming with the Realtime API" doc).
_EV_SESSION_CREATED = "session.created"
_EV_SESSION_UPDATED = "session.updated"
_EV_DELTA = "conversation.item.input_audio_transcription.delta"
_EV_INTERMEDIATE = "conversation.item.input_audio_transcription.intermediate"
_EV_COMPLETED = "conversation.item.input_audio_transcription.completed"
_EV_COMMITTED = "input_audio_buffer.committed"


def _extract_text(data: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string field among ``keys`` (or "")."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _error_detail(data: dict[str, Any]) -> str:
    """Extract a truncated error detail from an MAI error frame.

    Error messages never contain credentials; the value is bounded so a
    misbehaving server cannot flood the logs.
    """
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or "MAI stream error"
    elif isinstance(error, str) and error:
        message = error
    else:
        message = data.get("message") or "MAI stream error"
    return str(message)[:200]


class MaiTranscribeStreamingSession(StreamingSTTSession):
    """One realtime streaming session against the MAI Realtime API."""

    def __init__(self, config: StreamConfig) -> None:
        from app.core.config import settings

        self._config = config
        self._api_key = settings.mai_stt_api_key
        self._resource_url = settings.mai_stt_resource_url
        self._deployment = settings.mai_stt_deployment or _MAI_DEFAULT_DEPLOYMENT
        if not self._api_key:
            raise ValueError(
                "MAI-Transcribe API key not configured. Set MAI_STT_API_KEY."
            )
        if not self._resource_url:
            raise ValueError(
                "MAI-Transcribe resource URL not configured. "
                "Set MAI_STT_RESOURCE_URL."
            )
        self._open_timeout = min(settings.stt_timeout, _OPEN_TIMEOUT_CAP_S)
        self._update_timeout = min(settings.stt_timeout, _SESSION_UPDATE_TIMEOUT_S)
        # Browser audio arrives at the AudioContext native rate (48 kHz /
        # 44.1 kHz) and is resampled to the required 16 kHz exactly once,
        # here; 16 kHz input (telephony/test paths) passes through.
        self._resampler: PCM16Resampler | None = (
            PCM16Resampler(config.sample_rate, _MAI_SAMPLE_RATE)
            if config.sample_rate != _MAI_SAMPLE_RATE
            else None
        )
        self._ws: Any = None
        self._closed = False
        self._finished = False
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._commits_sent = 0
        self._commit_acks = 0
        # Uncommitted-audio / endpointing tracking (16 kHz domain).
        self._uncommitted_bytes = 0
        self._uncommitted_ms = 0.0
        self._silence_ms = 0.0
        self._has_speech = False
        # Provisional transcript state for the item being transcribed.
        self._item_id: str | None = None
        self._delta_text = ""
        self._interim_suffix = ""
        # Minimal latency instrumentation (existing monotonic-mark
        # convention); *_ms values are anchored at "session ready"
        # (session.updated received).
        self._ready_at: float | None = None
        self._first_partial_at: float | None = None
        self._first_final_at: float | None = None

    @property
    def provider_name(self) -> str:
        return "mai"

    def _build_ws_url(self) -> str:
        """Build the realtime WebSocket URL from the configured resource root."""
        base = self._resource_url.strip().rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        return f"{base}{_MAI_WS_PATH}"

    def _build_session_update(self) -> dict[str, Any]:
        """Build the documented ``session.update`` payload (sent once)."""
        return {
            "type": "session.update",
            "event_id": "configure-1",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": _MAI_SAMPLE_RATE},
                        "transcription": {
                            "model": self._deployment,
                            "language": self._config.language or None,
                        },
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }

    async def start(self) -> None:
        """Open the persistent MAI WebSocket and confirm ``session.updated``.

        Audio is only sent after the configuration acknowledgement, as the
        protocol requires. Any failure after the socket opened closes it
        before raising so a half-started session cannot leak a connection.
        """
        if self._ws is not None:
            raise StreamingSTTError("Streaming session already started")

        url = self._build_ws_url()
        started_at = time.monotonic()
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"api-key": self._api_key},
                open_timeout=self._open_timeout,
                ping_interval=_PING_INTERVAL_S,
                ping_timeout=_PING_TIMEOUT_S,
                close_timeout=5,
                max_size=None,
            )
        except Exception as e:
            logger.error(
                "[MAI:STT] connect_failed host=%s error_type=%s",
                self._resource_url,
                type(e).__name__,
            )
            raise StreamingSTTError(
                f"MAI streaming connection failed: {type(e).__name__}"
            ) from e

        try:
            await self._ws.send(json.dumps(self._build_session_update()))
            await asyncio.wait_for(
                self._await_session_updated(), timeout=self._update_timeout
            )
        except asyncio.CancelledError:
            await self._close_quietly()
            raise
        except TimeoutError as e:
            await self._close_quietly()
            raise StreamingSTTError(
                "MAI session.update was not confirmed (session.updated timeout)"
            ) from e
        except StreamingSTTError:
            await self._close_quietly()
            raise
        except Exception as e:
            await self._close_quietly()
            logger.error(
                "[MAI:STT] session_setup_failed error_type=%s", type(e).__name__
            )
            raise StreamingSTTError(
                f"MAI session setup failed: {type(e).__name__}"
            ) from e

        self._ready_at = time.monotonic()
        logger.info(
            "[MAI:STT] connected host=%s deployment=%s connect_ms=%.1f "
            "input_rate=%d output_rate=%d encoding=%s language=%s",
            self._resource_url,
            self._deployment,
            (self._ready_at - started_at) * 1000,
            self._config.sample_rate,
            _MAI_SAMPLE_RATE,
            self._config.encoding,
            self._config.language,
        )

    async def _await_session_updated(self) -> None:
        """Read handshake frames until ``session.updated`` (or failure).

        Called only from ``start()`` — the single-reader guarantee holds
        because the gateway pump is not started until ``start()`` returns.
        """
        ws = self._ws
        if ws is None:
            raise StreamingSTTError("MAI session not connected")
        while True:
            try:
                raw = await ws.recv()
            except ConnectionClosed as e:
                raise StreamingSTTError(
                    "MAI stream closed before session.updated "
                    f"(code={e.rcvd.code if e.rcvd else 'n/a'})"
                ) from e

            if isinstance(raw, bytes):
                logger.debug("[MAI:STT] binary_frame_ignored bytes=%d", len(raw))
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("[MAI:STT] non_json_frame_ignored")
                continue
            if not isinstance(data, dict):
                continue

            msg_type = data.get("type", "")
            if msg_type == _EV_SESSION_UPDATED:
                logger.debug("[MAI:STT] session_updated received")
                return
            if msg_type == "error":
                detail = _error_detail(data)
                logger.error("[MAI:STT] session_update_error detail=%s", detail)
                raise StreamingSTTError(f"MAI session.update rejected: {detail}")
            logger.debug("[MAI:STT] handshake_frame_ignored type=%s", msg_type)

    async def send_audio(self, chunk: bytes) -> None:
        """Resample (when needed) and forward one PCM16 chunk to MAI.

        Audio is sent as documented ``input_audio_buffer.append`` messages
        carrying base64 PCM16 (no WAV header). After forwarding, the
        endpoint check may issue an ``input_audio_buffer.commit``.
        """
        if self._ws is None:
            raise StreamingSTTError("Streaming session not started")
        if self._finished or self._closed:
            return

        pcm = self._resampler.process(chunk) if self._resampler is not None else chunk
        if not pcm:
            return

        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            )
        except ConnectionClosed as e:
            logger.warning(
                "[MAI:STT] closed_during_send code=%s",
                e.rcvd.code if e.rcvd else "n/a",
            )
            raise StreamingSTTError("MAI stream closed while sending") from e

        self._chunks_sent += 1
        self._bytes_sent += len(pcm)
        self._uncommitted_bytes += len(pcm)
        self._uncommitted_ms += len(pcm) / 2 / (_MAI_SAMPLE_RATE / 1000)

        reason = self._track_endpointing(pcm)
        if reason:
            await self._send_commit(reason)

    def _track_endpointing(self, pcm: bytes) -> str:
        """Track speech/silence on the forwarded 16 kHz stream.

        Minimal client-side endpoint check (see module docstring): frames
        below the RMS threshold extend the trailing-silence window once
        speech has been forwarded; a commit is due when that window reaches
        ``StreamConfig.endpointing_ms``. The max-uncommitted cap commits
        continuous speech that never pauses long enough.

        Returns the commit reason ("" = no commit due).
        """
        num_samples = len(pcm) // 2
        frame_ms = num_samples / (_MAI_SAMPLE_RATE / 1000)
        if num_samples:
            samples = struct.unpack(f"<{num_samples}h", pcm)
            square_sum = 0
            for sample in samples:
                square_sum += sample * sample
            rms = math.sqrt(square_sum / num_samples)
        else:
            rms = 0.0

        if rms >= _SPEECH_RMS_THRESHOLD:
            self._has_speech = True
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        if self._has_speech and self._silence_ms >= self._config.endpointing_ms:
            return "endpoint"
        if self._uncommitted_ms >= _MAX_UNCOMMITTED_MS:
            return "max_uncommitted"
        return ""

    async def _send_commit(self, reason: str) -> None:
        """Send ``input_audio_buffer.commit`` and reset the commit tracking."""
        if self._ws is None or self._closed:
            return
        if self._uncommitted_bytes <= 0:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except ConnectionClosed as e:
            logger.warning(
                "[MAI:STT] closed_during_commit code=%s",
                e.rcvd.code if e.rcvd else "n/a",
            )
            raise StreamingSTTError("MAI stream closed while committing") from e

        self._commits_sent += 1
        logger.info(
            "[MAI:STT] commit_sent n=%d reason=%s uncommitted_ms=%.0f "
            "silence_ms=%.0f",
            self._commits_sent,
            reason,
            self._uncommitted_ms,
            self._silence_ms,
        )
        self._uncommitted_bytes = 0
        self._uncommitted_ms = 0.0
        self._silence_ms = 0.0
        self._has_speech = False

    async def receive(self) -> StreamEvent | None:
        """Await the next *meaningful* provider event.

        Returns None ONLY when the provider stream has ended (or the
        session was closed) — ignorable frames (binary, session frames,
        commit acknowledgements, empty transcripts, unknown types) are
        consumed internally, matching the contract the gateway pump
        relies on.
        """
        if self._ws is None or self._closed:
            return None
        while True:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as e:
                logger.info(
                    "[MAI:STT] disconnected code=%s",
                    e.rcvd.code if e.rcvd else "n/a",
                )
                return None
            except asyncio.CancelledError:
                raise

            if isinstance(raw, bytes):
                # The transcription endpoint only sends JSON text frames;
                # ignore stray binary frames without ending the stream.
                logger.debug("[MAI:STT] binary_frame_ignored bytes=%d", len(raw))
                continue

            event = self._translate_message(raw)
            if event is not None:
                return event
            # Ignorable frame — keep waiting for the next meaningful event.
            continue

    def _translate_message(self, raw: str) -> StreamEvent | None:
        """Translate one MAI JSON frame into a StreamEvent (or None)."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[MAI:STT] non_json_frame_ignored")
            return None
        if not isinstance(data, dict):
            return None

        msg_type = data.get("type", "")

        if msg_type == _EV_DELTA:
            return self._translate_delta(data)
        if msg_type == _EV_INTERMEDIATE:
            return self._translate_intermediate(data)
        if msg_type == _EV_COMPLETED:
            return self._translate_completed(data)
        if msg_type == _EV_COMMITTED:
            # Commit acknowledgement — internal protocol event; finalization
            # continues afterwards (deltas / completed still follow). Never a
            # stream termination.
            self._commit_acks += 1
            logger.info(
                "[MAI:STT] commit_ack n=%d item_id=%s",
                self._commit_acks,
                data.get("item_id"),
            )
            return None
        if msg_type in (_EV_SESSION_CREATED, _EV_SESSION_UPDATED):
            logger.debug("[MAI:STT] session_frame_ignored type=%s", msg_type)
            return None
        if msg_type == "error":
            detail = _error_detail(data)
            logger.error("[MAI:STT] server_error detail=%s", detail)
            return StreamEvent(type="error", text=detail)

        logger.debug("[MAI:STT] unhandled_frame type=%s", msg_type)
        return None

    def _sync_item(self, data: dict[str, Any]) -> None:
        """Reset provisional text when transcription events switch items."""
        item_id = data.get("item_id")
        if isinstance(item_id, str) and item_id and item_id != self._item_id:
            self._item_id = item_id
            self._delta_text = ""
            self._interim_suffix = ""

    def _translate_delta(self, data: dict[str, Any]) -> StreamEvent | None:
        """Translate a ``delta`` frame into a partial (accumulated text).

        The documented semantics: append ``delta`` verbatim to the
        accumulated buffer ("don't insert or strip spaces"). Intermediates
        are relative to the *latest* delta, so a new delta supersedes any
        suffix received before it.
        """
        self._sync_item(data)
        text = _extract_text(data, "delta", "text")
        if not text:
            return None
        self._delta_text += text
        self._interim_suffix = ""
        self._log_partial(self._delta_text)
        return StreamEvent(type="partial", text=self._delta_text)

    def _translate_intermediate(self, data: dict[str, Any]) -> StreamEvent | None:
        """Translate an ``intermediate`` frame into a partial.

        The event carries the entire current provisional suffix, replacing
        the previous one; the current best hypothesis is
        ``"".join(deltas) + intermediate``.
        """
        self._sync_item(data)
        suffix = _extract_text(data, "intermediate", "text")
        if not suffix:
            return None
        self._interim_suffix = suffix
        display = f"{self._delta_text}{self._interim_suffix}"
        self._log_partial(display)
        return StreamEvent(type="partial", text=display)

    def _translate_completed(self, data: dict[str, Any]) -> StreamEvent | None:
        """Translate a ``completed`` frame into the turn's final event.

        ``transcript`` is the full final text for the committed interval
        (equivalent to the accumulated deltas) and is authoritative — the
        final is never re-derived by concatenating deltas, so text cannot
        be duplicated. ``speech_final=True`` routes the final through the
        gateway's existing early-release path.
        """
        self._sync_item(data)
        text = _extract_text(data, "transcript", "text")
        if not text:
            # Defensive: fall back to the accumulated deltas (or, last
            # resort, the provisional suffix) when the server omits the
            # transcript field.
            text = self._delta_text or self._interim_suffix
        self._delta_text = ""
        self._interim_suffix = ""
        if not text.strip():
            logger.debug("[MAI:STT] empty_completed_ignored")
            return None

        now = time.monotonic()
        if self._first_final_at is None:
            self._first_final_at = now
            logger.info(
                '[MAI:STT] final final_ms=%s text="%s"',
                self._final_ms(),
                text[:120],
            )
        else:
            logger.info('[MAI:STT] final text="%s"', text[:120])
        return StreamEvent(
            type="final",
            text=text,
            metadata={"speech_final": True},
        )

    def _log_partial(self, text: str) -> None:
        """Log a partial (the first one reports latency since ready)."""
        now = time.monotonic()
        if self._first_partial_at is None:
            self._first_partial_at = now
            logger.info(
                '[MAI:STT] first_partial first_partial_ms=%s text="%s"',
                self._partial_ms(),
                text[:120],
            )
        else:
            logger.info('[MAI:STT] partial text="%s"', text[:120])

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
        """Flush remaining audio with a final commit; the socket stays open.

        ``input_audio_buffer.commit`` requests the final transcript for all
        audio sent so far "as fast as possible". The connection is left
        open so ``receive()`` keeps delivering commit acknowledgements /
        deltas / ``completed`` until ``close()`` — the gateway's STOP path
        cancels the pump immediately after, matching the existing
        cancellation semantics.
        """
        if self._finished or self._ws is None or self._closed:
            return
        self._finished = True
        if self._uncommitted_bytes > 0:
            await self._send_commit("finish")
        else:
            logger.info("[MAI:STT] finish_sent no_uncommitted_audio")

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
                    "[MAI:STT] close_error error_type=%s", type(e).__name__
                )
        logger.info(
            "[MAI:STT] closed chunks=%d bytes=%d commits=%d commit_acks=%d",
            self._chunks_sent,
            self._bytes_sent,
            self._commits_sent,
            self._commit_acks,
        )

    async def _close_quietly(self) -> None:
        """Best-effort close used on start() failure paths (never raises)."""
        try:
            await self.close()
        except Exception as e:
            logger.warning("[MAI:STT] close_error error_type=%s", type(e).__name__)
