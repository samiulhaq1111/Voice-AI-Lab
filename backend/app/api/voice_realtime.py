"""Realtime streaming voice WebSocket gateway (Phase 6B).

Realtime voice path with AgentRuntime integration:

    Browser microphone (Int16 PCM)
        ↓  application WebSocket
    FastAPI realtime gateway (/api/v1/voice/realtime/ws)
        ↓  StreamingSTTSession abstraction
    Deepgram streaming WebSocket
        ↓
    transcript_partial / transcript_final / utterance_end events
        ↓
    AgentRuntime (on released utterance turn — UtteranceEnd + settle window)
        ↓
    agent_response event → browser

Client → Server protocol:
    {"type": "start", "sample_rate": 16000, "channels": 1,
     "encoding": "linear16", "language": "en", "model": "nova-3",
     "llm_provider": "openrouter", "llm_model": "openai/gpt-4o-mini",
     "utterance_end_ms": 2000}
    <binary PCM audio chunks>
    {"type": "browser_timing", "turn": 1, "ws_transit_ms": 3.0,
     "received_to_playing_ms": 15.0}  (Phase 6G playback report)
    {"type": "stop"}

Server → Client protocol:
    {"type": "session_started", "session_id": "...", "provider": "...",
     "model": "...", "sample_rate": 16000}
    {"type": "transcript_partial", "text": "...", "confidence": 0.9}
    {"type": "transcript_final", "text": "...", "confidence": 0.95}
    {"type": "utterance_end"}
    {"type": "agent_processing"}
    {"type": "agent_response", "text": "...", "tool_calls": 0, "iterations": 1}
    {"type": "tts_processing"}
    {"type": "audio", "format": "audio/mpeg", "data": "...", "turn": 1,
     "segment": 1, "sent_epoch_ms": 1727000000000}
    {"type": "turn_metrics", "data": {...}}
    {"type": "completed", "timings": {...}}
    {"type": "error", "stage": "stt"|"agent", "message": "..."}

Phase 6H: the TTS branch synthesizes one ordered audio segment per completed
LLM sentence — the first sentence is synthesized and delivered while the LLM
is still streaming the remainder of the response.
"""

import asyncio
import base64
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import logger
from app.providers.factory import get_llm_provider, get_tts_provider
from app.providers.llm.interface import LLMInterface
from app.providers.stt.streaming import (
    StreamConfig,
    StreamingSTTError,
    open_streaming_session,
)
from app.providers.tts.interface import TTSInterface
from app.providers.types import LLMMessage
from app.services.realtime_voice_service import (
    RealtimeVoiceError,
    create_realtime_session,
    process_realtime_utterance,
)
from app.services.sentence_buffer import SentenceBuffer

router = APIRouter()

# Turn-release settle window (Phase 6E).
#
# Deepgram's UtteranceEnd event only states that a >= utterance_end_ms word
# gap was detected — the speaker may still continue (e.g. a thinking pause in
# "…employee ID. [pause] E zero zero one."). UtteranceEnd therefore only
# *schedules* a turn release after this settle window instead of releasing
# immediately; any new transcript evidence (partial or final) within the
# window cancels the pending release and the utterance keeps accumulating.
_TURN_RELEASE_SETTLE_MS = 800


@dataclass
class UtteranceTimings:
    """Per-utterance timing milestones for one agent turn.

    All timestamps are ``time.monotonic()`` values — only useful for
    elapsed-time calculations, never wall-clock alignment.

    Phase 6G adds the release mark and LLM sub-stage marks so the gateway
    can distinguish "the LLM is slow to its first token" from "the LLM
    streamed early but TTS waited for the complete response".

    Phase 6H synthesizes one audio segment per completed LLM sentence as
    soon as SentenceBuffer yields it. On the provider streaming path
    ``tts_first_audio_at`` records the first audio chunk of the first
    sentence; on the buffered fallback (provider cannot stream)
    ``tts_first_audio_at`` stays None and ``tts_first_audio_reason``
    records why — the metric is reported as unavailable.
    """

    turn: int = 0
    utterance_end_at: float | None = None
    utterance_released_at: float | None = None
    agent_processing_started_at: float | None = None
    llm_request_started_at: float | None = None
    llm_first_token_at: float | None = None
    llm_first_sentence_at: float | None = None
    llm_completed_at: float | None = None
    tts_started_at: float | None = None
    tts_first_audio_at: float | None = None
    tts_completed_at: float | None = None
    audio_sent_at: float | None = None
    streamed_tokens: int = 0
    llm_sentence_count: int = 0
    # Phase 6H sentence-streaming TTS marks (first sentence + first delivery)
    tts_first_sentence_started_at: float | None = None
    tts_first_sentence_completed_at: float | None = None
    first_audio_sent_at: float | None = None
    tts_sentence_count: int = 0
    tts_sentence_failures: int = 0
    tts_audio_chunks: int = 0
    tts_mode: str | None = None

    def to_metrics_dict(self, base: float) -> dict[str, float | int | str | None]:
        """Serialize calculated durations as milliseconds.

        Args:
            base: The monotonic reference (ws_accepted_at) for offsets.

        Returns:
            Dict with latency durations (ms) and raw offsets (ms from base).
        """

        def _elapsed(start: float | None, end: float | None) -> float | None:
            if start is None or end is None:
                return None
            return round((end - start) * 1000, 1)

        def _offset(mark: float | None) -> float | None:
            if mark is None:
                return None
            return round((mark - base) * 1000, 1)

        return {
            "turn": self.turn,
            # Calculated durations (Phase 6E, preserved)
            "utterance_end_to_agent_ms": _elapsed(
                self.utterance_end_at, self.agent_processing_started_at
            ),
            "llm_first_token_ms": _elapsed(
                self.agent_processing_started_at, self.llm_first_token_at
            ),
            "agent_processing_ms": _elapsed(
                self.agent_processing_started_at, self.llm_completed_at
            ),
            "tts_duration_ms": _elapsed(self.tts_started_at, self.tts_completed_at),
            "utterance_end_to_audio_sent_ms": _elapsed(
                self.utterance_end_at, self.audio_sent_at
            ),
            # Phase 6G segment durations — release/LLM/TTS sub-stages
            "utterance_end_to_release_ms": _elapsed(
                self.utterance_end_at, self.utterance_released_at
            ),
            "release_to_agent_ms": _elapsed(
                self.utterance_released_at, self.agent_processing_started_at
            ),
            "agent_to_llm_request_ms": _elapsed(
                self.agent_processing_started_at, self.llm_request_started_at
            ),
            "llm_request_to_first_token_ms": _elapsed(
                self.llm_request_started_at, self.llm_first_token_at
            ),
            "llm_first_token_to_first_sentence_ms": _elapsed(
                self.llm_first_token_at, self.llm_first_sentence_at
            ),
            "llm_request_to_first_sentence_ms": _elapsed(
                self.llm_request_started_at, self.llm_first_sentence_at
            ),
            "llm_request_to_complete_ms": _elapsed(
                self.llm_request_started_at, self.llm_completed_at
            ),
            "first_sentence_to_tts_start_ms": _elapsed(
                self.llm_first_sentence_at, self.tts_started_at
            ),
            "tts_start_to_first_audio_ms": _elapsed(
                self.tts_started_at, self.tts_first_audio_at
            ),
            "tts_start_to_complete_ms": _elapsed(
                self.tts_started_at, self.tts_completed_at
            ),
            "tts_first_sentence_synth_ms": _elapsed(
                self.tts_first_sentence_started_at,
                self.tts_first_sentence_completed_at,
            ),
            "audio_encode_send_ms": _elapsed(
                self.tts_completed_at, self.audio_sent_at
            ),
            # First audible audio: the first streaming chunk of the first
            # sentence when the provider streams; otherwise the first server
            # audio segment actually written to the socket; otherwise the
            # full buffered blob (pre-6H behaviour).
            "utterance_to_first_audio_ms": _elapsed(
                self.utterance_end_at,
                self.tts_first_audio_at
                or self.first_audio_sent_at
                or self.audio_sent_at,
            ),
            "utterance_end_to_first_audio_sent_ms": _elapsed(
                self.utterance_end_at, self.first_audio_sent_at
            ),
            "llm_sentence_count": self.llm_sentence_count or None,
            "tts_first_audio_reason": (
                None
                if self.tts_first_audio_at is not None or self.tts_started_at is None
                else "unavailable_complete_mp3"
            ),
            "streamed_tokens": self.streamed_tokens or None,
            # Phase 6H sentence-streaming counters
            "tts_sentence_count": self.tts_sentence_count or None,
            "tts_sentence_failures": self.tts_sentence_failures or None,
            "tts_audio_chunks": self.tts_audio_chunks or None,
            "tts_mode": self.tts_mode,
            # Raw offsets from session start (for debugging)
            "utterance_end_offset_ms": _offset(self.utterance_end_at),
            "utterance_released_offset_ms": _offset(self.utterance_released_at),
            "agent_start_offset_ms": _offset(self.agent_processing_started_at),
            "llm_request_offset_ms": _offset(self.llm_request_started_at),
            "llm_first_token_offset_ms": _offset(self.llm_first_token_at),
            "llm_first_sentence_offset_ms": _offset(self.llm_first_sentence_at),
            "llm_completed_offset_ms": _offset(self.llm_completed_at),
            "tts_started_offset_ms": _offset(self.tts_started_at),
            "tts_completed_offset_ms": _offset(self.tts_completed_at),
            "tts_first_sentence_offset_ms": _offset(
                self.tts_first_sentence_started_at
            ),
            "first_audio_sent_offset_ms": _offset(self.first_audio_sent_at),
            "audio_sent_offset_ms": _offset(self.audio_sent_at),
        }


def _fmt_ms(value: Any) -> str:
    """Format a millisecond duration for log output ("n/a" when missing)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1f}"


def _as_ms(value: Any) -> float | None:
    """Coerce a browser-reported millisecond value; reject bools/negatives."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


@dataclass
class _SentenceSink:
    """Bridge from the LLM stream to the per-turn TTS consumer (Phase 6H).

    ``_stream_llm_response`` pushes every complete sentence here the moment
    SentenceBuffer yields it, so the TTS consumer can start synthesizing the
    first sentence while the LLM is still streaming the rest of the response.
    The sentinel ``None`` marks end-of-response.
    """

    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    count: int = 0

    def put(self, sentence: str) -> None:
        """Push one complete sentence for immediate TTS."""
        self.count += 1
        self.queue.put_nowait(sentence)


def _log_turn_latency_breakdown(
    turn: int, m: dict[str, Any], browser: dict[str, Any] | None
) -> None:
    """Emit the concise Phase 6G per-turn latency breakdown.

    Every value is a measured mark from the turn's metrics payload; the
    optional browser report contributes the WebSocket transit and the
    receive→playback segment so ``total`` covers the full journey from the
    user's final speech to the first audible audio. When no browser report
    arrived the browser fields are logged as n/a and ``total`` falls back to
    the backend-side utterance→audio-sent duration.
    """
    utterance_to_playing: float | None = None
    # Phase 6H: prefer the first segment's sent mark (first audible audio);
    # fall back to the whole-turn delivery mark for pre-6H sessions.
    sent = m.get("utterance_end_to_first_audio_sent_ms")
    if sent is None:
        sent = m.get("utterance_end_to_audio_sent_ms")
    if browser is not None:
        playing = browser.get("received_to_playing_ms")
        if playing is not None and sent is not None:
            utterance_to_playing = round(
                sent + (browser.get("ws_transit_ms") or 0.0) + playing, 1
            )
    total = utterance_to_playing if utterance_to_playing is not None else sent
    logger.info(
        "[REALTIME:LATENCY] turn_latency_breakdown turn=%d "
        "utterance_to_release=%s release_to_agent=%s agent_to_llm_request=%s "
        "llm_first_token=%s llm_first_sentence=%s llm_complete=%s "
        "tts_start=%s tts_first_audio=%s tts_complete=%s audio_sent=%s "
        "browser_received=%s browser_playing=%s total=%s "
        "first_audio=%s sentences=%s tts_mode=%s",
        turn,
        _fmt_ms(m.get("utterance_end_to_release_ms")),
        _fmt_ms(m.get("release_to_agent_ms")),
        _fmt_ms(m.get("agent_to_llm_request_ms")),
        _fmt_ms(m.get("llm_request_to_first_token_ms")),
        _fmt_ms(m.get("llm_request_to_first_sentence_ms")),
        _fmt_ms(m.get("llm_request_to_complete_ms")),
        _fmt_ms(m.get("first_sentence_to_tts_start_ms")),
        _fmt_ms(m.get("tts_start_to_first_audio_ms")),
        _fmt_ms(m.get("tts_start_to_complete_ms")),
        _fmt_ms(m.get("audio_encode_send_ms")),
        _fmt_ms(browser.get("ws_transit_ms")) if browser else "n/a",
        _fmt_ms(browser.get("received_to_playing_ms")) if browser else "n/a",
        _fmt_ms(total),
        _fmt_ms(m.get("utterance_to_first_audio_ms")),
        m.get("tts_sentence_count") or "n/a",
        m.get("tts_mode") or "n/a",
    )


@dataclass
class RealtimeTimings:
    """Basic in-memory timing marks for one realtime session."""

    ws_accepted_at: float = field(default_factory=time.monotonic)
    session_started_at: float | None = None
    first_audio_at: float | None = None
    first_partial_at: float | None = None
    first_final_at: float | None = None
    first_utterance_end_at: float | None = None
    completed_at: float | None = None
    audio_bytes: int = 0
    audio_chunks: int = 0
    partial_count: int = 0
    final_count: int = 0
    utterance_end_count: int = 0
    # Turns actually released for agent processing (one per finalized turn)
    turns_released: int = 0
    agent_response_count: int = 0
    tts_response_count: int = 0
    tts_audio_bytes: int = 0
    tts_characters: int = 0
    # Per-utterance timing records (one per completed turn)
    turn_timings: list[UtteranceTimings] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize as millisecond offsets from the WebSocket accept time."""
        base = self.ws_accepted_at

        def _ms(mark: float | None) -> float | None:
            return round((mark - base) * 1000, 1) if mark is not None else None

        return {
            "session_start_ms": _ms(self.session_started_at),
            "first_audio_ms": _ms(self.first_audio_at),
            "first_partial_ms": _ms(self.first_partial_at),
            "first_final_ms": _ms(self.first_final_at),
            "first_utterance_end_ms": _ms(self.first_utterance_end_at),
            "session_duration_ms": _ms(self.completed_at),
            "audio_bytes": self.audio_bytes,
            "audio_chunks": self.audio_chunks,
            "partial_count": self.partial_count,
            "final_count": self.final_count,
            "utterance_end_count": self.utterance_end_count,
            "turns_released": self.turns_released,
            "agent_response_count": self.agent_response_count,
            "tts_response_count": self.tts_response_count,
            "tts_audio_bytes": self.tts_audio_bytes,
            "tts_characters": self.tts_characters,
        }


@dataclass
class RealtimeSessionState:
    """Lifecycle ownership for one realtime voice session.

    Every background task and resource created for a connection is registered
    here so that ``_shutdown_session`` — the single cleanup path — can
    terminate all of them before the WebSocket handler returns.

    Lifecycle: RUNNING → STOPPING → CLOSED. After STOPPING no new work may
    start and no stale event may be sent; after CLOSED nothing session-owned
    may run at all.
    """

    connection_id: str = ""
    # Lifecycle flags (RUNNING → STOPPING → CLOSED)
    stop_requested: bool = False  # STOP seen: no new audio/agent work
    stopping: bool = False  # shutdown started: browser sends are gated
    closed: bool = False  # cleanup finished: nothing session-owned may run
    shutdown_reason: str | None = None
    # Serializes concurrent cleanup callers (idempotent shutdown)
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Session-owned tasks — created on START, reaped by _stop_session_tasks
    pump_task: asyncio.Task | None = None
    worker_task: asyncio.Task | None = None
    # Session-owned provider session (StreamingSTTSession)
    stt_session: Any = None
    voice_session_id: str | None = None
    llm_provider: str = ""
    llm_model: str | None = None
    llm: LLMInterface | None = None
    tts: TTSInterface | None = None
    # Current utterance accumulation (pump writes, worker reads snapshot)
    current_utterance_text: str = ""
    utterance_finalized: bool = False  # True after final transcript received
    agent_processing: bool = False  # True while AgentRuntime is running
    # Turn release state (Phase 6E) — UtteranceEnd schedules a settle release;
    # newer transcript evidence cancels it. The release task is session-owned.
    speech_final_at: float | None = None  # last final with speech_final=True
    pending_release: bool = False  # a settle release is scheduled
    release_task: asyncio.Task | None = None  # session-owned release task
    pending_release_turn: int = 0  # turn number the pending release will become
    utterance_generation: int = 0  # bumped on schedule/cancel; guards stragglers
    pending_utterance_end_at: float | None = None  # UE time of pending release
    # Queue-based serialized utterance processing
    # Items are (transcript, utterance_end_ts, released_ts) or None (sentinel)
    utterance_queue: asyncio.Queue[tuple[str, float, float] | None] = field(
        default_factory=lambda: asyncio.Queue()
    )
    # Phase 6G: completed-turn metrics awaiting the browser's playback report
    # (browser_timing message); flushed to the breakdown log on shutdown when
    # no report arrived.
    completed_turn_metrics: dict[int, dict] = field(default_factory=dict)
    # Phase 6H: per-turn sentence-streaming TTS consumer (session-owned task)
    tts_task: asyncio.Task | None = None
    # Turn currently owned by the agent worker (0 = none). Audio segments are
    # delivered mid-turn, so a browser report for this turn may arrive before
    # the metrics payload exists.
    active_turn: int = 0
    # Early browser_timing reports for the active turn, reconciled at metrics
    # registration (see _register_turn_metrics).
    turn_browser_reports: dict[int, dict] = field(default_factory=dict)
    # Database session (one per WebSocket connection)
    db: Session | None = None


def _register_turn_metrics(
    state: RealtimeSessionState, turn: int, metrics: dict[str, Any]
) -> None:
    """Store a completed turn's metrics and reconcile an early browser report.

    Phase 6H delivers audio segments mid-turn, so a ``browser_timing`` report
    can arrive while the agent worker is still finalizing (the metrics payload
    does not exist yet). The handler then stashes the report in
    ``state.turn_browser_reports``; here, at registration, it is popped and the
    ``[REALTIME:LATENCY]`` breakdown is logged immediately so the report is
    never lost to the race.
    """
    state.completed_turn_metrics[turn] = metrics
    report = state.turn_browser_reports.pop(turn, None)
    if report is not None:
        state.completed_turn_metrics.pop(turn, None)
        _log_turn_latency_breakdown(turn, metrics, report)


def _parse_start_message(msg: dict) -> tuple[StreamConfig, str, str | None]:
    """Parse and validate a START message.

    Returns:
        Tuple of (StreamConfig, llm_provider, llm_model).

    Raises:
        ValueError: When invalid.
    """
    sample_rate = msg.get("sample_rate", 16000)
    channels = msg.get("channels", 1)
    encoding = msg.get("encoding", "linear16")
    language = msg.get("language", "en")
    model = msg.get("model") or None
    # Optional turn-gap override (Phase 6E); the browser UI never sends it —
    # the StreamConfig default (2000 ms) applies unless explicitly provided.
    utterance_end_ms = msg.get("utterance_end_ms")

    # LLM configuration (optional)
    llm_provider = msg.get("llm_provider", "") or ""
    llm_model = msg.get("llm_model", "") or None

    if not isinstance(sample_rate, int):
        raise ValueError("sample_rate must be an integer")
    if not isinstance(channels, int):
        raise ValueError("channels must be an integer")
    if not isinstance(language, str) or not language:
        raise ValueError("language must be a non-empty string")
    if utterance_end_ms is not None and (
        isinstance(utterance_end_ms, bool) or not isinstance(utterance_end_ms, int)
    ):
        raise ValueError("utterance_end_ms must be an integer")

    cfg_kwargs: dict[str, Any] = {
        "model": model,
        "language": language,
        "sample_rate": sample_rate,
        "channels": channels,
        "encoding": encoding,
    }
    if utterance_end_ms is not None:
        cfg_kwargs["utterance_end_ms"] = utterance_end_ms
    cfg = StreamConfig(**cfg_kwargs)
    error = cfg.validate()
    if error:
        raise ValueError(error)
    return cfg, llm_provider, llm_model


async def _safe_ws_send(
    ws: WebSocket,
    state: RealtimeSessionState,
    payload: dict,
    *,
    allow_stopping: bool = False,
) -> bool:
    """Send one JSON event to the browser, lifecycle-safely.

    Used by the handler, the provider pump and the agent worker:

    - A stopping/closed session emits nothing (``stale_task_ignored``),
      except the protocol's final ``completed`` event (allow_stopping).
    - A client that vanished mid-send surfaces as ``WebSocketDisconnect``
      (Starlette converts the uvicorn transport error) or ``RuntimeError``
      (send after close) — both are normal lifecycle events, not failures.

    Returns:
        True when the payload was written to a live socket.
    """
    event_type = payload.get("type", "?")
    if state.closed or (state.stopping and not allow_stopping):
        logger.info(
            "[VOICE:REALTIME] stale_task_ignored connection_id=%s event=%s",
            state.connection_id,
            event_type,
        )
        return False
    try:
        await ws.send_json(payload)
        return True
    except WebSocketDisconnect:
        logger.info(
            "[VOICE:REALTIME] client_disconnected_during_send "
            "connection_id=%s event=%s",
            state.connection_id,
            event_type,
        )
        return False
    except RuntimeError as e:
        logger.info(
            "[VOICE:REALTIME] ws_send_after_close connection_id=%s event=%s "
            "error=%s",
            state.connection_id,
            event_type,
            e,
        )
        return False


def _cancel_pending_release(
    state: RealtimeSessionState,
    reason: str,
    *,
    event_type: str = "",
) -> asyncio.Task | None:
    """Cancel a scheduled turn release and invalidate its generation.

    Called when newer transcript evidence arrives (new partial/final) and from
    shutdown. Returns the cancelled task so the caller can await it
    (``_stop_session_tasks``); pump-side callers may ignore it — generation
    checks make a straggler harmless.
    """
    task = state.release_task
    if task is None and not state.pending_release:
        return None
    state.pending_release = False
    state.release_task = None
    state.pending_utterance_end_at = None
    state.utterance_generation += 1  # invalidate any straggler
    if task is not None and not task.done():
        task.cancel()
        logger.info(
            "[REALTIME] utterance_release_cancelled turn=%d reason=%s "
            "event=%s accumulated='%s'",
            state.pending_release_turn,
            reason,
            event_type or "evidence",
            state.current_utterance_text[:120],
        )
        return task
    return None


def _schedule_turn_release(
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    *,
    last_word_end: float | None = None,
) -> None:
    """Schedule the session-owned turn release after the settle window.

    The task handle is kept on the session state so it can never be garbage
    collected and is always cancellable/awaitable by ``_stop_session_tasks``.
    """
    state.utterance_generation += 1
    generation = state.utterance_generation
    state.pending_release = True
    state.pending_release_turn = timings.agent_response_count + 1
    state.pending_utterance_end_at = time.monotonic()
    state.release_task = asyncio.create_task(
        _release_turn_after_settle(timings, state, generation),
        name="realtime_turn_release",
    )
    logger.info(
        "[REALTIME] utterance_release_scheduled turn=%d settle_ms=%d "
        "last_word_end=%s speech_final=%s accumulated='%s'",
        state.pending_release_turn,
        _TURN_RELEASE_SETTLE_MS,
        last_word_end,
        state.speech_final_at is not None,
        state.current_utterance_text[:120],
    )


async def _release_turn_after_settle(
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    generation: int,
) -> None:
    """Release the accumulated utterance after the settle window elapses.

    Runs as a session-owned task. It only enqueues when this generation is
    still current, the release is still pending and the session is RUNNING.
    Everything after the sleep is synchronous — a cancellation that already
    ran can never race the enqueue.
    """
    try:
        await asyncio.sleep(_TURN_RELEASE_SETTLE_MS / 1000.0)
    except asyncio.CancelledError:
        raise
    turn = timings.agent_response_count + 1
    if generation != state.utterance_generation or not state.pending_release:
        return  # cancelled or superseded
    if state.stopping or state.closed or state.stop_requested:
        state.pending_release = False
        state.release_task = None
        state.pending_utterance_end_at = None
        state.utterance_generation += 1
        logger.info(
            "[REALTIME] utterance_release_reason turn=%d reason=session_stopping",
            turn,
        )
        return
    transcript = state.current_utterance_text.strip()
    utterance_end_at = state.pending_utterance_end_at or time.monotonic()
    state.pending_release = False
    state.release_task = None
    state.pending_utterance_end_at = None
    state.current_utterance_text = ""
    state.utterance_finalized = False
    state.speech_final_at = None
    if not transcript:
        logger.info(
            "[REALTIME] utterance_release_reason turn=%d reason=empty_buffer",
            turn,
        )
        return
    logger.info(
        "[REALTIME] utterance_release_reason turn=%d "
        "reason=utterance_end_settle settle_ms=%d",
        turn,
        _TURN_RELEASE_SETTLE_MS,
    )
    timings.turns_released += 1
    logger.info(
        "[REALTIME] utterance_released turn=%d chars=%d transcript='%s'",
        turn,
        len(transcript),
        transcript[:120],
    )
    # Enqueue for the serialized agent worker (unbounded queue — never blocks).
    state.utterance_queue.put_nowait(
        (transcript, utterance_end_at, time.monotonic())
    )


async def _stop_session_tasks(state: RealtimeSessionState) -> None:
    """Cancel and reap every session-owned background task.

    Idempotent: task references are cleared from the state, so repeated calls
    are no-ops. Cancellation is requested first, then awaited — after this
    returns, no pump/worker/release task of this session can run any code.
    """
    # Pending turn release first: after this, no stale turn can be enqueued.
    release_task = _cancel_pending_release(state, "session_stopping")
    if release_task is not None:
        logger.info(
            "[VOICE:REALTIME] cancelling_utterance_release connection_id=%s",
            state.connection_id,
        )
    if state.worker_task is not None:
        logger.info(
            "[VOICE:REALTIME] cancelling_agent_worker connection_id=%s",
            state.connection_id,
        )
        try:
            state.utterance_queue.put_nowait(None)  # graceful sentinel
        except Exception:
            # Unbounded queue — put_nowait cannot realistically fail. The
            # cancel() below terminates the worker regardless.
            pass
        state.worker_task.cancel()
    if state.tts_task is not None:
        logger.info(
            "[VOICE:REALTIME] cancelling_tts_consumer connection_id=%s",
            state.connection_id,
        )
        state.tts_task.cancel()
    if state.pump_task is not None:
        logger.info(
            "[VOICE:REALTIME] cancelling_deepgram_pump connection_id=%s",
            state.connection_id,
        )
        state.pump_task.cancel()

    tasks = [
        t
        for t in (
            release_task,
            state.worker_task,
            state.pump_task,
            state.tts_task,
        )
        if t is not None
    ]
    state.worker_task = None
    state.pump_task = None
    state.tts_task = None
    if tasks:
        logger.info(
            "[VOICE:REALTIME] awaiting_tasks connection_id=%s count=%d",
            state.connection_id,
            len(tasks),
        )
        await asyncio.gather(*tasks, return_exceptions=True)


async def _shutdown_session(state: RealtimeSessionState, reason: str) -> None:
    """Single idempotent cleanup path for one realtime session.

    Ordering: mark STOPPING → cancel/reap pump+worker → close Deepgram →
    close LLM/TTS providers → close DB session → mark CLOSED. Safe to call
    repeatedly (STOP, disconnect, handler exhaustion) and from concurrent
    callers — the cleanup lock serializes them after which the closed flag
    short-circuits.
    """
    async with state.cleanup_lock:
        if state.closed:
            return
        if not state.stopping:
            state.stopping = True
            state.stop_requested = True
            state.shutdown_reason = reason
            logger.info(
                "[VOICE:REALTIME] session_stopping connection_id=%s reason=%s",
                state.connection_id,
                reason,
            )

        await _stop_session_tasks(state)

        # Phase 6G: flush latency breakdowns that never got a browser report
        # (backend-only clients) so every completed turn is logged exactly once.
        if state.completed_turn_metrics:
            for _turn in sorted(state.completed_turn_metrics):
                _log_turn_latency_breakdown(
                    _turn, state.completed_turn_metrics[_turn], None
                )
            state.completed_turn_metrics.clear()
        # Phase 6H: early (mid-turn) reports have no metrics to pair with at
        # shutdown — drop them explicitly.
        state.turn_browser_reports.clear()

        if state.stt_session is not None:
            logger.info(
                "[VOICE:REALTIME] closing_deepgram connection_id=%s",
                state.connection_id,
            )
            try:
                await state.stt_session.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:REALTIME] stt_close_failed connection_id=%s "
                    "error_type=%s",
                    state.connection_id,
                    type(e).__name__,
                )
            state.stt_session = None

        if state.llm is not None:
            try:
                await state.llm.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:REALTIME] llm_close_failed connection_id=%s "
                    "error_type=%s",
                    state.connection_id,
                    type(e).__name__,
                )
            state.llm = None

        if state.tts is not None:
            try:
                await state.tts.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:REALTIME] tts_close_failed connection_id=%s "
                    "error_type=%s",
                    state.connection_id,
                    type(e).__name__,
                )
            state.tts = None

        if state.db is not None:
            try:
                state.db.close()
            except Exception as e:
                logger.warning(
                    "[VOICE:REALTIME] db_close_failed connection_id=%s "
                    "error_type=%s",
                    state.connection_id,
                    type(e).__name__,
                )
            state.db = None

        state.closed = True
        logger.info(
            "[VOICE:REALTIME] session_closed connection_id=%s reason=%s",
            state.connection_id,
            state.shutdown_reason or reason,
        )


async def _pump_provider_events(
    ws: WebSocket,
    session,  # StreamingSTTSession
    timings: RealtimeTimings,
    state: RealtimeSessionState,
) -> None:
    """Forward provider transcript events to the client until the stream ends.

    Turn finalization (Phase 6E): final segments keep accumulating; a
    UtteranceEnd *schedules* a release after ``_TURN_RELEASE_SETTLE_MS``
    (instead of releasing immediately), and any newer transcript evidence
    cancels that pending release.

    ``state.stop_requested`` distinguishes a normal post-STOP drain from a
    premature provider disconnect: the latter is reported to the client as an
    error instead of silently completing without transcripts.
    """
    try:
        while True:
            event = await session.receive()
            if event is None:
                if not state.stop_requested:
                    logger.error(
                        "[VOICE:REALTIME] Provider stream ended before STOP "
                        "(chunks=%d bytes=%d)",
                        timings.audio_chunks,
                        timings.audio_bytes,
                    )
                    await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "error",
                            "stage": "stt",
                            "message": "Provider stream ended unexpectedly",
                        },
                    )
                break

            # A stopping session must not emit provider events even if
            # Deepgram flushes buffered messages just before the pump is
            # reaped — those belong to the previous session's lifecycle.
            if state.stopping:
                logger.info(
                    "[VOICE:REALTIME] stale_task_ignored connection_id=%s "
                    "event=provider_%s",
                    state.connection_id,
                    event.type,
                )
                break

            if event.type == "partial":
                timings.partial_count += 1
                if timings.first_partial_at is None:
                    timings.first_partial_at = time.monotonic()
                # New speech evidence — a scheduled release is no longer safe.
                if event.text:
                    _cancel_pending_release(
                        state, "new_partial_evidence", event_type="partial"
                    )
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "transcript_partial",
                        "text": event.text,
                        "confidence": round(event.confidence, 4),
                    },
                ):
                    break
            elif event.type == "final":
                timings.final_count += 1
                if timings.first_final_at is None:
                    timings.first_final_at = time.monotonic()
                # New transcript evidence — a scheduled release is stale.
                _cancel_pending_release(
                    state, "new_final_evidence", event_type="final"
                )
                # speech_final is evidence (endpointing silence was seen), not
                # a dispatch trigger — the turn is released only via
                # UtteranceEnd + settle below.
                speech_final = bool((event.metadata or {}).get("speech_final"))
                if speech_final:
                    state.speech_final_at = time.monotonic()
                # Accumulate final transcript for the current utterance; other
                # finalized segments of the same turn must never be replaced.
                prev = state.current_utterance_text
                state.current_utterance_text = (
                    f"{prev} {event.text}".strip() if prev else event.text
                )
                state.utterance_finalized = True
                logger.info(
                    "[REALTIME] utterance_segment_final turn=%d "
                    "speech_final=%s segment='%s' accumulated='%s'",
                    timings.agent_response_count + 1,
                    speech_final,
                    event.text[:60],
                    state.current_utterance_text[:120],
                )
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "transcript_final",
                        "text": event.text,
                        "confidence": round(event.confidence, 4),
                    },
                ):
                    break
            elif event.type == "utterance_end":
                timings.utterance_end_count += 1
                if timings.first_utterance_end_at is None:
                    timings.first_utterance_end_at = time.monotonic()
                last_word_end = (event.metadata or {}).get("last_word_end")
                if not await _safe_ws_send(ws, state, {"type": "utterance_end"}):
                    break
                # A stale UtteranceEnd (Deepgram: last_word_end=-1 means the
                # confirming segment is already finalized) must never release
                # the turn — the accumulated buffer survives for the next
                # piece of evidence.
                if last_word_end is not None and last_word_end < 0:
                    logger.info(
                        "[REALTIME] utterance_end_received turn=%d stale=True "
                        "last_word_end=%s accumulated='%s'",
                        timings.agent_response_count + 1,
                        last_word_end,
                        state.current_utterance_text[:120],
                    )
                    continue
                logger.info(
                    "[REALTIME] utterance_end_received turn=%d stale=False "
                    "last_word_end=%s speech_final=%s accumulated='%s'",
                    timings.agent_response_count + 1,
                    last_word_end,
                    state.speech_final_at is not None,
                    state.current_utterance_text[:120],
                )
                if state.stop_requested or state.stopping:
                    continue
                if state.pending_release:
                    # A settle release is already scheduled — keep waiting.
                    continue
                if not state.current_utterance_text:
                    # No finalized text to release — reset and wait for speech.
                    state.utterance_finalized = False
                    logger.info(
                        "[REALTIME] utterance_release_reason turn=%d "
                        "reason=empty_buffer",
                        timings.agent_response_count + 1,
                    )
                    continue
                # UtteranceEnd means the turn MAY be complete: schedule a
                # session-owned release after the settle window. Any new
                # partial/final evidence within the window cancels it and
                # accumulation continues — one turn, one agent call.
                _schedule_turn_release(
                    timings, state, last_word_end=last_word_end
                )
            elif event.type == "error":
                if not await _safe_ws_send(
                    ws,
                    state,
                    {"type": "error", "stage": "stt", "message": event.text},
                ):
                    break
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(
            "[VOICE:REALTIME] Provider event pump failed connection_id=%s "
            "error_type=%s",
            state.connection_id,
            type(e).__name__,
        )
        await _safe_ws_send(
            ws,
            state,
            {
                "type": "error",
                "stage": "stt",
                "message": "Realtime STT stream error",
            },
        )


_REALTIME_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."

_STREAM_AUDIO_CONTENT_TYPE = "audio/mpeg"


async def _open_tts_stream(tts: TTSInterface, text: str) -> Any | None:
    """Open a provider streaming synthesis for one sentence, or None.

    Phase 6H strict capability check: only a genuine async generator counts
    as streamable. Mocked doubles (``MagicMock``) advertise arbitrary
    attributes but cannot be iterated — treating them as streamable would
    break every buffered path, so anything that is not
    ``inspect.isasyncgen()`` falls back to buffered ``synthesize()``.
    """
    stream_fn = getattr(tts, "stream_synthesize", None)
    if not callable(stream_fn):
        return None
    try:
        stream = stream_fn(text=text)
    except TypeError:
        return None
    if asyncio.iscoroutine(stream):
        # A coroutine, not an async generator (older provider signature) —
        # close it and use the buffered path instead.
        stream.close()
        return None
    if not inspect.isasyncgen(stream):
        return None
    return stream


async def _close_tts_stream(stream: Any) -> None:
    """Close a provider streaming synthesis generator, swallowing errors."""
    aclose = getattr(stream, "aclose", None)
    if not callable(aclose):
        return
    try:
        await aclose()
    except Exception as e:
        logger.warning(
            "[REALTIME:TTS] stream_close_failed error_type=%s", type(e).__name__
        )


async def _synthesize_sentence_segment(
    tts: TTSInterface,
    sentence: str,
    turn_timing: UtteranceTimings,
) -> tuple[bytes, str, int, str]:
    """Synthesize one sentence into one complete audio segment (Phase 6H).

    Streaming providers: the sentence's chunks are collected into one
    complete, valid audio file — a single chunked HTTP response, NOT a
    concatenation of independent files — then delivered as one segment.
    Providers without streaming support: buffered ``synthesize()`` fallback
    (the pre-6H path; first-audio latency unavailable).

    Returns:
        (audio_bytes, content_type, chunk_count, mode).
    """
    stream = await _open_tts_stream(tts, sentence)
    if stream is None:
        result = await tts.synthesize(text=sentence)
        return result.audio_data, result.content_type, 0, "buffered"
    try:
        chunks: list[bytes] = []
        async for chunk in stream:
            if not chunk:
                continue
            if turn_timing.tts_first_audio_at is None:
                # First provider audio byte of this turn — the genuine
                # "TTS started producing audio" mark (Phase 6H).
                turn_timing.tts_first_audio_at = time.monotonic()
            chunks.append(chunk)
        if not chunks:
            raise RealtimeVoiceError("TTS stream returned no audio")
        return b"".join(chunks), _STREAM_AUDIO_CONTENT_TYPE, len(chunks), "streaming"
    finally:
        await _close_tts_stream(stream)


async def _stream_llm_response(
    *,
    db: Session,
    voice_session_id: str,
    transcript: str,
    llm: LLMInterface,
    model: str | None,
    sentence_sink: _SentenceSink | None = None,
) -> dict:
    """Stream LLM response for conversational (no-tool) path.

    Builds messages (system prompt + history + user message), calls
    stream_chat(), accumulates tokens into a SentenceBuffer, and
    captures first-token timing.

    Phase 6H: every complete sentence is pushed to ``sentence_sink`` the
    moment SentenceBuffer yields it — the concurrent TTS consumer can start
    on sentence 1 while this stream is still producing sentences.

    Returns:
        Dict with response, tool_calls (empty), usage, iterations,
        first_token_ms, streamed_tokens and the absolute Phase 6G marks
        (llm_request_started_at, llm_first_token_at, llm_first_sentence_at,
        llm_completed_at, sentence_count).
    """
    from app.models.voice_session import VoiceSession
    from app.services.realtime_voice_service import _load_history, _save_message

    # Load conversation history
    history = _load_history(db, voice_session_id)

    # Build messages: system prompt + history + user message
    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=_REALTIME_DEFAULT_SYSTEM_PROMPT),
        *history,
        LLMMessage(role="user", content=transcript),
    ]

    logger.info(
        "[REALTIME:STREAM] llm_stream_start model=%s history_len=%d",
        model or "default",
        len(history),
    )

    first_token_at: float | None = None
    first_sentence_at: float | None = None
    accumulated_text = ""
    chunk_count = 0
    buffer = SentenceBuffer()
    sentences: list[str] = []

    stream_start = time.monotonic()
    try:
        async for chunk in llm.stream_chat(
            messages=messages,
            model=model,
            tools=None,  # No tools for streaming path
            temperature=0.7,
        ):
            if chunk.content:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                accumulated_text += chunk.content
                chunk_count += 1
                # Feed to sentence buffer
                new_sentences = buffer.add(chunk.content)
                if new_sentences:
                    if first_sentence_at is None:
                        first_sentence_at = time.monotonic()
                    sentences.extend(new_sentences)
                    if sentence_sink is not None:
                        for sentence in new_sentences:
                            sentence_sink.put(sentence)

            if chunk.finish_reason:
                break
    except Exception as e:
        logger.error(
            "[REALTIME:STREAM] llm_stream_failed error=%s",
            str(e),
        )
        raise RealtimeVoiceError(f"LLM streaming failed: {e}") from e

    # Flush any remaining text from buffer
    remaining = buffer.flush()
    if remaining:
        if first_sentence_at is None:
            first_sentence_at = time.monotonic()
        sentences.append(remaining)
        if sentence_sink is not None:
            sentence_sink.put(remaining)

    stream_completed_at = time.monotonic()
    stream_ms = (stream_completed_at - stream_start) * 1000
    first_token_ms = (
        (first_token_at - stream_start) * 1000 if first_token_at else None
    )
    first_sentence_ms = (
        (first_sentence_at - stream_start) * 1000 if first_sentence_at else None
    )

    logger.info(
        "[REALTIME:STREAM] llm_stream_complete model=%s duration_ms=%.0f "
        "first_token_ms=%s first_sentence_ms=%s chunks=%d sentences=%d",
        model or "default",
        stream_ms,
        f"{first_token_ms:.1f}" if first_token_ms else "n/a",
        f"{first_sentence_ms:.1f}" if first_sentence_ms else "n/a",
        chunk_count,
        len(sentences),
    )

    # Save messages to DB (user message + assistant response)
    voice_session = (
        db.query(VoiceSession).filter(VoiceSession.id == voice_session_id).first()
    )
    if voice_session:
        seq = voice_session.message_count
        _save_message(db, voice_session_id, "user", transcript, seq)
        _save_message(
            db,
            voice_session_id,
            "assistant",
            accumulated_text,
            seq + 1,
        )
        voice_session.message_count = seq + 2
        db.commit()

    return {
        "response": accumulated_text,
        "tool_calls": [],
        "usage": {},
        "iterations": 1,
        "streamed": True,
        "first_token_ms": first_token_ms,
        "streamed_tokens": chunk_count,
        "sentences": sentences,
        # Phase 6G: absolute monotonic marks for per-stage latency metrics.
        "llm_request_started_at": stream_start,
        "llm_first_token_at": first_token_at,
        "llm_first_sentence_at": first_sentence_at,
        "llm_completed_at": stream_completed_at,
        "sentence_count": len(sentences),
    }


async def _tts_sentence_consumer(
    ws: WebSocket,
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    turn: int,
    turn_timing: UtteranceTimings,
    sentence_sink: _SentenceSink,
) -> dict[str, Any]:
    """Consume complete LLM sentences and deliver ordered audio segments.

    Phase 6H: runs concurrently with ``_stream_llm_response`` — the first
    complete sentence is synthesized and sent while the LLM is still
    streaming the remaining sentences. Segments are strictly sequential:
    sentence N+1 synthesis starts only after segment N was handed to the
    socket, so audio order always matches sentence order. One failed
    sentence is logged and counted, never fatal to the turn or session.

    Runs as a session-owned task (``state.tts_task``); STOP/disconnect
    cancels it via ``_stop_session_tasks``.

    Returns:
        Summary dict: sentences, segments_sent, failures, chunks, and
        first_error (message of the first failed sentence, else None).
    """
    summary: dict[str, Any] = {
        "sentences": 0,
        "segments_sent": 0,
        "failures": 0,
        "chunks": 0,
        "first_error": None,
    }
    segment_no = 0
    while True:
        sentence = await sentence_sink.queue.get()
        if sentence is None:
            break
        summary["sentences"] += 1
        sentence_no = summary["sentences"]

        if state.stopping or state.closed or state.stop_requested:
            logger.info(
                "[REALTIME:TTS] sentence_skipped turn=%d sentence=%d "
                "reason=session_stopping",
                turn,
                sentence_no,
            )
            continue

        if sentence_no == 1:
            # The TTS stage opens with the first sentence; sent from here
            # (not the worker) so it precedes the first audio frame while
            # the worker is still awaiting the LLM stream.
            if not await _safe_ws_send(ws, state, {"type": "tts_processing"}):
                break
            turn_timing.tts_started_at = time.monotonic()
            turn_timing.tts_first_sentence_started_at = turn_timing.tts_started_at
            logger.info("[REALTIME] tts_processing turn=%d", turn)

        try:
            audio_data, content_type, chunks, mode = (
                await _synthesize_sentence_segment(state.tts, sentence, turn_timing)
            )
        except Exception as e:
            # One failed sentence must not kill the turn or the session:
            # log, count, and continue with the next sentence.
            summary["failures"] += 1
            if summary["first_error"] is None:
                summary["first_error"] = str(e)
            turn_timing.tts_sentence_failures += 1
            logger.error(
                "[REALTIME:TTS] sentence_synthesis_failed turn=%d sentence=%d "
                "error_type=%s error=%s",
                turn,
                sentence_no,
                type(e).__name__,
                str(e),
            )
            continue

        turn_timing.tts_completed_at = time.monotonic()
        if turn_timing.tts_mode is None:
            # Mode of the first successfully synthesized sentence.
            turn_timing.tts_mode = mode
        if sentence_no == 1:
            turn_timing.tts_first_sentence_completed_at = turn_timing.tts_completed_at
        turn_timing.tts_sentence_count += 1
        turn_timing.tts_audio_chunks += chunks

        segment_no += 1
        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        if not await _safe_ws_send(
            ws,
            state,
            {
                "type": "audio",
                "format": content_type,
                "data": audio_b64,
                "turn": turn,
                "segment": segment_no,
                "sent_epoch_ms": int(time.time() * 1000),
            },
        ):
            # Session is stopping/closed — later segments could never be
            # heard; stop consuming.
            break
        sent_at = time.monotonic()
        if turn_timing.first_audio_sent_at is None:
            turn_timing.first_audio_sent_at = sent_at
        turn_timing.audio_sent_at = sent_at
        summary["segments_sent"] += 1
        summary["chunks"] += chunks
        timings.tts_response_count += 1
        timings.tts_audio_bytes += len(audio_data)
        timings.tts_characters += len(sentence)
        logger.info(
            "[REALTIME] audio_segment_sent turn=%d segment=%d sentence=%d "
            "bytes=%d chunks=%d mode=%s",
            turn,
            segment_no,
            sentence_no,
            len(audio_data),
            chunks,
            mode,
        )
    logger.info(
        "[REALTIME:TTS] consumer_exit turn=%d sentences=%d segments=%d "
        "failures=%d chunks=%d",
        turn,
        summary["sentences"],
        summary["segments_sent"],
        summary["failures"],
        summary["chunks"],
    )
    return summary


async def _agent_worker(
    ws: WebSocket,
    timings: RealtimeTimings,
    state: RealtimeSessionState,
) -> None:
    """Serialized agent processing worker.

    Consumes finalized utterance transcripts from the utterance queue and
    processes them through AgentRuntime one at a time. This runs in a
    separate task from the provider event pump so that Deepgram event
    reception is never blocked by long-running LLM/tool calls.

    The queue guarantees:
    - Utterances are processed in order (FIFO).
    - No utterance is lost — even if a new one arrives during processing.
    - Exactly one AgentRuntime call per queued utterance.
    - The pump remains free to drain Deepgram events at all times.

    A None sentinel in the queue signals the worker to exit.
    """
    logger.info("[REALTIME:WORKER] Agent worker started")
    while True:
        try:
            item = await state.utterance_queue.get()
        except asyncio.CancelledError:
            logger.info("[REALTIME:WORKER] Worker cancelled while waiting")
            break

        # None sentinel — shutdown signal
        if item is None:
            state.utterance_queue.task_done()
            logger.info("[REALTIME:WORKER] Worker received shutdown sentinel")
            break

        # Unpack transcript + timestamps from the pump
        transcript, utterance_end_ts, released_at = item

        # Guard: do not start new processing if STOP was received
        if state.stop_requested:
            state.utterance_queue.task_done()
            logger.info(
                "[REALTIME:WORKER] Skipping item — stop_requested"
            )
            continue

        state.agent_processing = True
        utterance_start = time.monotonic()
        turn = timings.agent_response_count + 1

        # Create per-utterance timing record
        turn_timing = UtteranceTimings(
            turn=turn,
            utterance_end_at=utterance_end_ts,
            utterance_released_at=released_at,
        )

        # Phase 6H: browser playback reports for this turn may arrive while
        # the turn is still being finalized (audio segments are delivered
        # mid-turn) — tag the turn as active for the report handler.
        state.active_turn = turn
        # Per-turn TTS consumer state (created below when TTS is available;
        # declared here so the finally block can always reference them).
        sentence_sink: _SentenceSink | None = None
        tts_task: asyncio.Task | None = None

        logger.info(
            "[REALTIME] agent_start turn=%d transcript='%s'",
            turn,
            transcript[:80],
        )

        try:
            # Send agent_processing event; a dead/stopping session aborts
            # the turn before any LLM/TTS work is started.
            if not await _safe_ws_send(ws, state, {"type": "agent_processing"}):
                break
            turn_timing.agent_processing_started_at = time.monotonic()

            if state.db is None or state.voice_session_id is None:
                logger.error(
                    "[REALTIME] error turn=%d stage=agent "
                    "error=session_not_initialized",
                    turn,
                )
                await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": "Session not initialized",
                    },
                )
                continue

            # Get voice session
            from app.models.voice_session import VoiceSession

            voice_session = (
                state.db.query(VoiceSession)
                .filter(VoiceSession.id == state.voice_session_id)
                .first()
            )
            if voice_session is None:
                logger.error(
                    "[REALTIME] error turn=%d stage=agent "
                    "error=session_not_found",
                    turn,
                )
                await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": "Session not found",
                    },
                )
                continue

            # Process utterance via streaming LLM (no-tool conversational path)
            resolved_model = (
                voice_session.llm_model
                or settings.default_llm_model
                or "default"
            )
            logger.info(
                "[REALTIME] agent_processing turn=%d model=%s mode=streaming",
                turn,
                resolved_model,
            )
            if state.tts is not None:
                # Phase 6H: the consumer runs concurrently with the LLM
                # stream; the first complete sentence is synthesized while
                # the LLM keeps generating the rest.
                sentence_sink = _SentenceSink()
                tts_task = asyncio.create_task(
                    _tts_sentence_consumer(
                        ws, timings, state, turn, turn_timing, sentence_sink
                    ),
                    name=f"realtime_tts_consumer_turn_{turn}",
                )
                state.tts_task = tts_task
            result = await _stream_llm_response(
                db=state.db,
                voice_session_id=voice_session.id,
                transcript=transcript,
                llm=state.llm,
                model=resolved_model,
                sentence_sink=sentence_sink,
            )
            turn_timing.streamed_tokens = result.get("streamed_tokens", 0)
            turn_timing.llm_sentence_count = result.get(
                "sentence_count", len(result.get("sentences") or [])
            )
            # Phase 6G: the real streaming path reports absolute marks; mocked
            # or simplified results fall back to the pre-6G relative
            # reconstruction so every caller keeps working unchanged.
            turn_timing.llm_request_started_at = (
                result.get("llm_request_started_at")
                or turn_timing.agent_processing_started_at
            )
            turn_timing.llm_first_token_at = result.get("llm_first_token_at")
            first_token_ms = result.get("first_token_ms")
            if (
                turn_timing.llm_first_token_at is None
                and first_token_ms is not None
                and turn_timing.agent_processing_started_at is not None
            ):
                turn_timing.llm_first_token_at = (
                    turn_timing.agent_processing_started_at + first_token_ms / 1000
                )
            turn_timing.llm_first_sentence_at = result.get("llm_first_sentence_at")
            turn_timing.llm_completed_at = (
                result.get("llm_completed_at") or time.monotonic()
            )

            utterance_ms = (time.monotonic() - utterance_start) * 1000
            logger.info(
                "[REALTIME] agent_response turn=%d duration_ms=%.0f "
                "iterations=%d tool_calls=%d streamed=%s first_token_ms=%s",
                turn,
                utterance_ms,
                result["iterations"],
                len(result["tool_calls"]),
                result.get("streamed", False),
                f"{first_token_ms:.1f}" if first_token_ms else "n/a",
            )

            timings.agent_response_count += 1

            # Send agent_response event; if the session is gone, stop before
            # spending TTS on a response nobody can hear.
            if not await _safe_ws_send(
                ws,
                state,
                {
                    "type": "agent_response",
                    "text": result["response"],
                    "tool_calls": len(result["tool_calls"]),
                    "iterations": result["iterations"],
                },
            ):
                break

            # --- TTS synthesis: sentence-streaming (Phase 6H) ---
            response_text = result["response"]
            if sentence_sink is not None and tts_task is not None:
                # End-of-response marker + fallback parity: responses that
                # never streamed through SentenceBuffer (mocked or custom
                # results) are synthesized as one segment over the complete
                # text — the pre-6H behaviour.
                if sentence_sink.count == 0:
                    fallback = result.get("sentences") or []
                    if not fallback and response_text:
                        fallback = [response_text]
                    for sentence in fallback:
                        sentence_sink.put(sentence)
                sentence_sink.queue.put_nowait(None)

            if response_text and state.tts is not None and tts_task is not None:
                tts_summary: dict[str, Any] | None = None
                try:
                    tts_summary = await tts_task
                except asyncio.CancelledError:
                    # The consumer was cancelled by session shutdown (or
                    # this worker itself is being cancelled) — skip the
                    # metrics path; all sends are lifecycle-gated by now.
                    if not (state.stopping or state.stop_requested or state.closed):
                        raise
                    logger.info(
                        "[REALTIME:TTS] consumer_cancelled turn=%d "
                        "reason=session_stopping",
                        turn,
                    )
                tts_task = None
                if tts_summary is not None and (
                    tts_summary["segments_sent"] == 0 and tts_summary["failures"] > 0
                ):
                    # Every sentence failed — preserve the pre-6H error
                    # strategy for a totally failed turn.
                    logger.error(
                        "[REALTIME] error turn=%d stage=tts error=%s",
                        turn,
                        tts_summary["first_error"],
                    )
                    await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "error",
                            "stage": "tts",
                            "message": (
                                f"TTS synthesis failed: {tts_summary['first_error']}"
                            ),
                        },
                    )
                elif tts_summary is not None and tts_summary["segments_sent"] > 0:
                    # Store per-turn timings and send metrics event
                    timings.turn_timings.append(turn_timing)
                    metrics_payload = turn_timing.to_metrics_dict(
                        timings.ws_accepted_at
                    )
                    # Phase 6G: keep the payload until the browser reports
                    # its playback timing (or shutdown flushes it to the
                    # log). Phase 6H: an early report may already be stashed
                    # mid-turn — reconcile it at registration.
                    _register_turn_metrics(state, turn, metrics_payload)
                    if not await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "turn_metrics",
                            "data": metrics_payload,
                        },
                    ):
                        break
                    logger.info(
                        "[REALTIME:BENCH] turn_metrics turn=%d "
                        "utterance_to_agent_ms=%.1f agent_ms=%.1f "
                        "tts_ms=%.1f total_ms=%.1f",
                        turn,
                        metrics_payload.get("utterance_end_to_agent_ms") or 0,
                        metrics_payload.get("agent_processing_ms") or 0,
                        metrics_payload.get("tts_duration_ms") or 0,
                        metrics_payload.get("utterance_end_to_audio_sent_ms") or 0,
                    )
                # else: nothing audible and nothing failed loudly (session
                # stopping mid-turn) — skip the metrics event silently.
            elif response_text and state.tts is None:
                # No TTS — still record and send metrics for LLM-only turns
                timings.turn_timings.append(turn_timing)
                metrics_payload = turn_timing.to_metrics_dict(
                    timings.ws_accepted_at
                )
                # Phase 6G: keep the payload until the browser reports its
                # playback timing (or shutdown flushes it to the log).
                _register_turn_metrics(state, turn, metrics_payload)
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "turn_metrics",
                        "data": metrics_payload,
                    },
                ):
                    break
                logger.warning(
                    "[REALTIME] TTS skipped — no TTS provider"
                )
            else:
                # Empty response — still record metrics
                timings.turn_timings.append(turn_timing)
                metrics_payload = turn_timing.to_metrics_dict(
                    timings.ws_accepted_at
                )
                # Phase 6G: keep the payload until the browser reports its
                # playback timing (or shutdown flushes it to the log).
                _register_turn_metrics(state, turn, metrics_payload)
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "turn_metrics",
                        "data": metrics_payload,
                    },
                ):
                    break

        except RealtimeVoiceError as e:
            elapsed_ms = (time.monotonic() - utterance_start) * 1000
            logger.error(
                "[REALTIME] error turn=%d stage=agent error=%s duration_ms=%.0f",
                turn,
                e.message,
                elapsed_ms,
            )
            await _safe_ws_send(
                ws,
                state,
                {
                    "type": "error",
                    "stage": "agent",
                    "message": e.message,
                },
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - utterance_start) * 1000
            logger.error(
                "[REALTIME] error turn=%d stage=agent error_type=%s duration_ms=%.0f",
                turn,
                type(e).__name__,
                elapsed_ms,
                exc_info=True,
            )
            await _safe_ws_send(
                ws,
                state,
                {
                    "type": "error",
                    "stage": "agent",
                    "message": "Agent processing failed",
                },
            )
        finally:
            # Phase 6H: never leave a per-turn TTS consumer running.
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            state.active_turn = 0
            state.agent_processing = False
            state.utterance_queue.task_done()

    logger.info("[REALTIME:WORKER] Agent worker exited")


async def _process_utterance(
    ws: WebSocket,
    timings: RealtimeTimings,
    state: RealtimeSessionState,
) -> None:
    """Process a finalized utterance through AgentRuntime (legacy wrapper).

    Kept for backwards compatibility with tests that call this directly.
    New code uses _agent_worker() instead.
    """
    if not state.utterance_finalized or not state.current_utterance_text:
        logger.info("[REALTIME:AGENT] skipping — no finalized utterance")
        return

    transcript = state.current_utterance_text
    state.current_utterance_text = ""
    state.utterance_finalized = False
    state.agent_processing = True

    logger.info(
        "[REALTIME:AGENT] processing utterance text='%s' length=%d",
        transcript[:80],
        len(transcript),
    )

    try:
        await ws.send_json({"type": "agent_processing"})

        if state.db is None or state.voice_session_id is None:
            await ws.send_json(
                {
                    "type": "error",
                    "stage": "agent",
                    "message": "Session not initialized",
                }
            )
            return

        from app.models.voice_session import VoiceSession

        voice_session = (
            state.db.query(VoiceSession)
            .filter(VoiceSession.id == state.voice_session_id)
            .first()
        )
        if voice_session is None:
            await ws.send_json(
                {
                    "type": "error",
                    "stage": "agent",
                    "message": "Session not found",
                }
            )
            return

        result = await process_realtime_utterance(
            db=state.db,
            session=voice_session,
            transcript=transcript,
            llm=state.llm,
        )

        timings.agent_response_count += 1
        await ws.send_json(
            {
                "type": "agent_response",
                "text": result["response"],
                "tool_calls": len(result["tool_calls"]),
                "iterations": result["iterations"],
            }
        )

    except RealtimeVoiceError as e:
        await ws.send_json(
            {
                "type": "error",
                "stage": "agent",
                "message": e.message,
            }
        )
    except Exception as e:
        logger.error(
            "[REALTIME:AGENT] utterance processing failed error_type=%s",
            type(e).__name__,
            exc_info=True,
        )
        await ws.send_json(
            {
                "type": "error",
                "stage": "agent",
                "message": "Agent processing failed",
            }
        )
    finally:
        state.agent_processing = False


async def _handle_realtime_session(ws: WebSocket) -> None:
    """Handle one realtime voice client connection end-to-end.

    The handler is the single lifecycle owner: it creates both session-owned
    tasks, and every exit path funnels into ``_shutdown_session`` so that no
    session-owned task or resource can survive the handler.
    """
    timings = RealtimeTimings()
    connection_id = str(uuid.uuid4())
    state = RealtimeSessionState(connection_id=connection_id)

    try:
        while True:
            raw = await ws.receive()

            # Starlette surfaces the client disconnect as a raw ASGI message
            # here (unlike the receive_json-style helpers, which raise
            # WebSocketDisconnect). Treat it as a normal lifecycle event.
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(
                    code=raw.get("code", 1000),
                    reason=raw.get("reason", "") or "",
                )

            # ----- Binary audio chunk -----
            if "bytes" in raw:
                # Audio before START or after STOPPING is ignored.
                if state.stt_session is None or state.stopping:
                    continue
                chunk = raw["bytes"]
                timings.audio_chunks += 1
                timings.audio_bytes += len(chunk)
                if timings.first_audio_at is None:
                    timings.first_audio_at = time.monotonic()
                    logger.debug(
                        "[VOICE:REALTIME] client_audio first chunk bytes=%d",
                        len(chunk),
                    )
                try:
                    await state.stt_session.send_audio(chunk)
                except StreamingSTTError as e:
                    await _safe_ws_send(
                        ws,
                        state,
                        {"type": "error", "stage": "stt", "message": str(e)},
                    )
                    break
                continue

            if "text" not in raw:
                continue

            # ----- JSON control message -----
            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError:
                await _safe_ws_send(
                    ws, state, {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start":
                if state.stt_session is not None:
                    await _safe_ws_send(
                        ws,
                        state,
                        {"type": "error", "message": "Session already started"},
                    )
                    continue

                try:
                    cfg, llm_provider, llm_model = _parse_start_message(msg)
                except ValueError as e:
                    await _safe_ws_send(
                        ws, state, {"type": "error", "message": str(e)}
                    )
                    continue

                try:
                    state.stt_session = open_streaming_session(cfg)
                    await state.stt_session.start()
                except (StreamingSTTError, ValueError) as e:
                    logger.error(
                        "[VOICE:REALTIME] Session start failed error_type=%s",
                        type(e).__name__,
                    )
                    await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "error",
                            "stage": "stt",
                            "message": f"Failed to start stream: {e}",
                        },
                    )
                    state.stt_session = None
                    continue

                # Create database session and voice session
                state.db = SessionLocal()
                try:
                    voice_session = create_realtime_session(
                        db=state.db,
                        llm_provider=llm_provider or None,
                        llm_model=llm_model,
                    )
                    state.voice_session_id = voice_session.id
                    state.llm_provider = voice_session.llm_provider or ""
                    state.llm_model = voice_session.llm_model

                    # Pre-create LLM provider for reuse
                    if state.llm_provider:
                        try:
                            state.llm = get_llm_provider(
                                state.llm_provider,
                                model=state.llm_model,
                            )
                        except Exception as e:
                            logger.warning(
                                "[VOICE:REALTIME] Failed to create LLM provider: %s",
                                e,
                            )

                    # Pre-create TTS provider for reuse (session-level)
                    try:
                        state.tts = get_tts_provider()
                        logger.debug(
                            "[VOICE:REALTIME] TTS provider created provider=%s",
                            state.tts.provider_name,
                        )
                    except Exception as e:
                        logger.warning(
                            "[VOICE:REALTIME] Failed to create TTS provider: %s",
                            e,
                        )
                except Exception as e:
                    logger.error(
                        "[VOICE:REALTIME] Failed to create voice session: %s",
                        e,
                    )
                    await state.stt_session.close()
                    state.stt_session = None
                    state.db.close()
                    state.db = None
                    await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "error",
                            "stage": "agent",
                            "message": "Failed to create session",
                        },
                    )
                    continue

                timings.session_started_at = time.monotonic()
                logger.info(
                    "[VOICE:REALTIME] Session started connection_id=%s "
                    "voice_session_id=%s provider=%s model=%s sample_rate=%d "
                    "llm_provider=%s llm_model=%s",
                    connection_id,
                    state.voice_session_id,
                    state.stt_session.provider_name,
                    cfg.model,
                    cfg.sample_rate,
                    state.llm_provider,
                    state.llm_model,
                )
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "session_started",
                        "session_id": state.voice_session_id or connection_id,
                        "provider": state.stt_session.provider_name,
                        "model": cfg.model,
                        "sample_rate": cfg.sample_rate,
                        "encoding": cfg.encoding,
                    },
                ):
                    # Client vanished while the provider was connecting —
                    # stop the loop; the finally block tears this session
                    # down before any pump/worker is started.
                    break
                state.pump_task = asyncio.create_task(
                    _pump_provider_events(ws, state.stt_session, timings, state)
                )
                # Start the serialized agent processing worker.
                # This runs independently from the pump so that Deepgram
                # event reception is never blocked by AgentRuntime.
                state.worker_task = asyncio.create_task(
                    _agent_worker(ws, timings, state)
                )
                logger.info(
                    "[VOICE:REALTIME] session_start connection_id=%s "
                    "voice_session_id=%s",
                    connection_id,
                    state.voice_session_id,
                )

            elif msg_type == "stop":
                if state.stt_session is None:
                    await _safe_ws_send(
                        ws, state, {"type": "error", "message": "No active session"}
                    )
                    continue

                # STOPPING immediately: no further audio is forwarded and no
                # new agent/LLM/TTS work may be started from here on.
                if not state.stopping:
                    state.stopping = True
                    state.stop_requested = True
                    state.shutdown_reason = "stop"
                    logger.info(
                        "[VOICE:REALTIME] session_stopping connection_id=%s "
                        "reason=stop chunks=%d bytes=%d",
                        connection_id,
                        timings.audio_chunks,
                        timings.audio_bytes,
                    )

                # Signal end-of-audio so the provider can flush final results
                try:
                    await state.stt_session.finish()
                except StreamingSTTError as e:
                    logger.warning(
                        "[VOICE:REALTIME] finish() failed error_type=%s",
                        type(e).__name__,
                    )

                # --- Immediate cancellation (no graceful drain) ---
                # STOP is cancellation, not graceful completion: reap the
                # pump and worker before reporting completion to the browser.
                await _stop_session_tasks(state)

                timings.completed_at = time.monotonic()
                logger.info(
                    "[VOICE:REALTIME] Session completed connection_id=%s",
                    connection_id,
                )
                await _safe_ws_send(
                    ws,
                    state,
                    {"type": "completed", "timings": timings.to_dict()},
                    allow_stopping=True,
                )
                break

            elif msg_type == "browser_timing":
                # Phase 6G/6H: browser-side playback report for a turn. It can
                # arrive after the turn's metrics payload was registered
                # (normal path) or mid-turn while audio segments are still
                # being delivered (Phase 6H, segment queue) — an early report
                # is stashed and reconciled when metrics are registered.
                # Unknown/stale turns are ignored safely.
                turn_no = msg.get("turn")
                valid_turn = isinstance(turn_no, int) and not isinstance(
                    turn_no, bool
                )
                report = {
                    "ws_transit_ms": _as_ms(msg.get("ws_transit_ms")),
                    "received_to_playing_ms": _as_ms(
                        msg.get("received_to_playing_ms")
                    ),
                }
                if valid_turn and turn_no in state.completed_turn_metrics:
                    _log_turn_latency_breakdown(
                        turn_no,
                        state.completed_turn_metrics.pop(turn_no),
                        report,
                    )
                    continue
                if (
                    valid_turn
                    and turn_no == state.active_turn
                    and turn_no not in state.turn_browser_reports
                ):
                    # Mid-turn report: stash until the worker registers this
                    # turn's metrics (see _register_turn_metrics).
                    state.turn_browser_reports[turn_no] = report
                    continue
                logger.info(
                    "[VOICE:REALTIME] browser_timing_ignored turn=%s "
                    "connection_id=%s",
                    turn_no,
                    connection_id,
                )

            else:
                await _safe_ws_send(
                    ws,
                    state,
                    {"type": "error", "message": f"Unknown message type: {msg_type}"},
                )

    except WebSocketDisconnect as e:
        state.shutdown_reason = state.shutdown_reason or "client_disconnect"
        logger.info(
            "[VOICE:REALTIME] client_disconnected connection_id=%s code=%s",
            connection_id,
            e.code,
        )
    except Exception as e:
        state.shutdown_reason = state.shutdown_reason or "error"
        logger.error(
            "[VOICE:REALTIME] Gateway error connection_id=%s error_type=%s",
            connection_id,
            type(e).__name__,
        )
        await _safe_ws_send(
            ws, state, {"type": "error", "message": "Internal realtime error"}
        )
    finally:
        # Single idempotent cleanup path: reaps session-owned tasks, closes
        # Deepgram, providers and the DB session, then marks the session
        # CLOSED. Also runs after STOP (no-op for the parts already done).
        try:
            await _shutdown_session(
                state, reason=state.shutdown_reason or "handler_exit"
            )
        except asyncio.CancelledError:
            # The handler itself was cancelled (e.g. server shutdown):
            # request child cancellation synchronously so no session-owned
            # task is left running, then let the cancellation propagate.
            state.stopping = True
            state.stop_requested = True
            if state.worker_task is not None:
                state.worker_task.cancel()
            if state.pump_task is not None:
                state.pump_task.cancel()
            if state.release_task is not None:
                state.release_task.cancel()
            if state.tts_task is not None:
                state.tts_task.cancel()
            raise


@router.websocket("/api/v1/voice/realtime/ws")
async def realtime_voice_websocket(ws: WebSocket) -> None:
    """Realtime streaming voice over WebSocket (Phase 6B).

    Protocol is documented in this module's docstring. Binary frames are
    raw PCM (linear16, mono, 16 kHz by default) forwarded to the streaming
    STT provider; transcript events are streamed back as JSON. After an
    utterance_end and a short settle window, the accumulated transcript of
    the turn is released to AgentRuntime.
    """
    await ws.accept()
    logger.info("[VOICE:REALTIME] WebSocket connection accepted")
    await _handle_realtime_session(ws)
