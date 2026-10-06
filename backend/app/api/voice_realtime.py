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
    AgentRuntime (on released turn — speech_final early release with a short
    debounce; UtteranceEnd + settle window remains the fallback trigger)
        ↓
    agent_response event → browser

Client → Server protocol:
    {"type": "start", "sample_rate": 16000, "channels": 1,
     "encoding": "linear16", "language": "en", "model": "nova-3",
     "llm_provider": "openrouter", "llm_model": "openai/gpt-4o-mini",
     "utterance_end_ms": 1000, "tts_mode": "elevenlabs"|"browser",
     "audio_stream": true|false}
         (Phase 2: true advertises browser-side MediaSource MP3 support —
          each sentence is then delivered as audio_chunk/audio_end while it
          is still being synthesized. Omitted/false keeps the complete-MP3
          protocol.)
    <binary PCM audio chunks>
    {"type": "browser_timing", "turn": 1, "ws_transit_ms": 3.0,
     "received_to_playing_ms": 15.0}  (Phase 6G playback report)
    {"type": "tts_mode", "mode": "elevenlabs"|"browser"}  (Phase 6L switch;
     applied to the NEXT turn — elevenlabs keeps server audio, browser
     forwards tts_text and the browser speaks it via speechSynthesis)
    {"type": "stop"}

Server → Client protocol:
    {"type": "session_started", "session_id": "...", "provider": "...",
     "model": "...", "sample_rate": 16000}
    {"type": "transcript_partial", "text": "...", "confidence": 0.9}
    {"type": "transcript_final", "text": "...", "confidence": 0.95,
     "speech_final": true}
    {"type": "utterance_end"}
    {"type": "agent_processing"}
    {"type": "agent_delta", "text": "..."}
         (progressive assistant text — one raw LLM delta per event)
    {"type": "agent_response", "text": "...", "tool_calls": 0, "iterations": 1}
    {"type": "tool_progress", "tool_name": "get_employee",
     "tool_call_id": "call_abc123",
     "message": "Sure, let me check the employee details for you."}
     (deterministic ack sent right before each tool executes)
    {"type": "tool_result", "tool_name": "get_employee",
     "tool_call_id": "call_abc123", "duration_ms": 617, "success": true,
     "error": null}
     (one per executed tool call — fan-out calls each report; not deduplicated)
    {"type": "tts_processing"}
    {"type": "audio", "format": "audio/mpeg", "data": "...", "turn": 1,
     "segment": 1, "sent_epoch_ms": 1727000000000}  (elevenlabs mode —
     one complete segment per sentence; used unless the client sent
     "audio_stream": true)
    {"type": "audio_chunk", "format": "audio/mpeg", "data": "...", "turn": 1,
     "segment": 1, "seq": 0, "sent_epoch_ms": 1727000000000}
     (Phase 2 incremental delivery: one ElevenLabs request per sentence,
      provider chunks forwarded as they arrive — strictly ordered)
    {"type": "audio_end", "turn": 1, "segment": 1, "seq": 3}
     (Phase 2: closes an audio_chunk sequence so the browser can finalize
      playback)
    {"type": "tts_text", "turn": 1, "segment": 1, "text": "..."}
     (Phase 6L browser mode — sentence text for speechSynthesis; no audio)
    {"type": "turn_metrics", "data": {...}}
    {"type": "completed", "timings": {...}}
    {"type": "error", "stage": "stt"|"agent", "message": "..."}

Phase 6H: the TTS branch synthesizes one ordered audio segment per completed
LLM sentence — the first sentence is synthesized and delivered while the LLM
is still streaming the remainder of the response. Phase 6K: sentence TTS
pre-generates with bounded concurrency while earlier segments are sent, and
segments are still delivered strictly in sentence order. Phase 6L: in browser
TTS mode the same sentence stream is forwarded as tts_text events instead —
no ElevenLabs request, no MP3, no base64 audio. Phase 2 (audio_stream
clients): the complete-MP3 wait is removed from the ElevenLabs path — each
provider chunk is forwarded to the browser the moment it arrives
(audio_chunk / audio_end), still one provider request per sentence, one
strict segment order, and unchanged cancellation/barge-in semantics; every
other client keeps the complete-MP3 messages.
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
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor

router = APIRouter()

# Turn-release settle windows.
#
# Primary early release (turn-finalization v2): a final transcript with
# speech_final=True is Deepgram's endpointing proof that the speaker stopped,
# so the turn is released after this short debounce. Any new speech evidence
# (partial or final) inside the window cancels the release and the utterance
# keeps accumulating — keeping transcript_final -> agent_processing ~1 s.
_SPEECH_FINAL_RELEASE_SETTLE_MS = 600
#
# Fallback release (Phase 6E): Deepgram's UtteranceEnd only states that a
# >= utterance_end_ms word gap was detected (e.g. a thinking pause in
# "…employee ID. [pause] E zero zero one."). It is used when no speech_final
# final released the turn; it *schedules* a release after this settle window
# instead of releasing immediately — any new transcript evidence (partial or
# final) within the window cancels the pending release.
_TURN_RELEASE_SETTLE_MS = 800

# Phase 6K: maximum simultaneous per-sentence TTS syntheses. Sentence N+1
# pre-generates while segment N is being sent (and played back by the
# browser); completed audio waits in an ordered buffer and segments are
# still delivered strictly by sentence index. Bounded so STOP/disconnect
# cancellation and provider rate limits stay manageable.
_TTS_SENTENCE_CONCURRENCY = 2

# Phase 1 (latency): early TTS chunking thresholds. The realtime gateway
# configures SentenceBuffer with these so the FIRST TTS request can start
# while the LLM is still generating — without waiting for a complete
# sentence and without waiting for the stream to finish.
#
# ``_MIN_FIRST_CHUNK_CHARS``: chunks emitted off trailing punctuation or a
# stall flush are never shorter than this (tiny fragments like "e.g." or
# "Bye." keep accumulating; hard ". " boundaries still emit immediately).
# ``_MAX_TTS_CHUNK_CHARS``: boundary-less runs are soft-split at the last
# word boundary that fits — words are never split.
# ``_CHUNK_FLUSH_TIMEOUT_MS``: when no new token arrives for this long the
# chunk watchdog flushes buffered text at a safe word boundary. The
# watchdog only ever inspects the SentenceBuffer — it never touches, times
# out or cancels the LLM stream itself.
_MIN_FIRST_CHUNK_CHARS = 15
_MAX_TTS_CHUNK_CHARS = 160
_CHUNK_FLUSH_TIMEOUT_MS = 220


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

    Turn-finalization v2 adds the early-release marks: ``speech_final_at``
    (Deepgram endpointing proof of speech stop) and ``last_final_at`` (most
    recent final transcript of the turn), plus ``release_reason`` naming the
    trigger path.
    """

    turn: int = 0
    utterance_end_at: float | None = None
    utterance_released_at: float | None = None
    agent_processing_started_at: float | None = None
    # Turn-finalization v2: early speech_final-based release marks
    speech_final_at: float | None = None
    last_final_at: float | None = None
    release_reason: str | None = None
    llm_request_started_at: float | None = None
    llm_first_token_at: float | None = None
    llm_first_sentence_at: float | None = None
    # Phase 1: mark of the first TTS-ready chunk (a complete sentence, a
    # trailing-punctuation chunk or a watchdog/soft-split chunk — the first
    # text the TTS consumer could actually start on).
    llm_first_chunk_at: float | None = None
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
            # Turn-finalization v2: early speech_final-based release marks
            "speech_final_to_agent_ms": _elapsed(
                self.speech_final_at, self.agent_processing_started_at
            ),
            "transcript_final_to_agent_ms": _elapsed(
                self.last_final_at, self.agent_processing_started_at
            ),
            "release_reason": self.release_reason,
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
            # Phase 1 first-chunk metrics: the first TTS request can start on
            # a partial-sentence chunk now, so these anchors measure the wait
            # the old pipeline paid in full before any TTS request started.
            # Fall back to the first-sentence mark for older data.
            "llm_first_chunk_ms": _elapsed(
                self.agent_processing_started_at,
                self.llm_first_chunk_at or self.llm_first_sentence_at,
            ),
            "first_chunk_to_tts_request_ms": _elapsed(
                self.llm_first_chunk_at or self.llm_first_sentence_at,
                self.tts_started_at,
            ),
            "first_chunk_to_first_audio_ms": _elapsed(
                self.llm_first_chunk_at or self.llm_first_sentence_at,
                self.tts_first_audio_at,
            ),
            "first_chunk_to_first_audio_sent_ms": _elapsed(
                self.llm_first_chunk_at or self.llm_first_sentence_at,
                self.first_audio_sent_at or self.audio_sent_at,
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
            "llm_first_chunk_offset_ms": _offset(
                self.llm_first_chunk_at or self.llm_first_sentence_at
            ),
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


def _ms_since(mark: float | None, anchor: float) -> str:
    """Format a mark -> anchor gap in whole milliseconds ("n/a" when missing)."""
    if mark is None:
        return "n/a"
    return f"{(anchor - mark) * 1000:.0f}"


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
        "first_audio=%s sentences=%s tts_mode=%s "
        "first_chunk=%s first_chunk_to_tts=%s first_chunk_to_audio_sent=%s",
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
        _fmt_ms(m.get("llm_first_chunk_ms")),
        _fmt_ms(m.get("first_chunk_to_tts_request_ms")),
        _fmt_ms(m.get("first_chunk_to_first_audio_sent_ms")),
    )


@dataclass
class ReleasedTurn:
    """One finalized utterance handed from the release path to the worker.

    Carries the turn-finalization v2 marks alongside the transcript so the
    worker can report speech_final / transcript_final -> agent_processing
    without racing the pump's state resets.
    """

    transcript: str
    utterance_end_at: float
    released_at: float
    speech_final_at: float | None = None
    last_final_at: float | None = None
    release_reason: str = "utterance_end"


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
    # Turn release state (Phase 6E + v2) — a final with speech_final=True (or,
    # as fallback, UtteranceEnd) schedules a short-debounce settle release;
    # newer transcript evidence cancels it. The release task is session-owned.
    speech_final_at: float | None = None  # last final with speech_final=True
    last_final_at: float | None = None  # most recent final transcript mark
    pending_release: bool = False  # a settle release is scheduled
    release_task: asyncio.Task | None = None  # session-owned release task
    pending_release_turn: int = 0  # turn number the pending release will become
    pending_release_reason: str = ""  # "speech_final" | "utterance_end"
    utterance_generation: int = 0  # bumped on schedule/cancel; guards stragglers
    # UE time — or, on the speech_final path, the release-schedule time
    pending_utterance_end_at: float | None = None
    # Queue-based serialized utterance processing
    # Items are ReleasedTurn or None (sentinel)
    utterance_queue: asyncio.Queue[ReleasedTurn | None] = field(
        default_factory=lambda: asyncio.Queue()
    )
    # Phase 6G: completed-turn metrics awaiting the browser's playback report
    # (browser_timing message); flushed to the breakdown log on shutdown when
    # no report arrived.
    completed_turn_metrics: dict[int, dict] = field(default_factory=dict)
    # Phase 6H: per-turn sentence-streaming TTS consumer (session-owned task)
    tts_task: asyncio.Task | None = None
    # Phase 6L: response TTS mode for this session. "elevenlabs" (default)
    # synthesizes MP3 audio server-side; "browser" forwards sentence text
    # (tts_text events) and the browser speaks it via speechSynthesis — no
    # ElevenL request, no MP3, no base64 audio. Set from the START message
    # and switchable mid-session via the tts_mode control message; each
    # turn's consumer reads it at creation, so a switch applies to the next
    # turn (an in-flight response is never interrupted).
    tts_mode: str = "elevenlabs"
    # Phase 2: incremental audio delivery. Set from START ("audio_stream":
    # true — the browser advertises MediaSource MP3 support): the TTS
    # consumer then forwards each sentence's provider chunks as
    # audio_chunk/audio_end instead of one complete audio message. Legacy
    # clients (field absent/false) keep the complete-MP3 protocol.
    audio_stream: bool = False
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


def _parse_start_message(msg: dict) -> tuple[StreamConfig, str, str | None, str]:
    """Parse and validate a START message.

    Returns:
        Tuple of (StreamConfig, llm_provider, llm_model, tts_mode).

        tts_mode (Phase 6L) is "elevenlabs" (default) or "browser".

    Raises:
        ValueError: When invalid.
    """
    sample_rate = msg.get("sample_rate", 16000)
    channels = msg.get("channels", 1)
    encoding = msg.get("encoding", "linear16")
    language = msg.get("language", "en")
    model = msg.get("model") or None
    # Optional turn-gap override (Phase 6E); the browser UI never sends it —
    # the StreamConfig default (1000 ms fallback gap) applies unless
    # explicitly provided.
    utterance_end_ms = msg.get("utterance_end_ms")

    # LLM configuration (optional)
    llm_provider = msg.get("llm_provider", "") or ""
    llm_model = msg.get("llm_model", "") or None
    # Phase 6L: response TTS mode — "elevenlabs" (server MP3 audio, the
    # existing pipeline) or "browser" (tts_text events + speechSynthesis).
    tts_mode = msg.get("tts_mode", "elevenlabs")

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
    if tts_mode not in ("elevenlabs", "browser"):
        raise ValueError("tts_mode must be 'elevenlabs' or 'browser'")

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
    return cfg, llm_provider, llm_model, tts_mode


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
    state.pending_release_reason = ""
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
    settle_ms: int | None = None,
    reason: str = "utterance_end",
) -> None:
    """Schedule the session-owned turn release after the settle window.

    Turn-finalization v2: ``reason="speech_final"`` with the short 600 ms
    debounce is the primary path; ``reason="utterance_end"`` with the longer
    fallback settle is used when no speech_final final released the turn.

    The task handle is kept on the session state so it can never be garbage
    collected and is always cancellable/awaitable by ``_stop_session_tasks``.
    """
    # Resolve the fallback settle at call time so the module constant is
    # honored (never bind it as a default argument).
    if settle_ms is None:
        settle_ms = _TURN_RELEASE_SETTLE_MS
    state.utterance_generation += 1
    generation = state.utterance_generation
    state.pending_release = True
    state.pending_release_turn = timings.agent_response_count + 1
    state.pending_release_reason = reason
    state.pending_utterance_end_at = time.monotonic()
    state.release_task = asyncio.create_task(
        _release_turn_after_settle(timings, state, generation, settle_ms),
        name="realtime_turn_release",
    )
    logger.info(
        "[REALTIME] utterance_release_scheduled turn=%d reason=%s "
        "settle_ms=%d last_word_end=%s speech_final=%s accumulated='%s'",
        state.pending_release_turn,
        reason,
        settle_ms,
        last_word_end,
        state.speech_final_at is not None,
        state.current_utterance_text[:120],
    )


async def _release_turn_after_settle(
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    generation: int,
    settle_ms: int,
) -> None:
    """Release the accumulated utterance after the settle window elapses.

    Runs as a session-owned task. It only enqueues when this generation is
    still current, the release is still pending and the session is RUNNING.
    Everything after the sleep is synchronous — a cancellation that already
    ran can never race the enqueue.
    """
    try:
        await asyncio.sleep(settle_ms / 1000.0)
    except asyncio.CancelledError:
        raise
    turn = timings.agent_response_count + 1
    if generation != state.utterance_generation or not state.pending_release:
        return  # cancelled or superseded
    if state.stopping or state.closed or state.stop_requested:
        state.pending_release = False
        state.release_task = None
        state.pending_release_reason = ""
        state.pending_utterance_end_at = None
        state.utterance_generation += 1
        logger.info(
            "[REALTIME] utterance_release_reason turn=%d reason=session_stopping",
            turn,
        )
        return
    transcript = state.current_utterance_text.strip()
    utterance_end_at = state.pending_utterance_end_at or time.monotonic()
    # Capture the v2 release marks before resetting the per-turn state.
    release_reason = state.pending_release_reason or "utterance_end"
    speech_final_at = state.speech_final_at
    last_final_at = state.last_final_at
    state.pending_release = False
    state.release_task = None
    state.pending_release_reason = ""
    state.pending_utterance_end_at = None
    state.current_utterance_text = ""
    state.utterance_finalized = False
    state.speech_final_at = None
    state.last_final_at = None
    if not transcript:
        logger.info(
            "[REALTIME] utterance_release_reason turn=%d reason=empty_buffer",
            turn,
        )
        return
    logger.info(
        "[REALTIME] utterance_release_reason turn=%d "
        "reason=%s settle_ms=%d",
        turn,
        release_reason,
        settle_ms,
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
        ReleasedTurn(
            transcript=transcript,
            utterance_end_at=utterance_end_at,
            released_at=time.monotonic(),
            speech_final_at=speech_final_at,
            last_final_at=last_final_at,
            release_reason=release_reason,
        )
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

    Turn finalization (v2): a final with ``speech_final=True`` schedules the
    primary early release after ``_SPEECH_FINAL_RELEASE_SETTLE_MS``;
    UtteranceEnd only schedules the fallback release (after
    ``_TURN_RELEASE_SETTLE_MS``) when no speech_final final did. Any newer
    transcript evidence cancels a pending release and accumulation continues.

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
                final_at = time.monotonic()
                if timings.first_final_at is None:
                    timings.first_final_at = final_at
                # New transcript evidence — a scheduled release is stale.
                _cancel_pending_release(
                    state, "new_final_evidence", event_type="final"
                )
                # Turn-finalization v2: speech_final=True proves the
                # endpointing silence closed this segment, so the segment is
                # the primary early-release signal (short debounce below).
                speech_final = bool((event.metadata or {}).get("speech_final"))
                state.last_final_at = final_at
                if speech_final:
                    state.speech_final_at = final_at
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
                        "speech_final": speech_final,
                    },
                ):
                    break
                if (
                    speech_final
                    and not state.stop_requested
                    and not state.stopping
                ):
                    # Early release: ~600 ms after the endpointing final the
                    # turn is handed to the agent. New speech evidence inside
                    # the window cancels it and accumulation continues — one
                    # turn, one agent call.
                    _schedule_turn_release(
                        timings,
                        state,
                        settle_ms=_SPEECH_FINAL_RELEASE_SETTLE_MS,
                        reason="speech_final",
                    )
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
                # Fallback trigger (v2): UtteranceEnd only releases the turn
                # when no speech_final final scheduled/released it already.
                # Schedule a session-owned release after the fallback settle
                # window; any new partial/final evidence within the window
                # cancels it and accumulation continues — one turn, one call.
                _schedule_turn_release(
                    timings,
                    state,
                    last_word_end=last_word_end,
                    reason="utterance_end",
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
    *,
    diagnostics: dict[str, float] | None = None,
) -> tuple[bytes, str, int, str]:
    """Synthesize one sentence into one complete audio segment (Phase 6H).

    Streaming providers: the sentence's chunks are collected into one
    complete, valid audio file — a single chunked HTTP response, NOT a
    concatenation of independent files — then delivered as one segment.
    Providers without streaming support: buffered ``synthesize()`` fallback
    (the pre-6H path; first-audio latency unavailable).

    Measurement only: when ``diagnostics`` is provided, the per-segment
    marks ``request_started`` (provider request opened), ``first_chunk_at``
    (first provider audio byte), ``last_chunk_at`` (final provider audio
    byte) and ``assembled_at`` (MP3 join complete) are recorded into it.
    Timing-only — the audio bytes and control flow are never affected.

    Returns:
        (audio_bytes, content_type, chunk_count, mode).
    """
    diag: dict[str, float] = diagnostics if diagnostics is not None else {}
    stream = await _open_tts_stream(tts, sentence)
    if stream is None:
        diag["request_started"] = time.monotonic()
        result = await tts.synthesize(text=sentence)
        diag["assembled_at"] = time.monotonic()
        return result.audio_data, result.content_type, 0, "buffered"
    try:
        chunks: list[bytes] = []
        diag["request_started"] = time.monotonic()
        async for chunk in stream:
            if not chunk:
                continue
            if "first_chunk_at" not in diag:
                diag["first_chunk_at"] = time.monotonic()
            if turn_timing.tts_first_audio_at is None:
                # First provider audio byte of this turn — the genuine
                # "TTS started producing audio" mark (Phase 6H).
                turn_timing.tts_first_audio_at = time.monotonic()
            chunks.append(chunk)
        if not chunks:
            raise RealtimeVoiceError("TTS stream returned no audio")
        diag["last_chunk_at"] = time.monotonic()
        audio = b"".join(chunks)
        diag["assembled_at"] = time.monotonic()
        return audio, _STREAM_AUDIO_CONTENT_TYPE, len(chunks), "streaming"
    finally:
        await _close_tts_stream(stream)


async def _stream_sentence_segment(
    stream: Any,
    turn_timing: UtteranceTimings,
    outbox: asyncio.Queue,
    diagnostics: dict[str, float],
) -> tuple[int, int]:
    """Phase 2: forward provider chunks into the segment outbox as they arrive.

    One provider request, many chunks: every non-empty chunk is enqueued for
    the ordered sender while the provider stream is still open, so the
    browser receives the first audio bytes without waiting for the complete
    MP3. Measurement reuses the ``_synthesize_sentence_segment`` marks
    (``request_started`` / ``first_chunk_at`` / ``last_chunk_at``); there is
    deliberately no ``assembled_at`` — nothing is joined here.

    Returns:
        (total_bytes, chunk_count) delivered by the provider stream.
    """
    total_bytes = 0
    chunk_count = 0
    try:
        diagnostics["request_started"] = time.monotonic()
        async for chunk in stream:
            if not chunk:
                continue
            if "first_chunk_at" not in diagnostics:
                diagnostics["first_chunk_at"] = time.monotonic()
            if turn_timing.tts_first_audio_at is None:
                turn_timing.tts_first_audio_at = time.monotonic()
            outbox.put_nowait(
                {"kind": "chunk", "seq": chunk_count, "data": chunk}
            )
            chunk_count += 1
            total_bytes += len(chunk)
        if not chunk_count:
            raise RealtimeVoiceError("TTS stream returned no audio")
        diagnostics["last_chunk_at"] = time.monotonic()
    finally:
        await _close_tts_stream(stream)
    return total_bytes, chunk_count


async def _stream_llm_response(
    *,
    db: Session,
    voice_session_id: str,
    transcript: str,
    llm: LLMInterface,
    model: str | None,
    sentence_sink: _SentenceSink | None = None,
    ws: WebSocket | None = None,
    state: RealtimeSessionState | None = None,
) -> dict:
    """Stream LLM response with inline tool-call detection.

    Builds messages (system prompt + history + user message), calls
    stream_chat() with the registered tool schemas, accumulates tokens into
    a SentenceBuffer, and captures first-token timing.

    Phase 6H: every complete sentence is pushed to ``sentence_sink`` the
    moment SentenceBuffer yields it — the concurrent TTS consumer can start
    on sentence 1 while this stream is still producing sentences.

    Phase 1: SentenceBuffer runs with early chunking enabled, and a
    buffer-only watchdog flushes stalled text at a word boundary, so the
    first TTS chunk can start before the stream ends. The watchdog never
    awaits or cancels the LLM generator and is reaped in ``finally``
    before the stream-end flush.

    Tool calls (OpenAI-compatible streaming protocol): deltas are
    accumulated by tool-call index; when a turn calls tools the streamed
    filler is discarded and ``_complete_streamed_tool_turn`` executes the
    tools, runs a final streaming stream_chat() follow-up and streams the
    final answer through the same sink.

    Progressive UI: when ``ws``/``state`` are provided, every streamed
    content chunk is forwarded to the browser as an ``agent_delta`` event
    (lifecycle-safe send, no buffering); ``agent_response`` later carries
    the authoritative complete text.

    Returns:
        Dict with response, tool_calls, usage, iterations, first_token_ms,
        streamed_tokens and the absolute Phase 6G marks
        (llm_request_started_at, llm_first_token_at, llm_first_sentence_at,
        llm_first_chunk_at, llm_completed_at, sentence_count).
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

    # Tool schemas for function calling: the registered tools (including
    # Employee Details) ride along on the streaming request; an empty
    # registry keeps the pre-tool payload (tools=None).
    tool_schemas = get_tool_registry().get_schemas()

    logger.info(
        "[REALTIME:STREAM] llm_stream_start model=%s history_len=%d tools=%d",
        model or "default",
        len(history),
        len(tool_schemas),
    )

    first_token_at: float | None = None
    first_sentence_at: float | None = None
    accumulated_text = ""
    chunk_count = 0
    # Phase 1: early chunking — the first TTS chunk may be a complete
    # sentence, a trailing-punctuation chunk, a soft max-split or a stall
    # flush, all emitted while the LLM is still generating.
    buffer = SentenceBuffer(
        min_first_chunk_chars=_MIN_FIRST_CHUNK_CHARS,
        max_chunk_chars=_MAX_TTS_CHUNK_CHARS,
    )
    sentences: list[str] = []

    def _emit_chunks(new_chunks: list[str]) -> None:
        """Record + forward freshly emitted TTS chunks (order preserved)."""
        nonlocal first_sentence_at
        if not new_chunks:
            return
        if first_sentence_at is None:
            first_sentence_at = time.monotonic()
        sentences.extend(new_chunks)
        if sentence_sink is not None:
            for sentence in new_chunks:
                sentence_sink.put(sentence)

    async def _chunk_flush_watchdog() -> None:
        """Flush stalled buffer text at a safe word boundary (Phase 1).

        Inspects ONLY the SentenceBuffer — it never awaits, times out or
        cancels the LLM generator, so a slow chunk can never be aborted by
        this watchdog. Cancelled and reaped in the stream ``finally``
        before the stream-end flush, so no delayed flush can land after
        completion or cancellation.
        """
        while True:
            await asyncio.sleep(_CHUNK_FLUSH_TIMEOUT_MS / 1000.0)
            if tool_detected:
                # A tool turn discards the streamed filler — once tool-call
                # deltas appeared, never flush leftover filler text into the
                # TTS sink. The tool follow-up has its own watchdog.
                return
            _emit_chunks(buffer.flush_on_timeout())

    # Tool-call streaming state: deltas arrive split across chunks and are
    # accumulated by tool-call index (OpenAI streaming protocol).
    tool_detected = False
    tool_call_deltas: dict[int, dict[str, Any]] = {}

    stream_start = time.monotonic()
    watchdog_task = asyncio.create_task(_chunk_flush_watchdog())
    try:
        async for chunk in llm.stream_chat(
            messages=messages,
            model=model,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.7,
        ):
            # Accumulate tool call deltas from the stream; a turn that calls
            # tools discards any streamed filler in favour of the final
            # follow-up answer.
            if chunk.tool_calls:
                if not tool_detected:
                    tool_detected = True
                    logger.info(
                        "[REALTIME:STREAM] tool_calls detected in stream — "
                        "accumulating"
                    )
                for tc_delta in chunk.tool_calls:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    assembled = tool_call_deltas[idx]
                    if tc_delta.get("id"):
                        assembled["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        assembled["function"]["name"] = func["name"]
                    if func.get("arguments"):
                        assembled["function"]["arguments"] += func["arguments"]
                continue

            if chunk.content:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                accumulated_text += chunk.content
                chunk_count += 1
                # Progressive UI: forward the raw delta immediately —
                # lifecycle-safe send, no extra buffering. agent_response
                # later carries the authoritative complete text.
                if ws is not None and state is not None:
                    await _safe_ws_send(
                        ws,
                        state,
                        {"type": "agent_delta", "text": chunk.content},
                    )
                # Feed to sentence buffer (Phase 1: it may emit early
                # chunks — trailing punctuation, soft max-split or a
                # watchdog stall flush — before the stream completes).
                _emit_chunks(buffer.add(chunk.content))

            if chunk.finish_reason:
                break
    except Exception as e:
        logger.error(
            "[REALTIME:STREAM] llm_stream_failed error=%s",
            str(e),
        )
        raise RealtimeVoiceError(f"LLM streaming failed: {e}") from e
    finally:
        # Phase 1: reap the chunk watchdog BEFORE the stream-end flush so
        # no delayed watchdog flush can land after completion, failure or
        # cancellation. A session cancel (STOP/disconnect) raises
        # CancelledError past `except Exception` and is still cleaned up
        # here; the watchdog itself never touches the LLM generator.
        watchdog_task.cancel()
        await asyncio.gather(watchdog_task, return_exceptions=True)

    # Flush any remaining text from buffer (only when no tool calls were
    # detected — a tool turn discards the streamed filler text). The chunk
    # watchdog is already reaped, so the remainder is emitted exactly once.
    if not tool_detected:
        remaining = buffer.flush()
        if remaining:
            _emit_chunks([remaining])

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

    # Tool calls detected: execute them and generate the final spoken answer
    # via a streaming stream_chat() follow-up (streaming+tools pattern).
    if tool_detected:
        return await _complete_streamed_tool_turn(
            db=db,
            voice_session_id=voice_session_id,
            transcript=transcript,
            llm=llm,
            model=model,
            sentence_sink=sentence_sink,
            history=history,
            tool_call_deltas=tool_call_deltas,
            streamed_text=accumulated_text,
            stream_start=stream_start,
            first_token_at=first_token_at,
            req1_ms=stream_ms,
            streamed_tokens=chunk_count,
            ws=ws,
            state=state,
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
        "llm_first_chunk_at": first_sentence_at,
        "llm_completed_at": stream_completed_at,
        "sentence_count": len(sentences),
    }


# Deterministic tool-progress acks: shown/spoken while a tool executes so the
# caller gets immediate feedback without an extra LLM request. Exact tool-name
# matches win; well-known name fragments get a tailored ack; anything else
# gets the generic one. Never includes arguments, IDs or internal details.
_TOOL_PROGRESS_EXACT: dict[str, str] = {
    "get_employee": "Sure, let me check the employee details for you.",
    "get_weather": "Sure, let me check the latest weather for you.",
}
_TOOL_PROGRESS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("appointment", "Let me check that for you."),
    ("search", "Let me check that for you."),
)
_TOOL_PROGRESS_DEFAULT = "Sure, let me check that for you."


def _tool_progress_message(tool_name: str) -> str:
    if tool_name in _TOOL_PROGRESS_EXACT:
        return _TOOL_PROGRESS_EXACT[tool_name]
    lowered = tool_name.lower()
    for keyword, message in _TOOL_PROGRESS_KEYWORDS:
        if keyword in lowered:
            return message
    return _TOOL_PROGRESS_DEFAULT


async def _complete_streamed_tool_turn(
    *,
    db: Session,
    voice_session_id: str,
    transcript: str,
    llm: LLMInterface,
    model: str | None,
    sentence_sink: _SentenceSink | None,
    history: list[LLMMessage],
    tool_call_deltas: dict[int, dict[str, Any]],
    streamed_text: str,
    stream_start: float,
    first_token_at: float | None,
    req1_ms: float,
    streamed_tokens: int,
    ws: WebSocket | None = None,
    state: RealtimeSessionState | None = None,
) -> dict:
    """Execute streamed tool calls and stream the final answer back.

    Continues a turn whose initial stream_chat() request produced tool calls
    (OpenAI streaming protocol): the accumulated deltas are assembled, every
    tool runs through the shared ToolExecutor (same registry as the chat,
    voice and telephony paths), the results are sent back with a single
    streaming stream_chat() follow-up without tool schemas, and the final
    answer is pushed through SentenceBuffer -> sentence_sink sentence by
    sentence as it streams so the TTS consumer starts on the first complete
    sentence while the LLM is still generating.

    Mirrors the proven streaming+tools flow of the Telnyx agent.

    Tool progress UX: right before each executor call a deterministic
    ``tool_progress`` event is sent (lifecycle-safe, when ws/state are
    provided) and the ack is pushed through sentence_sink so the existing
    TTS consumer speaks it while the tool runs.

    Returns:
        Dict with the final response, detected tool calls, final usage,
        iterations (initial stream + final chat), the absolute Phase 6G
        marks and the tool-turn timing breakdown (req1_ms, tool_ms,
        final_ms, total_ms).
    """
    from app.models.voice_session import VoiceSession
    from app.services.realtime_voice_service import _save_message, _save_tool_call

    # Assemble complete tool calls from accumulated deltas
    detected_tool_calls = [
        {
            "id": tool_call_deltas[idx]["id"],
            "type": "function",
            "function": {
                "name": tool_call_deltas[idx]["function"]["name"],
                "arguments": tool_call_deltas[idx]["function"]["arguments"],
            },
        }
        for idx in sorted(tool_call_deltas.keys())
    ]
    logger.info(
        "[REALTIME:TOOL] assembled %d tool call(s) from stream deltas",
        len(detected_tool_calls),
    )

    # Execute each tool call with the shared executor
    tool_executor = ToolExecutor(get_tool_registry())
    tool_messages: list[LLMMessage] = []
    tool_records: list[dict[str, Any]] = []
    tool_execution_total_ms = 0.0
    # One progress ack per distinct tool name per turn: the model may fan out
    # several legitimate calls to the same tool (e.g. get_leave_balance per
    # leave type) — they all still execute below, but the caller hears one
    # acknowledgment per tool.
    acked_tools: set[str] = set()

    for tool_call in detected_tool_calls:
        tool_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call.get("id", "")
        # Immediate feedback: deterministic ack event + speech BEFORE the
        # potentially slow executor call. Lifecycle-safe send (suppressed on
        # STOP/closed); the ack rides the existing sentence_sink so the TTS
        # consumer speaks it while the tool runs. Only the first call per
        # tool name is acknowledged — repeat calls to the same tool still
        # execute in full below.
        if tool_name not in acked_tools:
            acked_tools.add(tool_name)
            progress_message = _tool_progress_message(tool_name)
            if ws is not None and state is not None:
                await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "tool_progress",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "message": progress_message,
                    },
                )
            if sentence_sink is not None:
                sentence_sink.put(progress_message)
        tool_start = time.monotonic()
        logger.info(
            "[REALTIME:TOOL] executing tool=%s args=%s",
            tool_name,
            arguments_str[:200],
        )
        tool_result = await tool_executor.execute_from_json(tool_name, arguments_str)
        tool_ms = (time.monotonic() - tool_start) * 1000
        tool_execution_total_ms += tool_ms
        if tool_result.success:
            raw_result = tool_result.output
        else:
            raw_result = {"error": tool_result.error}
        result_content = (
            json.dumps(raw_result)
            if not isinstance(raw_result, str)
            else raw_result
        )
        tool_messages.append(
            LLMMessage(
                role="tool",
                content=result_content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )
        tool_records.append(
            {
                "name": tool_name,
                "arguments": arguments_str,
                "id": tool_call_id,
                "content": result_content,
                "success": tool_result.success,
                "error": tool_result.error,
                "duration_ms": round(tool_ms),
            }
        )
        logger.info(
            "[REALTIME:TOOL] tool=%s duration_ms=%.0f success=%s",
            tool_name,
            tool_ms,
            tool_result.success,
        )
        # Trace support: report each executed tool call back to the client
        # (name, call id, duration, outcome). Unlike tool_progress this is
        # NOT deduplicated — fan-out calls execute once each and the client
        # trace shows every execution.
        if ws is not None and state is not None:
            await _safe_ws_send(
                ws,
                state,
                {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "duration_ms": round(tool_ms),
                    "success": tool_result.success,
                    "error": tool_result.error,
                },
            )

    # Final LLM call: conversation + tool results, no tool schemas — one
    # tool round per turn keeps latency bounded. Streamed (tools=None: tool
    # execution for this turn is already complete) and fed through the same
    # SentenceBuffer -> sentence_sink pipeline as the no-tool path so the
    # TTS consumer starts on the first complete sentence while the LLM is
    # still generating.
    final_messages: list[LLMMessage] = [
        LLMMessage(role="system", content=_REALTIME_DEFAULT_SYSTEM_PROMPT),
        *history,
        LLMMessage(role="user", content=transcript),
        LLMMessage(
            role="assistant",
            content=streamed_text or None,
            tool_calls=detected_tool_calls,
        ),
        *tool_messages,
    ]
    final_llm_start = time.monotonic()
    logger.info(
        "[REALTIME:TOOL] final_llm_start messages=%d mode=streaming",
        len(final_messages),
    )
    final_text = ""
    final_usage: dict[str, Any] = {}
    # Phase 1: same early chunking as the main stream — the final spoken
    # answer starts flowing to TTS while it is still being generated.
    final_buffer = SentenceBuffer(
        min_first_chunk_chars=_MIN_FIRST_CHUNK_CHARS,
        max_chunk_chars=_MAX_TTS_CHUNK_CHARS,
    )
    sentences: list[str] = []
    first_sentence_at: float | None = None

    def _emit_final_chunks(new_chunks: list[str]) -> None:
        """Record + forward TTS chunks for the tool follow-up answer."""
        nonlocal first_sentence_at
        if not new_chunks:
            return
        if first_sentence_at is None:
            first_sentence_at = time.monotonic()
        sentences.extend(new_chunks)
        if sentence_sink is not None:
            for sentence in new_chunks:
                sentence_sink.put(sentence)

    async def _final_chunk_flush_watchdog() -> None:
        """Buffer-only stall flush for the tool follow-up answer.

        Mirrors the main-stream watchdog: it inspects only the
        SentenceBuffer and is cancelled + reaped in the ``finally`` below.
        """
        while True:
            await asyncio.sleep(_CHUNK_FLUSH_TIMEOUT_MS / 1000.0)
            _emit_final_chunks(final_buffer.flush_on_timeout())

    final_watchdog_task = asyncio.create_task(_final_chunk_flush_watchdog())
    try:
        async for chunk in llm.stream_chat(
            messages=final_messages,
            model=model,
            tools=None,
        ):
            if chunk.content:
                final_text += chunk.content
                # Progressive UI: same per-chunk agent_delta as the no-tool
                # streaming path (lifecycle-safe send).
                if ws is not None and state is not None:
                    await _safe_ws_send(
                        ws,
                        state,
                        {"type": "agent_delta", "text": chunk.content},
                    )
                _emit_final_chunks(final_buffer.add(chunk.content))
            if chunk.finish_reason:
                break
    except Exception as e:
        logger.error(
            "[REALTIME:TOOL] final_llm_failed error=%s",
            str(e),
        )
        raise RealtimeVoiceError(f"LLM tool follow-up failed: {e}") from e
    finally:
        # Reap the watchdog before the trailing flush (see main stream).
        final_watchdog_task.cancel()
        await asyncio.gather(final_watchdog_task, return_exceptions=True)

    # Flush any trailing partial sentence, same as the no-tool path.
    trailing = final_buffer.flush()
    if trailing:
        _emit_final_chunks([trailing])

    final_llm_completed_at = time.monotonic()
    final_ms = (final_llm_completed_at - final_llm_start) * 1000
    logger.info(
        "[REALTIME:TOOL] final_llm_complete duration_ms=%.0f "
        "response_length=%d sentences=%d",
        final_ms,
        len(final_text),
        len(sentences),
    )

    # Persist the turn: user + assistant tool call + tool results + final
    # answer (llm_data keeps the full shape for history reload).
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
            streamed_text or None,
            seq + 1,
            llm_data={
                "role": "assistant",
                "content": streamed_text or None,
                "tool_calls": detected_tool_calls,
                "tool_call_id": None,
                "name": None,
            },
        )
        for offset, record in enumerate(tool_records):
            _save_message(
                db,
                voice_session_id,
                "tool",
                record["content"],
                seq + 2 + offset,
                llm_data={
                    "role": "tool",
                    "content": record["content"],
                    "tool_calls": None,
                    "tool_call_id": record["id"],
                    "name": record["name"],
                },
            )
        _save_message(
            db,
            voice_session_id,
            "assistant",
            final_text,
            seq + 2 + len(tool_records),
            token_count=final_usage.get("total_tokens"),
            llm_data={
                "role": "assistant",
                "content": final_text,
                "tool_calls": None,
                "tool_call_id": None,
                "name": None,
            },
        )
        voice_session.message_count = seq + 3 + len(tool_records)
        db.commit()

    # Persist tool call records (same table used by chat/voice paths)
    for record in tool_records:
        _save_tool_call(
            db,
            voice_session_id,
            record["name"],
            record["arguments"],
            record["content"],
            "success" if record["success"] else "error",
            error_message=record["error"],
            duration_ms=record["duration_ms"],
        )

    logger.info(
        "[REALTIME:TOOL] turn_complete tool_calls=%d tool_ms=%.0f final_ms=%.0f",
        len(detected_tool_calls),
        tool_execution_total_ms,
        final_ms,
    )
    return {
        "response": final_text,
        "tool_calls": detected_tool_calls,
        "usage": final_usage,
        "iterations": 2,  # initial stream + final chat
        "streamed": False,
        "first_token_ms": (
            (first_token_at - stream_start) * 1000 if first_token_at else None
        ),
        "streamed_tokens": streamed_tokens,
        "sentences": sentences,
        # Tool-turn timing breakdown: req1 (tool selection) -> tools -> req2
        # (final answer); total_ms spans req1 start -> req2 completion.
        "req1_ms": req1_ms,
        "tool_ms": tool_execution_total_ms,
        "final_ms": final_ms,
        "total_ms": (final_llm_completed_at - stream_start) * 1000,
        # Phase 6G marks: the initial stream start anchors the turn's LLM
        # request; llm_completed_at covers tool execution + final chat.
        "llm_request_started_at": stream_start,
        "llm_first_token_at": first_token_at,
        "llm_first_sentence_at": first_sentence_at,
        "llm_first_chunk_at": first_sentence_at,
        "llm_completed_at": final_llm_completed_at,
        "sentence_count": len(sentences),
    }


async def _tts_text_consumer(
    ws: WebSocket,
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    turn: int,
    turn_timing: UtteranceTimings,
    sentence_sink: _SentenceSink,
) -> dict[str, Any]:
    """Phase 6L browser TTS: deliver sentence TEXT, never audio.

    Used when the session's TTS mode is ``"browser"``: the backend does not
    call ElevenLabs, does not synthesize MP3, and does not emit base64
    ``audio`` segments. Each complete LLM sentence is forwarded as a
    ``tts_text`` event and the browser speaks it locally via
    ``speechSynthesis``. Sentence order is inherent (one send per popped
    sentence); STOP/disconnect cancels this consumer via
    ``_stop_session_tasks`` exactly like the Phase 6K audio consumer.

    Returns:
        Same summary shape as ``_tts_sentence_consumer``: sentences,
        segments_sent (text segments), failures, chunks (always 0), and
        first_error.
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
            # Same contract as the audio path: the TTS stage opens with the
            # first sentence so the browser UI flips to "speaking…".
            if not await _safe_ws_send(ws, state, {"type": "tts_processing"}):
                break
            started = time.monotonic()
            turn_timing.tts_started_at = started
            turn_timing.tts_first_sentence_started_at = started
            logger.info("[REALTIME] tts_processing turn=%d mode=browser", turn)
        segment_no += 1
        if not await _safe_ws_send(
            ws,
            state,
            {"type": "tts_text", "turn": turn, "segment": segment_no, "text": sentence},
        ):
            break
        sent_at = time.monotonic()
        # Timing marks mirror the audio path so turn_metrics stays populated
        # and comparable: "first audio sent" is the first tts_text handed to
        # the browser — the mark after which the user can hear the response.
        if turn_timing.first_audio_sent_at is None:
            turn_timing.first_audio_sent_at = sent_at
        turn_timing.audio_sent_at = sent_at
        turn_timing.tts_completed_at = sent_at
        if turn_timing.tts_mode is None:
            turn_timing.tts_mode = "browser"
        if sentence_no == 1:
            turn_timing.tts_first_sentence_completed_at = sent_at
        turn_timing.tts_sentence_count += 1
        summary["segments_sent"] += 1
        timings.tts_response_count += 1
        timings.tts_characters += len(sentence)
        logger.info(
            "[REALTIME] tts_text_sent turn=%d segment=%d sentence=%d chars=%d",
            turn,
            segment_no,
            sentence_no,
            len(sentence),
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


async def _tts_sentence_consumer(
    ws: WebSocket,
    timings: RealtimeTimings,
    state: RealtimeSessionState,
    turn: int,
    turn_timing: UtteranceTimings,
    sentence_sink: _SentenceSink,
) -> dict[str, Any]:
    """Consume complete LLM sentences and deliver ordered audio segments.

    Phase 6K: sentence synthesis pre-generates with bounded concurrency
    (``_TTS_SENTENCE_CONCURRENCY``) — sentence N+1's TTS runs while segment
    N is being sent (and played by the browser). Segments are handed to the
    socket strictly by sentence index: segment N+1 is never started before
    segment N. One failed sentence is logged and counted, never fatal to
    the turn or session.

    Phase 2: every sentence owns an outbox queue. Its producer enqueues
    ``chunk`` items (incremental ElevenLabs delivery when the client
    advertised ``audio_stream``) or one ``complete`` item (buffered
    fallback / legacy clients), followed by an ``end`` item. The ordered
    sender drains only the outbox of the next unsent sentence, so provider
    arrival order can never reorder segments — and the first chunk reaches
    the browser while the provider stream is still open.

    Runs as a session-owned task (``state.tts_task``); STOP/disconnect
    cancels it via ``_stop_session_tasks``. In-flight sentence synthesis
    tasks are cancelled and reaped by this consumer's ``finally`` — asyncio
    does not cancel child tasks automatically.

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
    # Phase 2: incremental delivery is only enabled when the client
    # advertised MediaSource MP3 support in START. Every other client keeps
    # the complete-MP3 protocol (legacy behavior, byte-for-byte).
    incremental = bool(state.audio_stream)
    # Phase 6K/2 ordered-concurrency state: per-sentence outboxes filled by
    # the synthesis producers ("chunk" / "complete" / "end" items),
    # in-flight synthesis tasks by sentence index, sentences waiting for a
    # free slot, and the next sentence index the sender may send.
    outboxes: dict[int, asyncio.Queue] = {}
    tasks: dict[int, asyncio.Task] = {}
    pending: list[tuple[int, str]] = []
    next_to_send = 1
    # Phase 2 delivery bookkeeping: "open" spans the items of one segment;
    # *_sent_at bound that segment's first/last socket write.
    send_open = False
    seg_first_sent_at: float | None = None
    seg_last_sent_at: float | None = None
    seg_chunks_sent = 0

    async def _synthesize(sentence_no: int, sentence: str) -> dict[str, Any]:
        """Synthesize one sentence; never raises (cancellation excepted)."""
        started_at = time.monotonic()
        outbox = outboxes[sentence_no]
        # Measurement-only per-segment marks (see _synthesize_sentence_segment).
        segment_diag: dict[str, float] = {}

        def _diag_delta(end_key: str, start: float | None) -> str:
            """Measured delta between diagnostic marks, or n/a when absent."""
            end = segment_diag.get(end_key)
            if end is None or start is None:
                return "n/a"
            return f"{(end - start) * 1000:.1f}"

        logger.info(
            "[REALTIME:TTS] segment_tts_started turn=%d sentence=%d chars=%d",
            turn,
            sentence_no,
            len(sentence),
        )
        try:
            if incremental:
                # Phase 2: one provider request, chunks forwarded to the
                # ordered sender the moment they arrive (no b"".join wait).
                stream = await _open_tts_stream(state.tts, sentence)
                if stream is not None:
                    sent_bytes, chunks = await _stream_sentence_segment(
                        stream, turn_timing, outbox, segment_diag
                    )
                    content_type = _STREAM_AUDIO_CONTENT_TYPE
                    mode = "streaming"
                    chunked = True
                else:
                    # Provider cannot stream — buffered fallback, delivered
                    # as one complete segment (Phase 6H shape).
                    (
                        audio_data,
                        content_type,
                        chunks,
                        mode,
                    ) = await _synthesize_sentence_segment(
                        state.tts,
                        sentence,
                        turn_timing,
                        diagnostics=segment_diag,
                    )
                    sent_bytes = len(audio_data)
                    chunked = False
                    outbox.put_nowait(
                        {
                            "kind": "complete",
                            "data": audio_data,
                            "content_type": content_type,
                        }
                    )
            else:
                # Legacy delivery: one complete MP3 per sentence (unchanged).
                (
                    audio_data,
                    content_type,
                    chunks,
                    mode,
                ) = await _synthesize_sentence_segment(
                    state.tts,
                    sentence,
                    turn_timing,
                    diagnostics=segment_diag,
                )
                sent_bytes = len(audio_data)
                chunked = False
                outbox.put_nowait(
                    {
                        "kind": "complete",
                        "data": audio_data,
                        "content_type": content_type,
                    }
                )
        except Exception as e:
            outbox.put_nowait({"kind": "end", "ok": False})
            return {
                "sentence_no": sentence_no,
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        completed_at = time.monotonic()
        turn_timing.tts_completed_at = completed_at
        if turn_timing.tts_mode is None:
            # Mode of the first successfully synthesized sentence.
            turn_timing.tts_mode = mode
        if sentence_no == 1:
            turn_timing.tts_first_sentence_completed_at = completed_at
        turn_timing.tts_sentence_count += 1
        turn_timing.tts_audio_chunks += chunks
        outbox.put_nowait(
            {
                "kind": "end",
                "ok": True,
                "mode": mode,
                "chunked": chunked,
                "bytes": sent_bytes,
                "chunks": chunks,
                "chars": len(sentence),
                "started_at": started_at,
                "completed_at": completed_at,
                "request_started": segment_diag.get("request_started"),
            }
        )
        logger.info(
            "[REALTIME:TTS] segment_tts_completed turn=%d sentence=%d "
            "bytes=%d chunks=%d mode=%s synthesis_ms=%.0f "
            "req_ms=%s first_byte_ms=%s stream_ms=%s assemble_ms=%s",
            turn,
            sentence_no,
            sent_bytes,
            chunks,
            mode,
            (completed_at - started_at) * 1000,
            _diag_delta("request_started", started_at),
            _diag_delta("first_chunk_at", segment_diag.get("request_started")),
            _diag_delta("last_chunk_at", segment_diag.get("first_chunk_at")),
            _diag_delta("assembled_at", segment_diag.get("last_chunk_at")),
        )
        return {
            "sentence_no": sentence_no,
            "ok": True,
        }

    def _record(task: asyncio.Task) -> None:
        """Fold one finished synthesis task into the turn statistics."""
        outcome = task.result()
        sentence_no = outcome["sentence_no"]
        tasks.pop(sentence_no, None)
        if outcome["ok"]:
            return
        # One failed sentence must not kill the turn or the session: log,
        # count, and let the ordered sender advance past the missing segment.
        summary["failures"] += 1
        turn_timing.tts_sentence_failures += 1
        if summary["first_error"] is None:
            summary["first_error"] = outcome["error"]
        logger.error(
            "[REALTIME:TTS] sentence_synthesis_failed turn=%d sentence=%d "
            "error_type=%s error=%s",
            turn,
            sentence_no,
            outcome["error_type"],
            outcome["error"],
        )

    def _start_task(sentence_no: int, sentence: str) -> None:
        # Phase 2: the outbox exists before the producer runs — the ordered
        # sender arms on it as soon as this sentence is next in order.
        outboxes[sentence_no] = asyncio.Queue()
        tasks[sentence_no] = asyncio.create_task(
            _synthesize(sentence_no, sentence),
            name=f"realtime_tts_sentence_{turn}_{sentence_no}",
        )

    def _schedule(sentence_no: int, sentence: str) -> None:
        """Start one sentence synthesis now, or hold it for a free slot."""
        if len(tasks) < _TTS_SENTENCE_CONCURRENCY:
            _start_task(sentence_no, sentence)
        else:
            pending.append((sentence_no, sentence))

    async def _handle_sentence(sentence: str) -> None:
        """Open the TTS stage / schedule synthesis for one new sentence."""
        nonlocal ws_dead
        summary["sentences"] += 1
        sentence_no = summary["sentences"]
        if state.stopping or state.closed or state.stop_requested:
            logger.info(
                "[REALTIME:TTS] sentence_skipped turn=%d sentence=%d "
                "reason=session_stopping",
                turn,
                sentence_no,
            )
            outbox = asyncio.Queue()
            outbox.put_nowait({"kind": "end", "ok": False, "skipped": True})
            outboxes[sentence_no] = outbox
            return
        if sentence_no == 1:
            # The TTS stage opens with the first sentence; sent from here
            # (not the worker) so it precedes the first audio frame while
            # the worker is still awaiting the LLM stream.
            if not await _safe_ws_send(ws, state, {"type": "tts_processing"}):
                ws_dead = True
                return
            started = time.monotonic()
            turn_timing.tts_started_at = started
            turn_timing.tts_first_sentence_started_at = started
            logger.info("[REALTIME] tts_processing turn=%d", turn)
        _schedule(sentence_no, sentence)

    async def _deliver(item: dict[str, Any]) -> None:
        """Deliver one outbox item: chunk, complete segment, or end marker.

        Ordering contract: items are pulled only from the outbox of
        ``next_to_send``, so provider arrival order can never reorder
        sentences — segment N+1 is never started before segment N ended.
        """
        nonlocal segment_no, next_to_send, ws_dead
        nonlocal send_open, seg_first_sent_at, seg_last_sent_at, seg_chunks_sent
        kind = item["kind"]
        if kind in ("chunk", "complete"):
            if not send_open:
                # First item of this segment: assign the segment number now
                # so every chunk of the segment carries the same number.
                send_open = True
                segment_no += 1
                seg_first_sent_at = None
                seg_last_sent_at = None
                seg_chunks_sent = 0
            if kind == "chunk":
                payload = {
                    "type": "audio_chunk",
                    "format": _STREAM_AUDIO_CONTENT_TYPE,
                    "data": base64.b64encode(item["data"]).decode("ascii"),
                    "turn": turn,
                    "segment": segment_no,
                    "seq": item["seq"],
                    "sent_epoch_ms": int(time.time() * 1000),
                }
            else:
                payload = {
                    "type": "audio",
                    "format": item["content_type"],
                    "data": base64.b64encode(item["data"]).decode("ascii"),
                    "turn": turn,
                    "segment": segment_no,
                    "sent_epoch_ms": int(time.time() * 1000),
                }
            if not await _safe_ws_send(ws, state, payload):
                # Session is stopping/closed — later segments could never
                # be heard; stop consuming.
                ws_dead = True
                return
            sent_at = time.monotonic()
            if seg_first_sent_at is None:
                seg_first_sent_at = sent_at
                if turn_timing.first_audio_sent_at is None:
                    turn_timing.first_audio_sent_at = sent_at
            seg_last_sent_at = sent_at
            seg_chunks_sent += 1
            turn_timing.audio_sent_at = sent_at
            return
        # end marker: this sentence's segment is complete (or failed/skipped).
        ok = bool(item.get("ok"))
        if send_open:
            if seg_chunks_sent > 0:
                # Chunked segment: close the sequence so the browser can
                # finalize playback. Also sent after a mid-stream failure so
                # a partially delivered segment still plays out and the
                # queue keeps moving.
                if not await _safe_ws_send(
                    ws,
                    state,
                    {
                        "type": "audio_end",
                        "turn": turn,
                        "segment": segment_no,
                        "seq": seg_chunks_sent - 1,
                    },
                ):
                    ws_dead = True
            if ok:
                sent_at = time.monotonic()
                completed_at = item.get("completed_at")
                item_started_at = item.get("started_at")
                request_started = item.get("request_started")
                synthesis_ms: float | None = None
                if completed_at is not None and item_started_at is not None:
                    synthesis_ms = (completed_at - item_started_at) * 1000
                first_sent_ms: float | None = None
                if seg_first_sent_at is not None and request_started is not None:
                    first_sent_ms = (seg_first_sent_at - request_started) * 1000
                sent_span_ms: float | None = None
                if seg_last_sent_at is not None and seg_first_sent_at is not None:
                    sent_span_ms = (seg_last_sent_at - seg_first_sent_at) * 1000
                # held_ms keeps its Phase 6K meaning (assembly -> send). A
                # chunked segment has no assembly wait left — report n/a
                # instead of a silent zero.
                held_ms: float | None = None
                if (
                    not item.get("chunked")
                    and completed_at is not None
                    and seg_last_sent_at is not None
                ):
                    held_ms = max(0.0, (seg_last_sent_at - completed_at) * 1000)
                summary["segments_sent"] += 1
                summary["chunks"] += int(item.get("chunks", 0))
                timings.tts_response_count += 1
                timings.tts_audio_bytes += int(item.get("bytes", 0))
                timings.tts_characters += int(item.get("chars", 0))
                logger.info(
                    "[REALTIME] audio_segment_sent turn=%d segment=%d sentence=%d "
                    "bytes=%d chunks=%d mode=%s synthesis_ms=%s held_ms=%s "
                    "first_sent_ms=%s sent_span_ms=%s",
                    turn,
                    segment_no,
                    next_to_send,
                    int(item.get("bytes", 0)),
                    int(item.get("chunks", 0)),
                    item.get("mode", "?"),
                    _fmt_ms(synthesis_ms),
                    _fmt_ms(held_ms),
                    _fmt_ms(first_sent_ms),
                    _fmt_ms(sent_span_ms),
                )
        outboxes.pop(next_to_send, None)
        next_to_send += 1
        send_open = False
        seg_first_sent_at = None
        seg_last_sent_at = None
        seg_chunks_sent = 0

    get_task: asyncio.Task | None = asyncio.ensure_future(sentence_sink.queue.get())
    sentinel_seen = False
    ws_dead = False
    # Phase 2: pending ordered-sender get() on the next unsent sentence.
    send_task: asyncio.Task | None = None
    try:
        while True:
            if ws_dead:
                break
            # Launch every held sentence that now fits the concurrency bound.
            while pending and len(tasks) < _TTS_SENTENCE_CONCURRENCY:
                held_no, held_sentence = pending.pop(0)
                _start_task(held_no, held_sentence)
            # Normal exit: end-of-response seen, nothing in flight or held,
            # no outbox left undelivered, and no segment mid-flight.
            if (
                sentinel_seen
                and not pending
                and not tasks
                and not outboxes
                and not send_open
            ):
                break
            wait_set: set[asyncio.Task] = set(tasks.values())
            if not sentinel_seen and get_task is not None:
                wait_set.add(get_task)
            if send_task is None and next_to_send in outboxes:
                # Arm the ordered sender: the next item of the next unsent
                # sentence is delivered the moment the producer enqueues it.
                send_task = asyncio.ensure_future(outboxes[next_to_send].get())
            if send_task is not None:
                wait_set.add(send_task)
            if not wait_set:
                break
            done, _ = await asyncio.wait(
                wait_set, return_when=asyncio.FIRST_COMPLETED
            )
            completed_get: asyncio.Task | None = None
            if get_task is not None and get_task in done:
                completed_get = get_task
                get_task = None
                sentence = completed_get.result()
                if sentence is None:
                    sentinel_seen = True
                else:
                    await _handle_sentence(sentence)
                if not sentinel_seen and not ws_dead and get_task is None:
                    get_task = asyncio.ensure_future(sentence_sink.queue.get())
            completed_send: asyncio.Task | None = None
            if send_task is not None and send_task in done:
                completed_send = send_task
                send_task = None
                await _deliver(completed_send.result())
            for finished in done:
                if finished is completed_get or finished is completed_send:
                    continue
                _record(finished)
    finally:
        # STOP/disconnect/dead socket: cancel and reap every task this
        # consumer created so no synthesis outlives the session and no task
        # is left pending on the event loop.
        stranded = list(tasks.values())
        if get_task is not None and not get_task.done():
            stranded.append(get_task)
        if send_task is not None and not send_task.done():
            stranded.append(send_task)
        for stranded_task in stranded:
            stranded_task.cancel()
        if stranded:
            await asyncio.gather(*stranded, return_exceptions=True)
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

        # Unpack the released turn from the release path
        transcript = item.transcript

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

        # Create per-utterance timing record (v2 marks carried by ReleasedTurn)
        turn_timing = UtteranceTimings(
            turn=turn,
            utterance_end_at=item.utterance_end_at,
            utterance_released_at=item.released_at,
            speech_final_at=item.speech_final_at,
            last_final_at=item.last_final_at,
            release_reason=item.release_reason,
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
            logger.info(
                "[REALTIME:TURN] turn=%d speech_final_to_agent=%sms "
                "transcript_to_agent=%sms release_reason=%s",
                turn,
                _ms_since(
                    turn_timing.speech_final_at,
                    turn_timing.agent_processing_started_at,
                ),
                _ms_since(
                    turn_timing.last_final_at,
                    turn_timing.agent_processing_started_at,
                ),
                turn_timing.release_reason,
            )

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

            # Process utterance via streaming LLM (inline tool detection)
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
            if state.tts is not None or state.tts_mode == "browser":
                # Phase 6H: the consumer runs concurrently with the LLM
                # stream; the first complete sentence is synthesized while
                # the LLM keeps generating the rest. Phase 6L: in browser mode
                # the sentence stream is forwarded as tts_text events by the
                # text consumer — no ElevenLabs request, no audio segments.
                sentence_sink = _SentenceSink()
                if state.tts_mode == "browser":
                    tts_task = asyncio.create_task(
                        _tts_text_consumer(
                            ws, timings, state, turn, turn_timing, sentence_sink
                        ),
                        name=f"realtime_tts_text_consumer_turn_{turn}",
                    )
                else:
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
                ws=ws,
                state=state,
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
            # Phase 1: the first chunk is what the first TTS request started
            # on; fall back to the first-sentence mark for streams that only
            # report sentences (e.g. older or mocked results).
            turn_timing.llm_first_chunk_at = (
                result.get("llm_first_chunk_at") or turn_timing.llm_first_sentence_at
            )
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

            # Tool-turn timing breakdown: one consolidated line per tool turn
            # (req1 tool-selection stream -> tools -> req2 follow-up).
            if result.get("iterations") == 2:
                logger.info(
                    "[REALTIME:TOOL] turn_metrics turn=%d req1_ms=%.0f "
                    "req1_ttft_ms=%s tool_ms=%.0f req2_ms=%.0f total_ms=%.0f",
                    turn,
                    result.get("req1_ms") or 0.0,
                    _fmt_ms(result.get("first_token_ms")),
                    result.get("tool_ms") or 0.0,
                    result.get("final_ms") or 0.0,
                    result.get("total_ms") or 0.0,
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

            # Phase 6L: browser mode may have no TTS provider (state.tts can
            # be None) but does have a text consumer to await — gate on the
            # task, not the provider.
            if response_text and tts_task is not None:
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
                    cfg, llm_provider, llm_model, tts_mode = _parse_start_message(msg)
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
                    # Phase 6L: response TTS mode for this session ("elevenlabs"
                    # keeps the Phase 6K audio pipeline; "browser" forwards
                    # tts_text and the browser speaks it).
                    state.tts_mode = tts_mode
                    # Phase 2: incremental audio delivery — enabled only when
                    # the browser advertised MediaSource MP3 support in START
                    # ("audio_stream": true). Absent/false keeps the
                    # complete-MP3 messages for this session.
                    state.audio_stream = msg.get("audio_stream") is True

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
                    "llm_provider=%s llm_model=%s tts_mode=%s",
                    connection_id,
                    state.voice_session_id,
                    state.stt_session.provider_name,
                    cfg.model,
                    cfg.sample_rate,
                    state.llm_provider,
                    state.llm_model,
                    state.tts_mode,
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

            elif msg_type == "tts_mode":
                # Phase 6L: mid-session TTS mode switch. Each turn's consumer
                # reads state.tts_mode at creation, so the change applies to
                # the NEXT turn — an in-flight ElevenLabs response keeps
                # playing; the browser cancels its own live speech on switch.
                mode = msg.get("mode")
                if mode not in ("elevenlabs", "browser"):
                    await _safe_ws_send(
                        ws,
                        state,
                        {
                            "type": "error",
                            "message": "tts_mode must be 'elevenlabs' or 'browser'",
                        },
                    )
                elif mode != state.tts_mode:
                    state.tts_mode = mode
                    logger.info(
                        "[VOICE:REALTIME] tts_mode_changed connection_id=%s mode=%s",
                        connection_id,
                        mode,
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
