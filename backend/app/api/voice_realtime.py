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
    AgentRuntime (on final utterance)
        ↓
    agent_response event → browser

Client → Server protocol:
    {"type": "start", "sample_rate": 16000, "channels": 1,
     "encoding": "linear16", "language": "en", "model": "nova-3",
     "llm_provider": "openrouter", "llm_model": "openai/gpt-4o-mini"}
    <binary PCM audio chunks>
    {"type": "stop"}

Server → Client protocol:
    {"type": "session_started", "session_id": "...", "provider": "...",
     "model": "...", "sample_rate": 16000}
    {"type": "transcript_partial", "text": "...", "confidence": 0.9}
    {"type": "transcript_final", "text": "...", "confidence": 0.95}
    {"type": "utterance_end"}
    {"type": "agent_processing"}
    {"type": "agent_response", "text": "...", "tool_calls": 0, "iterations": 1}
    {"type": "completed", "timings": {...}}
    {"type": "error", "stage": "stt"|"agent", "message": "..."}
"""

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

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
from app.services.realtime_voice_service import (
    RealtimeVoiceError,
    create_realtime_session,
    process_realtime_utterance,
)

router = APIRouter()

# How long to wait for the provider stream to drain after STOP
_STOP_DRAIN_TIMEOUT_S = 5.0
# Log every Nth audio chunk (avoid per-frame log spam)
_AUDIO_LOG_EVERY_N = 100


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
    agent_response_count: int = 0
    tts_response_count: int = 0
    tts_audio_bytes: int = 0
    tts_characters: int = 0

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
            "agent_response_count": self.agent_response_count,
            "tts_response_count": self.tts_response_count,
            "tts_audio_bytes": self.tts_audio_bytes,
            "tts_characters": self.tts_characters,
        }


@dataclass
class RealtimeSessionState:
    """Mutable state for one realtime voice session."""

    stop_requested: bool = False
    voice_session_id: str | None = None
    llm_provider: str = ""
    llm_model: str | None = None
    llm: LLMInterface | None = None
    tts: TTSInterface | None = None
    # Current utterance accumulation (pump writes, worker reads snapshot)
    current_utterance_text: str = ""
    utterance_finalized: bool = False  # True after final transcript received
    agent_processing: bool = False  # True while AgentRuntime is running
    # Queue-based serialized utterance processing
    utterance_queue: asyncio.Queue[str | None] = field(
        default_factory=lambda: asyncio.Queue()
    )
    # Database session (one per WebSocket connection)
    db: Session | None = None


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

    # LLM configuration (optional)
    llm_provider = msg.get("llm_provider", "") or ""
    llm_model = msg.get("llm_model", "") or None

    if not isinstance(sample_rate, int):
        raise ValueError("sample_rate must be an integer")
    if not isinstance(channels, int):
        raise ValueError("channels must be an integer")
    if not isinstance(language, str) or not language:
        raise ValueError("language must be a non-empty string")

    cfg = StreamConfig(
        model=model,
        language=language,
        sample_rate=sample_rate,
        channels=channels,
        encoding=encoding,
    )
    error = cfg.validate()
    if error:
        raise ValueError(error)
    return cfg, llm_provider, llm_model


async def _pump_provider_events(
    ws: WebSocket,
    session,  # StreamingSTTSession
    timings: RealtimeTimings,
    state: RealtimeSessionState,
) -> None:
    """Forward provider transcript events to the client until the stream ends.

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
                    try:
                        await ws.send_json(
                            {
                                "type": "error",
                                "stage": "stt",
                                "message": "Provider stream ended unexpectedly",
                            }
                        )
                    except Exception:
                        pass
                break

            if event.type == "partial":
                timings.partial_count += 1
                if timings.first_partial_at is None:
                    timings.first_partial_at = time.monotonic()
                await ws.send_json(
                    {
                        "type": "transcript_partial",
                        "text": event.text,
                        "confidence": round(event.confidence, 4),
                    }
                )
            elif event.type == "final":
                timings.final_count += 1
                if timings.first_final_at is None:
                    timings.first_final_at = time.monotonic()
                # Accumulate final transcript for current utterance
                prev = state.current_utterance_text
                state.current_utterance_text = (
                    f"{prev} {event.text}".strip() if prev else event.text
                )
                state.utterance_finalized = True
                logger.info(
                    "[REALTIME:STT] transcript_final text='%s' buffer='%s'",
                    event.text[:80],
                    state.current_utterance_text[:80],
                )
                await ws.send_json(
                    {
                        "type": "transcript_final",
                        "text": event.text,
                        "confidence": round(event.confidence, 4),
                    }
                )
            elif event.type == "utterance_end":
                timings.utterance_end_count += 1
                if timings.first_utterance_end_at is None:
                    timings.first_utterance_end_at = time.monotonic()
                logger.info(
                    "[REALTIME:AGENT] utterance_end received from provider "
                    "buffer='%s' finalized=%s",
                    state.current_utterance_text[:80],
                    state.utterance_finalized,
                )
                await ws.send_json({"type": "utterance_end"})
                # Capture the finalized utterance and enqueue for processing.
                # The pump must NOT block here — the agent worker processes
                # utterances from the queue in a separate task.
                if state.utterance_finalized and state.current_utterance_text:
                    transcript = state.current_utterance_text
                    state.current_utterance_text = ""
                    state.utterance_finalized = False
                    logger.info(
                        "[REALTIME:AGENT] enqueue_utterance text='%s' "
                        "queue_size=%d",
                        transcript[:80],
                        state.utterance_queue.qsize(),
                    )
                    state.utterance_queue.put_nowait(transcript)
                else:
                    # Empty utterance — reset state, do not enqueue
                    state.current_utterance_text = ""
                    state.utterance_finalized = False
                    logger.info(
                        "[REALTIME:AGENT] skipping empty utterance_end"
                    )
            elif event.type == "error":
                await ws.send_json(
                    {"type": "error", "stage": "stt", "message": event.text}
                )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(
            "[VOICE:REALTIME] Provider event pump failed error_type=%s",
            type(e).__name__,
        )
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "stage": "stt",
                    "message": "Realtime STT stream error",
                }
            )
        except Exception:
            pass


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
            transcript = await state.utterance_queue.get()
        except asyncio.CancelledError:
            logger.info("[REALTIME:WORKER] Worker cancelled while waiting")
            break

        # None sentinel — shutdown signal
        if transcript is None:
            state.utterance_queue.task_done()
            logger.info("[REALTIME:WORKER] Worker received shutdown sentinel")
            break

        state.agent_processing = True
        logger.info(
            "[REALTIME:WORKER] processing utterance text='%s' length=%d "
            "queue_remaining=%d",
            transcript[:80],
            len(transcript),
            state.utterance_queue.qsize(),
        )

        try:
            # Send agent_processing event
            await ws.send_json({"type": "agent_processing"})
            logger.info("[REALTIME:WORKER] agent_processing event sent")

            if state.db is None or state.voice_session_id is None:
                logger.error("[REALTIME:WORKER] cannot process — no session")
                await ws.send_json(
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": "Session not initialized",
                    }
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
                logger.error("[REALTIME:WORKER] voice session not found")
                await ws.send_json(
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": "Session not found",
                    }
                )
                continue

            # Process utterance through AgentRuntime
            logger.info("[REALTIME:WORKER] calling RealtimeVoiceService")
            result = await process_realtime_utterance(
                db=state.db,
                session=voice_session,
                transcript=transcript,
                llm=state.llm,
            )
            logger.info("[REALTIME:WORKER] AgentRuntime returned")

            timings.agent_response_count += 1
            logger.info(
                "[REALTIME:WORKER] agent response iterations=%d tool_calls=%d",
                result["iterations"],
                len(result["tool_calls"]),
            )

            # Send agent_response event
            await ws.send_json(
                {
                    "type": "agent_response",
                    "text": result["response"],
                    "tool_calls": len(result["tool_calls"]),
                    "iterations": result["iterations"],
                }
            )
            logger.info("[REALTIME:WORKER] agent_response sent")

            # --- TTS synthesis (Phase 6B.2) ---
            response_text = result["response"]
            if response_text and state.tts is not None:
                try:
                    await ws.send_json({"type": "tts_processing"})
                    logger.info(
                        "[REALTIME:WORKER] tts_processing event sent "
                        "text_length=%d",
                        len(response_text),
                    )

                    tts_start = time.monotonic()
                    tts_result = await state.tts.synthesize(
                        text=response_text,
                    )
                    tts_duration_ms = (time.monotonic() - tts_start) * 1000

                    audio_b64 = base64.b64encode(
                        tts_result.audio_data
                    ).decode("ascii")

                    await ws.send_json(
                        {
                            "type": "audio",
                            "format": tts_result.content_type,
                            "data": audio_b64,
                        }
                    )

                    timings.tts_response_count += 1
                    timings.tts_audio_bytes += len(tts_result.audio_data)
                    timings.tts_characters += len(response_text)

                    logger.info(
                        "[REALTIME:WORKER] audio sent format=%s "
                        "audio_bytes=%d characters=%d tts_latency_ms=%.0f",
                        tts_result.content_type,
                        len(tts_result.audio_data),
                        len(response_text),
                        tts_duration_ms,
                    )

                except Exception as e:
                    logger.error(
                        "[REALTIME:WORKER] TTS synthesis failed "
                        "error_type=%s error=%s",
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                    try:
                        await ws.send_json(
                            {
                                "type": "error",
                                "stage": "tts",
                                "message": f"TTS synthesis failed: {e}",
                            }
                        )
                    except Exception:
                        pass
            elif response_text and state.tts is None:
                logger.warning(
                    "[REALTIME:WORKER] TTS skipped — no TTS provider"
                )

        except RealtimeVoiceError as e:
            logger.error(
                "[REALTIME:WORKER] utterance processing failed error=%s",
                e.message,
            )
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": e.message,
                    }
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(
                "[REALTIME:WORKER] utterance processing failed "
                "error_type=%s",
                type(e).__name__,
                exc_info=True,
            )
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "stage": "agent",
                        "message": "Agent processing failed",
                    }
                )
            except Exception:
                pass
        finally:
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
    """Handle one realtime voice client connection end-to-end."""
    session = None
    pump_task: asyncio.Task | None = None
    worker_task: asyncio.Task | None = None
    timings = RealtimeTimings()
    connection_id = str(uuid.uuid4())
    state = RealtimeSessionState()
    db: Session | None = None

    try:
        while True:
            raw = await ws.receive()

            # ----- Binary audio chunk -----
            if "bytes" in raw:
                if session is None:
                    # Safe: ignore audio before START
                    continue
                chunk = raw["bytes"]
                timings.audio_chunks += 1
                timings.audio_bytes += len(chunk)
                if timings.first_audio_at is None:
                    timings.first_audio_at = time.monotonic()
                    logger.info(
                        "[VOICE:REALTIME] client_audio chunk=1 bytes=%d (first)",
                        len(chunk),
                    )
                elif timings.audio_chunks % _AUDIO_LOG_EVERY_N == 0:
                    logger.info(
                        "[VOICE:REALTIME] client_audio chunk=%d bytes=%d "
                        "total_bytes=%d",
                        timings.audio_chunks,
                        len(chunk),
                        timings.audio_bytes,
                    )
                try:
                    await session.send_audio(chunk)
                except StreamingSTTError as e:
                    await ws.send_json(
                        {"type": "error", "stage": "stt", "message": str(e)}
                    )
                    break
                continue

            if "text" not in raw:
                continue

            # ----- JSON control message -----
            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start":
                if session is not None:
                    await ws.send_json(
                        {"type": "error", "message": "Session already started"}
                    )
                    continue

                try:
                    cfg, llm_provider, llm_model = _parse_start_message(msg)
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue

                try:
                    session = open_streaming_session(cfg)
                    await session.start()
                except (StreamingSTTError, ValueError) as e:
                    logger.error(
                        "[VOICE:REALTIME] Session start failed error_type=%s",
                        type(e).__name__,
                    )
                    await ws.send_json(
                        {
                            "type": "error",
                            "stage": "stt",
                            "message": f"Failed to start stream: {e}",
                        }
                    )
                    session = None
                    continue

                # Create database session and voice session
                db = SessionLocal()
                state.db = db
                try:
                    voice_session = create_realtime_session(
                        db=db,
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
                        logger.info(
                            "[VOICE:REALTIME] TTS provider created "
                            "provider=%s",
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
                    await session.close()
                    session = None
                    db.close()
                    db = None
                    state.db = None
                    await ws.send_json(
                        {
                            "type": "error",
                            "stage": "agent",
                            "message": "Failed to create session",
                        }
                    )
                    continue

                timings.session_started_at = time.monotonic()
                logger.info(
                    "[VOICE:REALTIME] Session started connection_id=%s "
                    "voice_session_id=%s provider=%s model=%s sample_rate=%d "
                    "llm_provider=%s llm_model=%s",
                    connection_id,
                    state.voice_session_id,
                    session.provider_name,
                    cfg.model,
                    cfg.sample_rate,
                    state.llm_provider,
                    state.llm_model,
                )
                await ws.send_json(
                    {
                        "type": "session_started",
                        "session_id": state.voice_session_id or connection_id,
                        "provider": session.provider_name,
                        "model": cfg.model,
                        "sample_rate": cfg.sample_rate,
                        "encoding": cfg.encoding,
                    }
                )
                pump_task = asyncio.create_task(
                    _pump_provider_events(ws, session, timings, state)
                )
                # Start the serialized agent processing worker.
                # This runs independently from the pump so that Deepgram
                # event reception is never blocked by AgentRuntime.
                worker_task = asyncio.create_task(
                    _agent_worker(ws, timings, state)
                )

            elif msg_type == "stop":
                if session is None:
                    await ws.send_json(
                        {"type": "error", "message": "No active session"}
                    )
                    continue

                state.stop_requested = True
                logger.info(
                    "[VOICE:REALTIME] stop_received chunks=%d bytes=%d",
                    timings.audio_chunks,
                    timings.audio_bytes,
                )

                # Signal end-of-audio; provider flushes remaining finals
                try:
                    await session.finish()
                except StreamingSTTError as e:
                    logger.warning(
                        "[VOICE:REALTIME] finish() failed error_type=%s",
                        type(e).__name__,
                    )

                # Wait for the pump to drain remaining events
                if pump_task is not None:
                    logger.info("[VOICE:REALTIME] waiting_for_pump_drain")
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(pump_task), timeout=_STOP_DRAIN_TIMEOUT_S
                        )
                    except (TimeoutError, asyncio.CancelledError, Exception):
                        pump_task.cancel()

                # Wait for the agent processing queue to drain.
                # This ensures pending utterances are processed before shutdown.
                if worker_task is not None:
                    logger.info(
                        "[VOICE:REALTIME] waiting_for_agent_queue "
                        "queue_size=%d",
                        state.utterance_queue.qsize(),
                    )
                    try:
                        await asyncio.wait_for(
                            state.utterance_queue.join(),
                            timeout=60.0,  # generous: LLM+tools can be slow
                        )
                    except (TimeoutError, asyncio.CancelledError, Exception):
                        logger.warning(
                            "[VOICE:REALTIME] agent queue drain timed out"
                        )
                    # Send shutdown sentinel to stop the worker
                    state.utterance_queue.put_nowait(None)
                    try:
                        await asyncio.wait_for(worker_task, timeout=5.0)
                    except (TimeoutError, asyncio.CancelledError, Exception):
                        worker_task.cancel()

                await session.close()
                session = None
                pump_task = None
                worker_task = None

                timings.completed_at = time.monotonic()
                logger.info(
                    "[VOICE:REALTIME] Session completed connection_id=%s",
                    connection_id,
                )
                await ws.send_json(
                    {"type": "completed", "timings": timings.to_dict()}
                )

            else:
                await ws.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info(
            "[VOICE:REALTIME] Client disconnected connection_id=%s",
            connection_id,
        )
    except Exception as e:
        logger.error(
            "[VOICE:REALTIME] Gateway error connection_id=%s error_type=%s",
            connection_id,
            type(e).__name__,
        )
        try:
            await ws.send_json({"type": "error", "message": "Internal realtime error"})
        except Exception:
            pass
    finally:
        # Always clean up: cancel worker, cancel pump, close providers
        # Send shutdown sentinel to worker first (graceful)
        if worker_task is not None:
            try:
                state.utterance_queue.put_nowait(None)
            except Exception:
                pass
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
        if pump_task is not None:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
        if session is not None:
            await session.close()
        if state.llm is not None:
            await state.llm.close()
        if state.tts is not None:
            await state.tts.close()
        if db is not None:
            db.close()
        logger.info(
            "[VOICE:REALTIME] Resources cleaned connection_id=%s chunks=%d bytes=%d",
            connection_id,
            timings.audio_chunks,
            timings.audio_bytes,
        )


@router.websocket("/api/v1/voice/realtime/ws")
async def realtime_voice_websocket(ws: WebSocket) -> None:
    """Realtime streaming voice over WebSocket (Phase 6B).

    Protocol is documented in this module's docstring. Binary frames are
    raw PCM (linear16, mono, 16 kHz by default) forwarded to the streaming
    STT provider; transcript events are streamed back as JSON. On utterance_end,
    the finalized transcript is processed through AgentRuntime.
    """
    await ws.accept()
    logger.info("[VOICE:REALTIME] WebSocket connection accepted")
    await _handle_realtime_session(ws)
