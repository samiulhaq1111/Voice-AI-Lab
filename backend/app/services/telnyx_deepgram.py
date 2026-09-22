"""Telnyx → Deepgram realtime STT bridge (Phase 6C Milestone 5/6).

Connects Telnyx media stream audio to Deepgram for realtime transcription.
Accumulates final transcripts and emits complete utterances on utterance_end
for downstream agent processing.

Architecture:
    Telnyx WebSocket (PCMU 8kHz)
        ↓
    MediaStreamSession.handle_message()
        ↓ base64 decode → PCMU bytes
        ↓
    pcmu_8k_to_pcm_16k() [audio.py]
        ↓ linear PCM 16kHz
        ↓
    DeepgramStreamingSession.send_audio()
        ↓
    DeepgramStreamingSession.receive()
        ↓ StreamEvent (partial/final/utterance_end)
        ↓
    Utterance accumulation → utterance_queue
        ↓
    Telephony agent worker (consumes from utterance_queue)

The bridge uses async tasks to keep the Telnyx receive loop responsive
while Deepgram transcript events are processed concurrently.
"""

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import logger
from app.providers.stt.streaming import (
    StreamConfig,
    StreamEvent,
    StreamingSTTError,
    StreamingSTTSession,
    open_streaming_session,
)
from app.services.telephony_events import broadcast_telephony_event
from app.utils.audio import pcmu_8k_to_pcm_16k

# Queue size for audio packets (bounded to prevent memory growth)
# 100 packets × 20ms = 2 seconds of audio buffer
_AUDIO_QUEUE_MAX_SIZE = 100

# Queue size for completed utterances (bounded to prevent memory growth)
_UTTERANCE_QUEUE_MAX_SIZE = 20


@dataclass(frozen=True)
class Utterance:
    """A completed caller utterance with end-of-speech timing marks.

    Carries monotonic timestamps so the agent worker can compute
    speech_final → utterance_end → agent_processing deltas without
    depending on wall-clock alignment.
    """

    text: str
    speech_final_at: float | None  # monotonic, None if no speech_final seen
    utterance_end_at: float  # monotonic
    last_word_end: float | None  # Deepgram audio-time seconds, None if absent


class TelnyxDeepgramBridge:
    """Bridge between Telnyx media stream and Deepgram realtime STT.

    Manages the lifecycle of a Deepgram streaming session, handles
    audio conversion + forwarding, accumulates final transcripts,
    and emits complete utterances on utterance_end.
    """

    def __init__(self) -> None:
        self._session: StreamingSTTSession | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=_AUDIO_QUEUE_MAX_SIZE
        )
        # Completed utterances ready for agent processing
        self.utterance_queue: asyncio.Queue[Utterance | None] = asyncio.Queue(
            maxsize=_UTTERANCE_QUEUE_MAX_SIZE
        )
        self._audio_task: asyncio.Task | None = None
        self._transcript_task: asyncio.Task | None = None
        self._running = False
        self._stream_id: str = ""
        self._packets_in: int = 0
        self._bytes_converted: int = 0
        self._partials: int = 0
        self._finals: int = 0
        self._utterances_emitted: int = 0
        self._outbound_packets_skipped: int = 0
        # Utterance accumulation state
        self._current_utterance_text: str = ""
        self._utterance_finalized: bool = False
        # End-of-speech timing marks for the current accumulation
        self._speech_final_at: float | None = None
        self._last_word_end: float | None = None

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def packets_in(self) -> int:
        return self._packets_in

    @property
    def bytes_converted(self) -> int:
        return self._bytes_converted

    @property
    def partials(self) -> int:
        return self._partials

    @property
    def finals(self) -> int:
        return self._finals

    @property
    def utterances_emitted(self) -> int:
        return self._utterances_emitted

    async def start(self, stream_id: str = "") -> None:
        """Start the Deepgram streaming session and async tasks.

        Args:
            stream_id: Telnyx stream ID for logging.
        """
        if self._running:
            logger.warning("[VOICE:TELNYX:STT] Bridge already started")
            return

        self._stream_id = stream_id
        self._running = True

        # Create Deepgram session with telephony-appropriate config
        config = StreamConfig(
            model=None,  # Use default from settings
            language="en",
            sample_rate=16000,  # After conversion
            channels=1,
            encoding="linear16",
            interim_results=True,
            endpointing_ms=300,
            utterance_end_ms=1000,
        )

        try:
            self._session = open_streaming_session(config)
            await self._session.start()
            logger.debug(
                "[VOICE:TELNYX:STT] Deepgram session started "
                "stream_id=%s provider=%s",
                stream_id,
                self._session.provider_name,
            )
        except StreamingSTTError as e:
            self._running = False
            logger.error(
                "[VOICE:TELNYX:STT] Deepgram session failed to start "
                "error=%s",
                e,
            )
            raise

        # Start async tasks for audio forwarding and transcript processing
        self._audio_task = asyncio.create_task(self._audio_forwarder())
        self._transcript_task = asyncio.create_task(self._transcript_processor())

        logger.info("[VOICE:TELNYX:STT] streaming started")

    async def stop(self) -> None:
        """Stop the bridge and clean up resources."""
        if not self._running:
            return

        self._running = False
        logger.info(
            "[VOICE:TELNYX:STT] stopping "
            "inbound_packets=%d outbound_skipped=%d "
            "audio_bytes=%d finals=%d utterances=%d",
            self._packets_in,
            self._outbound_packets_skipped,
            self._bytes_converted,
            self._finals,
            self._utterances_emitted,
        )

        # Signal audio forwarder to stop
        await self._audio_queue.put(b"")  # Sentinel

        # Wait for tasks to complete
        if self._audio_task:
            try:
                await asyncio.wait_for(self._audio_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._audio_task.cancel()
            self._audio_task = None

        if self._transcript_task:
            try:
                await asyncio.wait_for(self._transcript_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._transcript_task.cancel()
            self._transcript_task = None

        # Close Deepgram session
        if self._session:
            try:
                await self._session.finish()
            except Exception as e:
                logger.warning("[VOICE:TELNYX:STT] finish() error=%s", e)
            try:
                await self._session.close()
            except Exception as e:
                logger.warning("[VOICE:TELNYX:STT] close() error=%s", e)
            self._session = None

        logger.debug("[VOICE:TELNYX:STT] Bridge stopped stream_id=%s", self._stream_id)

    async def process_media_packet(self, media_data: dict[str, Any]) -> None:
        """Process an inbound Telnyx media packet.

        Only INBOUND track audio is sent to Deepgram for STT.
        Outbound track audio (greeting, test tone) is ignored to
        prevent the assistant's own audio from being transcribed.

        Args:
            media_data: The "media" field from a Telnyx media event.
        """
        if not self._running:
            return

        # Track filtering: only process caller (inbound) audio
        track = media_data.get("track", "")
        if track and track != "inbound":
            # Outbound/unknown track — skip STT, count for diagnostics
            self._outbound_packets_skipped += 1
            return

        payload_b64 = media_data.get("payload", "")
        if not payload_b64:
            logger.warning("[VOICE:TELNYX:STT] Empty media payload ignored")
            return

        try:
            # Decode base64 → PCMU bytes
            pcmu_bytes = base64.b64decode(payload_b64)
        except Exception as e:
            logger.warning(
                "[VOICE:TELNYX:STT] base64 decode failed error=%s", e
            )
            return

        if not pcmu_bytes:
            return

        self._packets_in += 1

        # Convert PCMU 8kHz → PCM 16kHz
        pcm_bytes = pcmu_8k_to_pcm_16k(pcmu_bytes)
        self._bytes_converted += len(pcm_bytes)

        # Log first packet only at debug level
        if self._packets_in == 1:
            logger.debug(
                "[VOICE:TELNYX:STT] first_packet pcmu_bytes=%d pcm_bytes=%d",
                len(pcmu_bytes),
                len(pcm_bytes),
            )

        # Queue for forwarding (non-blocking)
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            logger.warning(
                "[VOICE:TELNYX:STT] Audio queue full — dropping packet "
                "queue_size=%d",
                self._audio_queue.qsize(),
            )

    async def _audio_forwarder(self) -> None:
        """Forward audio from queue to Deepgram session."""
        logger.debug("[VOICE:TELNYX:STT] Audio forwarder started")
        while self._running:
            try:
                pcm_bytes = await self._audio_queue.get()

                # Sentinel — shutdown signal
                if not pcm_bytes:
                    logger.debug("[VOICE:TELNYX:STT] Audio forwarder received sentinel")
                    break

                if self._session:
                    try:
                        await self._session.send_audio(pcm_bytes)
                    except StreamingSTTError as e:
                        logger.error(
                            "[VOICE:TELNYX:STT] Deepgram send_audio failed "
                            "error=%s",
                            e,
                        )
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "[VOICE:TELNYX:STT] Audio forwarder error=%s", e
                )
                break

        logger.debug("[VOICE:TELNYX:STT] Audio forwarder stopped")

    async def _transcript_processor(self) -> None:
        """Process transcript events from Deepgram.

        Accumulates final transcripts and emits complete utterances
        on utterance_end for downstream agent processing.
        """
        logger.debug("[VOICE:TELNYX:STT] Transcript processor started")
        while self._running and self._session:
            try:
                event = await self._session.receive()

                if event is None:
                    # Provider stream ended
                    logger.debug(
                        "[VOICE:TELNYX:STT] Deepgram stream ended "
                        "partials=%d finals=%d utterances=%d",
                        self._partials,
                        self._finals,
                        self._utterances_emitted,
                    )
                    break

                if event.type == "partial":
                    self._partials += 1
                    logger.debug(
                        "[VOICE:TELNYX:STT] partial text='%s' confidence=%.2f",
                        event.text[:80],
                        event.confidence,
                    )
                elif event.type == "final":
                    self._finals += 1
                    # Accumulate final transcript into current utterance
                    prev = self._current_utterance_text
                    self._current_utterance_text = (
                        f"{prev} {event.text}".strip() if prev else event.text
                    )
                    self._utterance_finalized = True
                    # Capture the FIRST speech_final mark for this accumulation
                    # (multi-segment utterances may have multiple speech_final finals;
                    # we anchor on the first one to avoid re-timing).
                    if (
                        self._speech_final_at is None
                        and event.metadata.get("speech_final")
                    ):
                        self._speech_final_at = time.monotonic()
                    logger.debug(
                        "[VOICE:TELNYX:STT] final text='%s' confidence=%.2f",
                        event.text[:120],
                        event.confidence,
                    )
                elif event.type == "utterance_end":
                    # Emit the accumulated utterance for agent processing
                    utterance_text = self._current_utterance_text.strip()
                    utterance_end_at = time.monotonic()
                    last_word_end = event.metadata.get("last_word_end")
                    if utterance_text and self._utterance_finalized:
                        self._utterances_emitted += 1
                        utterance = Utterance(
                            text=utterance_text,
                            speech_final_at=self._speech_final_at,
                            utterance_end_at=utterance_end_at,
                            last_word_end=last_word_end,
                        )
                        try:
                            self.utterance_queue.put_nowait(utterance)
                        except asyncio.QueueFull:
                            logger.warning(
                                "[VOICE:TELNYX:STT] Utterance queue full "
                                "— dropping utterance queue_size=%d",
                                self.utterance_queue.qsize(),
                            )
                        logger.info(
                            "[VOICE:TELNYX:STT] turn=%d transcript=\"%s\"",
                            self._utterances_emitted,
                            utterance_text[:120],
                        )
                        # Emit caller_speech_final with timing metadata.
                        # Only emitted when speech_final was observed for this
                        # accumulation — the frontend uses this to distinguish
                        # a clean end-of-speech from a forced flush.
                        speech_final_to_utterance_end_ms: float | None = None
                        if self._speech_final_at is not None:
                            speech_final_to_utterance_end_ms = (
                                utterance_end_at - self._speech_final_at
                            ) * 1000
                        await broadcast_telephony_event(
                            "caller_speech_final",
                            self._stream_id,
                            "Deepgram detected end of speech",
                            turn=self._utterances_emitted,
                            metadata={
                                "last_word_end": last_word_end,
                                "speech_final_to_utterance_end_ms": (
                                    round(speech_final_to_utterance_end_ms)
                                    if speech_final_to_utterance_end_ms is not None
                                    else None
                                ),
                            },
                        )
                        await broadcast_telephony_event(
                            "caller_transcript",
                            self._stream_id,
                            f'Caller: "{utterance_text}"',
                            turn=self._utterances_emitted,
                            metadata={"text": utterance_text},
                        )
                    else:
                        logger.debug(
                            "[VOICE:TELNYX:STT] utterance_end (empty)"
                        )
                    # Reset utterance accumulation for next utterance
                    self._current_utterance_text = ""
                    self._utterance_finalized = False
                    self._speech_final_at = None
                    self._last_word_end = None
                elif event.type == "error":
                    logger.error(
                        "[VOICE:TELNYX:STT] Deepgram error text='%s'",
                        event.text[:200],
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "[VOICE:TELNYX:STT] Transcript processor error=%s", e
                )
                break

        logger.debug("[VOICE:TELNYX:STT] Transcript processor stopped")

    def get_transcript_event(self) -> StreamEvent | None:
        """Get the next transcript event from the queue (non-blocking).

        Returns None if no event is available.
        Deprecated: utterance_queue is now the primary output mechanism.
        """
        return None
