"""Voice turn metrics collection and persistence.

Phase 5A: instrument the existing voice pipeline so that every voice turn
produces reliable timing and usage metrics.

Design notes:
- All elapsed durations use ``time.perf_counter()`` (monotonic), never wall clock.
- Persistence is best-effort: a database failure must never break a voice turn.
- No API keys or secrets are ever recorded.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import logger
from app.models.benchmark_result import BenchmarkResult

# Valid failure stages
ERROR_STAGE_STT = "stt"
ERROR_STAGE_LLM = "llm"
ERROR_STAGE_TOOL = "tool"
ERROR_STAGE_TTS = "tts"
ERROR_STAGE_WEBSOCKET = "websocket"
ERROR_STAGE_UNKNOWN = "unknown"

_MAX_ERROR_MESSAGE_LEN = 500


def new_turn_id() -> str:
    """Generate a unique identifier for a single voice turn."""
    return str(uuid.uuid4())


def _sanitize(message: str | None) -> str | None:
    """Truncate an error message. Provider adapters already strip secrets."""
    if not message:
        return None
    return message[:_MAX_ERROR_MESSAGE_LEN]


def _ms(start: float | None, end: float | None) -> float | None:
    """Convert a perf_counter start/end pair into milliseconds."""
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 2)


@dataclass
class VoiceTurnMetrics:
    """Collects timing and usage measurements for one voice turn.

    Timestamps are raw ``perf_counter()`` values used only for elapsed
    calculations — they are never exposed to clients.
    """

    turn_id: str = field(default_factory=new_turn_id)
    session_id: str | None = None

    # --- Monotonic timestamps (internal only) ---
    turn_started_at: float = field(default_factory=time.perf_counter)
    turn_completed_at: float | None = None
    stt_started_at: float | None = None
    stt_completed_at: float | None = None
    llm_started_at: float | None = None
    llm_completed_at: float | None = None
    tts_started_at: float | None = None
    tts_completed_at: float | None = None

    # --- Providers/models ---
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_output_format: str | None = None

    # --- STT usage ---
    stt_audio_bytes: int | None = None
    stt_audio_duration_seconds: float | None = None
    stt_audio_format: str | None = None
    transcript_length: int | None = None

    # --- LLM usage (None when the provider does not report usage) ---
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_iterations: int | None = None

    # --- Tool usage ---
    tool_count: int = 0
    tool_success_count: int = 0
    tool_execution_ms: float = 0.0

    # --- TTS usage ---
    tts_audio_bytes: int | None = None
    tts_characters: int | None = None

    # --- Outcome ---
    success: bool = False
    error_stage: str | None = None
    error_message: str | None = None

    # ------------------------------------------------------------------
    # Stage markers
    # ------------------------------------------------------------------

    def start_stt(self, *, provider: str, model: str, audio_bytes: int, audio_format: str) -> None:
        """Mark the STT stage as started."""
        self.stt_started_at = time.perf_counter()
        self.stt_provider = provider
        self.stt_model = model
        self.stt_audio_bytes = audio_bytes
        self.stt_audio_format = audio_format
        logger.info(
            "[VOICE:BENCH] turn_started turn_id=%s session_id=%s audio_bytes=%d",
            self.turn_id,
            self.session_id,
            audio_bytes,
        )

    def finish_stt(self, *, transcript_length: int, audio_duration_seconds: float | None) -> None:
        """Mark the STT stage as completed."""
        self.stt_completed_at = time.perf_counter()
        self.transcript_length = transcript_length
        self.stt_audio_duration_seconds = audio_duration_seconds
        logger.info(
            "[VOICE:BENCH] stt latency_ms=%s audio_duration_s=%s transcript_length=%d",
            self.stt_latency_ms,
            audio_duration_seconds if audio_duration_seconds is not None else "N/A",
            transcript_length,
        )

    def start_llm(self, *, provider: str, model: str) -> None:
        """Mark the LLM/agent stage as started."""
        self.llm_started_at = time.perf_counter()
        self.llm_provider = provider
        self.llm_model = model

    def finish_llm(self, *, usage: dict[str, int] | None, iterations: int) -> None:
        """Mark the LLM/agent stage as completed."""
        self.llm_completed_at = time.perf_counter()
        self.llm_iterations = iterations
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens") or None
            self.completion_tokens = usage.get("completion_tokens") or None
            self.total_tokens = usage.get("total_tokens") or None
        logger.info(
            "[VOICE:BENCH] llm latency_ms=%s input_tokens=%s output_tokens=%s iterations=%d",
            self.llm_latency_ms,
            self.prompt_tokens if self.prompt_tokens is not None else "N/A",
            self.completion_tokens if self.completion_tokens is not None else "N/A",
            iterations,
        )

    def record_tool_call(self, *, success: bool, duration_ms: float | None = None) -> None:
        """Record a single tool execution."""
        self.tool_count += 1
        if success:
            self.tool_success_count += 1
        if duration_ms:
            self.tool_execution_ms += duration_ms

    def log_tools(self) -> None:
        """Emit the aggregated tool metrics log line."""
        if self.tool_count:
            logger.info(
                "[VOICE:BENCH] tools count=%d success=%d duration_ms=%.2f",
                self.tool_count,
                self.tool_success_count,
                self.tool_execution_ms,
            )

    def start_tts(
        self,
        *,
        provider: str,
        model: str,
        voice: str | None,
        characters: int,
        output_format: str | None = None,
    ) -> None:
        """Mark the TTS stage as started."""
        self.tts_started_at = time.perf_counter()
        self.tts_provider = provider
        self.tts_model = model
        self.tts_voice = voice
        self.tts_characters = characters
        self.tts_output_format = output_format

    def finish_tts(self, *, audio_bytes: int) -> None:
        """Mark the TTS stage as completed."""
        self.tts_completed_at = time.perf_counter()
        self.tts_audio_bytes = audio_bytes
        logger.info(
            "[VOICE:BENCH] tts latency_ms=%s audio_bytes=%d characters=%s",
            self.tts_latency_ms,
            audio_bytes,
            self.tts_characters if self.tts_characters is not None else "N/A",
        )

    def fail(self, stage: str, message: str) -> None:
        """Record a failure for the current turn."""
        self.success = False
        self.error_stage = stage
        self.error_message = _sanitize(message)
        self.turn_completed_at = time.perf_counter()
        logger.info(
            "[VOICE:BENCH] failed turn_id=%s stage=%s total_processing_ms=%s",
            self.turn_id,
            stage,
            self.total_processing_ms,
        )

    def complete(self) -> None:
        """Mark the turn as successfully completed."""
        self.success = True
        self.turn_completed_at = time.perf_counter()
        logger.info(
            "[VOICE:BENCH] completed turn_id=%s total_processing_ms=%s",
            self.turn_id,
            self.total_processing_ms,
        )

    # ------------------------------------------------------------------
    # Derived durations
    # ------------------------------------------------------------------

    @property
    def stt_latency_ms(self) -> float | None:
        """STT elapsed time in milliseconds."""
        return _ms(self.stt_started_at, self.stt_completed_at)

    @property
    def llm_latency_ms(self) -> float | None:
        """LLM/agent elapsed time in milliseconds."""
        return _ms(self.llm_started_at, self.llm_completed_at)

    @property
    def tts_latency_ms(self) -> float | None:
        """TTS elapsed time in milliseconds."""
        return _ms(self.tts_started_at, self.tts_completed_at)

    @property
    def total_processing_ms(self) -> float | None:
        """Total server-side processing time in milliseconds."""
        end = self.turn_completed_at or time.perf_counter()
        return _ms(self.turn_started_at, end)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_event_payload(self) -> dict[str, Any]:
        """Build the payload for the WebSocket ``metrics`` event.

        Contains no secrets and no raw monotonic timestamps.
        """
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "stt_provider": self.stt_provider,
            "stt_model": self.stt_model,
            "stt_latency_ms": self.stt_latency_ms,
            "stt_audio_bytes": self.stt_audio_bytes,
            "stt_audio_duration_seconds": self.stt_audio_duration_seconds,
            "transcript_length": self.transcript_length,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_latency_ms": self.llm_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_iterations": self.llm_iterations,
            "tool_count": self.tool_count,
            "tool_success_count": self.tool_success_count,
            "tool_execution_ms": round(self.tool_execution_ms, 2) if self.tool_count else None,
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
            "tts_voice": self.tts_voice,
            "tts_output_format": self.tts_output_format,
            "tts_latency_ms": self.tts_latency_ms,
            "tts_audio_bytes": self.tts_audio_bytes,
            "tts_characters": self.tts_characters,
            "total_processing_ms": self.total_processing_ms,
            "success": self.success,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
        }

    def to_benchmark_result(self) -> BenchmarkResult:
        """Build (but do not persist) a BenchmarkResult row for this turn."""
        return BenchmarkResult(
            session_id=self.session_id,
            turn_id=self.turn_id,
            stt_provider=self.stt_provider,
            stt_model=self.stt_model,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            tts_provider=self.tts_provider,
            tts_model=self.tts_model,
            tts_voice=self.tts_voice,
            tts_output_format=self.tts_output_format,
            latency_ms=self.total_processing_ms,
            total_processing_ms=self.total_processing_ms,
            stt_latency_ms=self.stt_latency_ms,
            llm_latency_ms=self.llm_latency_ms,
            tts_latency_ms=self.tts_latency_ms,
            tool_execution_ms=round(self.tool_execution_ms, 2) if self.tool_count else None,
            stt_audio_bytes=self.stt_audio_bytes,
            stt_audio_duration_seconds=self.stt_audio_duration_seconds,
            transcript_length=self.transcript_length,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            token_usage=self.total_tokens,
            tts_audio_bytes=self.tts_audio_bytes,
            tts_characters=self.tts_characters,
            tool_calls_count=self.tool_count,
            tool_success_count=self.tool_success_count,
            conversation_success="true" if self.success else "false",
            error_stage=self.error_stage,
            error_message=self.error_message,
        )


def persist_metrics(db: Session, metrics: VoiceTurnMetrics) -> BenchmarkResult | None:
    """Persist turn metrics as a BenchmarkResult row (best effort).

    A persistence failure is logged and swallowed so that the voice
    response always reaches the browser.
    """
    try:
        record = metrics.to_benchmark_result()
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(
            "[VOICE:BENCH] persisted turn_id=%s benchmark_id=%s",
            metrics.turn_id,
            record.id,
        )
        return record
    except Exception as e:
        logger.error(
            "[VOICE:BENCH] persistence failed turn_id=%s error_type=%s",
            metrics.turn_id,
            type(e).__name__,
        )
        try:
            db.rollback()
        except Exception:
            pass
        return None
