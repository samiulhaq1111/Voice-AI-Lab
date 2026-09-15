"""Benchmark result model for provider comparison."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class BenchmarkResult(UUIDMixin, TimestampMixin, Base):
    """Records benchmark metrics for a session or provider combination.

    One row represents one complete voice turn. A session may contain
    multiple turns, each distinguished by ``turn_id``.
    """

    __tablename__ = "benchmark_results"

    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("voice_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Per-turn correlation identifier (a session contains many turns)
    turn_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    # Phase 5B benchmark metadata
    scenario_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    benchmark_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Provider/model info
    stt_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    stt_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tts_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tts_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tts_voice: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tts_output_format: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Cost metrics (populated in a later phase once pricing is centralized)
    stt_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Performance metrics
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    session_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_usage: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Per-stage latencies (monotonic clock, milliseconds)
    stt_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tool_execution_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # STT raw usage
    stt_audio_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stt_audio_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    transcript_length: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # LLM raw usage
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # TTS raw usage
    tts_audio_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tts_characters: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Quality metrics
    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # stored as string for flexibility
    conversation_success: Mapped[bool | None] = mapped_column(
        String(5),
        nullable=True,
    )
    # Failure stage: stt | llm | tool | tts | websocket | unknown
    error_stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    errors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
