"""Benchmark result model for provider comparison."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class BenchmarkResult(UUIDMixin, TimestampMixin, Base):
    """Records benchmark metrics for a session or provider combination."""

    __tablename__ = "benchmark_results"

    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("voice_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Provider/model info
    stt_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    stt_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tts_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tts_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Cost metrics
    stt_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Performance metrics
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    session_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_usage: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Quality metrics
    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # stored as string for flexibility
    conversation_success: Mapped[bool | None] = mapped_column(
        String(5),
        nullable=True,
    )
    errors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
