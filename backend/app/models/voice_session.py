"""Voice session model."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class VoiceSession(UUIDMixin, TimestampMixin, Base):
    """Represents a single voice conversation session."""

    __tablename__ = "voice_sessions"

    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Provider/model selections for this session
    stt_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    stt_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tts_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tts_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tts_voice: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Metrics
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
