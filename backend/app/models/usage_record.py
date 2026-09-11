"""Usage record model for tracking API usage and costs."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class UsageRecord(UUIDMixin, TimestampMixin, Base):
    """Tracks API usage and costs per provider call."""

    __tablename__ = "usage_records"

    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("voice_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    provider_type: Mapped[str] = mapped_column(String(20), nullable=False)  # stt, llm, tts
    provider_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Usage metrics
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
