"""Provider configuration model."""

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class ProviderConfiguration(UUIDMixin, TimestampMixin, Base):
    """Stores provider-specific configuration and credentials references."""

    __tablename__ = "provider_configurations"

    provider_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # stt, llm, tts
    provider_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # deepgram, openrouter, etc.
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)
