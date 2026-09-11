"""Tool call model for tracking tool invocations."""

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class ToolCall(UUIDMixin, TimestampMixin, Base):
    """Records a single tool invocation within a session."""

    __tablename__ = "tool_calls"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("voice_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_input: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    tool_output: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
    )  # pending, success, error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
