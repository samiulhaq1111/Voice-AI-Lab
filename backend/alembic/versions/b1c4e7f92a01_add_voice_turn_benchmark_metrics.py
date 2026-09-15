"""add voice turn benchmark metrics

Revision ID: b1c4e7f92a01
Revises: a86d9ad10a02
Create Date: 2026-09-11 22:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1c4e7f92a01"
down_revision: str | None = "a86d9ad10a02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add per-turn metric columns to benchmark_results."""
    with op.batch_alter_table("benchmark_results") as batch_op:
        # Correlation
        batch_op.add_column(sa.Column("turn_id", sa.String(length=36), nullable=True))
        # Providers
        batch_op.add_column(sa.Column("tts_voice", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("tts_output_format", sa.String(length=50), nullable=True))
        # Per-stage latencies
        batch_op.add_column(sa.Column("stt_latency_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("llm_latency_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("tts_latency_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("tool_execution_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("total_processing_ms", sa.Float(), nullable=True))
        # STT usage
        batch_op.add_column(sa.Column("stt_audio_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("stt_audio_duration_seconds", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("transcript_length", sa.Integer(), nullable=True))
        # LLM usage
        batch_op.add_column(sa.Column("prompt_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("completion_tokens", sa.Integer(), nullable=True))
        # TTS usage
        batch_op.add_column(sa.Column("tts_audio_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("tts_characters", sa.Integer(), nullable=True))
        # Failure detail
        batch_op.add_column(sa.Column("error_stage", sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))

    op.create_index(
        "ix_benchmark_results_turn_id",
        "benchmark_results",
        ["turn_id"],
    )


def downgrade() -> None:
    """Remove per-turn metric columns from benchmark_results."""
    op.drop_index("ix_benchmark_results_turn_id", table_name="benchmark_results")

    with op.batch_alter_table("benchmark_results") as batch_op:
        batch_op.drop_column("error_message")
        batch_op.drop_column("error_stage")
        batch_op.drop_column("tts_characters")
        batch_op.drop_column("tts_audio_bytes")
        batch_op.drop_column("completion_tokens")
        batch_op.drop_column("prompt_tokens")
        batch_op.drop_column("transcript_length")
        batch_op.drop_column("stt_audio_duration_seconds")
        batch_op.drop_column("stt_audio_bytes")
        batch_op.drop_column("total_processing_ms")
        batch_op.drop_column("tool_execution_ms")
        batch_op.drop_column("tts_latency_ms")
        batch_op.drop_column("llm_latency_ms")
        batch_op.drop_column("stt_latency_ms")
        batch_op.drop_column("tts_output_format")
        batch_op.drop_column("tts_voice")
        batch_op.drop_column("turn_id")
