"""Add benchmark scenario fields.

Phase 5B: scenario_id, run_id, benchmark_mode columns on benchmark_results.

Revision ID: c2d5e8f31b02
Revises: b1c4e7f92a01
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "c2d5e8f31b02"
down_revision = "b1c4e7f92a01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("benchmark_results") as batch_op:
        batch_op.add_column(sa.Column("scenario_id", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("run_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("benchmark_mode", sa.String(length=50), nullable=True))

    op.create_index("ix_benchmark_results_scenario_id", "benchmark_results", ["scenario_id"])
    op.create_index("ix_benchmark_results_run_id", "benchmark_results", ["run_id"])


def downgrade() -> None:
    with op.batch_alter_table("benchmark_results") as batch_op:
        batch_op.drop_index("ix_benchmark_results_run_id")
        batch_op.drop_index("ix_benchmark_results_scenario_id")
        batch_op.drop_column("benchmark_mode")
        batch_op.drop_column("run_id")
        batch_op.drop_column("scenario_id")
