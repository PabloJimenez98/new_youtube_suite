"""create ctr_predictions

Spec §4.1's `ctr_predictions`, plus `features`, `partial`, `source` and
the timestamps. Each row is one title/thumbnail candidate and its latest
heuristic score. See docs/decisions/0019-ctr-heuristic-v0.md §3.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CANDIDATE_SOURCES = ("manual", "show_notes")


def upgrade() -> None:
    op.create_table(
        "ctr_predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("upload_id", sa.BigInteger(), sa.ForeignKey("uploads.id", ondelete="CASCADE"), nullable=True),
        sa.Column("candidate_title", sa.Text(), nullable=False),
        sa.Column("candidate_thumbnail_path", sa.Text(), nullable=True),
        sa.Column("predicted_score", sa.Float(), nullable=True),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("partial", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "model_version_id",
            sa.BigInteger(),
            sa.ForeignKey("model_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source", sa.Text(), server_default="manual", nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(f"source IN {CANDIDATE_SOURCES!r}", name="ctr_predictions_source_check"),
    )
    op.create_index("ix_ctr_predictions_upload_id", "ctr_predictions", ["upload_id"])


def downgrade() -> None:
    op.drop_index("ix_ctr_predictions_upload_id", table_name="ctr_predictions")
    op.drop_table("ctr_predictions")
