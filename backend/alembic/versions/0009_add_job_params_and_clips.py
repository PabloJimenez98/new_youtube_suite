"""add jobs.params and create clips

`jobs.params` carries per-run tool options (M8: the shorts aspect ratio
and caption toggle). `clips` is spec §4's table plus rank, score, the
peak window, the relevance version used and the framing summary. See
docs/decisions/0013-shorts-generator.md §5-6.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ASPECT_RATIOS = ("9:16", "1:1", "4:5", "16:9")


def upgrade() -> None:
    op.add_column("jobs", sa.Column("params", postgresql.JSONB(), nullable=True))
    op.create_table(
        "clips",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("upload_id", sa.BigInteger(), sa.ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("peak_window_index", sa.Integer(), nullable=False),
        sa.Column("aspect_ratio", sa.Text(), nullable=False),
        sa.Column("caption_burned_in", sa.Boolean(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("crop_bounding_box_track", postgresql.JSONB(), nullable=False),
        sa.Column("framing", postgresql.JSONB(), nullable=False),
        sa.Column("model_version_id", sa.BigInteger(), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("upload_id", "rank", name="uq_clips_upload_rank"),
        sa.CheckConstraint(f"aspect_ratio IN {ASPECT_RATIOS!r}", name="clips_aspect_ratio_check"),
    )
    op.create_index("ix_clips_upload_id", "clips", ["upload_id"])


def downgrade() -> None:
    op.drop_index("ix_clips_upload_id", table_name="clips")
    op.drop_table("clips")
    op.drop_column("jobs", "params")
