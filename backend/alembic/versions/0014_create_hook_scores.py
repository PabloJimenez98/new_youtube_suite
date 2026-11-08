"""create hook_scores

Spec §4.1's `hook_scores`, plus `scored_at`, with one row per clip
(UNIQUE `clip_id`, cascading with the clip). See
docs/decisions/0020-hook-heuristic-v0.md §4.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "hook_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("clip_id", sa.BigInteger(), sa.ForeignKey("clips.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("feature_vector", postgresql.JSONB(), nullable=False),
        sa.Column(
            "model_version_id",
            sa.BigInteger(),
            sa.ForeignKey("model_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("scored_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("clip_id", name="hook_scores_clip_id_key"),
    )


def downgrade() -> None:
    op.drop_table("hook_scores")
