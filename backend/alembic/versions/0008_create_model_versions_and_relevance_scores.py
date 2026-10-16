"""create model_versions and relevance_scores tables

`model_versions` (spec §4) is created with relevance v0 (M7) rather
than in M15, with at most one live version per model. `relevance_scores`
keeps the heuristic and LLM components next to the combined score. See
docs/decisions/0012-relevance-v0.md §5.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("version_label", sa.Text(), nullable=False),
        sa.Column("architecture_notes", sa.Text(), nullable=True),
        sa.Column("training_data_reference", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("is_live", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("model_name", "version_label", name="uq_model_versions_name_label"),
    )
    op.create_index(
        "uq_model_versions_one_live", "model_versions", ["model_name"],
        unique=True, postgresql_where=sa.text("is_live"),
    )
    op.create_table(
        "relevance_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("upload_id", sa.BigInteger(), sa.ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window_index", sa.Integer(), nullable=False),
        sa.Column("window_start_seconds", sa.Float(), nullable=False),
        sa.Column("window_end_seconds", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("heuristic_score", sa.Float(), nullable=False),
        sa.Column("llm_score", sa.Float(), nullable=True),
        sa.Column("model_version_id", sa.BigInteger(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.UniqueConstraint("upload_id", "window_index", name="uq_relevance_scores_upload_window"),
    )
    op.create_index("ix_relevance_scores_upload_id", "relevance_scores", ["upload_id"])


def downgrade() -> None:
    op.drop_index("ix_relevance_scores_upload_id", table_name="relevance_scores")
    op.drop_table("relevance_scores")
    op.drop_index("uq_model_versions_one_live", table_name="model_versions")
    op.drop_table("model_versions")
