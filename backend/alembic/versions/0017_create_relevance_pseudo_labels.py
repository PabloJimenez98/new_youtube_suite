"""create relevance_pseudo_labels

M15's training targets, kept apart from `relevance_scores` (which a re-run
replaces, and which a live trained model would overwrite with its own
outputs). Backfilled from every `relevance_scores` row whose version is a
teacher (no `artifact_path`). See docs/decisions/0024-relevance-bilstm-m15.md §2.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "relevance_pseudo_labels",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("upload_id", sa.BigInteger(), sa.ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window_index", sa.Integer(), nullable=False),
        sa.Column("window_start_seconds", sa.Float(), nullable=False),
        sa.Column("window_end_seconds", sa.Float(), nullable=False),
        sa.Column("heuristic_score", sa.Float(), nullable=False),
        sa.Column("llm_score", sa.Float(), nullable=True),
        sa.Column("teacher_version_id", sa.BigInteger(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "upload_id", "window_index", "teacher_version_id", name="uq_relevance_pseudo_labels_upload_window_teacher"
        ),
    )
    op.create_index("ix_relevance_pseudo_labels_upload_id", "relevance_pseudo_labels", ["upload_id"])
    op.create_index("ix_relevance_pseudo_labels_teacher_version_id", "relevance_pseudo_labels", ["teacher_version_id"])
    op.execute(
        """
        INSERT INTO relevance_pseudo_labels
            (upload_id, window_index, window_start_seconds, window_end_seconds,
             heuristic_score, llm_score, teacher_version_id)
        SELECT rs.upload_id, rs.window_index, rs.window_start_seconds, rs.window_end_seconds,
               rs.heuristic_score, rs.llm_score, rs.model_version_id
        FROM relevance_scores rs
        JOIN model_versions mv ON mv.id = rs.model_version_id
        WHERE mv.artifact_path IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index("ix_relevance_pseudo_labels_teacher_version_id", table_name="relevance_pseudo_labels")
    op.drop_index("ix_relevance_pseudo_labels_upload_id", table_name="relevance_pseudo_labels")
    op.drop_table("relevance_pseudo_labels")
