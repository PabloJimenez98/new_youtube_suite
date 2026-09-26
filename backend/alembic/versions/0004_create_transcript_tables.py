"""create transcript_segments and transcript_words tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "upload_id",
            sa.BigInteger(),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("avg_logprob", sa.Float(), nullable=True),
        sa.Column("no_speech_prob", sa.Float(), nullable=True),
        sa.Column("speaker_label", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "upload_id", "segment_index", name="uq_transcript_segments_upload_index"
        ),
    )
    op.create_index(
        "ix_transcript_segments_upload_id", "transcript_segments", ["upload_id"]
    )

    op.create_table(
        "transcript_words",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "segment_id",
            sa.BigInteger(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("word_index", sa.Integer(), nullable=False),
        sa.Column("word", sa.Text(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=True),
        sa.Column("end_seconds", sa.Float(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("speaker_label", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "segment_id", "word_index", name="uq_transcript_words_segment_index"
        ),
    )
    op.create_index(
        "ix_transcript_words_segment_id", "transcript_words", ["segment_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_transcript_words_segment_id", table_name="transcript_words")
    op.drop_table("transcript_words")
    op.drop_index(
        "ix_transcript_segments_upload_id", table_name="transcript_segments"
    )
    op.drop_table("transcript_segments")
