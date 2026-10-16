"""create show_notes_drafts table

One editable show-notes draft per upload, produced by the `shownotes`
job (M6). Adds `language` and timestamps to spec §4's columns, and a
unique `upload_id` (a re-run replaces the draft). See
docs/decisions/0011-show-notes-local-llm.md §5.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "show_notes_drafts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "upload_id",
            sa.BigInteger(),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("title_suggestions", postgresql.JSONB(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("llm_model_used", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("show_notes_drafts")
