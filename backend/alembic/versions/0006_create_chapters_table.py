"""create chapters table

Chapter starts + titles produced by the `chapters` job (M5), with the
model bundle version that produced them -- a column spec §4 doesn't
have, added for traceability. See
docs/decisions/0009-chapter-generator-local-models.md §7.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-27

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chapters",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "upload_id",
            sa.BigInteger(),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.UniqueConstraint("upload_id", "order_index", name="uq_chapters_upload_index"),
    )
    op.create_index("ix_chapters_upload_id", "chapters", ["upload_id"])


def downgrade() -> None:
    op.drop_index("ix_chapters_upload_id", table_name="chapters")
    op.drop_table("chapters")
