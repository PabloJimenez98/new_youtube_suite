"""add uploads.youtube_video_id

Links an episode to its YouTube video, backfilled from the `[videoId]`
in existing filenames. See docs/decisions/0022-analytics-dashboards.md §1.

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept in sync with app.youtube_link.SQL_FILENAME_REGEX (copied, so the
# migration doesn't import app code that may change later).
_FILENAME_REGEX = r"\[([A-Za-z0-9_-]{11})\][^\[]*$"


def upgrade() -> None:
    op.add_column("uploads", sa.Column("youtube_video_id", sa.Text(), nullable=True))
    op.create_index("ix_uploads_youtube_video_id", "uploads", ["youtube_video_id"])
    op.execute(
        sa.text("UPDATE uploads SET youtube_video_id = substring(filename from :rx)").bindparams(
            rx=_FILENAME_REGEX
        )
    )


def downgrade() -> None:
    op.drop_index("ix_uploads_youtube_video_id", table_name="uploads")
    op.drop_column("uploads", "youtube_video_id")
