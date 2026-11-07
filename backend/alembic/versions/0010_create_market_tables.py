"""create market_channels, market_videos and market_video_snapshots

Spec §4.2's two market tables, plus `market_videos` (the set of known
videos) and the pull bookkeeping columns on `market_channels`. See
docs/decisions/0015-market-pipeline-v0.md §2.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-07

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DISCOVERED_VIA = ("backfill", "rss")


def upgrade() -> None:
    op.create_table(
        "market_channels",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("youtube_channel_id", sa.Text(), nullable=False, unique=True),
        sa.Column("channel_title", sa.Text(), nullable=False),
        sa.Column("added_date", sa.Date(), server_default=sa.func.current_date(), nullable=False),
        sa.Column("active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("uploads_playlist_id", sa.Text(), nullable=True),
        sa.Column("backfill_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rss_checked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "market_videos",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "market_channel_id",
            sa.BigInteger(),
            sa.ForeignKey("market_channels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("youtube_video_id", sa.Text(), nullable=False, unique=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("discovered_via", sa.Text(), nullable=False),
        sa.Column("unavailable_since", sa.Date(), nullable=True),
        sa.CheckConstraint(f"discovered_via IN {DISCOVERED_VIA!r}", name="market_videos_discovered_via_check"),
    )
    op.create_index("ix_market_videos_market_channel_id", "market_videos", ["market_channel_id"])
    op.create_table(
        "market_video_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "market_channel_id",
            sa.BigInteger(),
            sa.ForeignKey("market_channels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("youtube_video_id", sa.Text(), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("view_count", sa.BigInteger(), nullable=True),
        sa.Column("like_count", sa.BigInteger(), nullable=True),
        sa.Column("comment_count", sa.BigInteger(), nullable=True),
        sa.Column("category_id", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.UniqueConstraint("youtube_video_id", "snapshot_date", name="uq_market_video_snapshots_video_date"),
    )
    op.create_index(
        "ix_market_video_snapshots_market_channel_id", "market_video_snapshots", ["market_channel_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_market_video_snapshots_market_channel_id", table_name="market_video_snapshots")
    op.drop_table("market_video_snapshots")
    op.drop_index("ix_market_videos_market_channel_id", table_name="market_videos")
    op.drop_table("market_videos")
    op.drop_table("market_channels")
