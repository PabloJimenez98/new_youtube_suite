"""create oauth_credentials, analytics_videos, analytics_snapshots and analytics_retention_curve

Spec §4.2's `oauth_credentials`, `analytics_snapshots` and
`analytics_retention_curve`, plus `analytics_videos` (the connected
channel's catalogue) and the bookkeeping columns. See
docs/decisions/0017-oauth-analytics-v2.md §2, §4, §5.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-07

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CREDENTIAL_STATUSES = ("active", "needs_reconsent")


def upgrade() -> None:
    op.create_table(
        "oauth_credentials",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False, unique=True),
        sa.Column("access_token_enc", sa.Text(), nullable=False),
        sa.Column("refresh_token_enc", sa.Text(), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("connected_channel_id", sa.Text(), nullable=False),
        sa.Column("connected_channel_title", sa.Text(), server_default="", nullable=False),
        sa.Column("token_expiry", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="active", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("connected_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_refresh_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(f"status IN {CREDENTIAL_STATUSES!r}", name="oauth_credentials_status_check"),
    )
    op.create_table(
        "analytics_videos",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("youtube_video_id", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text(), server_default="", nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("privacy_status", sa.Text(), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("removed_since", sa.Date(), nullable=True),
        sa.Column("engagement_backfilled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retention_pulled_on", sa.Date(), nullable=True),
    )
    op.create_index("ix_analytics_videos_channel_id", "analytics_videos", ["channel_id"])
    op.create_table(
        "analytics_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("youtube_video_id", sa.Text(), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("views", sa.BigInteger(), nullable=True),
        sa.Column("estimated_minutes_watched", sa.Float(), nullable=True),
        sa.Column("average_view_duration_seconds", sa.Float(), nullable=True),
        sa.Column("average_view_percentage", sa.Float(), nullable=True),
        sa.Column("likes", sa.BigInteger(), nullable=True),
        sa.Column("comments", sa.BigInteger(), nullable=True),
        sa.Column("shares", sa.BigInteger(), nullable=True),
        sa.Column("subscribers_gained", sa.Integer(), nullable=True),
        sa.Column("subscribers_lost", sa.Integer(), nullable=True),
        sa.Column("pulled_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("youtube_video_id", "snapshot_date", name="uq_analytics_snapshots_video_date"),
    )
    op.create_index("ix_analytics_snapshots_channel_id", "analytics_snapshots", ["channel_id"])
    op.create_table(
        "analytics_retention_curve",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("youtube_video_id", sa.Text(), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("elapsed_video_time_ratio", sa.Float(), nullable=False),
        sa.Column("audience_watch_ratio", sa.Float(), nullable=True),
        sa.Column("relative_retention_performance", sa.Float(), nullable=True),
        sa.Column("started_watching", sa.Integer(), nullable=True),
        sa.Column("stopped_watching", sa.Integer(), nullable=True),
        sa.Column("total_segment_impressions", sa.Integer(), nullable=True),
        sa.UniqueConstraint(
            "youtube_video_id",
            "snapshot_date",
            "elapsed_video_time_ratio",
            name="uq_analytics_retention_video_date_ratio",
        ),
    )
    op.create_index("ix_analytics_retention_curve_channel_id", "analytics_retention_curve", ["channel_id"])


def downgrade() -> None:
    op.drop_index("ix_analytics_retention_curve_channel_id", table_name="analytics_retention_curve")
    op.drop_table("analytics_retention_curve")
    op.drop_index("ix_analytics_snapshots_channel_id", table_name="analytics_snapshots")
    op.drop_table("analytics_snapshots")
    op.drop_index("ix_analytics_videos_channel_id", table_name="analytics_videos")
    op.drop_table("analytics_videos")
    op.drop_table("oauth_credentials")
