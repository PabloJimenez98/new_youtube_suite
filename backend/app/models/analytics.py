from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Float, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AnalyticsVideo(Base):
    """The connected channel's own videos (docs/decisions/0017 §4: §4.2 has
    no catalogue table for them, like `market_videos` in 0015 §2).

    Built from the uploads playlist with the OAuth token, so unlisted and
    private videos are included. `removed_since` is set when a video
    leaves the playlist; it then leaves the pull. `channel_id` keeps two
    connected channels from mixing if the owner reconnects another one.
    """

    __tablename__ = "analytics_videos"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    privacy_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    removed_since: Mapped[date | None] = mapped_column(Date, nullable=True)
    # Pull bookkeeping (0017 §5): the first engagement pull reads from the
    # publish date, later ones the last 7 days; the retention due rule keys
    # on the last pull date even when it returned no points.
    engagement_backfilled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_pulled_on: Mapped[date | None] = mapped_column(Date, nullable=True)


class AnalyticsSnapshot(Base):
    """Per-video daily engagement from `reports.query` (spec §4.2/§6.2).

    `snapshot_date` is **the day the metrics cover** (the API's `day`
    dimension), not the pull date (0017 §5): every run re-reads the last
    7 days, because of the 48-72 h processing lag, and overwrites revised
    days. One row per video per day.
    """

    __tablename__ = "analytics_snapshots"
    __table_args__ = (
        UniqueConstraint("youtube_video_id", "snapshot_date", name="uq_analytics_snapshots_video_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    views: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    estimated_minutes_watched: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_view_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_view_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    likes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    comments: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    subscribers_gained: Mapped[int | None] = mapped_column(Integer, nullable=True)
    subscribers_lost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pulled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AnalyticsRetentionPoint(Base):
    """One point of a video's audience-retention curve (spec §4.2/§6.2).

    The API takes exactly one video per request and returns up to 100
    points (`elapsed_video_time_ratio` 0.01..1.0). `snapshot_date` is the
    **pull date**: each pull stores the lifetime curve as of that day.
    """

    __tablename__ = "analytics_retention_curve"
    __table_args__ = (
        UniqueConstraint(
            "youtube_video_id",
            "snapshot_date",
            "elapsed_video_time_ratio",
            name="uq_analytics_retention_video_date_ratio",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    elapsed_video_time_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    audience_watch_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Only populated above a view-count threshold (spec §4.2).
    relative_retention_performance: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_watching: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stopped_watching: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_segment_impressions: Mapped[int | None] = mapped_column(Integer, nullable=True)
