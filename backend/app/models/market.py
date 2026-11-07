from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

DISCOVERED_VIA = ("backfill", "rss")


class MarketChannel(Base):
    """A curated public channel (spec §4.2), plus the M10 pull bookkeeping
    columns (docs/decisions/0015 §2)."""

    __tablename__ = "market_channels"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    youtube_channel_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    channel_title: Mapped[str] = mapped_column(Text, nullable=False)
    added_date: Mapped[date] = mapped_column(Date, nullable=False, server_default=func.current_date())
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    uploads_playlist_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    backfill_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rss_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MarketVideo(Base):
    """The set of known videos of the tracked channels (0015 §2: not in
    §4.2, added so RSS discoveries and deleted videos are tracked)."""

    __tablename__ = "market_videos"
    __table_args__ = (
        CheckConstraint(f"discovered_via IN {DISCOVERED_VIA!r}", name="market_videos_discovered_via_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    market_channel_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("market_channels.id", ondelete="CASCADE"), nullable=False, index=True
    )
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    discovered_via: Mapped[str] = mapped_column(Text, nullable=False)
    # Set when videos.list stops returning the ID (deleted/private); the
    # video then leaves the pull.
    unavailable_since: Mapped[date | None] = mapped_column(Date, nullable=True)


class MarketVideoSnapshot(Base):
    """One video's public numbers on one (UTC) day -- spec §4.2 exactly.
    like/comment counts are NULL when the API hides them, never 0."""

    __tablename__ = "market_video_snapshots"
    __table_args__ = (
        UniqueConstraint("youtube_video_id", "snapshot_date", name="uq_market_video_snapshots_video_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    market_channel_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("market_channels.id", ondelete="CASCADE"), nullable=False, index=True
    )
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    view_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    like_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    comment_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    category_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    thumbnail_url: Mapped[str | None] = mapped_column(Text, nullable=True)
