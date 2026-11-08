from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

JOB_STATUSES = ("active", "expired", "superseded")
REPORT_STATUSES = ("pending", "downloaded", "superseded", "failed")
REACH_REPORT_TYPE = "channel_reach_basic_a1"


class ReportingJob(Base):
    """A YouTube Reporting API job: a **subscription**, not a one-shot task
    (docs/decisions/0018 §1). Google's Job has no status. Once created,
    YouTube posts one report file per day for as long as the job lives.

    `status` is our own lifecycle:

    - `active`: the job is in use.
    - `expired`: `expireTime` passed, or Google answered 404. The next run
      creates a new job.
    - `superseded`: a different channel was connected.

    At most one `active` row exists per (channel, report type).
    `reports_cursor` is the newest report `createTime` seen, the next
    `createdAfter`.
    """

    __tablename__ = "reporting_jobs"
    __table_args__ = (
        CheckConstraint(f"status IN {JOB_STATUSES!r}", name="reporting_jobs_status_check"),
        Index(
            "uq_reporting_jobs_active",
            "channel_id",
            "report_type",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[str] = mapped_column(Text, nullable=False)
    report_type: Mapped[str] = mapped_column(Text, nullable=False)
    youtube_job_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    job_created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expire_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reports_cursor: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ReportingReport(Base):
    """One report file of a job: one Pacific-time day of data (0018 §1).

    A correction (backfill) arrives as a new report ID with the same
    `start_time` and a newer `create_time`. Only the newest file for a
    date counts; the older ones become `superseded`. `failed` is a
    format error (the file can't be trusted). A failed download stays
    `pending` and is retried.
    """

    __tablename__ = "reporting_reports"
    __table_args__ = (
        CheckConstraint(f"status IN {REPORT_STATUSES!r}", name="reporting_reports_status_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reporting_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    youtube_report_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    download_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class AnalyticsReachSnapshot(Base):
    """Real thumbnail impressions and CTR per video per day, from
    `channel_reach_basic_a1` (spec §4.2/§6.3, 0018 §1).

    `report_date` is YouTube's day, in Pacific time. CTR is a **fraction
    0-1**. A correction for a date replaces that date's rows. Rows are
    not tied to `analytics_videos`: the report is authoritative.
    """

    __tablename__ = "analytics_reach_snapshots"
    __table_args__ = (
        UniqueConstraint("channel_id", "youtube_video_id", "report_date", name="uq_analytics_reach_video_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[str] = mapped_column(Text, nullable=False)
    youtube_video_id: Mapped[str] = mapped_column(Text, nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    video_thumbnail_impressions: Mapped[int] = mapped_column(BigInteger, nullable=False)
    video_thumbnail_impressions_ctr: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_report_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("reporting_reports.id", ondelete="SET NULL"), nullable=True
    )
    pulled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
