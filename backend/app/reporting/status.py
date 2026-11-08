"""Reach-report status for the connection card and the CLI (0018 §4)."""

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.reporting import REACH_REPORT_TYPE, AnalyticsReachSnapshot, ReportingJob, ReportingReport


@dataclass(frozen=True)
class ReachSummary:
    job_status: str
    youtube_job_id: str
    job_created_at: datetime
    last_polled_at: datetime | None
    latest_report_date: date | None
    reports_downloaded: int
    reports_pending: int
    reports_failed: int
    reach_rows: int
    last_error: str | None


async def reach_summary(sf: async_sessionmaker[AsyncSession], channel_id: str) -> ReachSummary | None:
    """The channel's newest reach job (the active one if any), or None."""
    async with sf() as db:
        job = (
            await db.execute(
                select(ReportingJob)
                .where(ReportingJob.channel_id == channel_id, ReportingJob.report_type == REACH_REPORT_TYPE)
                .order_by((ReportingJob.status == "active").desc(), ReportingJob.job_created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if job is None:
            return None
        counts = dict(
            (
                await db.execute(
                    select(ReportingReport.status, func.count())
                    .where(ReportingReport.job_id == job.id)
                    .group_by(ReportingReport.status)
                )
            ).all()
        )
        latest = await db.scalar(
            select(func.max(ReportingReport.report_date)).where(
                ReportingReport.job_id == job.id, ReportingReport.status == "downloaded"
            )
        )
        failed_error = await db.scalar(
            select(ReportingReport.last_error)
            .where(ReportingReport.job_id == job.id, ReportingReport.last_error.is_not(None))
            .order_by(ReportingReport.report_date.desc())
            .limit(1)
        )
        rows = await db.scalar(
            select(func.count())
            .select_from(AnalyticsReachSnapshot)
            .where(AnalyticsReachSnapshot.channel_id == channel_id)
        )
    return ReachSummary(
        job_status=job.status,
        youtube_job_id=job.youtube_job_id,
        job_created_at=job.job_created_at,
        last_polled_at=job.last_polled_at,
        latest_report_date=latest,
        reports_downloaded=counts.get("downloaded", 0),
        reports_pending=counts.get("pending", 0),
        reports_failed=counts.get("failed", 0),
        reach_rows=rows or 0,
        last_error=job.last_error or failed_error,
    )
