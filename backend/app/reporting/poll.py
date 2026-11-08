"""The daily Reporting API poll for the connected channel (0018 §2).

The run goes in five steps:

1. **Ensure the job:** reuse the `active` row, adopt a job that already
   exists at Google (`jobs.list`, e.g. after a DB reset), or `jobs.create`.
   Rows of another channel become `superseded`.
2. **List new reports:** `jobs.reports.list` with
   `createdAfter = cursor - CURSOR_MARGIN`. Unseen IDs land as `pending`.
   A 404 on the job marks it `expired`; the next run creates a new one.
3. **Resolve corrections** (`selection.resolve_reports`).
4. **Download and store** each remaining `pending` file. Each date's rows
   are replaced in one transaction, because a correction may also drop
   rows. A failed download stays `pending` and is retried next run; a
   format error is `failed`.
5. **Report the lag:**
   - waiting for the first report is normal for `FIRST_REPORT_EXPECTED`;
   - after `FIRST_REPORT_OVERDUE` it is a warning;
   - so is a `pending` file older than `PENDING_STALE`, which is close to
     the 60-day retention.

`ReconsentRequired` and `OAuthConfigError` propagate to the task.
`QuotaExceeded` stops the run cleanly; what was committed stays.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market.youtube_api import QuotaExceeded, YouTubeApiError
from app.models.reporting import REACH_REPORT_TYPE, AnalyticsReachSnapshot, ReportingJob, ReportingReport
from app.reporting.csv_parser import ParsedReport, ReportFormatError, parse_reach_csv
from app.reporting.reporting_api import RemoteJob, ReportingClient
from app.reporting.selection import ReportState, resolve_reports

log = logging.getLogger("app.reporting")

SessionFactory = async_sessionmaker[AsyncSession]
PACIFIC = ZoneInfo("America/Los_Angeles")
CURSOR_MARGIN = timedelta(hours=1)
FIRST_REPORT_EXPECTED = timedelta(hours=48)
FIRST_REPORT_OVERDUE = timedelta(hours=72)
PENDING_STALE = timedelta(days=45)


@dataclass
class PollReport:
    channel_id: str | None = None
    job_id: str | None = None
    job_action: str | None = None  # reused / adopted / created
    new_reports: int = 0
    downloaded: int = 0
    superseded: int = 0
    failed: int = 0
    pending: int = 0
    rows: int = 0
    channel_rows_skipped: int = 0
    api_calls: int = 0
    errors: int = 0
    waiting: str | None = None
    warnings: list[str] = field(default_factory=list)
    stopped: str | None = None

    def log_line(self) -> str:
        line = (
            f"reporting poll: channel={self.channel_id} job={self.job_id} ({self.job_action}) "
            f"new_reports={self.new_reports} downloaded={self.downloaded} superseded={self.superseded} "
            f"failed={self.failed} pending={self.pending} rows={self.rows} "
            f"channel_rows_skipped={self.channel_rows_skipped} calls={self.api_calls} errors={self.errors}"
        )
        if self.waiting:
            line += f" WAITING: {self.waiting}"
        for warning in self.warnings:
            line += f" WARNING: {warning}"
        if self.stopped:
            line += f" STOPPED: {self.stopped}"
        return line

    @property
    def is_warning(self) -> bool:
        return bool(self.stopped or self.errors or self.warnings)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def pacific_date(moment: datetime) -> date:
    return moment.astimezone(PACIFIC).date()


# ---- 1. the job ---------------------------------------------------------------


async def ensure_job(
    sf: SessionFactory, client: ReportingClient, channel_id: str, report_type: str, now: datetime, report: PollReport
) -> ReportingJob:
    async with sf() as db:
        await db.execute(
            update(ReportingJob)
            .where(
                ReportingJob.report_type == report_type,
                ReportingJob.status == "active",
                ReportingJob.channel_id != channel_id,
            )
            .values(status="superseded")
        )
        job = (
            await db.execute(
                select(ReportingJob).where(
                    ReportingJob.channel_id == channel_id,
                    ReportingJob.report_type == report_type,
                    ReportingJob.status == "active",
                )
            )
        ).scalar_one_or_none()
        if job is not None and job.expire_time is not None and job.expire_time <= now:
            job.status = "expired"
            job = None
        await db.commit()
        if job is not None:
            report.job_action = "reused"
            return job

    remote = _live_job(await client.list_jobs(), report_type, now)
    if remote is not None:
        report.job_action = "adopted"
    else:
        remote = await client.create_job(report_type)
        report.job_action = "created"
        log.info("reporting: created job %s for %s on channel %s", remote.id, report_type, channel_id)

    async with sf() as db:
        values = {
            "channel_id": channel_id,
            "report_type": report_type,
            "youtube_job_id": remote.id,
            "status": "active",
            "job_created_at": remote.create_time,
            "expire_time": remote.expire_time,
            "last_error": None,
        }
        stmt = insert(ReportingJob).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ReportingJob.youtube_job_id],
            set_={k: stmt.excluded[k] for k in values if k != "youtube_job_id"},
        ).returning(ReportingJob)
        job = (await db.execute(stmt)).scalar_one()
        await db.commit()
        return job


def _live_job(jobs: list[RemoteJob], report_type: str, now: datetime) -> RemoteJob | None:
    live = [j for j in jobs if j.report_type_id == report_type and (j.expire_time is None or j.expire_time > now)]
    return min(live, key=lambda j: j.create_time) if live else None


# ---- 2. listing ---------------------------------------------------------------


async def list_new_reports(
    sf: SessionFactory, client: ReportingClient, job: ReportingJob, now: datetime, report: PollReport
) -> bool:
    """Returns False when the job is gone at Google (marked `expired`)."""
    created_after = job.reports_cursor - CURSOR_MARGIN if job.reports_cursor else None
    try:
        remote = await client.list_reports(job.youtube_job_id, created_after)
    except YouTubeApiError as exc:
        if exc.status != 404:
            raise
        async with sf() as db:
            await db.execute(
                update(ReportingJob)
                .where(ReportingJob.id == job.id)
                .values(status="expired", last_error=f"job not found at Google ({exc})", last_polled_at=now)
            )
            await db.commit()
        report.warnings.append(f"job {job.youtube_job_id} no longer exists at Google; a new one is created next run")
        return False

    async with sf() as db:
        for r in remote:
            stmt = (
                insert(ReportingReport)
                .values(
                    job_id=job.id,
                    youtube_report_id=r.id,
                    start_time=r.start_time,
                    end_time=r.end_time,
                    report_date=pacific_date(r.start_time),
                    create_time=r.create_time,
                    download_url=r.download_url,
                )
                .on_conflict_do_nothing(index_elements=[ReportingReport.youtube_report_id])
                .returning(ReportingReport.id)
            )
            report.new_reports += (await db.execute(stmt)).scalar_one_or_none() is not None
        values: dict = {"last_polled_at": now}
        if remote:
            newest = max(r.create_time for r in remote)
            values["reports_cursor"] = max(newest, job.reports_cursor) if job.reports_cursor else newest
        expiries = [r.job_expire_time for r in remote if r.job_expire_time]
        if expiries:
            values["expire_time"] = min(expiries)
            report.warnings.append(f"Google set the job to expire at {min(expiries).isoformat()}")
        await db.execute(update(ReportingJob).where(ReportingJob.id == job.id).values(**values))
        await db.commit()
    return True


# ---- 3-4. resolve + download ----------------------------------------------------


async def _store(db: AsyncSession, channel_id: str, row: ReportingReport, parsed: ParsedReport, now: datetime) -> int:
    dates = parsed.dates | {row.report_date}
    await db.execute(
        delete(AnalyticsReachSnapshot).where(
            AnalyticsReachSnapshot.channel_id == channel_id,
            AnalyticsReachSnapshot.report_date.in_(dates),
        )
    )
    # One row per (video, date); a duplicate in the file keeps the last one.
    unique = {(r.youtube_video_id, r.report_date): r for r in parsed.rows}
    if unique:
        await db.execute(
            insert(AnalyticsReachSnapshot).values(
                [
                    {
                        "channel_id": channel_id,
                        "youtube_video_id": r.youtube_video_id,
                        "report_date": r.report_date,
                        "video_thumbnail_impressions": r.video_thumbnail_impressions,
                        "video_thumbnail_impressions_ctr": r.video_thumbnail_impressions_ctr,
                        "source_report_id": row.id,
                        "pulled_at": now,
                    }
                    for r in unique.values()
                ]
            )
        )
    return len(unique)


async def download_reports(
    sf: SessionFactory, client: ReportingClient, job: ReportingJob, now: datetime, report: PollReport
) -> None:
    async with sf() as db:
        rows = (await db.execute(select(ReportingReport).where(ReportingReport.job_id == job.id))).scalars().all()
        plan = resolve_reports([ReportState(r.id, r.report_date, r.create_time, r.status) for r in rows])
        if plan.supersede:
            await db.execute(
                update(ReportingReport).where(ReportingReport.id.in_(plan.supersede)).values(status="superseded")
            )
            await db.commit()
        report.superseded = len(plan.supersede)
        by_id = {r.id: r for r in rows}

    for report_id in plan.download:
        row = by_id[report_id]
        try:
            data = await client.download(row.download_url)
        except QuotaExceeded:
            raise
        except YouTubeApiError as exc:
            log.error("reporting: download of report %s (%s) failed: %s", row.youtube_report_id, row.report_date, exc)
            report.errors += 1
            await _set_report(sf, row.id, last_error=str(exc))
            continue
        try:
            parsed = parse_reach_csv(data)
        except ReportFormatError as exc:
            log.error("reporting: report %s (%s) is malformed: %s", row.youtube_report_id, row.report_date, exc)
            report.errors += 1
            report.failed += 1
            await _set_report(sf, row.id, status="failed", last_error=str(exc), downloaded_at=now)
            continue
        if parsed.header and report.downloaded == 0:
            log.info("reporting: CSV header %s (CTR read as %s)", parsed.header, parsed.ctr_scale)
        async with sf() as db:
            stored = await _store(db, job.channel_id, row, parsed, now)
            await db.execute(
                update(ReportingReport)
                .where(ReportingReport.id == row.id)
                .values(status="downloaded", row_count=stored, downloaded_at=now, last_error=None)
            )
            await db.commit()
        report.downloaded += 1
        report.rows += stored
        report.channel_rows_skipped += parsed.channel_level_rows


async def _set_report(sf: SessionFactory, report_id: int, **values) -> None:
    async with sf() as db:
        await db.execute(update(ReportingReport).where(ReportingReport.id == report_id).values(**values))
        await db.commit()


# ---- 5. lag -------------------------------------------------------------------


async def check_lag(sf: SessionFactory, job: ReportingJob, now: datetime, report: PollReport) -> None:
    async with sf() as db:
        counts = dict(
            (
                await db.execute(
                    select(ReportingReport.status, func.count())
                    .where(ReportingReport.job_id == job.id)
                    .group_by(ReportingReport.status)
                )
            ).all()
        )
        stale = (
            await db.execute(
                select(func.count()).where(
                    ReportingReport.job_id == job.id,
                    ReportingReport.status == "pending",
                    ReportingReport.create_time < now - PENDING_STALE,
                )
            )
        ).scalar_one()
    report.pending = counts.get("pending", 0)
    if not counts:
        age = now - job.job_created_at
        expected = job.job_created_at + FIRST_REPORT_EXPECTED
        if age < FIRST_REPORT_OVERDUE:
            report.waiting = (
                f"no reports yet; job created {job.job_created_at.isoformat()}, "
                f"first report expected by {expected.isoformat()}"
            )
        else:
            report.warnings.append(
                f"no reports {age.total_seconds() / 3600:.0f} h after job creation "
                f"(expected within {FIRST_REPORT_EXPECTED.total_seconds() / 3600:.0f} h)"
            )
    if stale:
        report.warnings.append(
            f"{stale} report(s) still pending after {PENDING_STALE.days} days; Google keeps them for 60"
        )


# ---- run --------------------------------------------------------------------


async def run_reporting_poll(
    sf: SessionFactory,
    client: ReportingClient,
    *,
    channel_id: str,
    report_type: str = REACH_REPORT_TYPE,
    now: datetime | None = None,
) -> PollReport:
    now = now or utcnow()
    report = PollReport(channel_id=channel_id)
    try:
        job = await ensure_job(sf, client, channel_id, report_type, now, report)
        report.job_id = job.youtube_job_id
        if await list_new_reports(sf, client, job, now, report):
            await download_reports(sf, client, job, now, report)
            await check_lag(sf, job, now, report)
    except QuotaExceeded as exc:
        report.stopped = str(exc)
    finally:
        report.api_calls = client.calls
    return report
