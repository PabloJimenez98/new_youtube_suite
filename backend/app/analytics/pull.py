"""The daily analytics pull for the connected channel (0017 §5).

The run goes in four steps:

1. **Video list:** page the uploads playlist into `analytics_videos`, then
   mark the videos that left it (`removed_since`) and clear that mark
   for any that came back.
2. **Engagement:** `day,video` in batches of ≤ 200 videos into
   `analytics_snapshots`. Videos on their first pull (a backfill, from
   their publish date) and the others (the last 7 days) go in separate
   batches.
3. **Retention:** one call per due video into `analytics_retention_curve`,
   with the lifetime curve as of today.
4. **Report:** one log line for the run (`RunReport`).

`ReconsentRequired` and `OAuthConfigError` propagate to the task, which
logs them. `QuotaExceeded` stops the run cleanly; what was committed
stays.

Only videos of the **currently connected** channel are pulled. A
different channel connected later never mixes with the old rows.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.analytics.analytics_api import ENGAGEMENT_BATCH, AnalyticsClient, OwnChannelClient, OwnVideo
from app.analytics.selection import OwnVideoState, engagement_window, retention_due
from app.market.batching import batch_ids
from app.market.youtube_api import QuotaExceeded, YouTubeApiError
from app.models.analytics import AnalyticsRetentionPoint, AnalyticsSnapshot, AnalyticsVideo
from app.models.oauth import OAuthCredential

log = logging.getLogger("app.analytics")

SessionFactory = async_sessionmaker[AsyncSession]


@dataclass
class RunReport:
    channel_id: str | None = None
    videos: int = 0
    new_videos: int = 0
    removed: int = 0
    engagement_rows: int = 0
    retention_videos: int = 0
    retention_rows: int = 0
    retention_empty: int = 0
    api_calls: int = 0
    errors: int = 0
    stopped: str | None = None

    def log_line(self) -> str:
        line = (
            f"analytics daily_pull: channel={self.channel_id} videos={self.videos} new={self.new_videos} "
            f"removed={self.removed} engagement_rows={self.engagement_rows} "
            f"retention_videos={self.retention_videos} retention_rows={self.retention_rows} "
            f"retention_empty={self.retention_empty} calls={self.api_calls} errors={self.errors}"
        )
        return f"{line} STOPPED: {self.stopped}" if self.stopped else line


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


async def sync_videos(
    sf: SessionFactory, channel_id: str, videos: Sequence[OwnVideo], today: date
) -> tuple[int, int]:
    """Upsert the uploads list; returns (new videos, newly removed)."""
    seen = {v.youtube_video_id for v in videos}
    async with sf() as db:
        known = set(
            (
                await db.execute(
                    select(AnalyticsVideo.youtube_video_id).where(AnalyticsVideo.channel_id == channel_id)
                )
            ).scalars()
        )
        new = 0
        for v in videos:
            stmt = insert(AnalyticsVideo).values(
                channel_id=channel_id,
                youtube_video_id=v.youtube_video_id,
                title=v.title,
                published_at=v.published_at,
                privacy_status=v.privacy_status,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[AnalyticsVideo.youtube_video_id],
                set_={
                    "channel_id": stmt.excluded.channel_id,
                    "title": stmt.excluded.title,
                    "published_at": stmt.excluded.published_at,
                    "privacy_status": stmt.excluded.privacy_status,
                    "removed_since": None,
                },
            )
            await db.execute(stmt)
            new += v.youtube_video_id not in known
        gone = known - seen
        removed = 0
        if gone:
            result = await db.execute(
                update(AnalyticsVideo)
                .where(
                    AnalyticsVideo.youtube_video_id.in_(gone),
                    AnalyticsVideo.removed_since.is_(None),
                )
                .values(removed_since=today)
            )
            removed = result.rowcount
        await db.commit()
    return new, removed


async def _video_states(sf: SessionFactory, channel_id: str) -> list[OwnVideoState]:
    async with sf() as db:
        rows = (
            await db.execute(select(AnalyticsVideo).where(AnalyticsVideo.channel_id == channel_id))
        ).scalars()
        return [
            OwnVideoState(
                youtube_video_id=v.youtube_video_id,
                published_at=v.published_at,
                removed=v.removed_since is not None,
                engagement_backfilled=v.engagement_backfilled_at is not None,
                retention_pulled_on=v.retention_pulled_on,
            )
            for v in rows
        ]


async def _upsert_engagement(db: AsyncSession, channel_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc)
    stmt = insert(AnalyticsSnapshot).values([{**r, "channel_id": channel_id, "pulled_at": now} for r in rows])
    stmt = stmt.on_conflict_do_update(
        constraint="uq_analytics_snapshots_video_date",
        set_={c: stmt.excluded[c] for c in rows[0] if c not in ("youtube_video_id", "snapshot_date")}
        | {"channel_id": stmt.excluded.channel_id, "pulled_at": stmt.excluded.pulled_at},
    )
    await db.execute(stmt)


async def _upsert_retention(
    db: AsyncSession, channel_id: str, video_id: str, today: date, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    values = [{**r, "channel_id": channel_id, "youtube_video_id": video_id, "snapshot_date": today} for r in rows]
    stmt = insert(AnalyticsRetentionPoint).values(values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_analytics_retention_video_date_ratio",
        set_={c: stmt.excluded[c] for c in rows[0] if c != "elapsed_video_time_ratio"},
    )
    await db.execute(stmt)


async def pull_engagement(
    sf: SessionFactory,
    api: AnalyticsClient,
    channel_id: str,
    states: list[OwnVideoState],
    today: date,
    report: RunReport,
) -> None:
    active = [s for s in states if not s.removed]
    # Backfills grouped apart from re-pulls, so a 7-day re-pull never
    # drags a 2-year window along with it.
    for backfill in (True, False):
        group = sorted(
            (s for s in active if s.engagement_backfilled != backfill),
            key=lambda s: s.published_at,
        )
        by_id = {s.youtube_video_id: s for s in group}
        for ids in batch_ids([s.youtube_video_id for s in group], ENGAGEMENT_BATCH):
            start = min(engagement_window(by_id[i], today) for i in ids)
            try:
                rows = await api.engagement(list(ids), start, today)
            except QuotaExceeded:
                raise
            except YouTubeApiError as exc:
                log.error("analytics engagement batch failed (%d videos): %s", len(ids), exc)
                report.errors += 1
                continue
            async with sf() as db:
                await _upsert_engagement(db, channel_id, rows)
                if backfill:
                    await db.execute(
                        update(AnalyticsVideo)
                        .where(AnalyticsVideo.youtube_video_id.in_(ids))
                        .values(engagement_backfilled_at=datetime.now(timezone.utc))
                    )
                await db.commit()
            report.engagement_rows += len(rows)


async def pull_retention(
    sf: SessionFactory,
    api: AnalyticsClient,
    channel_id: str,
    states: list[OwnVideoState],
    today: date,
    report: RunReport,
    *,
    recent_days: int,
    older_days: int,
) -> None:
    published = {s.youtube_video_id: s.published_at.date() for s in states}
    for video_id in retention_due(states, today, recent_days, older_days):
        try:
            rows = await api.retention(video_id, min(published[video_id], today), today)
        except QuotaExceeded:
            raise
        except YouTubeApiError as exc:
            log.error("analytics retention failed for %s: %s", video_id, exc)
            report.errors += 1
            continue
        async with sf() as db:
            await _upsert_retention(db, channel_id, video_id, today, rows)
            await db.execute(
                update(AnalyticsVideo)
                .where(AnalyticsVideo.youtube_video_id == video_id)
                .values(retention_pulled_on=today)
            )
            await db.commit()
        report.retention_videos += 1
        report.retention_rows += len(rows)
        if not rows:
            # No views yet, or below the anonymization threshold: not an error.
            report.retention_empty += 1


async def run_daily_pull(
    sf: SessionFactory,
    channel_api: OwnChannelClient,
    api: AnalyticsClient,
    *,
    recent_days: int,
    older_days: int,
    today: date | None = None,
) -> RunReport:
    today = today or utc_today()
    report = RunReport()
    try:
        channel = await channel_api.my_channel()
        if channel is None:
            report.stopped = "the connected account has no YouTube channel"
            return report
        report.channel_id = channel.channel_id
        await _note_channel_change(sf, channel.channel_id, channel.title)
        videos = await channel_api.uploads(channel.uploads_playlist_id) if channel.uploads_playlist_id else []
        report.new_videos, report.removed = await sync_videos(sf, channel.channel_id, videos, today)
        states = await _video_states(sf, channel.channel_id)
        report.videos = sum(not s.removed for s in states)
        await pull_engagement(sf, api, channel.channel_id, states, today, report)
        await pull_retention(
            sf, api, channel.channel_id, states, today, report, recent_days=recent_days, older_days=older_days
        )
    except QuotaExceeded as exc:
        report.stopped = str(exc)
    finally:
        report.api_calls = channel_api.calls + api.calls
    return report


async def _note_channel_change(sf: SessionFactory, channel_id: str, title: str) -> None:
    """Keep the credential row's channel in step with what the token sees
    (the owner can rename a channel)."""
    async with sf() as db:
        await db.execute(
            update(OAuthCredential)
            .where(OAuthCredential.connected_channel_id == channel_id)
            .values(connected_channel_title=title)
        )
        await db.commit()
