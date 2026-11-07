"""Market pull orchestration (docs/decisions/0015 §1, §3-5).

- `add_channel`: resolve a reference via `channels.list`, upsert the row.
- `backfill_channel`: page the uploads playlist into `market_videos`.
  Commits per page, idempotent (ON CONFLICT DO NOTHING), so an
  interrupted backfill simply resumes on the next run.
- `run_daily_pull`: resume unfinished backfills, then snapshot the due
  videos (§3) 50 at a time.
- `run_rss_poll`: free feeds; unknown IDs get `videos.list` right away.

A `QuotaExceeded`/`BudgetExhausted` stops a run cleanly: everything
committed so far stays, and the report says why it stopped.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market.batching import batch_ids
from app.market.rss import RssError, fetch_feed
from app.market.selection import VideoState, select_due
from app.market.youtube_api import (
    BudgetExhausted,
    QuotaExceeded,
    YouTubeApiError,
    YouTubeClient,
    parse_channel_ref,
    snapshot_fields,
)
from app.models.market import MarketChannel, MarketVideo, MarketVideoSnapshot

log = logging.getLogger("app.market")

SessionFactory = async_sessionmaker[AsyncSession]


class ChannelNotFound(Exception):
    pass


@dataclass
class RunReport:
    kind: str
    videos_due: int = 0
    rows_written: int = 0
    unavailable: int = 0
    new_videos: int = 0
    backfills_completed: int = 0
    units_used: int = 0
    errors: int = 0
    stopped: str | None = None

    def log_line(self) -> str:
        return (
            f"market {self.kind}: due={self.videos_due} new_videos={self.new_videos} "
            f"rows={self.rows_written} unavailable={self.unavailable} "
            f"backfills_completed={self.backfills_completed} errors={self.errors} units={self.units_used}"
            + (f" STOPPED: {self.stopped}" if self.stopped else "")
        )


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


# ---- Channels ------------------------------------------------------------


async def add_channel(sf: SessionFactory, yt: YouTubeClient, ref: str) -> MarketChannel:
    kind, value = parse_channel_ref(ref)
    items = await (yt.channels(handle=value) if kind == "handle" else yt.channels(channel_id=value))
    if not items:
        raise ChannelNotFound(f"no channel for {ref!r}")
    item = items[0]
    values = {
        "youtube_channel_id": item["id"],
        "channel_title": item["snippet"]["title"],
        "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        "active": True,
    }
    async with sf() as db:
        stmt = insert(MarketChannel).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[MarketChannel.youtube_channel_id],
            set_={k: stmt.excluded[k] for k in ("channel_title", "uploads_playlist_id", "active")},
        ).returning(MarketChannel)
        channel = (await db.execute(stmt)).scalar_one()
        await db.commit()
    return channel


async def backfill_channel(sf: SessionFactory, yt: YouTubeClient, channel: MarketChannel) -> int:
    """Discover the whole catalogue; returns the number of new videos."""
    if not channel.uploads_playlist_id:
        raise ValueError(f"channel {channel.id} has no uploads playlist")
    new, token = 0, None
    while True:
        try:
            videos, token = await yt.playlist_items_page(channel.uploads_playlist_id, token)
        except YouTubeApiError as exc:
            # A channel with no public uploads has no uploads playlist: an
            # empty catalogue, not an error to retry every day.
            if exc.reason != "playlistNotFound" or token is not None:
                raise
            log.warning("market backfill: %s has no uploads playlist; 0 videos", channel.channel_title)
            videos, token = [], None
        async with sf() as db:
            if videos:
                result = await db.execute(
                    insert(MarketVideo)
                    .values(
                        [
                            {
                                "market_channel_id": channel.id,
                                "youtube_video_id": vid,
                                "published_at": published,
                                "discovered_via": "backfill",
                            }
                            for vid, published in videos
                        ]
                    )
                    .on_conflict_do_nothing(index_elements=[MarketVideo.youtube_video_id])
                    .returning(MarketVideo.id)
                )
                new += len(result.all())
            if token is None:
                await db.execute(
                    update(MarketChannel)
                    .where(MarketChannel.id == channel.id)
                    .values(backfill_completed_at=func.now())
                )
            await db.commit()
        if token is None:
            return new


# ---- Snapshots -----------------------------------------------------------


async def _upsert_snapshots(
    db: AsyncSession, rows: list[dict[str, Any]], channel_by_video: dict[str, int], today: date
) -> int:
    if not rows:
        return 0
    values = [{**r, "market_channel_id": channel_by_video[r["youtube_video_id"]], "snapshot_date": today} for r in rows]
    stmt = insert(MarketVideoSnapshot).values(values)
    updatable = [c for c in values[0] if c not in ("youtube_video_id", "snapshot_date")]
    stmt = stmt.on_conflict_do_update(
        constraint="uq_market_video_snapshots_video_date",
        set_={c: stmt.excluded[c] for c in updatable},
    )
    await db.execute(stmt)
    return len(values)


async def snapshot_videos(
    sf: SessionFactory, yt: YouTubeClient, video_ids: Sequence[str], today: date, report: RunReport
) -> None:
    """Snapshot known videos; IDs the API no longer returns become unavailable."""
    for batch in batch_ids(video_ids):
        try:
            items = await yt.videos(batch)
        except (QuotaExceeded, BudgetExhausted):
            raise
        except YouTubeApiError as exc:
            # Skip this batch; its videos stay due and are retried next run.
            log.warning("market snapshot: batch of %d failed: %s", len(batch), exc)
            report.errors += 1
            continue
        async with sf() as db:
            channel_by_video = dict(
                (
                    await db.execute(
                        select(MarketVideo.youtube_video_id, MarketVideo.market_channel_id).where(
                            MarketVideo.youtube_video_id.in_(batch)
                        )
                    )
                ).all()
            )
            rows = [snapshot_fields(i) for i in items if i["id"] in channel_by_video]
            report.rows_written += await _upsert_snapshots(db, rows, channel_by_video, today)
            missing = set(batch) - {i["id"] for i in items}
            if missing:
                await db.execute(
                    update(MarketVideo)
                    .where(MarketVideo.youtube_video_id.in_(missing), MarketVideo.unavailable_since.is_(None))
                    .values(unavailable_since=today)
                )
                report.unavailable += len(missing)
            await db.commit()


async def due_video_ids(sf: SessionFactory, today: date, recent_days: int, older_days: int) -> list[str]:
    last = (
        select(
            MarketVideoSnapshot.youtube_video_id,
            func.max(MarketVideoSnapshot.snapshot_date).label("last_date"),
        )
        .group_by(MarketVideoSnapshot.youtube_video_id)
        .subquery()
    )
    query = (
        select(
            MarketVideo.youtube_video_id,
            MarketVideo.published_at,
            last.c.last_date,
            MarketVideo.unavailable_since,
            MarketChannel.active,
        )
        .join(MarketChannel, MarketChannel.id == MarketVideo.market_channel_id)
        .outerjoin(last, last.c.youtube_video_id == MarketVideo.youtube_video_id)
        .order_by(MarketVideo.published_at.desc())
    )
    async with sf() as db:
        rows = (await db.execute(query)).all()
    states = [
        VideoState(
            youtube_video_id=vid,
            published_at=published,
            last_snapshot_date=last_date,
            unavailable=unavailable is not None,
            channel_active=active,
        )
        for vid, published, last_date, unavailable, active in rows
    ]
    return select_due(states, today, recent_days, older_days)


# ---- Runs ----------------------------------------------------------------


async def run_daily_pull(
    sf: SessionFactory,
    yt: YouTubeClient,
    *,
    today: date | None = None,
    recent_days: int = 90,
    older_days: int = 7,
) -> RunReport:
    today = today or utc_today()
    report = RunReport(kind="daily_pull")
    try:
        async with sf() as db:
            pending = (
                await db.scalars(
                    select(MarketChannel).where(
                        MarketChannel.active, MarketChannel.backfill_completed_at.is_(None)
                    )
                )
            ).all()
        for channel in pending:
            try:
                report.new_videos += await backfill_channel(sf, yt, channel)
            except (QuotaExceeded, BudgetExhausted):
                raise
            except YouTubeApiError as exc:
                # One broken channel must not block the others (or the pull).
                log.warning("market backfill: %s failed: %s", channel.channel_title, exc)
                report.errors += 1
                continue
            report.backfills_completed += 1
        due = await due_video_ids(sf, today, recent_days, older_days)
        report.videos_due = len(due)
        await snapshot_videos(sf, yt, due, today, report)
    except (QuotaExceeded, BudgetExhausted) as exc:
        report.stopped = str(exc)
    report.units_used = yt.units_used
    return report


async def run_rss_poll(
    sf: SessionFactory, yt: YouTubeClient, http: httpx.AsyncClient, *, today: date | None = None
) -> RunReport:
    today = today or utc_today()
    report = RunReport(kind="rss_poll")
    async with sf() as db:
        channels = (await db.scalars(select(MarketChannel).where(MarketChannel.active))).all()
    try:
        for channel in channels:
            try:
                entries = await fetch_feed(http, channel.youtube_channel_id)
            except RssError as exc:
                log.warning("market rss: %s (%s): %s", channel.channel_title, channel.youtube_channel_id, exc)
                continue
            feed_ids = [e.video_id for e in entries]
            async with sf() as db:
                known = set(
                    (
                        await db.scalars(
                            select(MarketVideo.youtube_video_id).where(MarketVideo.youtube_video_id.in_(feed_ids))
                        )
                    ).all()
                ) if feed_ids else set()
            new_ids = [v for v in feed_ids if v not in known]
            for batch in batch_ids(new_ids):
                try:
                    items = await yt.videos(batch)
                except (QuotaExceeded, BudgetExhausted):
                    raise
                except YouTubeApiError as exc:
                    # Unknown IDs stay unknown, so the next poll retries them.
                    log.warning("market rss: %s lookup failed: %s", channel.channel_title, exc)
                    report.errors += 1
                    continue
                rows = [
                    snapshot_fields(i)
                    for i in items
                    if i.get("snippet", {}).get("channelId") == channel.youtube_channel_id
                ]
                async with sf() as db:
                    if rows:
                        inserted = await db.execute(
                            insert(MarketVideo)
                            .values(
                                [
                                    {
                                        "market_channel_id": channel.id,
                                        "youtube_video_id": r["youtube_video_id"],
                                        "published_at": r["published_at"],
                                        "discovered_via": "rss",
                                    }
                                    for r in rows
                                ]
                            )
                            .on_conflict_do_nothing(index_elements=[MarketVideo.youtube_video_id])
                            .returning(MarketVideo.id)
                        )
                        report.new_videos += len(inserted.all())
                        channel_by_video = {r["youtube_video_id"]: channel.id for r in rows}
                        report.rows_written += await _upsert_snapshots(db, rows, channel_by_video, today)
                    await db.commit()
            async with sf() as db:
                await db.execute(
                    update(MarketChannel).where(MarketChannel.id == channel.id).values(rss_checked_at=func.now())
                )
                await db.commit()
    except (QuotaExceeded, BudgetExhausted) as exc:
        report.stopped = str(exc)
    report.units_used = yt.units_used
    return report
