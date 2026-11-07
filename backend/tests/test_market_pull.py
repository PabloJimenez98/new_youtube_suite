"""M10 orchestration against the test database, with the fake API: add +
backfill, daily pull, idempotent re-runs, unavailable videos, quota stop,
resumed backfill, RSS discovery."""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market.pull import add_channel, backfill_channel, run_daily_pull, run_rss_poll
from app.models.market import MarketChannel, MarketVideo, MarketVideoSnapshot
from tests.market_fake import FakeYouTube, feed_xml, video_item

TODAY = date(2026, 11, 7)
SF = async_sessionmaker[AsyncSession]


async def _count(sf: SF, model, *where) -> int:
    async with sf() as db:
        return await db.scalar(select(func.count()).select_from(model).where(*where))


def _channel_with_mixed_ages(fake: FakeYouTube) -> None:
    # 120 videos, one every 7 days from 2026-11-01: 13 within the 90-day window.
    fake.add_channel(120)


async def test_add_backfill_then_daily_pull(session_factory: SF) -> None:
    fake = FakeYouTube()
    _channel_with_mixed_ages(fake)
    async with fake.http() as http:
        yt = fake.client(http)
        channel = await add_channel(session_factory, yt, "@chan0")
        assert channel.uploads_playlist_id == "UU" + "0" * 22
        new = await backfill_channel(session_factory, yt, channel)
        assert new == 120
        report = await run_daily_pull(session_factory, yt, today=TODAY)

    assert report.stopped is None
    assert report.videos_due == 120  # first pull: nothing snapshotted yet
    assert report.rows_written == 120
    assert await _count(session_factory, MarketVideoSnapshot, MarketVideoSnapshot.snapshot_date == TODAY) == 120
    async with session_factory() as db:
        ch = await db.scalar(select(MarketChannel))
        assert ch.backfill_completed_at is not None
        snap = await db.scalar(
            select(MarketVideoSnapshot).where(MarketVideoSnapshot.youtube_video_id == "c0v00000")
        )
    assert snap.view_count == 100 and snap.duration_seconds == 1800 and snap.like_count == 10
    assert snap.market_channel_id == ch.id
    # 1 channels + 3 playlist pages + 3 videos batches
    assert fake.calls.count("playlistItems") == 3 and fake.calls.count("videos") == 3


async def test_rerun_same_day_upserts_without_duplicates(session_factory: SF) -> None:
    fake = FakeYouTube()
    ch = fake.add_channel(3, newest=datetime(2026, 11, 5, tzinfo=timezone.utc), days_apart=1)
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        await run_daily_pull(session_factory, yt, today=TODAY)
        ch.videos[0]["statistics"]["viewCount"] = "555"
        # Same day: nothing is due any more...
        again = await run_daily_pull(session_factory, yt, today=TODAY)
        assert again.videos_due == 0
        # ...but a forced re-snapshot of the same day overwrites the row.
        from app.market.pull import RunReport, snapshot_videos

        await snapshot_videos(session_factory, yt, [ch.videos[0]["id"]], TODAY, RunReport("t"))
    assert await _count(session_factory, MarketVideoSnapshot) == 3
    async with session_factory() as db:
        views = await db.scalar(
            select(MarketVideoSnapshot.view_count).where(MarketVideoSnapshot.youtube_video_id == ch.videos[0]["id"])
        )
    assert views == 555


async def test_next_days_follow_recent_daily_and_older_weekly(session_factory: SF) -> None:
    fake = FakeYouTube()
    _channel_with_mixed_ages(fake)
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        await run_daily_pull(session_factory, yt, today=TODAY)
        next_day = await run_daily_pull(session_factory, yt, today=TODAY + timedelta(days=1))
        week = await run_daily_pull(session_factory, yt, today=TODAY + timedelta(days=7))
    # Published 2026-11-01 back in 7-day steps. On 11-07, 13 are within 90
    # days (the 13th, 08-09, exactly on the edge); on 11-08 it drops out.
    assert next_day.videos_due == 12
    assert week.videos_due == 120  # older ones are 7 days stale again


async def test_missing_video_becomes_unavailable_and_leaves_the_pull(session_factory: SF) -> None:
    fake = FakeYouTube()
    ch = fake.add_channel(3, newest=datetime(2026, 11, 5, tzinfo=timezone.utc), days_apart=1)
    gone = ch.videos[1]["id"]
    fake.hidden.add(gone)
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        report = await run_daily_pull(session_factory, yt, today=TODAY)
        tomorrow = await run_daily_pull(session_factory, yt, today=TODAY + timedelta(days=1))
    assert report.rows_written == 2 and report.unavailable == 1
    assert tomorrow.videos_due == 2
    async with session_factory() as db:
        video = await db.scalar(select(MarketVideo).where(MarketVideo.youtube_video_id == gone))
    assert video.unavailable_since == TODAY


async def test_quota_stop_keeps_rows_already_written(session_factory: SF) -> None:
    fake = FakeYouTube()
    _channel_with_mixed_ages(fake)
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        # channels + 3 pages done; allow 2 videos batches, then quota out.
        fake.quota_after = len(fake.calls) + 2
        report = await run_daily_pull(session_factory, yt, today=TODAY)
    assert report.stopped is not None and "quotaExceeded" in report.stopped
    assert report.rows_written == 100
    assert await _count(session_factory, MarketVideoSnapshot) == 100


async def test_budget_stop_is_clean(session_factory: SF) -> None:
    fake = FakeYouTube()
    _channel_with_mixed_ages(fake)
    async with fake.http() as http:
        yt = fake.client(http, budget=5)  # channels 1 + pages 3 + 1 videos batch
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        report = await run_daily_pull(session_factory, yt, today=TODAY)
    assert report.stopped and "budget" in report.stopped
    assert report.rows_written == 50


async def test_interrupted_backfill_resumes_on_next_pull(session_factory: SF) -> None:
    fake = FakeYouTube()
    _channel_with_mixed_ages(fake)
    async with fake.http() as http:
        yt = fake.client(http)
        channel = await add_channel(session_factory, yt, "@chan0")
        fake.quota_after = len(fake.calls) + 1  # one page lands, then quota out
        report = await run_daily_pull(session_factory, yt, today=TODAY)
        assert report.stopped
        assert await _count(session_factory, MarketVideo) == 50
        async with session_factory() as db:
            assert (await db.get(MarketChannel, channel.id)).backfill_completed_at is None

        fake.quota_after = None  # next day: fresh quota
        resumed = await run_daily_pull(session_factory, yt, today=TODAY + timedelta(days=1))
    assert resumed.stopped is None and resumed.backfills_completed == 1
    assert await _count(session_factory, MarketVideo) == 120
    assert resumed.rows_written == 120


async def test_inactive_channel_is_not_pulled(session_factory: SF) -> None:
    fake = FakeYouTube()
    fake.add_channel(3, newest=datetime(2026, 11, 5, tzinfo=timezone.utc), days_apart=1)
    async with fake.http() as http:
        yt = fake.client(http)
        channel = await add_channel(session_factory, yt, "@chan0")
        await backfill_channel(session_factory, yt, channel)
        async with session_factory() as db:
            (await db.get(MarketChannel, channel.id)).active = False
            await db.commit()
        report = await run_daily_pull(session_factory, yt, today=TODAY)
    assert report.videos_due == 0 and "videos" not in fake.calls


async def test_rss_new_id_gets_video_and_snapshot_known_id_nothing(session_factory: SF) -> None:
    fake = FakeYouTube()
    ch = fake.add_channel(3, newest=datetime(2026, 11, 5, tzinfo=timezone.utc), days_apart=1)
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        fresh = video_item("brand_new_01", ch.channel_id, datetime(2026, 11, 7, 6, tzinfo=timezone.utc))
        ch.videos.insert(0, fresh)
        fake.rss[ch.channel_id] = feed_xml(ch.channel_id, ["brand_new_01", ch.videos[1]["id"]])
        calls_before = len(fake.calls)
        report = await run_rss_poll(session_factory, yt, http, today=TODAY)
        assert fake.calls[calls_before:] == ["videos"]  # only the unknown ID is looked up
        calls_before = len(fake.calls)
        again = await run_rss_poll(session_factory, yt, http, today=TODAY)
        assert fake.calls[calls_before:] == []  # everything in the feed is known now

    assert report.new_videos == 1 and report.rows_written == 1
    assert again.new_videos == 0
    async with session_factory() as db:
        video = await db.scalar(select(MarketVideo).where(MarketVideo.youtube_video_id == "brand_new_01"))
        snap = await db.scalar(
            select(MarketVideoSnapshot).where(MarketVideoSnapshot.youtube_video_id == "brand_new_01")
        )
        ch_row = await db.scalar(select(MarketChannel))
    assert video.discovered_via == "rss" and snap.snapshot_date == TODAY
    assert ch_row.rss_checked_at is not None


async def test_rss_feed_failure_is_skipped(session_factory: SF) -> None:
    fake = FakeYouTube()
    fake.add_channel(1, idx=0)
    fake.add_channel(1, idx=1)
    ch1 = fake.channels[1]
    async with fake.http() as http:
        yt = fake.client(http)
        await add_channel(session_factory, yt, "@chan0")  # its feed 404s
        await add_channel(session_factory, yt, "@chan1")
        fake.rss[ch1.channel_id] = feed_xml(ch1.channel_id, [ch1.videos[0]["id"]])
        report = await run_rss_poll(session_factory, yt, http, today=TODAY)
    assert report.new_videos == 1


async def test_add_existing_channel_reactivates_without_duplicate(session_factory: SF) -> None:
    fake = FakeYouTube()
    fake.add_channel(1)
    async with fake.http() as http:
        yt = fake.client(http)
        first = await add_channel(session_factory, yt, "@chan0")
        async with session_factory() as db:
            (await db.get(MarketChannel, first.id)).active = False
            await db.commit()
        second = await add_channel(session_factory, yt, "UC" + "0" * 22)
    assert second.id == first.id and second.active
    assert await _count(session_factory, MarketChannel) == 1


async def test_channel_without_uploads_playlist_does_not_block_the_others(session_factory: SF) -> None:
    """Regression (M10 seed run, 2026-11-07): `playlistItems` answered 404
    `playlistNotFound` for one seed channel and aborted the whole seed."""
    fake = FakeYouTube()
    empty = fake.add_channel(0, idx=0)
    fake.add_channel(3, idx=1, newest=datetime(2026, 11, 5, tzinfo=timezone.utc), days_apart=1)
    fake.no_playlist.add(empty.channel_id)
    async with fake.http() as http:
        yt = fake.client(http)
        await add_channel(session_factory, yt, "@chan0")
        await add_channel(session_factory, yt, "@chan1")
        report = await run_daily_pull(session_factory, yt, today=TODAY)
    assert report.stopped is None and report.errors == 0
    assert report.backfills_completed == 2 and report.rows_written == 3
    async with session_factory() as db:
        channels = (await db.scalars(select(MarketChannel))).all()
    assert all(c.backfill_completed_at is not None for c in channels)


async def test_failed_videos_batch_is_skipped_and_retried_next_run(session_factory: SF) -> None:
    fake = FakeYouTube(bad_videos_calls={1})
    _channel_with_mixed_ages(fake)  # 120 videos -> 3 batches
    async with fake.http() as http:
        yt = fake.client(http)
        await backfill_channel(session_factory, yt, await add_channel(session_factory, yt, "@chan0"))
        report = await run_daily_pull(session_factory, yt, today=TODAY)
        assert report.errors == 1 and report.rows_written == 70 and report.stopped is None
        retry = await run_daily_pull(session_factory, yt, today=TODAY)
    assert retry.videos_due == 50 and retry.rows_written == 50
