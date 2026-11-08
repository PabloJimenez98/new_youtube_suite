"""M11 analytics pull (docs/decisions/0017 §4-5): response parsing, the
pure selection rules, the orchestration against `FakeGoogle`, and the
cron task's status gates."""

import logging
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select

from app.analytics import tasks
from app.analytics.analytics_api import AnalyticsClient, OwnChannelClient, report_rows
from app.analytics.pull import run_daily_pull
from app.analytics.selection import REPULL_DAYS, OwnVideoState, engagement_window, retention_due
from app.models.analytics import AnalyticsRetentionPoint, AnalyticsSnapshot, AnalyticsVideo
from app.worker.settings import WorkerSettings
from app.youtube_oauth.crypto import TokenCipher
from app.youtube_oauth.google_oauth import GoogleOAuthClient, TokenResponse
from app.youtube_oauth.tokens import TokenProvider, get_credential, store_connection
from tests.google_fake import CHANNEL_ID, FERNET_KEY, FakeGoogle, FakeVideo, oauth_settings

TODAY = date(2026, 11, 7)
NOW = datetime(2026, 11, 7, 10, 15, tzinfo=timezone.utc)


async def _no_sleep(_s: float) -> None:
    return None


# ---- parsing & selection (pure) -------------------------------------------------


def test_report_rows_maps_column_headers() -> None:
    payload = {"columnHeaders": [{"name": "day"}, {"name": "video"}, {"name": "views"}], "rows": [["2026-11-01", "v1", 5]]}
    assert report_rows(payload) == [{"day": "2026-11-01", "video": "v1", "views": 5}]
    assert report_rows({"columnHeaders": [{"name": "day"}]}) == []  # no data -> no `rows` key


def _state(vid: str, days_old: int, *, backfilled=True, retention: date | None = None, removed=False) -> OwnVideoState:
    return OwnVideoState(vid, NOW - timedelta(days=days_old), removed, backfilled, retention)


def test_retention_due_rule_matches_m10() -> None:
    videos = [
        _state("recent-new", 3),
        _state("recent-pulled-today", 3, retention=TODAY),
        _state("recent-pulled-yesterday", 3, retention=TODAY - timedelta(days=1)),
        _state("older-never", 200),
        _state("older-fresh", 200, retention=TODAY - timedelta(days=3)),
        _state("older-stale", 200, retention=TODAY - timedelta(days=7)),
        _state("removed", 3, removed=True),
    ]
    assert retention_due(videos, TODAY, 90, 7) == ["recent-new", "recent-pulled-yesterday", "older-never", "older-stale"]


def test_engagement_window_backfill_vs_repull() -> None:
    assert engagement_window(_state("a", 400, backfilled=False), TODAY) == (NOW - timedelta(days=400)).date()
    assert engagement_window(_state("a", 400), TODAY) == TODAY - timedelta(days=REPULL_DAYS)


# ---- orchestration ---------------------------------------------------------------


async def _connected(session_factory, google: FakeGoogle, cipher: TokenCipher) -> None:
    await store_connection(
        session_factory,
        cipher,
        TokenResponse(google.issue_access(), 3600, "rt-1", google.granted_scopes),
        channel_id=google.channel_id,
        channel_title=google.channel_title,
        now=datetime.now(timezone.utc),
    )


async def _run(session_factory, google: FakeGoogle, today: date = TODAY):
    cipher = TokenCipher(FERNET_KEY)
    if await get_credential(session_factory) is None:
        await _connected(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        oauth = GoogleOAuthClient(http, client_id="c", client_secret="client-secret", redirect_uri="x", sleep=_no_sleep)
        tokens = TokenProvider(session_factory, oauth, cipher)
        return await run_daily_pull(
            session_factory,
            OwnChannelClient(http, tokens, sleep=_no_sleep),
            AnalyticsClient(http, tokens, sleep=_no_sleep),
            recent_days=90,
            older_days=7,
            today=today,
        )


async def _count(session_factory, model) -> int:
    async with session_factory() as db:
        return await db.scalar(select(func.count()).select_from(model))


def _with_views(google: FakeGoogle, days: int = 10, retention: int = 100) -> None:
    for v in google.videos:
        v.daily_views = {TODAY - timedelta(days=d): 10 + d for d in range(1, days + 1)}
        v.retention_points = retention


async def test_empty_channel_runs_clean(session_factory) -> None:
    google = FakeGoogle()  # a brand-new test channel: no uploads
    report = await _run(session_factory, google)
    assert report.stopped is None and report.errors == 0
    assert report.channel_id == CHANNEL_ID and report.videos == 0
    assert await _count(session_factory, AnalyticsSnapshot) == 0


async def test_videos_without_views_store_nothing_and_are_not_errors(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(2, newest=NOW - timedelta(days=2))
    report = await _run(session_factory, google)
    assert report.new_videos == 2 and report.errors == 0
    assert report.engagement_rows == 0 and report.retention_videos == 2 and report.retention_empty == 2
    async with session_factory() as db:
        pulled = (await db.execute(select(AnalyticsVideo.retention_pulled_on))).scalars().all()
    assert pulled == [TODAY, TODAY]


async def test_full_pull_populates_both_tables(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(3, newest=NOW - timedelta(days=20), days_apart=50)  # 20, 70, 120 days old
    google.videos[2].privacy = "unlisted"
    _with_views(google)
    report = await _run(session_factory, google)
    assert report.errors == 0 and report.stopped is None
    assert report.engagement_rows == 30 == await _count(session_factory, AnalyticsSnapshot)
    assert report.retention_videos == 3 and report.retention_rows == 300
    async with session_factory() as db:
        snap = (
            await db.execute(select(AnalyticsSnapshot).where(AnalyticsSnapshot.snapshot_date == TODAY - timedelta(days=1)).limit(1))
        ).scalar_one()
        point = (await db.execute(select(AnalyticsRetentionPoint).order_by(AnalyticsRetentionPoint.elapsed_video_time_ratio).limit(1))).scalar_one()
        unlisted = (await db.execute(select(AnalyticsVideo).where(AnalyticsVideo.privacy_status == "unlisted"))).scalar_one()
    assert snap.views == 11 and snap.likes == 1 and snap.average_view_percentage == 41.5
    assert snap.channel_id == CHANNEL_ID and isinstance(snap.views, int)
    assert point.elapsed_video_time_ratio == 0.01 and point.snapshot_date == TODAY
    assert point.audience_watch_ratio == 1.0 and point.total_segment_impressions == 40
    assert unlisted.youtube_video_id == "own0002"


async def test_rerun_is_idempotent_and_repulls_only_seven_days(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(2, newest=NOW - timedelta(days=200))
    _with_views(google, days=30)
    await _run(session_factory, google)
    first = await _count(session_factory, AnalyticsSnapshot)
    google.api_calls.clear()
    google.videos[0].daily_views[TODAY - timedelta(days=2)] = 999  # a late revision
    report = await _run(session_factory, google)
    assert await _count(session_factory, AnalyticsSnapshot) == first == 60
    engagement = [p for path, p in google.api_calls if path == "/v2/reports" and p["dimensions"] == "day,video"]
    assert [p["startDate"] for p in engagement] == [(TODAY - timedelta(days=REPULL_DAYS)).isoformat()]
    assert report.retention_videos == 0  # older videos: weekly, pulled today already
    async with session_factory() as db:
        revised = await db.scalar(
            select(AnalyticsSnapshot.views).where(
                AnalyticsSnapshot.youtube_video_id == "own0000",
                AnalyticsSnapshot.snapshot_date == TODAY - timedelta(days=2),
            )
        )
    assert revised == 999


async def test_backfill_starts_at_the_publish_date(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(1, newest=NOW - timedelta(days=400))
    await _run(session_factory, google)
    params = next(p for path, p in google.api_calls if path == "/v2/reports" and p["dimensions"] == "day,video")
    assert params["startDate"] == (NOW - timedelta(days=400)).date().isoformat()
    assert params["endDate"] == TODAY.isoformat()


async def test_engagement_batches_of_200_and_one_retention_call_per_video(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(205, newest=NOW - timedelta(days=1), days_apart=0)
    await _run(session_factory, google)
    engagement = [p for path, p in google.api_calls if path == "/v2/reports" and p["dimensions"] == "day,video"]
    assert sorted(len(p["filters"].removeprefix("video==").split(",")) for p in engagement) == [5, 200]
    retention = [p for path, p in google.api_calls if path == "/v2/reports" and p["dimensions"] == "elapsedVideoTimeRatio"]
    assert len(retention) == 205 and all("," not in p["filters"] for p in retention)


async def test_paging_with_start_index(session_factory, monkeypatch) -> None:
    from app.analytics import analytics_api

    monkeypatch.setattr(analytics_api, "PAGE_SIZE", 4)
    google = FakeGoogle()
    google.add_videos(1, newest=NOW - timedelta(days=200))
    _with_views(google, days=10, retention=0)
    report = await _run(session_factory, google)
    assert report.engagement_rows == 10
    starts = [p["startIndex"] for path, p in google.api_calls if path == "/v2/reports" and p["dimensions"] == "day,video"]
    assert starts == ["1", "5", "9"]


async def test_removed_and_returning_videos(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(3, newest=NOW - timedelta(days=200))
    await _run(session_factory, google)
    gone = google.videos.pop(1)
    report = await _run(session_factory, google, today=TODAY + timedelta(days=1))
    assert report.removed == 1 and report.videos == 2
    async with session_factory() as db:
        removed = await db.scalar(select(AnalyticsVideo.removed_since).where(AnalyticsVideo.youtube_video_id == gone.vid))
    assert removed == TODAY + timedelta(days=1)
    google.videos.insert(1, gone)
    report = await _run(session_factory, google, today=TODAY + timedelta(days=2))
    assert report.removed == 0 and report.videos == 3


async def test_deleted_video_placeholder_is_skipped_and_marked_removed(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(3, newest=NOW - timedelta(days=200))
    await _run(session_factory, google)
    # The owner deletes a video: it lingers in the playlist as a placeholder.
    google.videos[1].privacy = "privacyStatusUnspecified"
    google.videos[1].title = "Deleted video"
    deleted = google.videos[1].vid
    before = len(google.api_calls)
    report = await _run(session_factory, google, today=TODAY + timedelta(days=1))
    assert report.removed == 1 and report.videos == 2
    async with session_factory() as db:
        removed = await db.scalar(select(AnalyticsVideo.removed_since).where(AnalyticsVideo.youtube_video_id == deleted))
    assert removed == TODAY + timedelta(days=1)
    assert all(deleted not in p.get("filters", "") for path, p in google.api_calls[before:] if path == "/v2/reports")


async def test_another_channel_never_mixes_with_the_old_rows(session_factory) -> None:
    google = FakeGoogle()
    google.add_videos(2, newest=NOW - timedelta(days=200))
    _with_views(google)
    await _run(session_factory, google)
    other = FakeGoogle(channel_id="UC" + "o" * 22, channel_title="Other")
    other.videos = [FakeVideo("other1", NOW - timedelta(days=5))]
    async with session_factory() as db:  # the owner reconnected another account
        from app.models.oauth import OAuthCredential

        await db.execute(OAuthCredential.__table__.delete())
        await db.commit()
    report = await _run(session_factory, other)
    assert report.channel_id == other.channel_id and report.videos == 1
    calls = [p["filters"] for path, p in other.api_calls if path == "/v2/reports"]
    assert all("own" not in f for f in calls)


async def test_quota_stop_keeps_committed_rows(session_factory) -> None:
    google = FakeGoogle(quota_after=2)
    google.add_videos(3, newest=NOW - timedelta(days=10), days_apart=1)
    _with_views(google)
    report = await _run(session_factory, google)
    assert report.stopped and "quotaExceeded" in report.stopped
    assert await _count(session_factory, AnalyticsSnapshot) == 30  # engagement committed before the stop
    assert report.retention_videos == 1


async def test_dead_grant_mid_run_marks_reconsent(session_factory) -> None:
    from app.youtube_oauth.tokens import ReconsentRequired

    google = FakeGoogle(api_401_always=True)
    with pytest.raises(ReconsentRequired):
        await _run(session_factory, google)
    assert (await get_credential(session_factory)).status == "needs_reconsent"


# ---- cron task -------------------------------------------------------------------


def test_analytics_cron_registered_at_10_15_utc() -> None:
    jobs = {c.name: c for c in WorkerSettings.cron_jobs}
    daily = jobs["cron:analytics_daily_pull"]
    assert daily.hour == 10 and daily.minute == 15 and daily.unique


@pytest.mark.parametrize(
    ("settings_kw", "connect", "status", "level", "text"),
    [
        ({"google_oauth_client_id": None}, False, None, logging.INFO, "not configured"),
        ({}, False, None, logging.INFO, "no YouTube channel connected"),
        ({}, True, "needs_reconsent", logging.WARNING, "needs_reconsent"),
    ],
)
async def test_task_status_gates_log_and_skip(
    session_factory, monkeypatch, caplog, settings_kw, connect, status, level, text
) -> None:
    monkeypatch.setattr(tasks, "get_settings", lambda: oauth_settings(**settings_kw))
    if connect:
        google = FakeGoogle()
        await _connected(session_factory, google, TokenCipher(FERNET_KEY))
        async with session_factory() as db:
            from app.models.oauth import OAuthCredential

            row = (await db.execute(select(OAuthCredential))).scalar_one()
            row.status, row.last_error = status, "invalid_grant: revoked"
            await db.commit()
    with caplog.at_level(logging.INFO, logger="app.analytics"):
        assert await tasks.analytics_daily_pull({"session_factory": session_factory}) is None
    record = next(r for r in caplog.records if r.name == "app.analytics")
    assert record.levelno == level and text in record.getMessage()


async def test_task_runs_end_to_end_and_logs_the_report(session_factory, monkeypatch, caplog) -> None:
    monkeypatch.setattr(tasks, "get_settings", lambda: oauth_settings())
    google = FakeGoogle()
    google.add_videos(2, newest=NOW - timedelta(days=200))
    _with_views(google)
    await _connected(session_factory, google, TokenCipher(FERNET_KEY))
    async with httpx.AsyncClient(transport=google.transport()) as http:
        with caplog.at_level(logging.INFO, logger="app.analytics"):
            report = await tasks.analytics_daily_pull({"session_factory": session_factory, "http": http})
    assert report is not None and report.engagement_rows > 0
    assert "analytics daily_pull: channel=" in caplog.text


async def test_task_revoked_grant_is_loud_and_sticky(session_factory, monkeypatch, caplog) -> None:
    """The manual 'revoke at Google' check, simulated: the next run must
    WARN and leave `needs_reconsent`, never pass silently."""
    monkeypatch.setattr(tasks, "get_settings", lambda: oauth_settings())
    google = FakeGoogle()
    await _connected(session_factory, google, TokenCipher(FERNET_KEY))
    google.access_tokens.clear()
    google.refresh_tokens.clear()  # revoked: refresh -> invalid_grant
    async with httpx.AsyncClient(transport=google.transport()) as http:
        with caplog.at_level(logging.INFO):
            assert await tasks.analytics_daily_pull({"session_factory": session_factory, "http": http}) is None
    assert any(r.levelno == logging.WARNING and "needs_reconsent" in r.getMessage() for r in caplog.records)
    row = await get_credential(session_factory)
    assert row.status == "needs_reconsent" and "invalid_grant" in row.last_error
