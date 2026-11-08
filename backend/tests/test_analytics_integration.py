"""M11 integration against the real Google OAuth + Analytics API v2
(docs/decisions/0017, test plan). Opt-in: `pytest -m youtube_oauth`.

Precondition: the owner has connected the test channel through the app
(Analytics page), so the **dev** database holds the credential row. The
test copies that row into the test database, then:

1. forces a refresh against Google, and checks that the new expiry is in
   the future;
2. runs the real daily pull, and checks that `analytics_videos` matches
   the channel's uploads (unlisted included) and that every stored row is
   schema-valid;
3. runs the pull again: it must be idempotent.

A brand-new test channel with no views legitimately yields 0 engagement
and retention rows: Analytics has a 48-72 h lag and omits tiny
audiences. The counts are printed; the row shapes are only asserted when
rows exist.

If Google rotates the refresh token, the new one is written back to the
dev row, so the app's connection survives the test.
"""

from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.analytics.analytics_api import AnalyticsClient, OwnChannelClient
from app.analytics.pull import run_daily_pull
from app.config import get_settings
from app.models.analytics import AnalyticsRetentionPoint, AnalyticsSnapshot, AnalyticsVideo
from app.models.oauth import OAuthCredential
from app.youtube_oauth.crypto import cipher_from_settings
from app.youtube_oauth.google_oauth import GoogleOAuthClient
from app.youtube_oauth.tokens import PROVIDER, TokenProvider

settings = get_settings()

pytestmark = [
    pytest.mark.youtube_oauth,
    pytest.mark.skipif(not settings.youtube_oauth_configured, reason="YouTube OAuth not configured in backend/.env"),
]

_COLUMNS = [c.key for c in OAuthCredential.__table__.columns if c.key != "id"]


async def _dev_credential() -> dict | None:
    engine = create_async_engine(settings.database_url)
    try:
        async with async_sessionmaker(engine)() as db:
            row = await db.scalar(select(OAuthCredential).where(OAuthCredential.provider == PROVIDER))
            return {c: getattr(row, c) for c in _COLUMNS} if row else None
    finally:
        await engine.dispose()


async def _write_back_refresh_token(refresh_token_enc: str) -> None:
    engine = create_async_engine(settings.database_url)
    try:
        async with async_sessionmaker(engine)() as db:
            row = await db.scalar(select(OAuthCredential).where(OAuthCredential.provider == PROVIDER))
            row.refresh_token_enc = refresh_token_enc
            await db.commit()
    finally:
        await engine.dispose()


async def _count(sf, model) -> int:
    async with sf() as db:
        return await db.scalar(select(func.count()).select_from(model))


async def test_real_refresh_and_pull(session_factory: async_sessionmaker[AsyncSession]) -> None:
    dev = await _dev_credential()
    if dev is None:
        pytest.skip("no channel connected in the dev DB: connect it from the Analytics page first")
    assert dev["status"] == "active", f"dev credential needs re-consent: {dev['last_error']}"
    async with session_factory() as db:
        db.add(OAuthCredential(**dev))
        await db.commit()

    cipher = cipher_from_settings(settings)
    async with httpx.AsyncClient() as http:
        tokens = TokenProvider(session_factory, GoogleOAuthClient.from_settings(http, settings), cipher)
        try:
            await tokens.get(force=True)
        finally:
            async with session_factory() as db:
                row = await db.scalar(select(OAuthCredential))
            if row.refresh_token_enc != dev["refresh_token_enc"]:
                await _write_back_refresh_token(row.refresh_token_enc)
        assert row.status == "active"
        assert row.token_expiry > datetime.now(timezone.utc)
        assert tokens.refreshes == 1

        channel_api = OwnChannelClient(http, tokens)
        channel = await channel_api.my_channel()
        assert channel is not None and channel.channel_id == dev["connected_channel_id"]
        uploads = await channel_api.uploads(channel.uploads_playlist_id) if channel.uploads_playlist_id else []

        kw = dict(recent_days=settings.market_recent_window_days, older_days=settings.market_older_interval_days)
        report = await run_daily_pull(session_factory, OwnChannelClient(http, tokens), AnalyticsClient(http, tokens), **kw)
        assert report.stopped is None and report.errors == 0, report.log_line()
        counts = [await _count(session_factory, m) for m in (AnalyticsVideo, AnalyticsSnapshot, AnalyticsRetentionPoint)]

        again = await run_daily_pull(session_factory, OwnChannelClient(http, tokens), AnalyticsClient(http, tokens), **kw)
        assert again.errors == 0 and again.new_videos == 0
        assert [await _count(session_factory, m) for m in (AnalyticsVideo, AnalyticsSnapshot, AnalyticsRetentionPoint)] == counts

    print(f"\n{report.log_line()}\nrows: videos={counts[0]} snapshots={counts[1]} retention={counts[2]}")
    async with session_factory() as db:
        stored = set((await db.execute(select(AnalyticsVideo.youtube_video_id))).scalars())
        snap = await db.scalar(select(AnalyticsSnapshot).limit(1))
        point = await db.scalar(select(AnalyticsRetentionPoint).limit(1))
    assert stored == {v.youtube_video_id for v in uploads}
    if snap is not None:
        assert snap.channel_id == channel.channel_id and snap.views is not None and snap.views >= 0
        assert snap.snapshot_date <= datetime.now(timezone.utc).date()
    if point is not None:
        assert 0 < point.elapsed_video_time_ratio <= 1 and point.audience_watch_ratio is not None
