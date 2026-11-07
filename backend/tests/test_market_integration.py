"""M10 integration against the real YouTube Data API v3. Skipped without
`YOUTUBE_API_KEY`. Spends < 10 units: channels 1 + playlistItems 1 +
videos 1 (the snapshot run) + the pull's own calls on a 5-video channel."""

from datetime import date

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.market.pull import RunReport, add_channel, snapshot_videos
from app.market.youtube_api import UnitLedger, YouTubeClient
from app.models.market import MarketChannel, MarketVideo, MarketVideoSnapshot

settings = get_settings()

pytestmark = [
    pytest.mark.youtube_api,
    pytest.mark.skipif(settings.youtube_api_key is None, reason="YOUTUBE_API_KEY not set"),
]

ME_AT_THE_ZOO = "jNQXAC9IVRw"  # stable public video, 19 s, channel "jawed"


async def test_real_api_resolve_page_and_snapshot(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with httpx.AsyncClient() as http:
        yt = YouTubeClient(settings.youtube_api_key.get_secret_value(), http, UnitLedger(budget=10))

        channel = await add_channel(session_factory, yt, "@jawed")
        assert channel.youtube_channel_id.startswith("UC")
        assert channel.uploads_playlist_id.startswith("UU")

        page, _token = await yt.playlist_items_page(channel.uploads_playlist_id, max_results=5)
        assert any(vid == ME_AT_THE_ZOO for vid, _ in page)

        async with session_factory() as db:
            db.add(
                MarketVideo(
                    market_channel_id=channel.id,
                    youtube_video_id=ME_AT_THE_ZOO,
                    published_at=next(p for v, p in page if v == ME_AT_THE_ZOO),
                    discovered_via="backfill",
                )
            )
            await db.commit()
        today = date.today()
        report = RunReport("integration")
        await snapshot_videos(session_factory, yt, [ME_AT_THE_ZOO], today, report)
        assert yt.units_used < 10

    assert report.rows_written == 1
    async with session_factory() as db:
        snap = await db.scalar(select(MarketVideoSnapshot))
        ch = await db.scalar(select(MarketChannel))
    assert snap.title and snap.duration_seconds == 19 and snap.view_count > 0
    assert snap.thumbnail_url and snap.market_channel_id == ch.id
