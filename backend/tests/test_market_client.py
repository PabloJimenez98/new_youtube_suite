"""M10 Data API client tests over `httpx.MockTransport`: pagination, unit
counting, retries, quota/budget stops. No network."""

from datetime import date

import pytest

from app.market.youtube_api import BudgetExhausted, QuotaExceeded, UnitLedger, YouTubeApiError
from tests.market_fake import FakeYouTube


async def test_pagination_follows_next_page_token_and_counts_units() -> None:
    fake = FakeYouTube()
    ch = fake.add_channel(120)
    async with fake.http() as http:
        yt = fake.client(http)
        seen, token = [], None
        while True:
            page, token = await yt.playlist_items_page(ch.uploads, token)
            seen += [vid for vid, _ in page]
            if token is None:
                break
    assert len(seen) == 120 and len(set(seen)) == 120
    assert fake.calls == ["playlistItems"] * 3
    assert yt.units_used == 3


async def test_5xx_is_retried_then_succeeds() -> None:
    fake = FakeYouTube(fail_next={"videos": [500, 503]})
    ch = fake.add_channel(2)
    async with fake.http() as http:
        yt = fake.client(http)
        items = await yt.videos([v["id"] for v in ch.videos])
    assert len(items) == 2
    assert fake.calls == ["videos"] * 3
    assert yt.units_used == 1  # one logical call


async def test_persistent_5xx_gives_up_after_retries() -> None:
    fake = FakeYouTube(fail_next={"videos": [500] * 10})
    ch = fake.add_channel(1)
    async with fake.http() as http:
        with pytest.raises(YouTubeApiError):
            await fake.client(http).videos([ch.videos[0]["id"]])
    assert fake.calls == ["videos"] * 4


async def test_quota_exceeded_is_not_retried() -> None:
    fake = FakeYouTube(quota_after=0)
    ch = fake.add_channel(1)
    async with fake.http() as http:
        with pytest.raises(QuotaExceeded):
            await fake.client(http).videos([ch.videos[0]["id"]])
    assert fake.calls == ["videos"]


async def test_other_4xx_fails_fast() -> None:
    fake = FakeYouTube(fail_next={"videos": [400]})
    ch = fake.add_channel(1)
    async with fake.http() as http:
        with pytest.raises(YouTubeApiError) as exc:
            await fake.client(http).videos([ch.videos[0]["id"]])
    assert not isinstance(exc.value, QuotaExceeded)
    assert fake.calls == ["videos"]


async def test_budget_cap_stops_before_the_call() -> None:
    fake = FakeYouTube()
    ch = fake.add_channel(1)
    async with fake.http() as http:
        yt = fake.client(http, budget=2)
        await yt.videos([ch.videos[0]["id"]])
        await yt.videos([ch.videos[0]["id"]])
        with pytest.raises(BudgetExhausted):
            await yt.videos([ch.videos[0]["id"]])
    assert fake.calls == ["videos", "videos"]


def test_ledger_resets_on_a_new_quota_day() -> None:
    day = [date(2026, 11, 7)]
    ledger = UnitLedger(budget=1, clock=lambda: day[0])
    ledger.reserve(1)
    with pytest.raises(BudgetExhausted):
        ledger.reserve(1)
    day[0] = date(2026, 11, 8)
    ledger.reserve(1)
    assert ledger.spent_today == 1


async def test_videos_rejects_more_than_50_ids() -> None:
    fake = FakeYouTube()
    async with fake.http() as http:
        with pytest.raises(ValueError):
            await fake.client(http).videos([f"v{i}" for i in range(51)])
    assert fake.calls == []
