"""Insights analyses (docs/decisions/0022): pure functions on synthetic
frames with known effects, plus the /insights API on a seeded DB."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.insights import analysis
from app.insights.store import clean_json, refresh_insights
from app.insights.frame import MarketFrame
from app.models.insight import InsightResult
from app.models.market import MarketChannel, MarketVideoSnapshot
from app.models.upload import Upload

SNAP = date(2026, 11, 8)


def make_frame(n_per_channel: int = 200, channels: int = 4, seed: int = 1, **effects: float) -> MarketFrame:
    """Channels with very different popularity; views ∝ age^0.3, plus optional effects:
    `duration` (log10 views per log10 minute) and `aliens` (log10 lift of titles with 'aliens')."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(channels):
        base = 3 + c  # 1k, 10k, 100k, 1M
        for i in range(n_per_channel):
            age = int(rng.integers(1, 1500))
            minutes = float(rng.uniform(20, 180))
            word = ["aliens", "money", "health", "sports"][i % 4]
            title = f"Episode {i} about {word} with guest"
            logv = base + 0.3 * np.log10(age) + effects.get("duration", 0) * np.log10(minutes) + rng.normal(0, 0.2)
            if word == "aliens":
                logv += effects.get("aliens", 0)
            published = datetime(2026, 11, 8, 18, tzinfo=timezone.utc) - timedelta(days=age)
            rows.append((c + 1, f"v{c}_{i}", title, published, minutes * 60, 10**logv))
    published = [r[3] for r in rows]
    views = np.array([r[5] for r in rows])
    return MarketFrame(
        snapshot_date=SNAP,
        channel_ids=np.array([r[0] for r in rows]),
        channel_titles={c + 1: f"Channel {chr(65 + c)}" for c in range(channels)},
        video_ids=[r[1] for r in rows],
        titles=[r[2] for r in rows],
        published=np.array([np.datetime64(p.replace(tzinfo=None), "s") for p in published]),
        snapshot_dates=np.full(len(rows), np.datetime64(SNAP, "D")),
        duration_s=np.array([r[4] for r in rows]),
        views=views,
        likes=views * 0.03,
        comments=views * 0.002,
        tag_counts=np.zeros(len(rows), dtype=int),
        pub_weekday=np.array([p.weekday() for p in published]),
        pub_hour=np.array([p.hour for p in published]),
        own_channel_id=1,
        velocity={},
    )


def test_format_selection_splits_shorts_at_three_minutes() -> None:
    f = make_frame(n_per_channel=4, channels=1)
    f.duration_s[:] = [30, 180, 181, np.nan]
    assert analysis.select_format(f, "short").tolist() == [True, True, False, False]
    assert analysis.select_format(f, "long").tolist() == [False, False, True, True]
    f.views[0] = np.nan
    assert analysis.select_format(f, "all").tolist() == [False, True, True, True]


def test_residuals_remove_channel_popularity_and_age() -> None:
    f = make_frame()
    mask = analysis.select_format(f, "long")
    r = analysis.residuals(f, mask)
    # Raw log-views differ by ~1 per channel; residuals are centred on 0 in every channel.
    for c in range(1, 5):
        assert abs(np.mean(r[f.channel_ids == c])) < 0.01
    assert np.std(r) == pytest.approx(0.2, abs=0.03)


def test_fit_log_age_recovers_the_growth_exponent() -> None:
    f = make_frame(channels=1, n_per_channel=600)
    fit = analysis.fit_log_age(analysis.age_days(f), f.views)
    assert fit.slope == pytest.approx(0.3, abs=0.03)
    assert fit.r2 > 0.2


def test_drivers_find_a_planted_duration_effect_with_a_ci() -> None:
    f = make_frame(duration=1.0)
    out = analysis.drivers(f, analysis.select_format(f, "long"))
    by = {r["feature"]: r for r in out["correlations"]}
    assert by["duration_min"]["rho"] > 0.5
    assert by["duration_min"]["ci_low"] < by["duration_min"]["rho"] < by["duration_min"]["ci_high"]
    assert abs(by["title_chars"]["rho"]) < 0.2
    assert len(out["heatmap"]["cells"]) == 7 * 24


def test_drivers_report_unavailable_on_too_little_data() -> None:
    f = make_frame(n_per_channel=5, channels=1)
    out = analysis.drivers(f, analysis.select_format(f, "long"))
    assert out["available"] is False


def test_keywords_find_a_planted_title_lift() -> None:
    f = make_frame(aliens=0.5)
    out = analysis.keywords(f, analysis.select_format(f, "long"), min_videos=40)
    top = out["positive"][0]
    assert top["token"] == "aliens"
    assert top["multiplier"] == pytest.approx(10**0.5, rel=0.35)  # lift 0.5 in log10
    assert top["ci_low"] > 0


def test_keywords_skip_channel_names_and_single_channel_vocabulary() -> None:
    f = make_frame(channels=4)
    f.channel_titles = {1: "Money Talks", 2: "B", 3: "C", 4: "D"}
    # 'health' only in channel 1's titles: one channel's vocabulary.
    f.titles = [t if (cid == 1 or "health" not in t) else t.replace("health", "news") for t, cid in zip(f.titles, f.channel_ids)]
    out = analysis.keywords(f, analysis.select_format(f, "long"), min_videos=20)
    tokens = {r["token"] for r in out["positive"] + out["negative"]}
    assert "money" not in tokens  # part of a channel name
    assert "health" not in tokens  # one channel holds all of them


def test_channel_overview_and_clusters() -> None:
    f = make_frame(channels=6)
    over = analysis.channel_overview(f, analysis.select_format(f, "long"))
    assert [r["videos"] for r in over] == [200] * 6
    assert over[0]["own"] is True and over[1]["own"] is False
    assert over[0]["views_q1"] <= over[0]["views_median"] <= over[0]["views_q3"]
    assert over[0]["like_rate"] == pytest.approx(0.03)

    clusters = analysis.cluster_channels(over)
    assert clusters["available"] is True
    assert 2 <= clusters["k"] <= 3  # ≤ 3: the colours stay CVD-safe in a scatter
    assert len(clusters["points"]) == 6
    assert sum(clusters["explained_variance"]) <= 1.0001


def test_clusters_need_four_channels() -> None:
    f = make_frame(channels=3)
    assert analysis.cluster_channels(analysis.channel_overview(f, analysis.select_format(f, "long")))["available"] is False


def test_growth_bins_and_velocity_buckets() -> None:
    f = make_frame()
    f.velocity = {vid: (float(i % 50), 1) for i, vid in enumerate(f.video_ids)}
    out = analysis.growth(f, analysis.select_format(f, "long"), sample=100)
    # The pooled fit mixes channels 10× apart in popularity; each channel's own fit recovers 0.3.
    assert out["fit"]["slope"] > 0
    assert all(c["slope"] == pytest.approx(0.3, abs=0.06) for c in out["per_channel"])
    # 100 sampled from the other channels, plus all 200 of the owner's (channel 1).
    assert len(out["points"]) == 300
    assert sum(p["own"] for p in out["points"]) == 200
    assert all(b["q1"] <= b["median"] <= b["q3"] for b in out["binned"])
    assert out["velocity"]["available"] is True
    assert out["velocity"]["top"][0]["views_per_day"] == 49.0


def test_growth_without_velocity_explains_why() -> None:
    f = make_frame()
    assert analysis.growth(f, analysis.select_format(f, "long"))["velocity"]["available"] is False


# ---- Precomputed results (docs/decisions/0023) --------------------------------------


async def _seed_market(db: AsyncSession, videos: int = 60) -> None:
    rng = np.random.default_rng(3)
    for c in range(4):
        ch = MarketChannel(youtube_channel_id=f"UC{c}", channel_title=f"Show {c}")
        db.add(ch)
        await db.flush()
        for i in range(videos):
            age = int(rng.integers(2, 900))
            db.add(
                MarketVideoSnapshot(
                    market_channel_id=ch.id,
                    youtube_video_id=f"v{c}{i:09d}",
                    snapshot_date=SNAP,
                    title=f"Episode {i} {['aliens', 'money', 'health'][i % 3]}",
                    published_at=datetime(2026, 11, 8, tzinfo=timezone.utc) - timedelta(days=age),
                    duration_seconds=int(rng.integers(1200, 9000)),
                    view_count=int(10 ** (3 + c + rng.normal(0, 0.3))),
                    like_count=100,
                    comment_count=10,
                    tags=["a", "b"],
                )
            )
    await db.commit()


def test_clean_json_drops_non_finite_floats() -> None:
    assert clean_json({"a": float("nan"), "b": [1.5, float("inf")], "c": "x"}) == {"a": None, "b": [1.5, None], "c": "x"}


async def test_refresh_computes_every_result_then_skips_when_unchanged(db_session: AsyncSession, session_factory) -> None:
    await _seed_market(db_session)
    first = await refresh_insights(session_factory)
    assert len(first.computed) == 5 * 3 + 1  # 5 analyses × 3 formats + the channel summary
    rows = (await db_session.execute(select(InsightResult))).scalars().all()
    assert len(rows) == 16 and all(r.duration_ms >= 0 for r in rows)

    second = await refresh_insights(session_factory)
    assert second.computed == [] and len(second.skipped) == 16

    # A new snapshot changes the market fingerprint: the market group recomputes, the channel doesn't.
    ch = (await db_session.execute(select(MarketChannel))).scalars().first()
    db_session.add(MarketVideoSnapshot(market_channel_id=ch.id, youtube_video_id="vNEW0000001", snapshot_date=SNAP, title="New", published_at=datetime(2026, 11, 1, tzinfo=timezone.utc), view_count=10))
    await db_session.commit()
    third = await refresh_insights(session_factory)
    assert len(third.computed) == 15 and third.skipped == ["channel"]

    forced = await refresh_insights(session_factory, force=True)
    assert len(forced.computed) == 16


async def test_relinking_an_episode_changes_the_market_fingerprint(db_session: AsyncSession, session_factory) -> None:
    await _seed_market(db_session)
    await refresh_insights(session_factory)
    db_session.add(Upload(filename="Ep [v0000000003].mkv", storage_path="k", youtube_video_id="v0000000003"))
    await db_session.commit()
    again = await refresh_insights(session_factory)
    assert len(again.computed) == 15  # the owner's channel is part of every market result


@pytest.mark.parametrize("path", ["market/overview", "market/clusters", "market/drivers", "market/growth", "market/keywords", "channel"])
async def test_missing_results_answer_pending_and_enqueue_one_refresh(
    client: AsyncClient, auth_headers: dict[str, str], arq_redis_pool, path: str
) -> None:
    body = (await client.get(f"/insights/{path}", headers=auth_headers)).json()
    assert body["pending"] is True and body["computed_at"] is None
    # A second request while pending doesn't queue a second refresh.
    await client.get(f"/insights/{path}", headers=auth_headers)
    queued = await arq_redis_pool.queued_jobs()
    assert [j.function for j in queued] == ["insights_refresh"]


async def test_endpoints_serve_the_stored_results(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, session_factory, arq_redis_pool
) -> None:
    await _seed_market(db_session)
    db_session.add(Upload(filename="Ep [v0000000003].mkv", storage_path="k", youtube_video_id="v0000000003"))
    await db_session.commit()
    await refresh_insights(session_factory)

    overview = (await client.get("/insights/market/overview", headers=auth_headers)).json()
    assert overview["pending"] is False and overview["computed_at"]
    assert overview["videos"] == 240
    assert overview["own_channel"] == "Show 0"
    assert [c["own"] for c in overview["channels"]] == [True, False, False, False]
    for path in ["clusters", "drivers", "growth"]:
        assert (await client.get(f"/insights/market/{path}", headers=auth_headers)).json()["available"] is True, path
    assert (await client.get("/insights/market/overview?format=short", headers=auth_headers)).json()["videos"] == 0
    channel = (await client.get("/insights/channel", headers=auth_headers)).json()
    assert channel["connected"] is False and channel["pending"] is False
    assert await arq_redis_pool.queued_jobs() == []  # nothing missing, nothing enqueued


async def test_insights_require_auth(client: AsyncClient) -> None:
    assert (await client.get("/insights/market/overview")).status_code == 401
    assert (await client.get("/insights/channel")).status_code == 401
