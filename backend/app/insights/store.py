"""Precompute the Analytics results into `insight_results` (docs/decisions/0023).

The analyses are too heavy to run inside a request: loading 47k videos
and bootstrapping correlations blocked the API when five charts asked at
once. The worker runs `refresh_insights` hourly (and on demand); it
recomputes a group only when its source tables changed, and the API only
reads the stored rows.
"""

import asyncio
import hashlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.insights import analysis
from app.insights.frame import MarketFrame, load_market_frame
from app.models.analytics import AnalyticsRetentionPoint, AnalyticsSnapshot, AnalyticsVideo
from app.models.insight import InsightResult
from app.models.market import MarketVideoSnapshot
from app.models.oauth import OAuthCredential
from app.models.reporting import AnalyticsReachSnapshot
from app.models.upload import Upload

log = logging.getLogger("app.insights")

FORMATS = ("long", "short", "all")
MARKET_ANALYSES = ("overview", "clusters", "drivers", "growth", "keywords")
CHANNEL = ("channel", "-")


def _meta(frame: MarketFrame, mask: np.ndarray) -> dict:
    return {
        "snapshot_date": frame.snapshot_date.isoformat() if frame.snapshot_date else None,
        "videos": int(mask.sum()),
        "own_channel": frame.channel_titles.get(frame.own_channel_id) if frame.own_channel_id else None,
    }


def _overview(frame: MarketFrame, mask: np.ndarray) -> dict:
    return {**_meta(frame, mask), "channels": analysis.channel_overview(frame, mask), "monthly": analysis.monthly_output(frame, mask)}


MARKET: dict[str, Callable[[MarketFrame, np.ndarray], dict]] = {
    "overview": _overview,
    "clusters": lambda f, m: {**_meta(f, m), **analysis.cluster_channels(analysis.channel_overview(f, m))},
    "drivers": lambda f, m: {**_meta(f, m), **analysis.drivers(f, m)},
    "growth": lambda f, m: {**_meta(f, m), **analysis.growth(f, m)},
    "keywords": lambda f, m: {**_meta(f, m), **analysis.keywords(f, m)},
}


# ---- Source fingerprints --------------------------------------------------------------


async def market_source_key(db: AsyncSession) -> str:
    """Changes when a pull adds snapshots or an episode is (re)linked (own channel)."""
    S = MarketVideoSnapshot
    day, rows = (await db.execute(select(func.max(S.snapshot_date), func.count(S.id)))).one()
    ordered_ids = func.string_agg(Upload.youtube_video_id, aggregate_order_by(",", Upload.youtube_video_id))
    linked = (await db.execute(select(func.count(Upload.id), ordered_ids))).one()
    # A stable digest (Python's hash() changes per process).
    links = hashlib.sha1((linked[1] or "").encode()).hexdigest()[:12]
    return f"market:{day}:{rows}:{linked[0]}:{links}"


async def channel_source_key(db: AsyncSession) -> str:
    parts = []
    for model, col in [
        (AnalyticsSnapshot, AnalyticsSnapshot.pulled_at),
        (AnalyticsReachSnapshot, AnalyticsReachSnapshot.pulled_at),
        (AnalyticsRetentionPoint, AnalyticsRetentionPoint.snapshot_date),
    ]:
        n, last = (await db.execute(select(func.count(model.id), func.max(col)))).one()
        parts.append(f"{n}:{last}")
    connected = (await db.execute(select(func.count(OAuthCredential.id)))).scalar_one()
    videos = (await db.execute(select(func.count(AnalyticsVideo.id)))).scalar_one()
    return f"channel:{connected}:{videos}:" + "|".join(parts)


# ---- The channel summary (SQL aggregates) -----------------------------------------------


async def compute_channel_summary(db: AsyncSession) -> dict:
    """The connected channel's private data (M11/M12): daily engagement, reach, retention."""
    connected = (await db.execute(select(func.count(OAuthCredential.id)))).scalar_one() > 0
    videos = (await db.execute(select(func.count(AnalyticsVideo.id)))).scalar_one()

    A = AnalyticsSnapshot
    daily = (
        await db.execute(
            select(
                A.snapshot_date,
                func.sum(A.views),
                func.sum(A.estimated_minutes_watched),
                func.sum(A.likes),
                func.sum(A.comments),
                func.sum(A.subscribers_gained) - func.sum(A.subscribers_lost),
            )
            .group_by(A.snapshot_date)
            .order_by(A.snapshot_date)
        )
    ).all()

    R = AnalyticsReachSnapshot
    reach = (
        await db.execute(
            select(
                R.report_date,
                func.sum(R.video_thumbnail_impressions),
                # Impression-weighted CTR across the channel's videos.
                func.sum(R.video_thumbnail_impressions * R.video_thumbnail_impressions_ctr)
                / func.nullif(func.sum(R.video_thumbnail_impressions), 0),
            )
            .group_by(R.report_date)
            .order_by(R.report_date)
        )
    ).all()

    P = AnalyticsRetentionPoint
    latest = select(P.youtube_video_id, func.max(P.snapshot_date).label("d")).group_by(P.youtube_video_id).subquery()
    retention = (
        await db.execute(
            select(P.elapsed_video_time_ratio, func.avg(P.audience_watch_ratio), func.count(func.distinct(P.youtube_video_id)))
            .join(latest, (P.youtube_video_id == latest.c.youtube_video_id) & (P.snapshot_date == latest.c.d))
            .group_by(P.elapsed_video_time_ratio)
            .order_by(P.elapsed_video_time_ratio)
        )
    ).all()

    return {
        "connected": connected,
        "videos": videos,
        "daily": [
            {
                "date": d.isoformat(),
                "views": v,
                "minutes_watched": float(mw) if mw is not None else None,
                "likes": lk,
                "comments": cm,
                "subscribers_net": sn,
            }
            for d, v, mw, lk, cm, sn in daily
        ],
        "reach": [{"date": d.isoformat(), "impressions": imp, "ctr": float(ctr) if ctr is not None else None} for d, imp, ctr in reach],
        "retention": {
            "videos": max((n for *_, n in retention), default=0),
            "points": [{"ratio": float(r), "watch_ratio": float(w) if w is not None else None} for r, w, _ in retention],
        },
    }


# ---- Refresh -------------------------------------------------------------------------------


@dataclass
class RefreshReport:
    computed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    seconds: float = 0.0


async def _stored_keys(db: AsyncSession) -> dict[tuple[str, str], str]:
    rows = (await db.execute(select(InsightResult.name, InsightResult.format, InsightResult.source_key))).all()
    return {(n, f): k for n, f, k in rows}


def clean_json(value):
    """NaN/inf → None, recursively: JSONB (and JSON) reject them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


async def _upsert(db: AsyncSession, name: str, fmt: str, payload: dict, source_key: str, ms: int) -> None:
    payload = clean_json(payload)
    stmt = insert(InsightResult).values(name=name, format=fmt, payload=payload, source_key=source_key, duration_ms=ms)
    await db.execute(
        stmt.on_conflict_do_update(
            constraint="uq_insight_results_name_format",
            set_={"payload": stmt.excluded.payload, "source_key": stmt.excluded.source_key, "duration_ms": stmt.excluded.duration_ms, "computed_at": func.now()},
        )
    )


def _compute_market(frame: MarketFrame) -> list[tuple[str, str, dict, int]]:
    """CPU-bound: every market analysis × format from one loaded frame."""
    out = []
    for fmt in FORMATS:
        mask = analysis.select_format(frame, fmt)
        for name in MARKET_ANALYSES:
            t = time.perf_counter()
            payload = MARKET[name](frame, mask)
            out.append((name, fmt, payload, int((time.perf_counter() - t) * 1000)))
    return out


async def refresh_insights(session_factory: async_sessionmaker[AsyncSession], force: bool = False) -> RefreshReport:
    """Recompute the groups whose sources changed (or everything with `force`)."""
    report = RefreshReport()
    start = time.perf_counter()
    async with session_factory() as db:
        stored = await _stored_keys(db)

        market_key = await market_source_key(db)
        market_rows = [(n, f) for f in FORMATS for n in MARKET_ANALYSES]
        if force or any(stored.get(k) != market_key for k in market_rows):
            frame = await load_market_frame(db)
            for name, fmt, payload, ms in await asyncio.to_thread(_compute_market, frame):
                await _upsert(db, name, fmt, payload, market_key, ms)
                report.computed.append(f"{name}/{fmt}")
        else:
            report.skipped += [f"{n}/{f}" for n, f in market_rows]

        channel_key = await channel_source_key(db)
        if force or stored.get(CHANNEL) != channel_key:
            t = time.perf_counter()
            payload = await compute_channel_summary(db)
            await _upsert(db, *CHANNEL, payload, channel_key, int((time.perf_counter() - t) * 1000))
            report.computed.append("channel")
        else:
            report.skipped.append("channel")
        await db.commit()
    report.seconds = time.perf_counter() - start
    log.info("insights refresh: %d computed, %d unchanged in %.1fs", len(report.computed), len(report.skipped), report.seconds)
    return report
