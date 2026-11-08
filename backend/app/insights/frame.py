"""Load the market data into numpy columns (docs/decisions/0022 §2).

One row per video: its latest snapshot. Velocity uses the snapshots of
the last `VELOCITY_WINDOW_DAYS` only, so the load stays bounded as the
daily pull accumulates history.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market import MarketChannel, MarketVideoSnapshot
from app.models.upload import Upload

VELOCITY_WINDOW_DAYS = 14
#: The owner's time zone, for the publish day×hour heatmap.
PUBLISH_TZ = ZoneInfo("Europe/Madrid")


@dataclass
class MarketFrame:
    snapshot_date: date | None
    channel_ids: np.ndarray  # int, per video
    channel_titles: dict[int, str]
    video_ids: list[str]
    titles: list[str]
    published: np.ndarray  # datetime64[s] (UTC)
    snapshot_dates: np.ndarray  # datetime64[D]
    duration_s: np.ndarray  # float, NaN when unknown
    views: np.ndarray  # float, NaN when hidden
    likes: np.ndarray
    comments: np.ndarray
    tag_counts: np.ndarray  # int
    #: Publication weekday (0 = Monday) and hour, in PUBLISH_TZ.
    pub_weekday: np.ndarray
    pub_hour: np.ndarray
    #: The tracked channel the owner's episodes belong to, if any are linked.
    own_channel_id: int | None
    #: Views gained per day over the velocity window: video_id → (views/day, days).
    velocity: dict[str, tuple[float, int]]

    def __len__(self) -> int:
        return len(self.video_ids)


def _f(values: list[int | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


async def frame_key(db: AsyncSession) -> tuple:
    """Cheap fingerprint of the snapshot table: a new pull changes it."""
    row = (
        await db.execute(select(func.max(MarketVideoSnapshot.snapshot_date), func.count(MarketVideoSnapshot.id)))
    ).one()
    return tuple(row)


async def load_market_frame(db: AsyncSession) -> MarketFrame:
    S = MarketVideoSnapshot
    latest = (
        await db.execute(
            select(
                S.market_channel_id,
                S.youtube_video_id,
                S.title,
                S.published_at,
                S.snapshot_date,
                S.duration_seconds,
                S.view_count,
                S.like_count,
                S.comment_count,
                func.cardinality(S.tags),
            )
            .order_by(S.youtube_video_id, S.snapshot_date.desc())
            .distinct(S.youtube_video_id)
        )
    ).all()
    channels = {cid: title for cid, title in (await db.execute(select(MarketChannel.id, MarketChannel.channel_title))).all()}

    linked = [v for (v,) in (await db.execute(select(Upload.youtube_video_id).where(Upload.youtube_video_id.is_not(None)))).all()]
    by_video = {r[1]: r[0] for r in latest}
    owners = [by_video[v] for v in linked if v in by_video]
    own = max(set(owners), key=owners.count) if owners else None

    snapshot_date = max((r[4] for r in latest), default=None)
    velocity: dict[str, tuple[float, int]] = {}
    if snapshot_date is not None:
        since = snapshot_date - timedelta(days=VELOCITY_WINDOW_DAYS)
        rows = (
            await db.execute(
                select(S.youtube_video_id, S.snapshot_date, S.view_count)
                .where(S.snapshot_date >= since, S.view_count.is_not(None))
                .order_by(S.youtube_video_id, S.snapshot_date)
            )
        ).all()
        first: dict[str, tuple[date, int]] = {}
        last: dict[str, tuple[date, int]] = {}
        for vid, day, views in rows:
            first.setdefault(vid, (day, views))
            last[vid] = (day, views)
        for vid, (d0, v0) in first.items():
            d1, v1 = last[vid]
            days = (d1 - d0).days
            if days >= 1:
                velocity[vid] = ((v1 - v0) / days, days)

    return MarketFrame(
        snapshot_date=snapshot_date,
        channel_ids=np.array([r[0] for r in latest], dtype=int),
        channel_titles=channels,
        video_ids=[r[1] for r in latest],
        titles=[r[2] for r in latest],
        published=np.array([np.datetime64(r[3].replace(tzinfo=None), "s") for r in latest], dtype="datetime64[s]"),
        snapshot_dates=np.array([np.datetime64(r[4], "D") for r in latest], dtype="datetime64[D]"),
        duration_s=_f([r[5] for r in latest]),
        views=_f([r[6] for r in latest]),
        likes=_f([r[7] for r in latest]),
        comments=_f([r[8] for r in latest]),
        tag_counts=np.array([r[9] or 0 for r in latest], dtype=int),
        pub_weekday=np.array([r[3].astimezone(PUBLISH_TZ).weekday() for r in latest], dtype=int),
        pub_hour=np.array([r[3].astimezone(PUBLISH_TZ).hour for r in latest], dtype=int),
        own_channel_id=own,
        velocity=velocity,
    )
