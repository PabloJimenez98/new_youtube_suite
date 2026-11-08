"""`/insights`: the Analytics results, read from `insight_results`
(docs/decisions/0022, 0023). No request runs an analysis: the worker
precomputes them. A missing result enqueues a refresh and answers
`pending`, so the page can say "being computed" instead of blocking."""

import logging
from typing import Literal

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.insights.store import CHANNEL, market_source_key
from app.insights.tasks import JOB_ID_PREFIX
from app.models.insight import InsightResult
from app.models.user import User
from app.worker.pool import get_arq_pool

router = APIRouter(prefix="/insights", tags=["insights"])
log = logging.getLogger("app.insights")

Format = Literal["long", "short", "all"]
Analysis = Literal["overview", "clusters", "drivers", "growth", "keywords"]

PENDING_REASON = "The analyses are being computed in the background. This page refreshes them in a moment."

#: What each analysis returns while there is no stored result yet.
_EMPTY: dict[str, dict] = {
    "overview": {"channels": [], "monthly": []},
    "clusters": {"available": False},
    "drivers": {"available": False, "correlations": [], "heatmap": None},
    "growth": {"available": False},
    "keywords": {"available": False, "positive": [], "negative": []},
    "channel": {"connected": False, "videos": 0, "daily": [], "reach": [], "retention": {"videos": 0, "points": []}},
}


async def _enqueue_refresh(db: AsyncSession, pool: ArqRedis) -> None:
    # One job id per source state: concurrent page loads don't pile up refreshes.
    key = await market_source_key(db)
    try:
        await pool.enqueue_job("insights_refresh", _job_id=f"{JOB_ID_PREFIX}:{key}")
    except Exception:  # Redis down: the hourly cron still fills the table
        log.warning("could not enqueue insights_refresh", exc_info=True)


async def _read(name: str, fmt: str, db: AsyncSession, pool: ArqRedis) -> dict:
    row = (
        await db.execute(select(InsightResult).where(InsightResult.name == name, InsightResult.format == fmt))
    ).scalar_one_or_none()
    if row is None:
        await _enqueue_refresh(db, pool)
        meta = {} if name == "channel" else {"snapshot_date": None, "videos": 0, "own_channel": None}
        return {**meta, **_EMPTY[name], "pending": True, "reason": PENDING_REASON, "computed_at": None}
    return {**row.payload, "pending": False, "computed_at": row.computed_at.isoformat()}


@router.get("/market/{analysis}")
async def market(
    analysis: Analysis,
    format: Format = Query("long"),
    db: AsyncSession = Depends(get_db),
    pool: ArqRedis = Depends(get_arq_pool),
    _user: User = Depends(get_current_user),
) -> dict:
    return await _read(analysis, format, db, pool)


@router.get("/channel")
async def channel_summary(
    db: AsyncSession = Depends(get_db),
    pool: ArqRedis = Depends(get_arq_pool),
    _user: User = Depends(get_current_user),
) -> dict:
    """The connected channel's private data (M11/M12), precomputed like the market results."""
    return await _read(*CHANNEL, db, pool)
