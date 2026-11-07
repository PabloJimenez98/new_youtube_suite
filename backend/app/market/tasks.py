"""arq cron tasks for the market pipeline (docs/decisions/0015 §4).

Registered on the existing worker: `market_rss_poll` hourly at :05,
`market_daily_pull` daily at 09:30 UTC (after Google's midnight-Pacific
quota reset). No `jobs` rows: each run writes one log line. Without
`YOUTUBE_API_KEY` both log a skip and return.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.market.pull import RunReport, run_daily_pull, run_rss_poll
from app.market.youtube_api import UnitLedger, YouTubeClient

log = logging.getLogger("app.market")

_ledger: UnitLedger | None = None


def process_ledger(settings: Settings) -> UnitLedger:
    """One ledger per process, so the daily budget spans every run."""
    global _ledger
    if _ledger is None or _ledger.budget != settings.market_daily_unit_budget:
        _ledger = UnitLedger(budget=settings.market_daily_unit_budget)
    return _ledger


@asynccontextmanager
async def youtube_client(settings: Settings) -> AsyncIterator[tuple[YouTubeClient, httpx.AsyncClient]]:
    if settings.youtube_api_key is None:
        raise RuntimeError("YOUTUBE_API_KEY is not set")
    async with httpx.AsyncClient() as http:
        yield YouTubeClient(settings.youtube_api_key.get_secret_value(), http, process_ledger(settings)), http


def _session_factory(ctx: dict[str, Any]):
    if "session_factory" in ctx:
        return ctx["session_factory"]
    from app.database import async_session_factory

    return async_session_factory


async def market_daily_pull(ctx: dict[str, Any]) -> RunReport | None:
    settings = get_settings()
    if settings.youtube_api_key is None:
        log.info("market daily_pull: skipped, YOUTUBE_API_KEY not set")
        return None
    async with youtube_client(settings) as (yt, _http):
        report = await run_daily_pull(
            _session_factory(ctx),
            yt,
            recent_days=settings.market_recent_window_days,
            older_days=settings.market_older_interval_days,
        )
    (log.warning if report.stopped else log.info)(report.log_line())
    return report


async def market_rss_poll(ctx: dict[str, Any]) -> RunReport | None:
    settings = get_settings()
    if settings.youtube_api_key is None:
        log.info("market rss_poll: skipped, YOUTUBE_API_KEY not set")
        return None
    async with youtube_client(settings) as (yt, http):
        report = await run_rss_poll(_session_factory(ctx), yt, http)
    (log.warning if report.stopped else log.info)(report.log_line())
    return report
