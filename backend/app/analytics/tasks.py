"""arq cron task for the connected channel's analytics (0017 §5).

`analytics_daily_pull` runs daily at 10:15 UTC on the existing worker.
There are no `jobs` rows; each run writes one log line.

Status gates:

- **Not configured** (client id/secret/key missing) or **not connected:**
  INFO skip.
- **`needs_reconsent`:** WARNING skip. It is never silent: the
  connection card shows the same status.
- **The grant dies mid-run** (`ReconsentRequired`): WARNING; the row is
  already marked.
- **Bad client credentials** (`OAuthConfigError`): ERROR; the status is
  unchanged.
- **Anything else failing:** ERROR; the status is unchanged.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from app.analytics.analytics_api import AnalyticsClient, OwnChannelClient
from app.analytics.pull import RunReport, run_daily_pull
from app.config import Settings, get_settings
from app.market.youtube_api import YouTubeApiError
from app.youtube_oauth.crypto import cipher_from_settings
from app.youtube_oauth.google_oauth import GoogleOAuthClient, OAuthConfigError, OAuthError
from app.youtube_oauth.tokens import ReconsentRequired, TokenProvider, get_credential

log = logging.getLogger("app.analytics")


def _session_factory(ctx: dict[str, Any]):
    if "session_factory" in ctx:
        return ctx["session_factory"]
    from app.database import async_session_factory

    return async_session_factory


@asynccontextmanager
async def token_provider(
    settings: Settings, sf, http: httpx.AsyncClient | None = None
) -> AsyncIterator[tuple[TokenProvider, httpx.AsyncClient]]:
    async with httpx.AsyncClient() if http is None else _borrowed(http) as client:
        oauth = GoogleOAuthClient.from_settings(client, settings)
        yield TokenProvider(sf, oauth, cipher_from_settings(settings)), client


@asynccontextmanager
async def _borrowed(http: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    yield http


async def analytics_daily_pull(ctx: dict[str, Any]) -> RunReport | None:
    settings = get_settings()
    if not settings.youtube_oauth_configured:
        log.info("analytics daily_pull: skipped, YouTube OAuth not configured (GOOGLE_OAUTH_* / key)")
        return None
    sf = _session_factory(ctx)
    credential = await get_credential(sf)
    if credential is None:
        log.info("analytics daily_pull: skipped, no YouTube channel connected")
        return None
    if credential.status == "needs_reconsent":
        log.warning(
            "analytics daily_pull: skipped, needs_reconsent (%s); reconnect from the Analytics page",
            credential.last_error,
        )
        return None
    try:
        async with token_provider(settings, sf, ctx.get("http")) as (tokens, http):
            report = await run_daily_pull(
                sf,
                OwnChannelClient(http, tokens),
                AnalyticsClient(http, tokens),
                recent_days=settings.market_recent_window_days,
                older_days=settings.market_older_interval_days,
            )
    except ReconsentRequired as exc:
        log.warning("analytics daily_pull: stopped, needs_reconsent (%s)", exc)
        return None
    except OAuthConfigError as exc:
        log.error("analytics daily_pull: OAuth client misconfigured (%s); check GOOGLE_OAUTH_CLIENT_*", exc)
        return None
    except OAuthError as exc:
        log.error("analytics daily_pull: token refresh failed (%s); will retry on the next run", exc)
        return None
    except YouTubeApiError as exc:
        log.error("analytics daily_pull: API error (%s); will retry on the next run", exc)
        return None
    (log.warning if report.stopped or report.errors else log.info)(report.log_line())
    return report
