"""arq cron task for the Reporting API (0018 §2).

`reporting_daily_poll` runs daily at 11:00 UTC on the existing worker,
after M11's 10:15 analytics pull. YouTube posts one report per day, so
polling more often only finds nothing new. There are no `jobs` rows;
each run writes one log line.

The status gates are the same as `analytics_daily_pull` (0017 §5):

- not configured / not connected: INFO skip
- `needs_reconsent`: WARNING skip
- the grant dies mid-run: WARNING
- bad client credentials: ERROR
- any other failure: ERROR
"""

import logging
from typing import Any

from app.analytics.tasks import _session_factory, token_provider
from app.config import get_settings
from app.market.youtube_api import YouTubeApiError
from app.reporting.poll import PollReport, run_reporting_poll
from app.reporting.reporting_api import ReportingClient
from app.youtube_oauth.google_oauth import OAuthConfigError, OAuthError
from app.youtube_oauth.tokens import ReconsentRequired, get_credential

log = logging.getLogger("app.reporting")


async def reporting_daily_poll(ctx: dict[str, Any]) -> PollReport | None:
    settings = get_settings()
    if not settings.youtube_oauth_configured:
        log.info("reporting poll: skipped, YouTube OAuth not configured (GOOGLE_OAUTH_* / key)")
        return None
    sf = _session_factory(ctx)
    credential = await get_credential(sf)
    if credential is None:
        log.info("reporting poll: skipped, no YouTube channel connected")
        return None
    if credential.status == "needs_reconsent":
        log.warning(
            "reporting poll: skipped, needs_reconsent (%s); reconnect from the Analytics page",
            credential.last_error,
        )
        return None
    try:
        async with token_provider(settings, sf, ctx.get("http")) as (tokens, http):
            report = await run_reporting_poll(
                sf, ReportingClient(http, tokens), channel_id=credential.connected_channel_id
            )
    except ReconsentRequired as exc:
        log.warning("reporting poll: stopped, needs_reconsent (%s)", exc)
        return None
    except OAuthConfigError as exc:
        log.error("reporting poll: OAuth client misconfigured (%s); check GOOGLE_OAUTH_CLIENT_*", exc)
        return None
    except OAuthError as exc:
        log.error("reporting poll: token refresh failed (%s); will retry on the next run", exc)
        return None
    except YouTubeApiError as exc:
        log.error("reporting poll: API error (%s); will retry on the next run", exc)
        return None
    (log.warning if report.is_warning else log.info)(report.log_line())
    return report
