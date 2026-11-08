"""M12 integration against the real YouTube Reporting API
(docs/decisions/0018, test plan). Opt-in: `pytest -m youtube_reporting -s`.

Precondition: the test channel is connected through the app, as for
`-m youtube_oauth`. The dev credential row is copied into the test DB.

**This test has two phases, because of the real reporting lag:**

- **Phase 1 (any time, e.g. the first run):**
  - The poll creates the `channel_reach_basic_a1` job at Google, or adopts
    it if it already exists. This also checks that the granted scopes are
    enough for the Reporting API.
  - YouTube posts the first report **within 48 h** of job creation, and the
    30 days of history "within a couple of days".
  - While the job is younger than 48 h and has no reports, the test
    **passes** and prints when to re-run it. Waiting is the documented
    state, not a flake.
- **Phase 2 (re-run ≥ 48 h after the job was created):**
  - The same test now requires at least one report, every report
    downloaded or superseded (0 failed), schema-valid rows and an
    idempotent second run.
  - It prints the first CSV header and the CTR scale it read, which
    settles the `CTR_SCALE` / header / date-format checks (0018).

The test DB is wiped per test. Every run therefore **adopts** the real
job via `jobs.list` and re-downloads every report still available. That
also exercises adoption against Google, and no duplicate job is created.

The test channel has no impressions yet, so header-only reports (0 rows)
are the expected outcome until the M11 follow-up (real views) happens.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.models.oauth import OAuthCredential
from app.models.reporting import AnalyticsReachSnapshot, ReportingJob, ReportingReport
from app.reporting.poll import FIRST_REPORT_EXPECTED, run_reporting_poll
from app.reporting.reporting_api import ReportingClient
from app.youtube_oauth.crypto import cipher_from_settings
from app.youtube_oauth.google_oauth import GoogleOAuthClient
from app.youtube_oauth.tokens import TokenProvider
from tests.test_analytics_integration import _dev_credential, _write_back_refresh_token

settings = get_settings()

pytestmark = [
    pytest.mark.youtube_reporting,
    pytest.mark.skipif(not settings.youtube_oauth_configured, reason="YouTube OAuth not configured in backend/.env"),
]


async def _statuses(sf) -> dict[str, int]:
    async with sf() as db:
        return dict(
            (await db.execute(select(ReportingReport.status, func.count()).group_by(ReportingReport.status))).all()
        )


async def test_real_reach_job_and_reports(session_factory: async_sessionmaker[AsyncSession]) -> None:
    dev = await _dev_credential()
    if dev is None:
        pytest.skip("no channel connected in the dev DB: connect it from the Analytics page first")
    assert dev["status"] == "active", f"dev credential needs re-consent: {dev['last_error']}"
    async with session_factory() as db:
        db.add(OAuthCredential(**dev))
        await db.commit()

    async with httpx.AsyncClient() as http:
        tokens = TokenProvider(session_factory, GoogleOAuthClient.from_settings(http, settings), cipher_from_settings(settings))
        try:
            report = await run_reporting_poll(
                session_factory, ReportingClient(http, tokens), channel_id=dev["connected_channel_id"]
            )
            print(f"\n{report.log_line()}")
            assert report.stopped is None, report.log_line()
            assert report.job_action in ("created", "adopted"), report.job_action

            async with session_factory() as db:
                job = await db.scalar(select(ReportingJob))
            assert job.report_type == "channel_reach_basic_a1" and job.status == "active"
            age = datetime.now(timezone.utc) - job.job_created_at
            statuses = await _statuses(session_factory)
            print(f"job {job.youtube_job_id} created {job.job_created_at.isoformat()} ({age.total_seconds() / 3600:.1f} h ago)")
            print(f"reports by status: {statuses}")

            if not statuses:
                if age < FIRST_REPORT_EXPECTED:
                    rerun = job.job_created_at + FIRST_REPORT_EXPECTED
                    print(
                        f"PHASE 1 OK: job exists, no reports yet (expected). Re-run after {rerun.isoformat()} "
                        "for phase 2; the 30-day history follows within a few days."
                    )
                    return
                pytest.fail(f"no reports {age} after job creation (expected within {FIRST_REPORT_EXPECTED})")

            # ---- phase 2 ----
            assert statuses.get("failed", 0) == 0 and report.errors == 0, report.log_line()
            assert statuses.get("pending", 0) == 0, f"undownloaded reports: {statuses}"
            async with session_factory() as db:
                first = await db.scalar(select(ReportingReport).order_by(ReportingReport.report_date).limit(1))
                rows = (await db.execute(select(AnalyticsReachSnapshot))).scalars().all()
            print(f"PHASE 2: {sum(statuses.values())} reports from {first.report_date}; {len(rows)} reach rows")
            print(f"first CSV (header + up to 3 lines):\n{(await ReportingClient(http, tokens).download(first.download_url)).decode()[:500]}")
            for r in rows:
                assert r.channel_id == dev["connected_channel_id"] and r.youtube_video_id
                assert r.video_thumbnail_impressions >= 0
                assert r.video_thumbnail_impressions_ctr is None or 0 <= r.video_thumbnail_impressions_ctr <= 1
                assert r.report_date <= datetime.now(timezone.utc).date() + timedelta(days=1)

            again = await run_reporting_poll(
                session_factory, ReportingClient(http, tokens), channel_id=dev["connected_channel_id"]
            )
            assert (again.job_action, again.new_reports, again.downloaded, again.errors) == ("reused", 0, 0, 0)
            async with session_factory() as db:
                assert await db.scalar(select(func.count()).select_from(AnalyticsReachSnapshot)) == len(rows)
        finally:
            async with session_factory() as db:
                row = await db.scalar(select(OAuthCredential))
            if row.refresh_token_enc != dev["refresh_token_enc"]:
                await _write_back_refresh_token(row.refresh_token_enc)
