"""M12 Reporting API (docs/decisions/0018): the CSV parser, the correction
rule, the poll pipeline against `FakeGoogle`, and the cron task's gates."""

import logging
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import select

from app.models.reporting import AnalyticsReachSnapshot, ReportingJob, ReportingReport
from app.reporting import tasks
from app.reporting.csv_parser import ReportFormatError, parse_reach_csv
from app.reporting.poll import pacific_date, run_reporting_poll
from app.reporting.reporting_api import ReportingClient, parse_time
from app.reporting.selection import ReportState, resolve_reports
from app.worker.settings import WorkerSettings
from app.youtube_oauth.crypto import TokenCipher
from app.youtube_oauth.google_oauth import GoogleOAuthClient, TokenResponse
from app.youtube_oauth.tokens import TokenProvider, get_credential, mark_reconsent, store_connection
from tests.google_fake import CHANNEL_ID, FERNET_KEY, REACH_HEADER, FakeGoogle, FakeReporting, reach_csv, oauth_settings

NOW = datetime(2026, 11, 8, 11, tzinfo=timezone.utc)
D1, D2, D3 = date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)


async def _no_sleep(_s: float) -> None:
    return None


# ---- CSV parser (pure) -----------------------------------------------------------


def test_parse_happy_path() -> None:
    parsed = parse_reach_csv(reach_csv(D1, [("v1", 1000, 0.052), ("v2", 10, 0.0)]))
    assert [(r.youtube_video_id, r.report_date, r.video_thumbnail_impressions, r.video_thumbnail_impressions_ctr) for r in parsed.rows] == [
        ("v1", D1, 1000, 0.052),
        ("v2", D1, 10, 0.0),
    ]
    assert parsed.ctr_scale == "fraction"


def test_parse_maps_by_header_name_reordered_and_extra_columns() -> None:
    data = b"video_thumbnail_impressions_ctr,new_metric,video_id,date,channel_id,video_thumbnail_impressions\n0.1,7,v1,2026-11-01,UC,50\n"
    (row,) = parse_reach_csv(data).rows
    assert (row.youtube_video_id, row.report_date, row.video_thumbnail_impressions, row.video_thumbnail_impressions_ctr) == ("v1", D1, 50, 0.1)


def test_parse_header_only_is_a_valid_empty_day() -> None:
    parsed = parse_reach_csv((REACH_HEADER + "\n").encode())
    assert parsed.rows == [] and parsed.channel_level_rows == 0


def test_parse_bom_and_both_date_formats() -> None:
    data = ("﻿" + REACH_HEADER + "\n20261101,UC,v1,5,0.2\n2026-11-02,UC,v1,6,0.3\n").encode()
    assert [r.report_date for r in parse_reach_csv(data).rows] == [D1, D2]


def test_parse_percent_scale_is_converted_to_fraction() -> None:
    parsed = parse_reach_csv(reach_csv(D1, [("v1", 100, 5.2), ("v2", 100, 0.4)]))
    assert parsed.ctr_scale == "percent"
    assert [r.video_thumbnail_impressions_ctr for r in parsed.rows] == pytest.approx([0.052, 0.004])
    fixed = parse_reach_csv(reach_csv(D1, [("v1", 100, 0.4)]), ctr_scale="percent")
    assert fixed.rows[0].video_thumbnail_impressions_ctr == pytest.approx(0.004)


def test_parse_skips_and_counts_channel_level_rows() -> None:
    parsed = parse_reach_csv(reach_csv(D1, [("", 900, 0.05), ("v1", 100, 0.05)]))
    assert [r.youtube_video_id for r in parsed.rows] == ["v1"] and parsed.channel_level_rows == 1


def test_parse_empty_ctr_is_null() -> None:
    (row,) = parse_reach_csv(f"{REACH_HEADER}\n20261101,UC,v1,0,\n".encode()).rows
    assert row.video_thumbnail_impressions == 0 and row.video_thumbnail_impressions_ctr is None


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty file"),
        (b"date,channel_id,video_id,video_thumbnail_impressions\n", "missing column(s) video_thumbnail_impressions_ctr"),
        (f"{REACH_HEADER}\n20261101,UC,v1,lots,0.1\n".encode(), "line 2: bad impressions"),
        (f"{REACH_HEADER}\n20261101,UC,v1,5,0.1\nnot-a-date,UC,v2,5,0.1\n".encode(), "line 3: bad date"),
        (f"{REACH_HEADER}\n20261101,UC,v1\n".encode(), "line 2: 3 fields"),
        (f"{REACH_HEADER}\n20261101,UC,v1,-5,0.1\n".encode(), "negative"),
        (b"\xff\xfe\x00bad", "not UTF-8"),
    ],
)
def test_parse_rejects_malformed_files(data: bytes, message: str) -> None:
    with pytest.raises(ReportFormatError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_reach_csv(data)


def test_parse_time_handles_nanoseconds_and_z() -> None:
    assert parse_time("2026-11-08T10:00:00.123456789Z") == datetime(2026, 11, 8, 10, 0, 0, 123456, tzinfo=timezone.utc)
    assert parse_time("2026-11-08T10:00:00Z") == datetime(2026, 11, 8, 10, tzinfo=timezone.utc)
    assert parse_time(None) is None


def test_pacific_date_of_a_report_start() -> None:
    # Reports run midnight-to-midnight Pacific: 08:00 UTC in winter, 07:00 in summer.
    assert pacific_date(datetime(2026, 11, 1, 8, tzinfo=timezone.utc)) == D1
    assert pacific_date(datetime(2026, 7, 1, 7, tzinfo=timezone.utc)) == date(2026, 7, 1)


# ---- correction rule (pure) ------------------------------------------------------


def _rs(id_: int, day: date, hour: int, status: str = "pending") -> ReportState:
    return ReportState(id_, day, NOW.replace(hour=hour), status)


def test_resolve_one_report_per_date_downloads_pending_oldest_first() -> None:
    plan = resolve_reports([_rs(2, D2, 1), _rs(1, D1, 1), _rs(3, D3, 1, "downloaded")])
    assert plan.download == [1, 2] and plan.supersede == []


def test_resolve_newest_correction_wins() -> None:
    plan = resolve_reports([_rs(1, D1, 1), _rs(2, D1, 5)])
    assert plan.download == [2] and plan.supersede == [1]


def test_resolve_correction_after_download_replaces_it() -> None:
    plan = resolve_reports([_rs(1, D1, 1, "downloaded"), _rs(2, D1, 5)])
    assert plan.download == [2] and plan.supersede == [1]


def test_resolve_failed_newest_stays_failed_and_old_superseded_is_left() -> None:
    plan = resolve_reports([_rs(1, D1, 1, "superseded"), _rs(2, D1, 5, "failed")])
    assert plan.download == [] and plan.supersede == []


# ---- pipeline --------------------------------------------------------------------


async def _connect(sf, google: FakeGoogle) -> None:
    await store_connection(
        sf,
        TokenCipher(FERNET_KEY),
        TokenResponse(google.issue_access(), 3600, "rt-1", google.granted_scopes),
        channel_id=google.channel_id,
        channel_title=google.channel_title,
        now=datetime.now(timezone.utc),
    )


async def _poll(sf, google: FakeGoogle, now: datetime = NOW):
    if await get_credential(sf) is None:
        await _connect(sf, google)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        oauth = GoogleOAuthClient(http, client_id="c", client_secret="client-secret", redirect_uri="x", sleep=_no_sleep)
        tokens = TokenProvider(sf, oauth, TokenCipher(FERNET_KEY))
        return await run_reporting_poll(
            sf, ReportingClient(http, tokens, sleep=_no_sleep), channel_id=google.channel_id, now=now
        )


async def _all(sf, model):
    async with sf() as db:
        return (await db.execute(select(model).order_by(model.id))).scalars().all()


async def _reach(sf) -> list[tuple]:
    rows = await _all(sf, AnalyticsReachSnapshot)
    return sorted((r.youtube_video_id, r.report_date, r.video_thumbnail_impressions, r.video_thumbnail_impressions_ctr) for r in rows)


async def test_first_run_creates_a_job_and_waits_for_reports(session_factory) -> None:
    google = FakeGoogle()
    report = await _poll(session_factory, google)
    assert google.reporting.created == [{"reportTypeId": "channel_reach_basic_a1", "name": "youtube-suite-channel_reach_basic_a1"}]
    assert report.job_action == "created" and report.new_reports == 0
    assert report.waiting and "first report expected by" in report.waiting and not report.is_warning
    (job,) = await _all(session_factory, ReportingJob)
    assert (job.channel_id, job.status, job.youtube_job_id) == (CHANNEL_ID, "active", "job-1")
    assert job.last_polled_at == NOW


async def test_existing_google_job_is_adopted_not_duplicated(session_factory) -> None:
    google = FakeGoogle()
    google.reporting.add_job("channel_basic_a3")  # other type: ignored
    expired = google.reporting.add_job(expire=NOW - timedelta(days=1))
    live = google.reporting.add_job(created=NOW - timedelta(days=10))
    report = await _poll(session_factory, google)
    assert report.job_action == "adopted" and report.job_id == live != expired
    assert google.reporting.created == []
    # Next run reuses the DB row without listing jobs again.
    assert (await _poll(session_factory, google)).job_action == "reused"
    assert google.reporting.created == []


async def test_download_stores_rows_and_rerun_is_idempotent(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 1000, 0.05), ("", 5000, 0.04)]), created=NOW - timedelta(days=2))
    google.reporting.add_report(job, D2, reach_csv(D2, [("v1", 800, 0.06), ("v2", 50, 0.1)]), created=NOW - timedelta(days=1))
    google.reporting.add_report(job, D3, reach_csv(D3, []), created=NOW - timedelta(hours=2))  # header-only

    report = await _poll(session_factory, google)
    assert (report.new_reports, report.downloaded, report.rows, report.channel_rows_skipped) == (3, 3, 3, 1)
    assert report.waiting is None and not report.is_warning
    assert await _reach(session_factory) == [("v1", D1, 1000, 0.05), ("v1", D2, 800, 0.06), ("v2", D2, 50, 0.1)]
    reports = await _all(session_factory, ReportingReport)
    assert [(r.report_date, r.status, r.row_count) for r in reports] == [(D1, "downloaded", 1), (D2, "downloaded", 2), (D3, "downloaded", 0)]

    downloads = len(google.reporting.downloads)
    again = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert (again.new_reports, again.downloaded) == (0, 0)
    assert len(google.reporting.downloads) == downloads
    assert len(await _reach(session_factory)) == 3
    # The cursor drove `createdAfter` (minus the 1 h margin).
    assert google.reporting.report_list_calls[-1]["createdAfter"].startswith("2026-11-08T08:00:00")


async def test_report_listing_pages(session_factory) -> None:
    google = FakeGoogle()
    google.reporting.page_size = 2
    job = google.reporting.add_job(created=NOW - timedelta(days=10))
    for i in range(5):
        day = D1 + timedelta(days=i)
        google.reporting.add_report(job, day, reach_csv(day, [("v1", i + 1, 0.1)]), created=NOW - timedelta(days=5 - i))
    report = await _poll(session_factory, google)
    assert (report.new_reports, report.downloaded, report.rows) == (5, 5, 5)


async def test_correction_replaces_the_days_rows_including_dropped_ones(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 1000, 0.05), ("v2", 30, 0.1)]), created=NOW - timedelta(days=2))
    google.reporting.add_report(job, D2, reach_csv(D2, [("v1", 5, 0.2)]), created=NOW - timedelta(days=2))
    await _poll(session_factory, google)

    # YouTube re-issues D1 with revised numbers and v2 gone.
    google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 1100, 0.045)]), created=NOW + timedelta(hours=5))
    report = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert (report.new_reports, report.downloaded, report.superseded) == (1, 1, 1)
    assert await _reach(session_factory) == [("v1", D1, 1100, 0.045), ("v1", D2, 5, 0.2)]
    statuses = [(r.report_date, r.status) for r in await _all(session_factory, ReportingReport)]
    assert statuses == [(D1, "superseded"), (D2, "downloaded"), (D1, "downloaded")]


async def test_two_files_for_one_day_in_the_same_listing_downloads_only_the_newest(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    old = google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 1, 0.1)]), created=NOW - timedelta(days=2))
    new = google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 2, 0.1)]), created=NOW - timedelta(days=1))
    await _poll(session_factory, google)
    assert google.reporting.downloads == [new.id] and old.id not in google.reporting.downloads
    assert await _reach(session_factory) == [("v1", D1, 2, 0.1)]


async def test_failed_download_stays_pending_and_is_retried(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    bad = google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 10, 0.1)]), created=NOW - timedelta(days=1))
    google.reporting.download_fail.add(bad.id)
    report = await _poll(session_factory, google)
    assert (report.errors, report.downloaded, report.pending) == (1, 0, 1) and report.is_warning
    (row,) = await _all(session_factory, ReportingReport)
    assert row.status == "pending" and "HTTP 500" in row.last_error

    google.reporting.download_fail.clear()
    report = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert (report.errors, report.downloaded, report.pending) == (0, 1, 0)
    (row,) = await _all(session_factory, ReportingReport)
    assert row.status == "downloaded" and row.last_error is None


async def test_malformed_file_is_failed_and_stores_nothing(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    google.reporting.add_report(job, D1, b"date,video_id\n20261101,v1\n", created=NOW - timedelta(days=1))
    report = await _poll(session_factory, google)
    assert (report.failed, report.errors) == (1, 1)
    (row,) = await _all(session_factory, ReportingReport)
    assert row.status == "failed" and "missing column" in row.last_error
    assert await _reach(session_factory) == []


async def test_plain_body_without_gzip_also_works(session_factory) -> None:
    google = FakeGoogle()
    google.reporting.download_gzip = False
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    google.reporting.add_report(job, D1, reach_csv(D1, [("v1", 10, 0.1)]), created=NOW - timedelta(days=1))
    assert (await _poll(session_factory, google)).rows == 1


async def test_job_gone_at_google_is_expired_then_recreated(session_factory) -> None:
    google = FakeGoogle()
    await _poll(session_factory, google)  # creates job-1
    google.reporting.gone_jobs.add("job-1")
    report = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert report.is_warning and "no longer exists" in report.warnings[0]
    (job,) = await _all(session_factory, ReportingJob)
    assert job.status == "expired" and "not found" in job.last_error

    report = await _poll(session_factory, google, NOW + timedelta(days=2))
    assert report.job_action == "created" and report.job_id != "job-1"
    jobs = await _all(session_factory, ReportingJob)
    assert [(j.youtube_job_id, j.status) for j in jobs] == [("job-1", "expired"), (report.job_id, "active")]


async def test_job_expiry_from_google_is_stored_warned_and_acted_on(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=3))
    google.reporting.add_report(job, D1, reach_csv(D1, []), created=NOW - timedelta(days=1))
    google.reporting.job_expire_time = NOW + timedelta(hours=12)
    report = await _poll(session_factory, google)
    assert any("expire" in w for w in report.warnings)
    google.reporting.gone_jobs.add(job)  # it really expired
    report = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert report.job_action == "created"


async def test_other_channel_job_is_superseded(session_factory) -> None:
    google = FakeGoogle()
    await _poll(session_factory, google)
    google.channel_id = "UC" + "o" * 22
    google.reporting = FakeReporting(_n=100)  # the other account sees only its own jobs
    report = await _poll(session_factory, google, NOW + timedelta(days=1))
    assert report.job_action == "created"
    jobs = await _all(session_factory, ReportingJob)
    assert [(j.channel_id, j.status) for j in jobs] == [(CHANNEL_ID, "superseded"), (google.channel_id, "active")]


async def test_no_reports_after_72h_is_a_warning(session_factory) -> None:
    google = FakeGoogle()
    google.reporting.add_job(created=NOW - timedelta(hours=73))
    report = await _poll(session_factory, google)
    assert report.waiting is None and report.is_warning and "no reports 73 h" in report.warnings[0]


async def test_stale_pending_report_is_a_warning(session_factory) -> None:
    google = FakeGoogle()
    job = google.reporting.add_job(created=NOW - timedelta(days=60))
    bad = google.reporting.add_report(job, D1, reach_csv(D1, []), created=NOW - timedelta(days=46))
    google.reporting.download_fail.add(bad.id)
    report = await _poll(session_factory, google)
    assert any("still pending after 45 days" in w for w in report.warnings)


async def test_a_401_refreshes_once_and_retries(session_factory) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google)
    google.access_tokens.clear()  # the stored token is no longer accepted
    report = await _poll(session_factory, google)
    assert report.job_action == "created" and google.refresh_count == 1


# ---- cron task -------------------------------------------------------------------


async def _task(session_factory, google: FakeGoogle, monkeypatch, settings=None):
    monkeypatch.setattr(tasks, "get_settings", lambda: settings or oauth_settings())
    async with httpx.AsyncClient(transport=google.transport()) as http:
        return await tasks.reporting_daily_poll({"session_factory": session_factory, "http": http})


async def test_task_skips_when_not_configured_or_not_connected(session_factory, monkeypatch, caplog) -> None:
    google = FakeGoogle()
    caplog.set_level(logging.INFO, logger="app.reporting")
    unconfigured = oauth_settings(oauth_token_encryption_key=None)
    assert await _task(session_factory, google, monkeypatch, unconfigured) is None
    assert await _task(session_factory, google, monkeypatch) is None
    assert "not configured" in caplog.text and "no YouTube channel connected" in caplog.text
    assert google.reporting.created == []


async def test_task_skips_loudly_on_needs_reconsent(session_factory, monkeypatch, caplog) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google)
    await mark_reconsent(session_factory, "invalid_grant: Token has been expired or revoked.")
    caplog.set_level(logging.INFO, logger="app.reporting")
    assert await _task(session_factory, google, monkeypatch) is None
    assert any(r.levelno == logging.WARNING and "needs_reconsent" in r.message for r in caplog.records)
    assert google.reporting.created == []


async def test_task_runs_and_logs_one_line(session_factory, monkeypatch, caplog) -> None:
    google = FakeGoogle()
    # The task reads the real clock, so the fake's job must be created "now":
    # with the fake's fixed 2026-11-08 clock this test expired 48 h later.
    google.reporting.now = datetime.now(timezone.utc)
    await _connect(session_factory, google)
    caplog.set_level(logging.INFO, logger="app.reporting")
    report = await _task(session_factory, google, monkeypatch)
    assert report is not None and report.job_action == "created"
    assert "reporting poll: channel=" in caplog.text and "WAITING" in caplog.text


async def test_task_grant_dies_mid_run_is_a_warning(session_factory, monkeypatch, caplog) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google)
    google.access_tokens.clear()
    google.refresh_error = "invalid_grant"
    caplog.set_level(logging.INFO, logger="app.reporting")
    assert await _task(session_factory, google, monkeypatch) is None
    assert (await get_credential(session_factory)).status == "needs_reconsent"
    assert any(r.levelno == logging.WARNING and "stopped, needs_reconsent" in r.message for r in caplog.records)


def test_cron_is_registered_daily_after_the_analytics_pull() -> None:
    (job,) = [c for c in WorkerSettings.cron_jobs if c.name.endswith("reporting_daily_poll")]
    assert job.hour == 11 and job.minute == 0
