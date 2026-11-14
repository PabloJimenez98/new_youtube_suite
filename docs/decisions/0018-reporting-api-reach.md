# Decision: YouTube Reporting API integration, real CTR/impressions (M12)

**Date:** 2026-11-08
**Status:** Approved 2026-11-08 (two-phase integration test, the two-table data model, and one status line on the connection card).

## Context

- **What the plan asks for in M12:**
  - a `reporting_jobs` table and a create → poll → download pipeline for
    `channel_reach_basic_a1`
  - CSV parsing into `analytics_reach_snapshots` (§4.2, §6.3)
  - a design that treats this as a structurally different integration from M11: an async bulk
    job plus CSV download, not a live query
  - a poll cadence that tolerates the reporting lag
- **Builds on M11 (0017):**
  - the single connected account
  - `TokenProvider` / `get_access_token`, including the 401 → forced refresh → `needs_reconsent`
    handling
  - the `google_get` retry and quota policy
  - the status gates of the cron task

### API facts this design relies on

Sources: the Reporting API docs, read 2026-11-08 (`reports`, `channel_reports`, `jobs`,
`jobs.reports`, `jobs.reports/list`, `metrics`). The revision-history page returned 404. Points
marked *verify* are checked on the first live call.

- **Jobs:**
  - A job is created once per report type (`jobs.create {reportTypeId, name}`). YouTube
    "does not begin to generate your report until you create a reporting job".
  - **A Job has no status field.** Its fields are `id`, `reportTypeId`, `name`, `createTime`,
    `expireTime` and `systemManaged`.
  - `expireTime` is set if the report type is deprecated, or if the job's reports "have not
    been downloaded for a prolonged period". The docs give no duration.
- **Reports:**
  - Each Report covers one 24-hour period, midnight to midnight **Pacific** time. Its fields
    are `id`, `jobId`, `startTime` (inclusive), `endTime` (exclusive), `createTime`,
    `jobExpireTime` and `downloadUrl`.
  - Download is a GET on `downloadUrl` with the bearer token. `Accept-Encoding: gzip` is
    supported.
- **Timing:**
  - The first report becomes available within **48 h** of job creation. The jobs page says
    24 h; we plan for 48.
  - On creation YouTube also generates **30 days of historical reports**, "typically within a
    couple of days".
  - After that, one new report per day.
- **Retention:**
  - Daily reports are downloadable for **60 days** after generation.
  - Historical reports are downloadable for **30 days**.
- **Backfill:**
  - A correction arrives as a **new report ID** with the same `startTime`/`endTime` and a newer
    `createTime`.
  - The client should replace that period's data and keep only the newest `createTime`.
  - `createdAfter` on `jobs.reports.list` catches both new and backfilled reports.
- **Empty days:** a day with no data still gets a report containing just the header row. Rows
  with no metrics are omitted.
- **`channel_reach_basic_a1` columns:**
  - dimensions `date`, `channel_id`, `video_id`
  - metrics `video_thumbnail_impressions`, `video_thumbnail_impressions_ctr`
  - Columns must be mapped **by header name**, because new columns can appear.
  - A `video_id` of `""` means a channel-level row or a deleted video.
  - *Verify:* the exact header spelling, the date format (`YYYY-MM-DD` per the overview; we
    accept `YYYYMMDD` too) and the **CTR scale** (the docs say "percentage", but not whether
    0–1 or 0–100).
- **Scope:**
  - The docs list `yt-analytics.readonly` and `yt-analytics-monetary.readonly`. We already hold
    `yt-analytics.readonly`, and reach is not monetary.
  - *Verify* on the first `jobs.create`. A 403 for insufficient scope would mean a re-consent
    with an extra scope, which would go back to the owner.
- **Not documented:** quota, a per-channel job limit, the CSV encoding (we assume UTF-8, with a
  BOM tolerated).

## Decisions

### 1. Data model: the job is a subscription, the report file is the unit of work

The §4.2 `reporting_jobs.status` enum (`created/pending/ready/failed`) assumed a job finishes
once. In the real API a job runs indefinitely and readiness belongs to each daily report.
Therefore:

**`reporting_jobs`**: one row per (channel, report type) subscription

| Column | Type | Notes |
|---|---|---|
| id | PK | |
| channel_id | TEXT | the connected channel the job belongs to |
| report_type | TEXT | `channel_reach_basic_a1` |
| youtube_job_id | TEXT UNIQUE | |
| status | TEXT (`active`/`expired`/`superseded`) | CHECK. Our lifecycle, not Google's: `expired` = `expireTime` passed or the job returned 404; `superseded` = a different channel was connected. |
| job_created_at | TIMESTAMPTZ | Google's `createTime`; drives the "waiting for first report" messages |
| expire_time | TIMESTAMPTZ NULL | from Google |
| reports_cursor | TIMESTAMPTZ NULL | the newest report `createTime` seen; the next `createdAfter` |
| last_polled_at | TIMESTAMPTZ NULL | (§4.2) |
| last_error | TEXT NULL | |

A partial unique index allows only one `active` job per (channel_id, report_type).

**`reporting_reports`** (new; §4.2 doesn't have it): one row per report file

| Column | Type | Notes |
|---|---|---|
| id | PK | |
| job_id | FK → reporting_jobs | |
| youtube_report_id | TEXT UNIQUE | |
| start_time, end_time | TIMESTAMPTZ | |
| report_date | DATE | the Pacific date of `start_time` |
| create_time | TIMESTAMPTZ | |
| status | TEXT (`pending`/`downloaded`/`superseded`/`failed`) | CHECK. This is where §4.2's "pending/ready/failed" lives. |
| row_count | INT NULL | rows stored (0 for a header-only file) |
| downloaded_at | TIMESTAMPTZ NULL | |
| last_error | TEXT NULL | |

**`analytics_reach_snapshots`**: the §4.2 columns plus bookkeeping

| Column | Type | Notes |
|---|---|---|
| id | PK | |
| channel_id | TEXT | as on every analytics table (0017 §4) |
| youtube_video_id | TEXT | |
| report_date | DATE | the metric day, Pacific (YouTube's day) |
| video_thumbnail_impressions | BIGINT | |
| video_thumbnail_impressions_ctr | FLOAT | stored as a **fraction 0–1**; the scale is converted in the parser once verified |
| source_report_id | FK → reporting_reports | which file the row came from |
| pulled_at | TIMESTAMPTZ | |

- **Unique key:** (`channel_id`, `youtube_video_id`, `report_date`).
- **Unmatched videos:** a row is not tied to `analytics_videos`. A video id we don't know is
  still stored, since the report is authoritative.
- **Channel-level rows** (`video_id = ""`) are skipped and counted in the log line. The spec's
  CTR predictor needs per-video pairs.

### 2. The pipeline: one daily task, `reporting_daily_poll`

The cron runs on the existing worker **daily at 11:00 UTC**, after M11's 10:15. Reports appear
once a day, so polling more often only finds nothing new. It writes no `jobs` rows and one log
line per run, as in 0015 and 0017. Steps:

1. **Status gate:** the same as `analytics_daily_pull`: not configured, not connected,
   `needs_reconsent` (WARNING), `ReconsentRequired` mid-run, `OAuthConfigError`.
2. **Ensure the job:**
   - If there is no `active` row for (connected channel, `channel_reach_basic_a1`), call
     `jobs.list`. If a job for that type already exists at Google (for example after a DB
     reset), adopt it. Otherwise `jobs.create`.
   - Store `createTime` and `expireTime`.
   - A job row for a different channel becomes `superseded`. We can't delete it at Google,
     because the token belongs to the other account; it expires on its own.
3. **List new reports:**
   - `jobs.reports.list` with `createdAfter = reports_cursor − 1 h`. The margin guards against
     equal timestamps. Page with `pageToken`.
   - Insert unseen report IDs as `pending`.
   - A 404 on the job marks it `expired`, and the next run creates a new job.
4. **Resolve backfills:** if a date has more than one report, only the newest `create_time`
   counts. Older `pending` ones become `superseded` without being downloaded, and an older
   `downloaded` one is replaced in step 5.
5. **Download and store:** for each remaining `pending` report, oldest date first:
   - GET `downloadUrl` (bearer token, gzip)
   - parse the CSV by its header
   - in one transaction, **replace that date's rows** for the channel (delete, then insert),
     because a correction may also drop rows
   - mark the report `downloaded` with its `row_count` and advance `reports_cursor`
   - If a download fails, the report stays `pending` with `last_error`, the run continues, and
     the next run retries it.
6. **Report the lag, never silently:**
   - No reports and a job younger than 48 h is the normal waiting state. It gives an INFO line:
     "job created at X, first report expected by X+48 h".
   - **No reports after 72 h** gives a WARNING.
   - A `pending` report older than **45 days**, which is near the 60-day retention, gives a
     WARNING too.

**Missed runs:** reports stay downloadable for 60 days (30 for historical ones), and the cursor
starts from the last report we saw. A worker that is down for weeks loses nothing. A daily
download also stops Google from expiring the job for inactivity.

**Cost:** about one `jobs.list`/`jobs.create` call the first time, then one or two list calls
and one download per day. This is negligible, and the Reporting API's quota is separate from
the Data API's 10k pool.

### 3. CSV parser: pure, header-driven, strict about required columns

`parse_reach_csv(bytes) -> ParsedReport` is the function behind the unit tests:

- Decodes UTF-8, strips a BOM and maps columns **by header name**. Unknown extra columns are
  ignored.
- A missing required column raises `ReportFormatError`, so the report becomes `failed` instead
  of storing wrong data.
- A header-only file → 0 rows, valid.
- `date` is accepted as `YYYY-MM-DD` or `YYYYMMDD`.
- **CTR scale:** one constant (`CTR_SCALE`), set once the live data is checked. Until then, a
  value above 1 is treated as a percentage and divided by 100. A guard warns if a file mixes
  values above and below 1 in a way that makes the scale ambiguous.
- A malformed number in a row → `ReportFormatError` naming the line.
- `video_id = ""` rows are counted, not returned.

### 4. Code layout

- **`app/reporting/`:**
  - `reporting_api.py`: the httpx client for `jobs.list/create`, `jobs.reports.list` and the
    download, using `TokenProvider` and `google_get`
  - `csv_parser.py`
  - `selection.py`: the pure backfill/supersede rule
  - `poll.py`: orchestration
  - `tasks.py`: the cron entry
- **Models and migrations:** `app/models/reporting.py`; migration
  `0012_create_reporting_tables.py`.
- **CLI:**
  - `python -m app.cli reporting poll`: one run now
  - `python -m app.cli reporting status`: the job, its age, report counts by status, the
    newest `report_date` and the reach row count
- **Frontend: one status line on the YouTube connection card** (owner, 2026-11-08):
  - `GET /youtube/connection` gains a `reach` object, or `null` if there is no job:
    `{job_status, job_created_at, latest_report_date, reports_pending, reports_failed,
    last_error}`.
  - The card shows one line:
    - "Reach reports: waiting for first report (job created …)"
    - "Reach reports: up to <date>"
    - a warning-coloured "<n> failed" / `last_error` when there are failures
  - No buttons. The Analytics charts stay on dummy data (M10 backlog).

## Test plan

**Unit (default suite, no network: `httpx.MockTransport` + `tests/google_fake.py` extended
with the Reporting endpoints, injectable clock):**

- **CSV parsing (the plan's item):**
  - the happy path
  - reordered columns
  - an extra unknown column
  - a header-only file
  - a BOM
  - gzip and plain bodies
  - `YYYY-MM-DD` and `YYYYMMDD` dates
  - CTR as a fraction and as a percentage
  - a `video_id=""` row skipped and counted
  - a missing required column → `ReportFormatError`
  - a non-numeric metric → an error naming the line
- **Backfill selection (pure):**
  - one report per date
  - two reports for the same date → only the newest counts
  - a newer one arriving after the older one was downloaded → replace
- **Pipeline:**
  - ensure-job: create when none exists; adopt an existing Google job via `jobs.list`; reuse
    the DB row
  - a different channel → the old row is `superseded` and a new job is created
  - a 404 job → `expired`, then recreated on the next run
  - report listing with `createdAfter` and paging; the cursor advances
  - download + store; a re-run is idempotent (no duplicate rows, no re-download)
  - a backfill replaces the day's rows, including a dropped row
  - a failed download stays `pending` and is retried on the next run
  - a header-only report → `downloaded`, `row_count=0`
  - the lag messages: < 48 h INFO; > 72 h with no reports WARNING; a `pending` report
    > 45 days WARNING
  - a 401 → refresh-retry, the M11 path, reused
  - skips for not configured, not connected and `needs_reconsent`
- **Migration 0012:** upgrade/downgrade, the CHECK constraints, and the partial unique index on
  `active`.

**Integration (opt-in marker `youtube_reporting`, deselected by default like `youtube_oauth`;
real Google, needs the connected test channel):**

The reporting lag makes a single in-process "poll to completion" impractical (up to 48 h), so
the test is **two phases**, with the wait written into the test:

1. **Phase 1, any time:**
   - Run `reporting_daily_poll` against the real channel.
   - Assert that a real job exists (created or adopted), with `channel_reach_basic_a1` and a
     `createTime`.
   - This also verifies the scope.
   - If the job is younger than 48 h and has no reports, the test **passes with an explicit
     printed message**: "job created at X; first reports expected by X+48 h, 30-day history
     within a few days; re-run then".
2. **Phase 2, ≥ 48 h after job creation:**
   - The same test now asserts at least one `reporting_reports` row and that every report is
     `downloaded` or `superseded`, with 0 `failed`.
   - The reach rows are schema-valid: CTR in 0–1, impressions ≥ 0.
   - A second run is idempotent.
   - Test channel data: the 7 private shorts have no impressions, so header-only reports
     (0 rows) are the expected outcome, as in M11. Real rows depend on the open M11 follow-up
     (views on the channel).
   - The test prints the first downloaded CSV's header row, which settles the *verify* points:
     header spelling, date format and CTR scale.

**Frontend:** card spec for the reach line: absent (`null`), waiting, up to date, with failures; router test for the `reach` payload.

**Regression:** the full CPU suite + `ng test`. The GPU suite isn't touched.

## Alternatives rejected

- **The §4.2 status enum on `reporting_jobs`:** it doesn't match the API, where a job never
  becomes "ready" (see §1).
- **`channel_reach_combined_a1` now:** the plan names basic. The traffic-source/device
  breakdown is a later addition if the CTR model wants it (brainstorm).
- **Polling several times a day:** reports are produced daily, so extra polls find nothing.
- **A single long-running integration test that sleeps until the report exists (up to 48 h):**
  it would look like a hang or a flake. The two-phase test makes the wait explicit instead.
- **`google-api-python-client` for the media download:** the download is one authorized GET,
  as with the rest of M10/M11 in plain `httpx`.

## Implementation notes (2026-11-08)

### Deviations and additions to the plan above

- **`reporting_reports.download_url`:** stored with each report, so a failed download can be
  retried on a later run without listing the report again.
- **Shared HTTP policy:** `google_get` now wraps a new `google_request(method, …)`, which
  returns the raw response. That gives `jobs.create` (POST) and the CSV download (bytes) the
  same retry, quota and 401 handling. The M10/M11 callers are unchanged.
  - `jobs.create` is a POST and is retried on 5xx. A duplicate that slips through is harmless:
    the next run adopts the oldest live job via `jobs.list`.
- **Adoption picks the oldest live job** of the type. An expired one (`expireTime` in the past)
  is never adopted.
- **`jobExpireTime` on a report** is copied to `reporting_jobs.expire_time` and raised as a
  warning. Once it has passed, the next run marks the row `expired` and creates a new job.
- **`createdAfter` margin:** 1 h before the cursor. A report seen twice is a no-op, because of
  `ON CONFLICT DO NOTHING` on `youtube_report_id`.
- **Duplicate (video, date) rows within one file:** the last one wins. This should not occur
  in basic_a1.
- **Negative numbers** in the CSV are a `ReportFormatError`, and so is a CTR above 100 %.
- **The integration test adopts the real job on every run.** The test DB is wiped per test, so
  each run lists every report still available. This exercises adoption against Google without
  creating duplicates.

### Tests

- **New default-suite files:**
  - `tests/test_reporting.py`: parser, correction rule, pipeline, task gates, cron
  - a reach-payload test in `tests/test_youtube_router.py`
  - `test_0012_…` in `tests/test_migrations.py`, covering the partial unique index and the
    status CHECK
- **Fake:** `FakeReporting` in `tests/google_fake.py` (jobs, paged report lists,
  `createdAfter`, gzip downloads, 404 jobs, failing downloads).
- **Opt-in:** `tests/test_reporting_integration.py`, marker `youtube_reporting`, deselected by
  default.
- **Frontend:** five reach-line specs in `youtube-connection.spec.ts`.
- **Regression run (2026-11-08, WSL clone):**
  - backend: 520 passed, 10 skipped, 4 failed
  - frontend: 155/155
  - The 4 failures are diarization tests that need `HF_TOKEN`, missing because the WSL clone
    has no `backend/.env` (brainstorm). They are unrelated to M12.
- **Live phases:** not run yet. They need `backend/.env` in the WSL clone (phase 1), then
  ≥ 48 h of waiting (phase 2).

## Live phase 1 (2026-11-08)

- **Setup gap found and fixed:**
  - The first run failed with HTTP 403 "YouTube Reporting API has not been used in project …
    or it is disabled". The API has to be enabled in the Cloud project, like the Analytics API
    in M11.
  - The owner enabled it, and the README setup guide now lists it.
  - The 403 is non-retryable, so it surfaced as one clear error rather than as retries.
- **Scope verified:** `jobs.list` and `jobs.create` succeed with the M11 grant
  (`yt-analytics.readonly` + `youtube.readonly`), so no new consent is needed.
- **Real job:** `14f8221d-2fce-44d3-9801-39b3aa45b520` (`channel_reach_basic_a1`, channel
  `UCJEybqMK_7qMTzvKm4jyp5g`), created 2026-11-08 12:38 UTC by
  `pytest -m youtube_reporting -s`, which passed phase 1.
- **Dev DB adoption:** `python -m app.cli reporting poll` then **adopted** the same job in the
  dev DB (no duplicate). The card shows "waiting for the first report".
- **Phase 2:** done on 2026-11-14 (see below). The original note: re-run `pytest -m youtube_reporting -s` after **2026-11-10 12:38 UTC**,
  then pin `CTR_SCALE` from the printed CSV header and values. Header-only files are expected
  while the test channel has no impressions.

## Live phase 2 (2026-11-14): milestone closed

- **Run:** `pytest -m youtube_reporting -s` in the WSL clone, 140.6 h after job creation.
  **1 passed.**
- **Job:** `14f8221d-…` was adopted again (no duplicate job).
- **Reports:** 33 new reports, starting from 2026-10-09 (the ~30-day backfill). All 33 are
  `downloaded`: 0 superseded, 0 failed, 0 pending. The run made 35 API calls with 0 errors.
- **Rows:** 0 reach rows and 0 channel-level rows. Every file has a header only, as expected
  while the test channel has no thumbnail impressions.
- **Header verified:**
  `date,channel_id,video_id,video_thumbnail_impressions,video_thumbnail_impressions_ctr`.
  It matches `REQUIRED_COLUMNS`. `channel_id` is extra and is ignored, as designed.
- **Not verifiable yet:** the date format and the CTR scale need at least one data row.
  - `CTR_SCALE` stays `"auto"`. That is its designed pre-verification mode: a file with any
    value above 1 is read as percentages.
  - **Follow-up (open, non-blocking):** when the first non-empty reach file arrives, check
    the date format and the CTR values, then pin `CTR_SCALE` to `"fraction"` or `"percent"`.
- **M12 is done.** No code changed in this phase, so the regression baseline stays the one from
  the 0021–0023 commit.
