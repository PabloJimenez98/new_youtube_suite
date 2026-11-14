# Decision: OAuth + YouTube Analytics API v2 (M11)

**Date:** 2026-11-07
**Status:** Approved 2026-11-07; implemented (see "Implementation notes" at the end). The live checks against the real test channel are pending until the owner connects it.

## Context

- **What the plan asks for in M11:**
  - an OAuth consent flow for **one** platform-level connected account (not multi-tenant,
    `v1-scope.md`)
  - encrypted credential storage
  - daily `reports.query` pulls into `analytics_snapshots` and `analytics_retention_curve`
    (§4.2, §6.2); retention needs one call per video
  - explicit test coverage of token refresh and expiry, a classic silent-failure source
- **Decided with the owner (2026-11-07):**
  - **Channel:** a **new test channel**, created for this. With no videos or views, the tables
    stay empty; see "Test channel data" below.
  - **Consent:** a button in the app: backend endpoints plus a connection card in the
    frontend that shows the status.
  - **Google Cloud:** Claude writes the step-by-step guide (README) and the owner creates the
    client.
  - **Retention cadence:** the same rule as M10. Videos published ≤ 90 days ago are pulled
    daily, older ones weekly.

### API facts this design relies on

Sources: the Analytics API docs, read 2026-11-07 (`channel_reports`, `reports/query`,
`data_model`). Points marked *verify* are checked with the first live call during
development.

- **Scope for `reports.query`:** the query reference says requests "now require access to the
  `youtube.readonly` scope", alongside `yt-analytics.readonly`.
- **Per-video daily data:**
  - `dimensions=day,video` is allowed with a multi-value `video==id1,id2,…` filter (up to 500
    IDs) and returns one row per video per day.
  - All nine §6.2 metrics are listed for day reports. *Verify* likes, comments and shares
    (one summary contradicted that).
- **Retention:**
  - `dimensions=elapsedVideoTimeRatio`, filtered to **exactly one** video. It returns 100
    points.
  - Metrics: `audienceWatchRatio`, `relativeRetentionPerformance`, `startedWatching`,
    `stoppedWatching`, `totalSegmentImpressions`. The last three were added in May 2024.
  - The docs don't say whether the curve is cumulative over startDate–endDate. *Verify*.
- **Data lag:** 48–72 hours. Responses stop at the last day for which every metric is
  complete. Small audiences may be omitted (anonymization).
- **Quota:** 1 unit per request, from the Analytics API's own quota (Cloud console). The Data
  API's 10k pool is not touched.
- **Google OAuth "Testing" status:**
  - Refresh tokens expire **7 days** after consent for any scope beyond openid, email and
    profile.
  - "In production" without verification shows an *unverified app* warning and caps the app
    at 100 users. Tokens don't expire on a timer.
  - **Choice: publish the app as "In production", unverified**, for personal use. The warning
    is acceptable and the weekly forced re-consent is not.
  - **Amended 2026-11-08: publishing is deferred to the end of the project.** "In production"
    for an external app requires a public homepage URL and privacy-policy URL on an authorized
    domain (e.g. a `github.io` page), which the owner chose not to set up yet. Until then the
    app stays in **Testing** with the owner's account as a test user, and the owner re-connects
    the channel from `/analytics` whenever the 7-day expiry flips it to `needs_reconsent`. No
    code depends on the publishing status. Missing a few days costs no data: pulls re-read
    windows from the publish date or the last 7 days, and retention curves are lifetime. The
    cost is manual re-consent (and stopped daily pulls until then). Publishing is a console-only
    change; after it, re-connect once to get a refresh token without the 7-day limit.

## Decisions

### 1. Consent flow: OAuth web-server flow behind the Angular proxy

**Endpoints** (new router `app/routers/youtube.py`, all under `/youtube`, and `/youtube` is
added to `frontend/proxy.conf.json`):

| Endpoint | Auth | What it does |
|---|---|---|
| `GET /youtube/connection` | app user | Returns the status: `not_configured` (no client id/secret/key in `.env`), `disconnected`, `connected` or `needs_reconsent`. Also returns the channel id and title, the scopes, `connected_at`, `last_refresh_at` and `last_error`. |
| `POST /youtube/oauth/start` | app user | Returns the Google authorization URL. It carries `access_type=offline`, `prompt=consent` (so a refresh token always comes back), `include_granted_scopes=true` and a signed `state`. |
| `GET /youtube/oauth/callback` | none (a browser redirect from Google) | 1. Checks `state`. 2. Exchanges the code. 3. Checks that both scopes were granted and that a refresh token is present. 4. Calls `channels.list mine=true`. 5. Upserts the credential row. 6. Redirects to `/analytics?youtube=connected`, or to `?youtube=error&reason=…`. |
| `DELETE /youtube/connection` | app user | Revokes the token at Google (`oauth2.googleapis.com/revoke`, best effort) and deletes the row. The analytics rows already pulled are kept. |

- **Redirect URI:** `http://localhost:4200/youtube/oauth/callback` by default. It goes through
  the Angular dev proxy, so the browser stays on one origin and no CORS change is needed.
  Configurable with `GOOGLE_OAUTH_REDIRECT_URI`.
- **`state`:** a JWT signed with `JWT_SECRET`. It holds the app user id, a nonce, a
  `purpose: youtube_oauth` claim and a 10-minute expiry. A tampered, expired or foreign state
  is rejected and redirects with `reason=state`.
- **PKCE:** not used. This is a confidential web client with a secret, and putting the
  verifier in `state` would defeat it.
- **Scopes:** `yt-analytics.readonly` + `youtube.readonly`, both read-only.
  - `youtube.readonly` is required by `reports.query` (above). It also gives `channels.list
    mine=true` (the channel id) and the uploads playlist **including unlisted and private
    videos**, which the public API key can't see.
  - §4.2 lists `yt-analytics.readonly` as an example. Adding `youtube.readonly` is the change
    recorded here.
  - M12's Reporting API uses the same scope, so no new consent will be needed.
- **No client library:** a hand-written `httpx` client, like M10. The flow is three POSTs (token
  exchange, refresh, revoke). Neither `google-auth-oauthlib` nor `google-api-python-client` is
  added.

### 2. Credential storage: `oauth_credentials`, Fernet-encrypted

**`oauth_credentials`** keeps the §4.2 columns and adds bookkeeping:

| Column | Type | Notes |
|---|---|---|
| id | PK | |
| provider | TEXT UNIQUE | `youtube`. The unique key enforces the single row. |
| access_token_enc, refresh_token_enc | TEXT | Fernet tokens (encrypted at rest) |
| scopes | TEXT[] | as granted |
| connected_channel_id | TEXT | from `channels.list mine=true` |
| connected_channel_title | TEXT | shown on the card |
| token_expiry | TIMESTAMPTZ | access-token expiry |
| status | TEXT (`active`/`needs_reconsent`) | CHECK constraint |
| last_error | TEXT NULL | why re-consent is needed, e.g. `invalid_grant: Token has been expired or revoked.` |
| connected_at, last_refresh_at, updated_at | TIMESTAMPTZ | |

- **Encryption:** `cryptography.Fernet` (AES-128-CBC + HMAC).
  - The key comes from `OAUTH_TOKEN_ENCRYPTION_KEY` in `.env`; the README shows how to generate
    it.
  - `cryptography` is already installed through `python-jose[cryptography]`. It gets an
    explicit pin in `requirements.txt` because we now import it directly.
  - **A missing key** gives status `not_configured`: the endpoints refuse to connect and the
    cron logs a skip.
  - **A wrong key**, for example one rotated by mistake, gives `InvalidToken`. That is
    reported as `needs_reconsent` with `last_error="token decryption failed"`, never as a
    crash. Re-consent rewrites the row with the new key. Key rotation is not built.
- **Config:** `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` (SecretStr),
  `GOOGLE_OAUTH_REDIRECT_URI`, `OAUTH_TOKEN_ENCRYPTION_KEY` (SecretStr) and
  `FRONTEND_BASE_URL` (default `http://localhost:4200`, used for the post-callback redirect).
  An empty value means unset, as with `YOUTUBE_API_KEY`.

### 3. Token refresh: one function, failures made loud

- **`app/youtube_oauth/tokens.py: get_access_token(session)`** is the only way to get an access
  token.
  - **More than 120 s of validity left:** returns the decrypted token, with no network call.
  - **Otherwise:** refreshes it at `oauth2.googleapis.com/token`, then stores the new access
    token, its expiry, `last_refresh_at`, and the refresh token if Google rotated it. The row
    is read with `SELECT … FOR UPDATE`, so two concurrent callers refresh only once.
- **Failure handling:**

| Failure | Handling |
|---|---|
| `400 invalid_grant` (revoked, expired, password changed, 7-day Testing expiry) | Sets `status=needs_reconsent` and `last_error`, logs a WARNING and raises `ReconsentRequired`. |
| Decryption failure | Handled the same way as `invalid_grant`. |
| `invalid_client` or `unauthorized_client` (bad client id/secret in `.env`) | Raises `OAuthConfigError`. The status is unchanged: re-consent wouldn't help, so this is a config problem. |
| Network errors and 5xx | 3 retries with backoff, then a plain error. The status is unchanged, so a transient outage never asks for re-consent. |
| `401` from the Analytics or Data API with a token we believed valid | One forced refresh and one retry. A second 401 sets `needs_reconsent`. |

- **Never silent:** `needs_reconsent` is a stored status.
  - The cron run logs at WARNING and stops.
  - `GET /youtube/connection` reports it, and the frontend card shows a "Reconnect" call to
    action with `last_error`.
  - A run that fails for another reason logs at ERROR and changes nothing.
- **Clock:** injectable (`now`), so the expiry tests don't sleep.

### 4. Own-channel video list: `analytics_videos`

The Analytics API reports metrics, not the catalogue, and §4.2 has no table for the connected
channel's videos. As with `market_videos` (0015 §2), M11 adds one:

**`analytics_videos`**
| Column | Type | Notes |
|---|---|---|
| id | PK | |
| youtube_video_id | TEXT UNIQUE | |
| title | TEXT | latest |
| published_at | TIMESTAMPTZ | drives the 90-day rule and the start of each video's window |
| privacy_status | TEXT | `public`/`unlisted`/`private` |
| discovered_at | TIMESTAMPTZ | |
| removed_since | DATE NULL | set when the video leaves the uploads playlist; it then leaves the pull |

- **How the list is built:** every daily run pages through the uploads playlist
  (`playlistItems.list` with OAuth, 1 Data API unit per 50 videos), then calls `videos.list
  part=status` for privacy. For a test channel this is a handful of units.
- **No link to `uploads`:** `analytics_videos` is not linked to the app's `uploads` table.
  Matching an upload to a published video is open (brainstorm).
- **Changing the connected channel:** connecting a different channel than the stored
  `connected_channel_id` keeps the old rows but drops them from the pull, so the analytics of
  two channels don't mix. Every analytics table carries `channel_id`.

### 5. The pulls

New cron task `analytics_daily_pull` on the existing worker, **daily at 10:15 UTC** (after
M10's 09:30). It does no `jobs` rows and writes one log line per run, as in 0015 §4.

1. **Status gate:** `not_configured`, no row, or `needs_reconsent` → log a skip (WARNING for
   `needs_reconsent`) and return.
2. **Refresh the video list** (§4).
3. **Engagement → `analytics_snapshots`:**
   - `dimensions=day,video`, `filters=video==<≤200 ids>`, with the nine §6.2 metrics and
     `startIndex`/`maxResults` paging.
   - **Window:** from the video's publish date on its first pull (a backfill), then the last
     **7** days on every later run. The lag is 48–72 h and late days get revised.
   - **Upsert key:** (`youtube_video_id`, `snapshot_date`). Here `snapshot_date` **is the
     metric day**, not the pull date, so the table is one row per video per day and re-pulls
     overwrite revised days. The meaning is written in the model docstring.
   - Adds a `pulled_at` column.
4. **Retention → `analytics_retention_curve`:**
   - For each **due** video (M10's rule, 0015 §3: published ≤ 90 days and no curve pulled
     today yet; or older and its latest curve is ≥ 7 days old), one call: `dimensions=elapsedVideoTimeRatio`,
     `filters=video==<id>`, the five metrics, startDate = publish date, endDate = today. That
     is the lifetime curve.
   - `snapshot_date` is the **pull date**. A pull stores up to 100 rows. Upsert key:
     (`youtube_video_id`, `snapshot_date`, `elapsed_video_time_ratio`).
   - Videos with no retention data yet (no views, or anonymized) return 0 rows. That counts
     as "pulled, empty", is logged, and is not an error.
   - `audienceType` is not filtered (all traffic), so the numbers match YouTube Studio's default.
5. **API errors:** 403 `quotaExceeded` stops the run and keeps the rows already written. 429
   and 5xx get 3 retries. This reuses the M10 `youtube_api` retry helper, refactored to take
   a bearer token as well as the key.
6. **Cost, for a channel with V videos:** about ⌈V/200⌉ engagement calls, at most V retention
   calls (fewer after the first week), and ⌈V/50⌉ + 1 Data API units. That is negligible for
   both quotas.

A missed run is not lost:
- Engagement re-reads 7 days, so a run that is down for up to 4–5 days self-heals.
- Retention uses the snapshot-age rule.

The M10 limitation therefore doesn't apply to these tables.

### 6. Frontend: a "YouTube connection" card on the Analytics page

- **Location:** the top of `/analytics`. The page's charts keep their dummy data; wiring
  them to real data is still the backlog item from M10.
- **States:**
  - not configured: points to the README
  - disconnected: a "Connect YouTube" button
  - connected: channel title and id, last refresh, and "Disconnect", behind a confirm dialog
  - needs re-consent: a warning banner with `last_error` and a "Reconnect" button
- **Connect:** `POST /youtube/oauth/start`, then `window.location.assign(url)`. On return,
  `?youtube=connected|error` shows a snackbar and the card re-reads its status.
- **Manual pull:** none in the UI. The CLI gets `python -m app.cli analytics pull` (one run
  now) and `analytics status`.

### 7. Code layout

- **`app/youtube_oauth/`:**
  - `crypto.py`: Fernet wrapper
  - `state.py`: the signed state JWT
  - `google_oauth.py`: the httpx token, refresh and revoke client
  - `tokens.py`: `get_access_token` and the status transitions
- **`app/analytics/`:**
  - `analytics_api.py`: the `reports.query` client and the columnHeaders→rows parser
  - `pull.py`: orchestration
  - `selection.py`: the retention due-rule, pure, reusing M10's rule
  - `tasks.py`: the cron entry
- **Models and migrations:**
  - `app/models/analytics.py` and `app/models/oauth.py`
  - migration `0011_create_oauth_and_analytics_tables.py`
- **Other:** `app/routers/youtube.py`, the CLI subcommand, and the frontend card component.

## Test plan

**Unit (default suite, no network: `httpx.MockTransport`, injectable clock):**

- **Token refresh** (the plan's flagged item):
  - Valid token: no HTTP call.
  - Simulated expiry (clock past `token_expiry - 120 s`): one refresh, with the new token and
    expiry stored.
  - Refresh-token rotation is stored.
  - `invalid_grant` → `needs_reconsent` + `last_error` + `ReconsentRequired`.
  - `invalid_client` → config error, status unchanged.
  - 5xx/network ×3 → error, status unchanged.
  - A 401 from the API → one forced refresh and a retry; a second 401 → `needs_reconsent`.
  - Two concurrent callers → one refresh (`FOR UPDATE`).
  - Undecryptable token → `needs_reconsent`.
- **Crypto:** roundtrip; wrong key → `InvalidToken`; stored values aren't plaintext (the test
  reads the DB column directly).
- **State:** a valid state works; expired, tampered, wrong-purpose and missing state are each
  rejected.
- **Callback:**
  - Happy path: one row, scopes stored, redirect to `connected`.
  - Missing refresh token, missing scope, Google `error=access_denied` and a bad state each
    redirect with a reason and write no row.
  - Re-consent overwrites the row and resets the status to `active`.
  - A different channel → the old videos drop out of the pull.
- **Endpoints:** the status payload for all four states; `DELETE` revokes and deletes; auth is
  required except on the callback; no endpoint ever returns tokens.
- **Analytics parsing and pull:**
  - columnHeaders→rows mapping
  - `startIndex` paging
  - batching ≤ 200 IDs
  - window selection: backfill from the publish date vs 7 days
  - upserts idempotent on re-run
  - empty retention → 0 rows, no error
  - quota stop
  - cron skip for `not_configured` and `needs_reconsent`
  - video-list refresh: new, removed and privacy change
- **Retention due-rule:** M10-style cases: recent, older, older-and-fresh, removed.

**Integration (opt-in marker `youtube_oauth`, real Google, like M10's `youtube_api`):**

- **Precondition:** the owner has connected the test channel through the app.
- Force a refresh against Google and check that the new expiry is in the future.
- Run `analytics_daily_pull` against the real channel:
  - `analytics_videos` matches the channel's uploads, unlisted included
  - the engagement and retention calls succeed, with schema-valid rows or 0
  - a second run is idempotent
- **Test channel data:** with no views, 0 rows is the expected, documented outcome. For the
  tables to get real rows, the owner uploads 1–2 short videos (public or unlisted), watches
  them from another account or device, and waits **48–72 h** for the Analytics lag. The test
  prints the row counts and asserts on their shape only when rows exist.

**Manual (owner, written as a checklist in the research or README):**

1. Connect from `/analytics`.
2. Revoke at myaccount.google.com/permissions.
3. Run `python -m app.cli analytics pull`. It must log a WARNING `needs_reconsent`, and the
   card must show the reconnect banner (**not silent**).
4. Reconnect; the next pull succeeds.

**Frontend:** card specs for the four states, connect redirect, query-param snackbar,
disconnect confirm.

**Regression:** full CPU suite + `ng test`. The GPU suite isn't touched by M11, so running it
is optional.

## Alternatives rejected

- **CLI loopback flow:** the owner chose the in-app button, which makes re-consent visible.
- **`google-auth-oauthlib` / `google-api-python-client`:** too heavy for three POSTs and one
  GET, and the M10 code is plain `httpx`.
- **Leaving the OAuth app in "Testing" permanently:** it forces re-consent every 7 days. It is
  accepted as a temporary state until the end of the project (amendment of 2026-11-08 above).
- **Storing tokens unencrypted, or in a file:** §4.2 says encrypted at rest.
- **Keying `analytics_snapshots` by pull date:** it would duplicate every day's numbers across
  7 re-pulls. Keying by the metric day gives one clean daily series.
- **Retention for every video every day:** the owner chose the M10 cadence.

## Implementation notes (2026-11-07)

### Deviations from the plan above

- **Video list:**
  - Built from `playlistItems.list part=snippet,contentDetails,status` alone. Privacy comes
    from the playlist item's `status.privacyStatus`, so the planned `videos.list part=status`
    call was dropped (one fewer call per 50 videos).
  - A never-published private video has no `videoPublishedAt`. Its playlist-insert time
    (`snippet.publishedAt`) stands in.
- **Two bookkeeping columns on `analytics_videos`:**
  - `engagement_backfilled_at`: without it, a video with no views (no rows) would be
    re-backfilled from its publish date on every run.
  - `retention_pulled_on`: the retention due rule keys on it, so an empty curve still counts
    as pulled.
- **Shared HTTP policy:**
  - M10's retry and quota loop moved into `app.market.youtube_api.google_get`. Both the
    API-key client and the bearer-token clients use it.
  - New subclass `Unauthorized` (401). The M10 client behaves as before, because a 401 was
    already a non-retried `YouTubeApiError`.
- **`get_session_factory`** (in `app.database`): a dependency for endpoints that hand a factory
  to code shared with the worker (`store_connection`, `get_credential`).
- **Integration marker:** `youtube_oauth` is **deselected by default** (`addopts`), unlike
  M10's `youtube_api`. It needs a manually connected channel in the dev DB, and it refreshes
  that account's real token.

### To verify on the first live run

These are constants in `app/analytics/analytics_api.py` that are cheap to adjust:
- `PAGE_SIZE` (10,000 rows per `startIndex` page)
- `sort=day` on the `day,video` report
- `sort=elapsedVideoTimeRatio` on retention
- that likes, comments and shares are accepted with `day,video`

A rejected combination shows up as an HTTP 400 in the pull's ERROR log line, not as missing
data.

### Tests

- **New default-suite files:**
  - `tests/test_youtube_tokens.py`
  - `tests/test_youtube_router.py`
  - `tests/test_analytics_pull.py`
  - `tests/test_migrations.py` (0011)
- **Fakes:** `tests/google_fake.py` fakes the token, revoke, Data API and Analytics endpoints.
  `oauth_settings()` keeps the tests independent of `backend/.env`.
- **Opt-in integration:** `tests/test_analytics_integration.py`.
- **Frontend:** `analytics/youtube-connection/*.spec.ts`.

## First live run (2026-11-08) — milestone closed under option A

The owner chose **option A**: close M11 on a working connection, a real pull (even with 0
rows), and the revocation check. Verifying real rows is a follow-up (below).

- **Setup:** OAuth app in Testing (amendment above), owner account as test user, test channel
  "BIT BIT" (`UCJEybqMK_7qMTzvKm4jyp5g`) with 10 private shorts uploaded the same day.
- **Connect:** the in-app flow worked end to end (unverified-app warning, both scopes granted,
  303 back to `/analytics`, card shows the channel).
- **Real pull:** `videos=10 calls=13 errors=0`, with 0 engagement and 0 retention rows. That is
  correct: the videos were private and minutes old. Google accepted every query shape (no
  HTTP 400): `day,video` with likes/comments/shares and `sort=day`, and retention with
  `sort=elapsedVideoTimeRatio`. `PAGE_SIZE` paging past page 1 was not exercised.
- **`pytest -m youtube_oauth`:** passed (real refresh, pull, idempotent re-run).
- **Found and fixed: deleted-video placeholders.** A deleted video stays in the uploads
  playlist as "Deleted video" with `privacyStatus=privacyStatusUnspecified`. It was stored as
  a live video and got a daily retention call. Now `uploads()` skips it (`DELETED_PRIVACY`),
  so a known one is marked `removed_since`. Test:
  `test_deleted_video_placeholder_is_skipped_and_marked_removed`. The re-run on the real
  channel gave `videos=7 removed=3`.
- **Revocation:** the owner removed access at myaccount.google.com/permissions. The next pull
  logged the WARNING, stopped with `needs_reconsent (invalid_grant: Token has been expired or
  revoked.)`, and stored that in `last_error`. Not silent.

**Follow-up (closed 2026-11-14, owner confirmed): verify real rows.** Once the channel has views (≥ 48–72 h after them),
run `python -m app.cli analytics pull` and `pytest -m youtube_oauth`, then check that
`analytics_snapshots` and `analytics_retention_curve` have rows with plausible values.
