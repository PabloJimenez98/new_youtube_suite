# Decision: Market-Analysis Pipeline v0 (M10)

**Date:** 2026-11-06
**Status:** Done 2026-11-07. Seed list approved and loaded (see "Seed and first pull" at the end); regression 429 passed, 6 skipped (missing M5 chapter bundle, accepted by the owner).

## Context

- **What the plan asks for in M10:** seed a curated list of podcast channels; plan the
  quota budget (`videos.list` batched at 50 as the daily backbone,
  `search.list` for sparse discovery, RSS as the free "what's new"
  trigger). Then build `market_channels`/`market_video_snapshots`, a scheduled
  daily pull, ISO-8601 duration parsing and RSS polling.
- **Decided with the owner (2026-11-06):**
  - **Channels:** Claude proposes a list and the owner reviews it before it is loaded (§7).
  - **Snapshot cadence:** videos published in the last 90 days get a snapshot every day; older
    videos get one every week.
  - **History:** when a channel is added, its whole catalogue is backfilled through its uploads
    playlist.
  - **Scope:** backend and CLI only. The Analytics page keeps its dummy
    data (the pending wiring is logged in `brainstorm.md`). No new HTTP endpoints.
- **Quota facts:** see `docs/research/youtube-data-api-quota-notes.md` and
  `project-description.md` §6.1.

## Decisions

### 1. Discovery and backfill: `playlistItems.list` replaces `search.list`

- **Uploads playlist:** every channel has one, and `channels.list
  part=contentDetails` returns its ID (`relatedPlaylists.uploads`, i.e. `UU…`).
  `playlistItems.list` pages through it at **1 unit per 50 videos**,
  newest first, and reaches the full catalogue.
- **Why not `search.list`:** it costs 100× more under the old billing, or is capped at about
  100 calls/day under the 2026 change (this is unverified either way, and §6.1 asked for a recheck). It also tops out
  at around 500 results per query. Using `playlistItems.list` means the recheck no longer
  blocks anything.
- **`search.list` is not used in M10.** Its §6.1 role, finding *new candidate
  channels*, is not part of M10's scope.
- **RSS** (`feeds/videos.xml?channel_id=…`, free, last ~15 uploads) stays the
  frequent "new upload" trigger, polled **hourly**. A video ID it surfaces that we do not know yet
  gets a `videos.list` call right away, so its first snapshot is taken within
  about an hour of publication. That first day of the growth curve is the most valuable one.
- **Backfill** runs once when a channel is added and is resumable: a channel whose
  backfill did not finish is picked up again on the next run.

### 2. Schema: add `market_videos` alongside the two §4.2 tables

§4.2 has no table for the set of known videos. Deriving that set from the snapshots would
not work well: a video discovered by RSS would not exist until its first
snapshot, and a deleted or private video would be requested forever. **Addition:**

**`market_videos`**
| Column | Type | Notes |
|---|---|---|
| id | PK | |
| market_channel_id | FK → market_channels | |
| youtube_video_id | TEXT UNIQUE | |
| published_at | TIMESTAMPTZ | drives the 90-day "recent" window |
| discovered_at | TIMESTAMPTZ | |
| discovered_via | TEXT | `backfill` \| `rss` |
| unavailable_since | DATE NULL | set when `videos.list` stops returning the ID (deleted or private); the video then leaves the pull |

**`market_channels`** keeps the §4.2 columns and adds:
- `uploads_playlist_id TEXT`
- `backfill_completed_at TIMESTAMPTZ NULL`
- `rss_checked_at TIMESTAMPTZ NULL`

**`market_video_snapshots`** keeps the §4.2 columns exactly.
- Title, tags and thumbnail stay per snapshot on purpose, because title and thumbnail changes are
  signal for the M13 CTR model.
- `like_count`/`comment_count` are **nullable**, because the API omits them when they are
  hidden or disabled. They are stored as NULL, never as 0.
- Upsert key: `(youtube_video_id, snapshot_date)`. A second pull on the same day
  overwrites the row, so re-runs are idempotent.
- `snapshot_date` is the **UTC** date of the pull.

The `channels.list` subscriber statistics in §6.1 have no table in §4.2. They are **not
stored in M10** (logged in `brainstorm.md` for later).

### 3. Which videos get pulled each day

A video is due on a given day if it is **available** and either:
- **recent:** published ≤ 90 days ago, and it has no snapshot today yet; or
- **older:** its latest snapshot is ≥ 7 days old, or it has none.

The 7-day rule for older videos uses the date of the last snapshot, not a fixed weekday. So it
self-heals: a missed day or a worker that was down simply makes the videos due on the next run.
Right after a backfill, every older video falls on the same weekday. That burst is
small (≈ 700 calls, see §5), so no spreading logic is added.

**Shorts are stored like any other video.** `duration_seconds` is what
tells them apart later. No filtering happens in M10.

### 4. Scheduling: arq `cron_jobs` on the existing worker

- **No new process or library.** arq has cron support built in, and the M2
  worker (`app/worker/settings.py`) gains two cron tasks.
  - `market_rss_poll`: every hour, at minute 5. RSS check for every active channel,
    then `videos.list` for the new IDs, then snapshot rows.
  - `market_daily_pull`: daily at **09:30 UTC**. The Data API quota resets at midnight
    Pacific time (07:00/08:00 UTC), so the pull runs on a fresh quota. Order: resume
    unfinished backfills, select the due videos (§3), batch them by 50, call `videos.list`
    and upsert.
- **Single-worker assumption** (v1 is single-operator). arq's `unique=True` default stops
  overlapping runs of the same cron.
- **Not stored in the `jobs` table.** `jobs` is tied to an upload's
  `tool_type`, and market pulls belong to no upload. Each run writes one log line instead
  (videos due, calls made, units used, rows written, unavailable count).
- **Known limitation:** if the worker is down at 09:30, that day's snapshots for recent videos
  are lost. Older videos recover by themselves (§3). This is acceptable for v0.

### 5. Quota budget

Estimate for about 25 channels with roughly 1,500 videos each (shorts included):

| Item | Calls/units |
|---|---|
| One-off backfill: `channels.list` 25 + `playlistItems.list` ≈ 750 + `videos.list` ≈ 750 | ≈ 1,525 (once) |
| Daily, recent videos (≈ 25 × 60 in 90 days → 1,500 IDs / 50) | ≈ 30 |
| Daily, older videos (≈ 36k IDs / 7 days / 50) | ≈ 105 on average, up to ≈ 720 on the burst day |
| Hourly RSS follow-up `videos.list` | ≈ 24 at most |
| **Typical day** | **≈ 160 of 10,000 units (< 2 %)** |

- **Rows:** ≈ 6.5k snapshots/day, ≈ 2.4M/year. That is fine for Postgres with the unique
  index. Retention and downsampling are logged for later, not built.
- **Safeguards:**
  - an in-process unit counter, with a soft cap of `MARKET_DAILY_UNIT_BUDGET` (default 5,000)
  - an HTTP 403 with `quotaExceeded` stops the run cleanly, keeps the rows already written,
    and logs the error, with no retry
  - other 5xx/429 responses get 3 retries with exponential backoff

### 6. Code layout and tooling

- **Package:** `app/market/`
  - `youtube_api.py`: thin `httpx` client for `channels`, `playlistItems` and `videos`, with
    the unit counter
  - `rss.py`: stdlib `xml.etree` Atom parsing
  - `durations.py`: ISO-8601 parsing
  - `batching.py`
  - `selection.py`: due-video rule, pure
  - `pull.py`: orchestration
- **`httpx` moves** from `requirements-dev.txt` to `requirements.txt`. It is
  already the test client, and adding no new library keeps the dependency list as it is.
- **Duration parser:** hand-written, about 15 lines, no `isodate` dependency.
  - Handles `PT#H#M#S`, `P#DT…` (streams longer than a day) and `P0D`/`PT0S` (live and
    upcoming).
  - Anything else raises an error and the field is stored as NULL, logged, so the row
    still lands.
- **CLI:** `python -m app.cli market …`
  - `add <@handle|channel URL|UC… id>`: resolved via `channels.list`
    `forHandle`/`id`, then backfills
  - `seed`: loads `app/market/seed_channels.json`, the approved list (§7)
  - `list`
  - `deactivate <id>`: stops pulling the channel and keeps its history
  - `pull`: the daily pull on demand
  - `rss`: the RSS poll on demand
- **Config:** `YOUTUBE_API_KEY` (a SecretStr, read-only key with YouTube Data API v3 enabled).
  `MARKET_RECENT_WINDOW_DAYS=90`, `MARKET_OLDER_INTERVAL_DAYS=7`,
  `MARKET_DAILY_UNIT_BUDGET=5000`. A missing key makes the cron tasks log a skip and
  return. The worker does not crash, so the M3–M8 tools are unaffected.

### 7. Proposed seed list (pending owner review)

- **Mix:** Spanish- and English-language video podcasts of different sizes and formats
  (long interviews, comedy, business/tech, wellbeing), plus the M9 test channel as a
  known reference.
- **Handles are unverified:** each one is resolved with `channels.list forHandle` on
  `seed`. One that does not resolve is reported and skipped, never guessed.

**Spanish:**
| # | Podcast | Handle (to verify) | Format |
|---|---|---|---|
| 1 | Sastre y Maldonado | — (resolve from the M9 playlist) | conversation/comedy, reference |
| 2 | The Wild Project | @TheWildProject | long interviews |
| 3 | La Pija y la Quinqui | @lapijaylaquinqui | conversation/comedy |
| 4 | Estirando el Chicle | @estirandoelchicle | conversation/comedy |
| 5 | Lo Que Tú Digas (Álex Fidalgo) | to be resolved | long interviews |
| 6 | Poco se Habla | @pocosehabla | conversation |
| 7 | Cracks Podcast (Oso Trava) | @CracksPodcast | business/interviews (MX) |
| 8 | Leyendas Legendarias | @LeyendasLegendarias | true crime/comedy (MX) |
| 9 | La Cotorrisa | @LaCotorrisa | comedy (MX) |
| 10 | Se Regalan Dudas | @SeRegalanDudas | conversation (MX) |
| 11 | Hablando Huevadas | @HablandoHuevadas | comedy (PE) |

**English:**
| # | Podcast | Handle (to verify) | Format |
|---|---|---|---|
| 12 | The Joe Rogan Experience | @joerogan | long interviews |
| 13 | Lex Fridman Podcast | @lexfridman | long interviews/tech |
| 14 | The Diary of a CEO | @TheDiaryOfACEO | business/interviews |
| 15 | Huberman Lab | @hubermanlab | science/wellbeing |
| 16 | Modern Wisdom | @ChrisWillx | interviews |
| 17 | Theo Von — This Past Weekend | @TheoVon | comedy/interviews |
| 18 | Flagrant | @flagrant | comedy panel |
| 19 | Call Her Daddy | @CallHerDaddy | conversation |
| 20 | My First Million | @MyFirstMillionPod | business |
| 21 | Lenny's Podcast | @LennysPodcast | product/tech, mid-size |
| 22 | Dwarkesh Podcast | @DwarkeshPatel | tech/AI, mid-size |

## Test plan

**Unit tests** (no network, no API key):
- **`durations.py`:**
  - `PT15M33S` → 933, `PT1H2M3S` → 3723, `PT45S`, `PT2H`, `P1DT2H3M` → 93780
  - `P0D`/`PT0S` → 0
  - invalid input (`15:33`, `""`, `PT`) → error
- **`batching.py`:** 0, 1, 50, 51 and 120 IDs → sizes `[]`, `[1]`, `[50]`, `[50,1]`,
  `[50,50,20]`; duplicate IDs are removed and order is kept.
- **`selection.py`** (fixed "today"):
  - recent video with no snapshot today → due
  - recent video already snapshotted today → not due
  - older video last seen 6 days ago → not due; 7 days ago → due; never → due
  - unavailable video → not due
  - inactive channel → not due
  - exactly at the 90-day edge
- **`rss.py`** on a fixture Atom XML (saved real feed shape): video IDs, titles and
  `published` are extracted; an empty feed and an invalid XML feed are handled.
- **Response mapping** on fixture `videos.list` JSON:
  - string → int casts
  - missing `likeCount`/`commentCount` → NULL
  - missing `tags` → empty array
  - `high` thumbnail missing → falls back to the next available size
  - an ID requested but absent from the response → `unavailable_since` set
- **Client** (with `httpx.MockTransport`):
  - pagination with `nextPageToken`
  - the unit counter increments per call
  - 403 `quotaExceeded` stops the run and keeps the rows already written
  - 500 → retried, then the call succeeds
  - the budget cap stops the run
- **Pull orchestration** against the test database with a mocked client:
  - backfill → `market_videos` rows; daily pull → snapshot rows
  - re-running the same day upserts and does not duplicate
  - RSS with one new ID → video + snapshot inserted; a known ID → nothing happens
  - a backfill interrupted mid-way resumes on the next run
- **Worker settings:** both cron jobs are registered with the expected schedule. A missing
  API key → the task logs a skip and returns without raising.

**Integration test** (`@pytest.mark.youtube_api`, skipped without `YOUTUBE_API_KEY`):
- resolve one seed handle with `forHandle`
- `videos.list` on a stable public video (`jNQXAC9IVRw`, "Me at the zoo") → a snapshot
  row lands with a non-null title, a duration of 19 s and a `view_count` > 0
- a short `playlistItems.list` page (`maxResults=5`) on one channel
- spends fewer than 10 units in total

**Manual check:** after the first real daily pull, pick one known video and compare its
`view_count`/`like_count`/`comment_count`/title in the database with the YouTube page
(within the normal drift of the public counters), and check the logged unit count against the
Cloud Console quota page.

**Regression:** full backend `pytest` (all M0–M9 suites) plus frontend `ng test`.
Also check that the worker still starts and processes a `dummy`/`transcribe` job with the
cron tasks registered.

## Alternatives rejected

- **`search.list` for backfill/discovery:** expensive or tightly capped (still unverified), and it
  stops at about 500 results. The uploads playlist does the same job for about 1 % of the cost.
- **Snapshotting every video every day:** inside the quota, but it writes about 2.5× the
  rows for videos whose counters barely move.
- **RSS only, no backfill:** the catalogue would take months to build up, and with
  ~15 items per feed nothing from before the first poll would ever be seen.
- **A separate scheduler (APScheduler, system cron):** a second process
  to run and monitor. arq `cron_jobs` already runs inside the worker.
- **Logging market pulls as rows in `jobs`:** `jobs` requires an upload
  and a `tool_type`, so fitting market runs in would distort that schema.
- **`isodate`/`feedparser` libraries:** both formats here are simple. The stdlib plus about
  15 lines is enough, so there is no new dependency.

## Implementation notes (2026-11-07)

Details settled while building, within the plan above:
- **API key in a header.** Requests send `X-Goog-Api-Key` instead of `?key=`, so the key
  never appears in URLs, request logs or exception messages.
- **`published_at` from the playlist.** `playlistItems.list part=contentDetails` already
  returns `videoPublishedAt`, so backfill fills `market_videos.published_at` with no extra
  call. Playlist entries without it (private/deleted) are skipped.
- **Backfill writes videos, not snapshots.** The first snapshots come from the daily pull
  that follows (every new video is due). `market add` only backfills; `market pull` takes
  the snapshots.
- **Resume = re-page from the top.** An interrupted backfill restarts the playlist from the
  first page on the next run; the per-page commit plus `ON CONFLICT DO NOTHING` makes that
  idempotent. Re-paging costs at most a few dozen units, so no page token is stored.
- **Budget scope.** The unit counter is keyed by the quota day (midnight Pacific) and lives
  in the process, so in the worker it spans the hourly RSS runs and the daily pull. A CLI
  run has its own counter.
- **403 `rateLimitExceeded`** (a short-term per-user limit) is retried like a 429. Only
  `quotaExceeded`/`dailyLimitExceeded` stop the run.
- **Cron timezone.** arq evaluates cron schedules in the host timezone by default; the
  worker sets `timezone = UTC` explicitly.
- **Integration test channel:** `@jawed` ("Me at the zoo"). It spends 3 units.

### Seed and first pull (2026-11-07)
- **Handles verified before loading.** Each seed handle was checked with `channels.list`
  (title, subscribers, video count). Of the §7 guesses, 15 were right. 7 were corrected:
  - @pijayquinqui
  - @pocosehablaelpodcast
  - @OsoTrava (Cracks is published there)
  - @leyendaspodcast
  - @HablandoHuevadasOficial
  - @officialflagrant
  - @LoQueTuDigas, for Lo Que Tú Digas

  @lapijaylaquinqui pointed to an empty "CLIPS" channel, which was loaded by mistake and
  then deactivated (#3). **La Cotorrisa is unresolved:** @La_Cotorrisa is empty and the only
  ID found is an unofficial channel. It stays `ref: null` until the owner supplies the
  right one.
- **Errors are isolated.** One seed channel's uploads playlist answered 404
  `playlistNotFound` and aborted the whole seed. Now:
  - A missing uploads playlist on the first page counts as an empty catalogue: the backfill
    is marked complete with 0 videos.
  - Any other non-quota API error skips that channel's backfill, that `videos.list` batch,
    or that channel's RSS lookup. It is counted in the report's `errors` and retried on the
    next run.
- **`add`/`seed` skip channels that are already backfilled,** so re-running the seed costs
  1 unit per channel, not a full re-page.
- **An empty `YOUTUBE_API_KEY=`** counts as unset.
- **Real numbers:**
  - Backfill: 21 channels, 47,012 videos, about 970 units including the first, aborted attempt.
  - First daily pull: 47,012 rows, 941 units, 0 errors, 2 min 15 s.
  - The catalogue is about 30% larger than the §5 estimate, so the weekly older-video load
    is about 135 calls a day. A typical day is still about 200 units (2% of the quota).
- **Manual check.** 15 videos from three channels were compared with the public RSS
  numbers. Views were within ±0.3%, the drift between the two reads. Likes were equal or
  within ±2, titles were identical, and durations were plausible.
