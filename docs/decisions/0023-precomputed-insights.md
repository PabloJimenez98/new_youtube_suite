# Decision: Precomputed Analytics results (`insight_results`), lazy canvas charts

**Date:** 2026-11-08
**Status:** Implemented (2026-11-08). Follows 0022, outside the milestone plan.

## Context

- **The owner reported the Analytics page blocking** while its charts were computed.
- **Cause (0022 design):** the five market analyses ran inside the API requests, cached only in
  process memory.
  - A cold page fired five requests at once. Each loaded the 47k-video frame, and part of that
    load is CPU work on the event loop.
  - Then each ran its analysis: the drivers bootstrap takes about 2.5 s per format, the TF-IDF
    keywords about 0.4–0.7 s.
  - Every API restart, and every new snapshot, started cold again.
- **The browser side added to it:** about 15 charts initialised at once with the SVG renderer,
  including a scatter of more than 2,000 points (one DOM node each).

## Decisions

### 1. `insight_results` table (migration 0016)

- **One row per (analysis, format):** `overview`, `clusters`, `drivers`, `growth`, `keywords` ×
  `long`, `short`, `all`, plus `channel/-` for the own-channel summary.
- **Columns:** the JSON `payload`, a `source_key` fingerprint, `computed_at` and `duration_ms`.
  Unique on `(name, format)`.
- **NaN/inf are stored as null** (`clean_json`), because JSONB rejects them.

### 2. The worker computes; the API only reads

- **`app/insights/store.refresh_insights`** loads the frame once and computes all 15 market
  results in one `asyncio.to_thread`. It also computes the channel summary, which was moved out of
  the router.
- **Fingerprints decide what to recompute:**
  - **market:** max snapshot date + snapshot count + linked episodes (count and a SHA-1 of the
    ordered IDs). Re-linking an episode changes the owner's channel in every market result.
  - **channel:** counts and the latest pull of `analytics_snapshots`, `analytics_reach_snapshots`
    and `analytics_retention_curve`, plus the connection and video count.
  - **Stable across processes:** the market digest uses SHA-1 rather than Python's `hash()`
    (randomised per process), and `string_agg` is ordered.
- **Unchanged sources → no-op:** 0.03–0.1 s. A full recompute takes about 9.5 s on the dev data
  (drivers ≈ 7.4 s of it, for its 3 formats).
- **When it runs:**
  - **worker cron hourly at :45**, plus `run_at_startup`; it catches the 09:30 daily pull and the
    hourly RSS polls
  - **arq function `insights_refresh`**, which the API enqueues when a result is missing. The job
    id is `insights_refresh:<market fingerprint>`, so concurrent page loads queue one refresh, not
    one each.
  - **CLI:** `python -m app.cli insights refresh [--force]` and `insights status` (per-row compute
    time and fingerprint).
- **`/insights/*` reads the stored row and adds `pending: false` and `computed_at`.**
  - A missing row answers at once with `pending: true`, a reason and empty data of the same shape,
    and enqueues the refresh.
  - The in-process caches are removed.
- **Measured:** all six endpoints in parallel take 126 ms in-process; in the browser each request
  takes 116–142 ms.

### 3. Frontend

- **Analytics:**
  - shows "Precomputed <time>"
  - while any result is `pending`, shows "Computing in the background…" and re-requests that
    result every 5 s, for at most 2 min, until it is stored
  - a format change cancels the retries of the previous format
  - card empty states say "Computing…" instead of "not enough data" while pending
- **`Chart`:**
  - **Canvas renderer** instead of SVG (thousands of points stay fast).
  - **Lazy drawing:** each chart initialises only when it comes within 300 px of the viewport
    (IntersectionObserver).
  - **No 2D canvas** (jsdom): skips drawing; the table view keeps the data.
  - **Measured on the page:** 2 charts drawn on load, the rest on scroll; the worst main-thread
    task was 81 ms.
- **Growth payload:** points are rounded to whole days and views (161 → 153 KB; most of the rest
  is repeated channel names).

### 4. Other candidates reviewed

- **The channel summary (`/insights/channel`):** precomputed too. Its SQL aggregates grow with
  every daily pull (retention is 100 points per video per day).
- **Dashboard and Manage:** left live. They use `GET /uploads` (a `DISTINCT ON` over the indexed
  snapshot table for the ~63 linked IDs) and `GET /jobs`, both cheap.
- **Chapters, relevance and the other tool results:** already stored per episode by their jobs.

### 5. The real freeze: video streams starving the browser's connections (found 2026-11-08)

- **The precomputation alone didn't unfreeze the app:** the owner still saw nothing load.
- **Cause:** 0022 gave every Library card (and every unlinked Manage row) a `<video
  preload="metadata">` "frame".
  - Each one opened a stream of the **whole episode file** (`/uploads/N/file`, .mkv files of more
    than 100 MB).
  - With about 12 cards in view, they took Chrome's 6 HTTP/1.1 connections to `localhost:4200`, so
    every API call (uploads, insights, login) queued behind the video streams.
  - Library is the landing page after login, so the whole platform looked frozen.
  - The episode page was close to the limit too: the main player plus 5 shorts that preload even in
    their hidden tab.
- **Fix:**
  - **Library cards and Manage rows:** the YouTube thumbnail (`youtube.thumbnail_url`, an
    `i.ytimg.com` image) or a ▶ placeholder. **No `<video>` in lists.**
  - **Shorts players:** `preload="none"`; they load on play.
  - **The episode's main player:** still `preload="metadata"`, the only stream on that page.
- **Measured after the fix:** 0 file streams on Library, `GET /uploads` takes 9 ms, and Analytics
  loads at once.
- **Rule:** never put a `<video>` of an uploaded episode in a list; use an image.

## Alternatives rejected

- **A longer in-process cache:** still cold after every restart and every new pull, and still
  computed inside a request.
- **Postgres materialized views:** the analyses are numpy/scipy (k-means, bootstrap, TF-IDF), not
  SQL.
- **Computing right after `market_daily_pull` only:** it would miss RSS-discovered snapshots and
  re-linked episodes. The fingerprinted hourly run covers every source and costs ~0.1 s when idle.

## Tests (run 2026-11-08)

- **Backend:** 702 passed. New tests:
  - **refresh:** computes 16 rows, then skips all 16; a new snapshot recomputes only the 15 market
    rows; `force` recomputes everything; re-linking an episode recomputes the market group
  - **API:** a missing result answers `pending` and enqueues exactly one refresh across repeated
    requests; stored results are served with nothing enqueued
  - **migration 0016:** unique `(name, format)`, revert
  - **worker:** schedule and functions; NaN cleaning
- **Frontend:** 312 passed. New tests: the "Precomputed" line, and pending → poll → stored → no
  further polling. `ng build` has no warnings (440 kB initial).
- **Manual:** in Chrome against the dev stack:
  - the worker's startup refresh was a no-op (0.03 s)
  - the page has no event-loop lag, and API requests take 116–142 ms
  - lazy drawing has a worst long task of 81 ms
