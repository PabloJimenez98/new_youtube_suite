# Decision: Analytics & Dashboard data visualisation, episode ↔ YouTube link (out-of-plan pass)

**Date:** 2026-11-08
**Status:** Implemented (2026-11-08). Outside the milestone plan, like 0016 and 0021. M12's live
phase 2 and M15 are unchanged.

## Context

- **The Analytics page still showed mock data.** M10, M11 and M12 collect real data:
  - **Market:** 47,001 videos of 21 tracked channels.
  - **Own channel:** the OAuth channel, with 0 Analytics rows so far.
  - **No read endpoints:** the backend had none for any of it, so nothing reached the UI.
- **Owner's request (2026-11-08):**
  - **Analytics:**
    - an informative dashboard of both sources
    - data-science analyses that add information
    - an explanatory tooltip on every chart
  - **Dashboard:** more visual, with charts of the videos' data.
  - **Manage list:**
    - a video preview
    - filters by publication date and views
    - column headers
  - **Episode page:**
    - an ⓘ next to every pipeline tool
    - the transcript readable in the app, with speakers when diarized
    - hook-score information in the Shorts tab
- **Decided with the owner:**
  - **Link:** the YouTube ID read from the filename and stored in a column.
  - **Charts:** Apache ECharts.
  - **Analyses:** channel clustering; what drives views; growth and title keywords. The owner did
    not pick the "breakout videos" analysis.
  - **List preview:** the YouTube thumbnail.

## Decisions

### 1. Episode ↔ YouTube video: `uploads.youtube_video_id` (migration 0015)

- **The rule:** the last `[...]` group of exactly 11 YouTube-ID characters in the filename
  (`app/youtube_link.py`).
- **When it applies:**
  - the 0015 backfill: SQL `substring(filename from regex)`, and all 63 episodes are linked
  - every new upload
- **Editable** through `PATCH /uploads/{id}` (validated: 11 characters or null), and from the
  episode page ("Change" / "Link a video").
- **`UploadOut.youtube`:** the linked video's latest market snapshot (title, published_at, views,
  likes, comments, thumbnail), attached in list/get with a single `DISTINCT ON` query.
- **Partial answer to the open M11/M12 linking question** (uploads ↔ YouTube videos), for the
  market data. Joining `analytics_videos` and the reach rows for the owner's *own* channel uses the
  same column once that channel has published episodes.

### 2. `/insights` API (`app/insights/`)

- **Split:**
  - `frame.py` loads the latest snapshot per video into numpy columns, plus a 14-day window for
    velocity
  - `analysis.py` holds pure, seeded numpy/scipy functions
  - `router.py` serves them
- **Cache:** results are cached in-process per (snapshot table fingerprint = max date + row count,
  analysis, format), so a new daily pull invalidates them. The work runs in `asyncio.to_thread`.
  On the dev data the load takes 0.6 s and the slowest analysis about 2.5 s, once.
- **Endpoints** (`format = long | short | all`; long > 3 min, since Shorts can run to 3 min):
  - `/insights/market/overview`
  - `/insights/market/clusters`
  - `/insights/market/drivers`
  - `/insights/market/growth`
  - `/insights/market/keywords`
  - `/insights/channel`: the own channel's daily engagement, impression-weighted CTR, and average
    retention curve
- **Empty data:** every analysis reports `available: false` with a reason instead of failing.

### 3. The analyses (data science)

- **Shared building block: the age- and channel-adjusted residual.**
  - Per channel with ≥ 20 videos, least squares `log10(views+1) ~ a + b·log10(age_days)`; smaller
    channels use the pooled fit. The residual is the "surprise".
  - Without it, raw views mostly measure channel size and age.
- **Channel clustering:**
  - six channel features: log median views, likes/view, comments/view, uploads/week (90 d), median
    length, IQR of log views
  - z-scored, then k-means (scipy `kmeans2`, seeded) with k chosen by silhouette
  - **k ≤ 3**: the first 3 palette slots are the only ones validated colour-blind-safe for
    all-pairs (scatter) use
  - PCA (SVD) to 2D for the map
  - **On the dev data:** k = 2, silhouette 0.21. The tooltip says this is a weak tendency.
- **What drives views:**
  - Spearman ρ between the residual and video length, title length / question / number / capitals /
    separator, tag count, and weekend.
  - **95% CI = ρ ± 1.96 · bootstrap SE.** The SE comes from 300 resamples of up to 6,000 rows,
    rescaled to n. A percentile interval was tried first and rejected: ties from resampling biased
    binary features off-centre (e.g. ρ 0.157 with an interval of [0.156, 0.205]).
  - Plus a weekday × hour heatmap of the median residual, in Madrid time; slots with fewer than 15
    videos are left blank.
  - **Dev data:** length ρ 0.30, a number in the title 0.16, a question in the title −0.06.
- **Growth:**
  - views vs age, log-log: the binned medians with an IQR band, every video of the owner's channel
    plus a 2,000-point sample of the others, and the pooled fit
  - per-channel exponents b
  - velocity: views/day between snapshots in the last 14 days, by age bucket. With 2 days of
    snapshots it is not available yet, and it says so.
- **Title keywords:**
  - **Candidates:** the top 400 summed-TF-IDF title tokens with ≥ 40 videos across ≥ 3 channels.
    Excluded: tokens in a channel name, and tokens with more than 50 % of their videos in one
    channel. Before this filter, "estirando" and "chicle" (a channel's own name) topped the list.
  - **Lift:** mean residual with the word minus without it, Welch SE for the 95% CI, shown as
    "% vs expected".
  - **Dev data:** guests and topics on top (Jordan Peterson, Eric Weinstein, aliens); covid and
    "live" at the bottom. The tooltip says these describe what drew audiences, not words to copy.

### 4. Episode page

- **Stepper:** every tool node has its ⓘ, which opens the 0021 popup.
- **Transcript viewer** (`GET /uploads/{id}/transcript`, same 404/409 rules as the subtitles):
  - turns grouped by speaker (or 60 s paragraphs before diarization)
  - each speaker's share of speaking time
  - search with highlighting
  - click a timestamp to seek
  - refetched when a newer transcribe or diarize job succeeds
- **Hook summary** (Shorts & hooks tab), above the gallery:
  - the weights
  - every scored clip ranked, with its four sub-scores as bars, the strongest and weakest signal,
    and the measured inputs (lead silence, words/s, opening question)
  - click a clip to seek

### 5. Charts: Apache ECharts 5.6.0 (pinned)

- **Bundle:** modular imports (bar, boxplot, custom, heatmap, line, scatter, and only the
  components in use) with the SVG renderer.
  - Only the lazily loaded pages import it (Analytics, Dashboard, now lazy too).
  - The initial bundle stays at 440 kB, under the 500 kB budget.
- **Palette:** the dataviz skill's reference palette, dark steps, validated against our panel
  surface `#15122f`. All checks pass: worst adjacent CVD ΔE 8.4, normal-vision ΔE 19.3, all
  ≥ 3:1 contrast.
- **Colour rules:**
  - the owner's channel is always blue; other channels are context gray
  - diverging charts go red ← gray → blue around "as expected"
  - market-wide aggregates are violet
  - status colours are used only for job outcomes
  - one y-axis per chart
- **Every chart sits in a `ChartCard`:**
  - an ⓘ tooltip in three parts: what it shows / method / how to read it, including what it does
    not say
  - an empty state that says why there's no data
  - a **Table** toggle, so no value is readable only through a hover
- **Option builders** are pure functions (`analytics-charts.ts`, `dashboard-charts.ts`) and are
  unit-tested.

### 6. Dashboard and Manage list

- **Dashboard:**
  - tiles: episodes, total and median views, runtime, shorts
  - views per episode in publication order, with the median line; click a bar to open the episode
  - top 5 episodes with their thumbnails
  - length vs views
  - tool runs per day (done / failed)
  - pipeline coverage and recent activity
- **Manage:** a table with headers (Preview, Episode, Published, Views, Length, Pipeline, Next
  step).
  - **Sorting:** sortable headers with `aria-sort`; missing values always sort last.
  - **Filters:** search, published from/to, views ≥ / ≤. A date or views filter excludes episodes
    without YouTube data.
  - **Preview:** the YouTube thumbnail (`thumbnail_url`), or a frame of the file when not linked.
- **Removed:** the M1/M10 mock code (`analytics-dummy-data.ts`, `market-chart`, `preview-banner`,
  `sparkline`).

## Alternatives rejected

- **Parsing the ID on every request** (no column): breaks if a file is renamed, and can't be fixed
  by hand.
- **Chart.js:** box plots and heatmaps need plugins.
- **Hand-written SVG:** too much axis/tooltip/zoom code for about 15 charts.
- **Computing the analyses in the browser:** shipping 47k rows to the client, and losing the
  tested Python and scipy.
- **Breakout-videos analysis** (outlier z-scores): offered, and not chosen by the owner.
- **More than 3 cluster colours:** fails the all-pairs CVD check.

## Test plan (run 2026-11-08)

- **Backend:**
  - `test_insights.py` (19): synthetic frames with planted effects. Checks that:
    - residuals centre each channel
    - fits recover the growth exponent
    - Spearman finds a planted length effect with a CI around ρ
    - keywords find a planted ×3.16 lift and skip channel names and one channel's vocabulary
    - clusters give k ≤ 3; growth sampling keeps all of the owner's points
    - the API works on an empty DB and on a seeded one, and requires auth
  - plus upload link, PATCH and snapshot tests, the JSON transcript tests, and migration 0015
    (backfill, revert)
  - **Total:** 698 passed, 10 deselected.
- **Frontend:**
  - chart builders (colours, ordering, the IQR polygon, the heatmap dimension, tables)
  - Analytics page (loads, format switch, tooltips on every card, empty states, table view)
  - Dashboard builders
  - Manage table (headers, sort, filters, thumbnails, row click)
  - transcript viewer, hook summary, stepper ⓘ, episode YouTube link
  - **Total:** 38 files, 310 tests passed. `ng build` has no warnings.
- **Manual check (Chrome, dev data):** Analytics, Dashboard, Manage, episode transcript and Shorts
  tabs. It found and fixed:
  - the heatmap coloured by the count (ECharts `visualMap` defaults to the last dimension)
  - the IQR band filled to the log axis minimum, then a placeholder 0 broke the log axis
  - the owner's points were missing from the age sample
  - the axis name collided with the legend
  - the score cell broke the table row (`display: flex` on a `<td>`)

## Consequences

- **More data, better charts.** As the daily pull adds snapshots, velocity becomes available and
  the cross-sectional age curve can be complemented by real per-video trajectories.
- **Own-channel charts** fill in once the OAuth channel has views (M11 follow-up). **Closed 2026-11-14:** the owner confirmed it.
- **The analyses are descriptive** (correlation, not causation). The tooltips say so. Their
  residuals and features are natural inputs for M15's CTR and hook models.
