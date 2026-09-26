# YouTube Suite — Development Plan (Phase 4)

**Date:** 2026-08-13
**Status:** Phase 4 output — based on `docs/project-description.md` (Phase 3).

## How to read this plan

16 milestones (M0–M15), ordered so the **simplest possible end-to-end
path ships first** (upload → one working tool → see a result), before
the full tool suite, then the market-analysis pipeline, then the
custom-model training loop that depends on data the earlier milestones
generate. Each milestone is a self-contained unit of three sub-phases,
matching the incremental-verification approach requested:

1. **Planning** — lock the design/test decisions *before* writing code (this is where any still-open architecture choices from Phase 3 get resolved).
2. **Development** — build the milestone's scope.
3. **Testing** — implement and run the tests that prove this milestone works *and* that nothing earlier broke (a regression pass, not just new-feature tests).

Every milestone ends in something demonstrable — never "half a pipeline."

## Progress

Updated as milestones complete — this table is the source of truth for
"what's the current milestone," used by the `milestone-dev` skill.

| Milestone | Status |
|---|---|
| M0 — Backend & database skeleton | Done |
| M1 — Frontend skeleton & upload flow | Done |
| M2 — Async job pipeline | Done |
| M3 — Transcript/subtitle generator | Not started |
| M4 — Speaker diarization | Not started |
| M5 — Chapter generator | Not started |
| M6 — Show notes & metadata generator | Not started |
| M7 — Relevance-detection model v0 | Not started |
| M8 — Shorts generator | Not started |
| M9 — Full-suite hardening | Not started |
| M10 — Market-analysis pipeline v0 | Not started |
| M11 — OAuth + Analytics API v2 | Not started |
| M12 — YouTube Reporting API | Not started |
| M13 — Title/thumbnail CTR predictor v0 | Not started |
| M14 — Hook-strength scorer v0 | Not started |
| M15 — Model retraining pipeline | Not started |

## Milestones

### M0 — Backend & database skeleton
- **Depends on:** nothing (first milestone)
- **Estimate:** 4–5 days
- **1. Planning:** Decide blob-storage approach for v1 — recommend **local filesystem behind a thin storage-abstraction interface** (a small class with `save`/`get`/`delete` methods), so swapping to S3-compatible storage later doesn't touch calling code. This resolves one of Phase 3's explicitly-open infra decisions. Define the `users`/`uploads`/`jobs` table migrations per the Phase 3 schema.
- **2. Development:** FastAPI app skeleton, Postgres connection + migrations (`users`, `uploads`, `jobs`), single-user auth (email/password, no multi-tenancy), storage-abstraction module.
- **3. Testing:** unit tests for the auth flow (login/token issuance/rejection) and the storage abstraction (save/get/delete round-trip); a migration-up/migration-down smoke test.

### M1 — Frontend skeleton & upload flow
- **Depends on:** M0
- **Estimate:** 3–4 days
- **1. Planning:** Define the upload UI's contract with the backend (accepted file types/size limits, progress reporting) and what "upload succeeded" looks like in the UI.
- **2. Development:** Angular app shell, login screen, upload screen (calls M0's upload endpoint), a bare uploads-list view.
- **3. Testing:** component tests for the upload form (validation, progress states); one manual end-to-end check (log in, upload a file, see it listed).
- **Note (2026-08-14):** before M1's PR was opened, the project was
  deliberately expanded beyond this milestone's original scope — see
  `docs/decisions/0004-full-app-shell-pivot.md`. What's above is still
  what M1 itself delivered (kept `Done` below); the expansion is
  additive on the same branch. Later milestones' frontend
  "Development" bullets (M3, M5, M6, M7, M8, M10, M13, M14) now mean
  wiring real data into an already-existing screen from that pivot,
  not building a screen from scratch — check the decision doc before
  assuming a milestone needs new frontend scaffolding.

### M2 — Async job pipeline (first real end-to-end path)
- **Depends on:** M0, M1
- **Estimate:** 4–5 days
- **1. Planning:** Decide the job-queue technology — recommend a **lightweight queue+worker** (e.g. `arq` or `RQ` over Redis) rather than Celery, given single-operator scale and lower operational overhead; this resolves another explicitly-open Phase 3 decision. Define the job-status state machine (`queued → running → succeeded/failed`) and how progress is reported to the frontend (polling vs. websockets — recommend simple polling for v1).
- **2. Development:** wire the queue+worker; implement one **trivial dummy job** (e.g. sleep N seconds, write a placeholder result) triggered on upload; expose job status via API; show live status in the Angular upload-list view.
- **3. Testing:** integration test: upload a file → assert a job is enqueued → assert the worker picks it up → assert status transitions land in the DB correctly. This milestone's test *is* the "does the skeleton work" checkpoint for everything after it — treat it as the regression baseline all later milestones test against.
- This is the milestone that satisfies "simplest possible end-to-end path first" — nothing AI-related yet, but upload → job → visible result is fully proven.

### M3 — Transcript/subtitle generator
- **Depends on:** M2
- **Estimate:** 1 week
- **1. Planning:** Decide GPU vs. CPU deployment target for `faster-whisper large-v3-turbo` (affects hosting cost/latency — flag as a real decision, not a placeholder default). Define the `transcript_segments`/`transcript_words` migrations. Write the test plan: a short fixture audio clip with known correct transcript text to test against.
- **2. Development:** ffmpeg audio extraction, faster-whisper integration with `word_timestamps=True`, segment/word persistence, `.srt`/`.vtt` generation endpoints.
- **3. Testing:** unit tests for `.srt`/`.vtt` formatting from mock segment data; integration test running the real fixture clip through the full pipeline and asserting transcript accuracy within a tolerance; regression check that M2's job-status flow still works with a real (not dummy) job.

### M4 — Speaker diarization
- **Depends on:** M3
- **Estimate:** 3–4 days
- **1. Planning:** Explicitly plan around the version-pairing gotcha from Phase 3 (WhisperX 3.8.6 defaulting to `pyannote/speaker-diarization-community-1`, not `3.1`) — write a **startup healthcheck test** into the plan now, not as an afterthought.
- **2. Development:** pyannote 4.0.7 pipeline integration, WhisperX word-speaker merge, `speaker_label` population on existing rows.
- **3. Testing:** the healthcheck test (assert non-empty diarization output on a 2+-speaker fixture clip, run at service startup); unit test for the word/segment speaker-label merge logic on synthetic overlapping-turn data.

### M5 — Chapter generator
- **Depends on:** M3
- **Estimate:** 3–4 days
- **1. Planning:** Confirm the LLM choice (Claude API) and prompt contract (input: transcript chunk; output: boundary phrase + title). Plan the TF-IDF lexical-realignment step explicitly as its own tested unit, since Phase 3 flagged this as the most common DIY chaptering bug.
- **2. Development:** chaptering LLM call, TF-IDF realignment/timestamp-snap logic, `chapters` persistence.
- **3. Testing:** unit test the realignment logic in isolation with synthetic paraphrased boundary text against known-good source text; manual QA on a real podcast transcript.

### M6 — Show notes & metadata generator
- **Depends on:** M3 (M5 optional input)
- **Estimate:** 2–3 days
- **1. Planning:** Define the structured-output JSON schema and the retry/repair policy for malformed LLM responses.
- **2. Development:** prompt assembly, LLM call, schema validation + retry, `show_notes_drafts` persistence, frontend review/edit view.
- **3. Testing:** unit tests for schema validation and the retry path (mocked malformed responses); manual QA on a real transcript. Lowest-risk milestone in the plan — good place to also catch up on any test-debt from earlier milestones if the schedule is tight.

### M7 — Relevance-detection model v0 (heuristic + zero-shot bootstrap)
- **Depends on:** M3
- **Estimate:** 1 week
- **1. Planning:** This ships the **bootstrap version only** (heuristic rules + zero-shot LLM scoring per window) — the trained BiLSTM comes later in M15, once real data exists to train it on. Write this distinction explicitly into the plan so it isn't mistaken for the final model. Define the heuristic rule set and the zero-shot prompt contract.
- **2. Development:** sliding-window chunking (10–20s, 50% overlap) over transcript timestamps, heuristic scoring rules, zero-shot LLM scoring call, `relevance_scores` persistence tagged with a `heuristic-v0` `model_version`.
- **3. Testing:** unit tests for the windowing/chunking logic; a manual spot-check comparing v0 scores against human judgment on 2–3 real episodes (not a rigorous benchmark yet — there's no held-out labeled set to benchmark against this early).
- **Risk note:** heuristic quality ceiling is low *by design* here — that's expected, not a defect, and the shorts tool (M8) needs to be tolerant of mediocre v0 scoring.

### M8 — Shorts generator (mechanical layer)
- **Depends on:** M3, M7
- **Estimate:** 1.5–2 weeks (largest milestone before M15)
- **1. Planning:** Review `Autocrop-vertical`/`openshorts` (Phase 2 references) as implementation references for the subject-detection crop before building from scratch. Plan the peak-finding/boundary-snap/crop/caption-burn-in steps as **separately testable units** — this milestone has the most moving parts in the mechanical pipeline, so decomposing the test plan matters more here than anywhere else so far.
- **2. Development:** peak-finding over the score curve, boundary expansion + sentence-snap, duration constraints/dedup, ffmpeg cut, subject-detection + smoothed crop, optional caption burn-in, `clips` persistence.
- **3. Testing:** unit tests for peak-finding and boundary-snap logic on synthetic score curves (deterministic, no model/video needed); integration test producing an actual clip file from a fixture video and asserting duration/aspect-ratio/file-validity; manual visual QA of crop quality on 2–3 real episodes.
- **Risk note:** subject-detection crop quality is the most likely "looks worse than expected" surprise in the whole plan — budget iteration time beyond the estimate above rather than treating it as a fixed-cost task.

### M9 — Full-suite hardening / regression checkpoint
- **Depends on:** M3–M8
- **Estimate:** 1 week
- **1. Planning:** Not a new feature — plan which real podcast episodes (aim for 3–5, varying length/speaker-count/audio quality) will be run end-to-end through the entire pipeline, and what "acceptable" looks like per tool.
- **2. Development:** run the full pipeline on the chosen episodes; fix integration bugs surfaced (e.g. timestamp drift between tools, job-failure edge cases); add basic error handling/retry to job orchestration; polish the frontend result viewers for all tools built so far.
- **3. Testing:** the "development" here largely *is* the test — treat every bug found as a new regression test added to the suite, not just a silent fix. This is the explicit checkpoint the user asked for: proof that nothing has broken as the tool count grew.

### M10 — Market-analysis pipeline v0 (public curated-channel data)
- **Depends on:** M0 (independent of M3–M9's tool-suite track — can be parallelized if working on both at once)
- **Estimate:** 1 week
- **1. Planning:** Seed the initial curated channel list (a first batch of podcast channels to track). Plan the quota budget explicitly: `videos.list` batched at 50 IDs/call as the daily backbone, `search.list` reserved for sparse discovery only, RSS feeds as the free "what's new" trigger.
- **2. Development:** `market_channels`/`market_video_snapshots` tables, scheduled daily pull job, ISO-8601 duration parsing, RSS-feed polling for discovery.
- **3. Testing:** unit tests for duration parsing and ID-batching logic; integration test against the real Data API (rate-limit-aware) verifying snapshot rows land correctly; manual check of a day's pull against YouTube's own public numbers for a known video.

### M11 — OAuth + Analytics API v2 integration (own channel, retention/engagement)
- **Depends on:** M10
- **Estimate:** 1–1.5 weeks
- **1. Planning:** Design the OAuth consent flow for the single platform-level connected account (not multi-tenant, per the Phase 1 decision). Plan explicit test coverage for token refresh/expiry — flagged as a classic silent-failure source.
- **2. Development:** OAuth flow + encrypted credential storage, `reports.query` daily pulls for `analytics_snapshots` and `analytics_retention_curve` (the single-video-filter requirement for retention curves needs its own loop, not a batch call).
- **3. Testing:** unit tests for token refresh logic (including simulated expiry); integration test against the real connected test channel verifying both snapshot tables populate correctly; manual check that re-consent after token revocation is handled gracefully, not silently.

### M12 — YouTube Reporting API integration (real CTR/impressions)
- **Depends on:** M11
- **Estimate:** 1–1.5 weeks
- **1. Planning:** Explicitly plan for this being a **structurally different integration** (async bulk job + CSV download vs. live query) — the Phase 3 correction that surfaced this requirement. Plan the `reporting_jobs` poll cadence and how the pipeline tolerates the inherent reporting lag (jobs aren't ready instantly).
- **2. Development:** `reporting_jobs` table + job creation/poll/download pipeline for `channel_reach_basic_a1`, CSV parsing into `analytics_reach_snapshots`.
- **3. Testing:** unit tests for CSV parsing; integration test creating a real scheduled job and polling it to completion (this test will take longer to run than others, due to real reporting lag — document the expected wait in the test plan itself, don't just let it look "flaky").
- **Risk note:** the newest, least-documented API surface in the whole plan (added Jan 2026) — budget extra buffer beyond the estimate.

### M13 — Title/thumbnail CTR predictor v0 (heuristic-only)
- **Depends on:** M0 (technically independent of M10–M12 for the v0 heuristic version — can be pulled earlier if useful for scheduling, though it's sequenced here narratively since the *trained* version needs M12's data)
- **Estimate:** 1 week
- **1. Planning:** Explicitly scope this as **heuristic-only for now** — no trained fusion MLP yet, since there won't be enough labeled (title, thumbnail, real-CTR) pairs this early even with M12 shipped. Define the engineered feature set and a simple weighted-combination scoring function as the v0 stand-in.
- **2. Development:** feature extractors (title length/word-count/number-and-question detection, thumbnail face-count/size via a lightweight detector, thumbnail text-overlap via OCR), weighted heuristic scorer, `ctr_predictions` persistence.
- **3. Testing:** unit tests per feature extractor; manual QA comparing heuristic rankings against intuition on real candidate title/thumbnail sets.
- **Fast-follow (not in this milestone):** the trained SigLIP-2 + BGE fusion-MLP version from Phase 3, once M12 has accumulated enough real CTR pairs — that's part of M15's territory, not v1.

### M14 — Hook-strength scorer v0 (heuristic-only)
- **Depends on:** M8 (needs `clips` to score), M3
- **Estimate:** 4–5 days
- **1. Planning:** Same reasoning as M13 — ship heuristic-only, since LightGBM needs labeled data that doesn't exist yet. Define the full engineered feature list from Phase 3 (§2.8) as the concrete implementation checklist.
- **2. Development:** feature extractors (question/exclamation counts, superlative/numeric-claim detection, direct-address/imperative detection, sentiment/emotion classifier, pacing from word timestamps, title-keyword overlap), weighted heuristic scorer, `hook_scores` persistence.
- **3. Testing:** unit tests per feature extractor; manual QA spot-checking scores against intuition on real clips from M8/M9.
- **Fast-follow:** LightGBM-trained version once usage/performance data accumulates — M15 territory.

### M15 — Model retraining pipeline & first real relevance-model training run
- **Depends on:** M7, M9, M10, M11 (needs both the heuristic-bootstrap baseline and real accumulated data)
- **Estimate:** 2–3 weeks
- **1. Planning:** This is where the **actual custom BiLSTM relevance model** (frozen `bge-base-en-v1.5` embeddings → BiLSTM → sigmoid head, per Phase 3 §2.4/§5) gets trained for the first time, replacing M7's heuristic v0. Plan the pseudo-label generation job, the held-out evaluation set, and — critically — the **rollback path**: M7's heuristic scorer stays live until the trained model demonstrably beats it, not swapped in blindly.
- **2. Development:** pseudo-label generation (heuristic + zero-shot LLM, same as M7 but now used as training targets), BiLSTM training script with the temporal-smoothness regularizer, `model_versions` artifact storage, live-model swap mechanism.
- **3. Testing:** offline evaluation against the held-out pseudo-labeled set; manual A/B-style comparison of BiLSTM-v1 scores vs. the M7 heuristic on several real episodes before flipping `is_live`; regression check that M8 (shorts generator) still produces sane output when fed the new model's scores.
- **Risk note:** explicitly the **highest-uncertainty milestone in the entire plan** — there's no guarantee the trained model beats the heuristic baseline on the first attempt. This is expected and acceptable *because* M7's heuristic keeps the product fully functional regardless of this milestone's outcome — nothing else in the plan is blocked on M15 succeeding. Budget for at least one full iteration cycle (train → evaluate → fail to beat baseline → adjust → retrain), not a single pass.

## Highest-risk milestones (called out together)

| Milestone | Risk | Why it's flagged |
|---|---|---|
| M8 — Shorts generator | Subject-detection crop quality may look worse than expected | Most moving parts of any mechanical milestone; visual quality is subjective and hard to unit-test |
| M12 — Reporting API integration | Newest, least-documented API surface (added Jan 2026) | Confirmed by Phase 3 research as a genuine surprise vs. the original assumption |
| M15 — Relevance model training | No guarantee the trained model beats the M7 heuristic baseline on the first attempt | The core differentiator's actual ML risk — but non-blocking, since M7 keeps the product working either way |

## v1 Definition of Done

**M0 through M14** constitute a working, usable first version: every
in-scope v1 tool from `docs/decisions/v1-scope.md` is present and
functioning — transcription, diarization, chapters, show notes,
relevance-based shorts, and heuristic-v0 versions of the CTR predictor
and hook-strength scorer — plus the full market-analysis data pipeline
(public + OAuth + real CTR) actively accumulating data.

**M15 (trained relevance model) is deliberately excluded from the v1
cut** — it depends on data volume that can only accumulate *after* v1
has been running for a while, so it cannot honestly be scheduled as
part of the initial build. The product is fully usable without it
(M7's heuristic keeps working), which is exactly why it's safe to
treat as a fast-follow rather than a blocker.
