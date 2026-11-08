# Brainstorm Log

Running log of ideas, open questions, and options under consideration.
Append new entries with a date header; don't delete old ones (mark
them resolved/rejected instead if superseded).

## Open questions
- ~~What exactly does the custom model do? (task definition)~~ **Resolved 2026-08-13** — three custom models, not one: relevance-detection (transcript → per-window score), title/thumbnail CTR predictor, hook-strength scorer. See `docs/project-description.md` §2, §5.
- ~~What data will it train on, and where does that data come from?~~ **Resolved 2026-08-13** — heuristic + zero-shot-LLM bootstrap initially, migrating to market-analysis-pipeline data (public curated channels + OAuth-connected channel's Analytics/Reporting API data) as it accumulates. See `docs/project-description.md` §6, `docs/development-plan.md` M15.
- ~~Build from scratch vs. fine-tune an existing model?~~ **Resolved 2026-08-13** — hybrid per-tool: transcription/diarization/show-notes use external tools (Whisper, pyannote/WhisperX, an LLM API) as-is; relevance/CTR/hook models are custom but built on frozen pretrained encoders (BGE, SigLIP-2), not trained from scratch. See `docs/project-description.md` §5.
- ~~What does "success" look like for a first version?~~ **Resolved 2026-08-13** — see "v1 Definition of Done" in `docs/development-plan.md` (M0–M14).

## 2026-08-13 — Phase 1–4 brainstorm complete
Full scope → research → spec → dev-plan cycle done for v1. Outputs:
- `docs/decisions/v1-scope.md` — Phase 1 scope
- `docs/research/bibliography.md` (+ 5 supporting files) — Phase 2 research
- `docs/project-description.md` — Phase 3 full spec
- `docs/development-plan.md` — Phase 4 milestone plan
New open item raised during Phase 3, not yet resolved: whether the
hook-strength scorer should ever take audio/video input (currently
assumed text-only for consistency with the relevance model, but this
was an inferred call, not an explicit decision).

## 2026-08-14 — Full app shell pivot before M1's PR
Deliberate scope expansion beyond M1's original spec, decided with the
project owner before M1's PR was opened — see
`docs/decisions/0004-full-app-shell-pivot.md` for the full reasoning.
Backend gained tool-trigger/management endpoints for all 8 v1 tools
(queued-only, nothing processes them yet — that's still M2+); frontend
gained all four of v1's eventual screens (library, dashboard,
analytics, manage) with real data where available and clearly-labeled
dummy data elsewhere, plus a vaporwave/Y2K design system. **Flag for
future milestones**: when M3/M5/M6/M7/M8/M10/M13/M14 come up, check
whether their frontend "Development" bullet is already half-done by
this pivot (a dummy-data file to swap for real data, a "preview" badge
to remove) before scaffolding a new screen.

## Ideas
(nothing logged yet)

## 2026-09-26 — M2 done (async job pipeline)
`arq` + Redis worker, per-`tool_type` handler registry (all 8 on the
dummy handler), 3s polling on the Manage page, manual re-run once a job
is terminal — see `docs/decisions/0006-async-job-pipeline.md`.
Open items noted for later milestones (not built now):
- The API's arq pool (`app/worker/pool.py`) is never closed on app
  shutdown — harmless for dev, worth a FastAPI lifespan hook in M9.
- If Redis is down, `POST /uploads/{id}/jobs` commits the job row and
  then 500s on enqueue, leaving a `queued` row nothing will pick up.
  M9 (hardening) should decide: enqueue-failure → mark `failed`, or a
  periodic re-enqueue sweep of stale `queued` rows.

## 2026-09-26 — M3 (transcript/subtitle generator)
Implemented per `docs/decisions/0007-transcription-pipeline.md`.
Deferred / noted, not built in M3:
- **Detected language isn't stored** (no column in the §4 schema).
  M5/M6 LLM prompts may want it. Add `uploads.language` or a
  transcript-level column when a milestone actually needs it.
- **`compression_ratio` isn't stored** either. It's only useful as a
  hallucination signal. Revisit if M9 hardening wants quality flags.
- **`progress_pct` stays NULL.** Whisper's segment generator plus
  `info.duration` could drive it, but M3 didn't ask for it. Candidate
  for M9 (a 1 h episode takes ~4 min on the GTX 1060).
- **One subtitle cue per Whisper segment** (as spec'd). Segments can be
  long. Re-cueing from word timestamps (~42 chars/line, 2 lines) would
  make better captions. Candidate for M8, whose caption burn-in needs
  clip-scoped cues anyway, or M9.
- **Transcript viewer/editor** (spec §3 "result viewers") has no
  screen yet. M3 only added `.srt`/`.vtt` download buttons in Manage.
- **Input video is read through `LocalStorageBackend.path_for`**, which
  is local-only. An S3 backend needs a download-to-temp step in the
  transcribe handler.
- **GPU contention with M4.** The transcribe handler serializes itself
  with an in-process lock, but pyannote (M4) will also want the 3 GB
  GPU. M4's planning should decide on a shared GPU lock or a
  dedicated queue/`max_jobs`.
- **arq `job_timeout` is global** (3 h now, raised from 300 s for
  transcription). Per-function timeouts may be worth it once more real
  handlers exist.

## 2026-09-26 — M4 (speaker diarization)
Implemented per `docs/decisions/0008-speaker-diarization.md`, with the
measured GPU/CPU spike in `docs/research/m4-diarization-spike.md`.
Deferred / noted, not built in M4:
- **Re-transcribing drops speaker labels.** `transcribe` replaces the
  transcript rows, so `diarize` has to be re-run by hand. Automatic
  chaining (transcribe → diarize), or re-diarizing after a
  re-transcription, belongs to M9 hardening or a pipeline-orchestration
  change.
- **Speakers are anonymous** (`SPEAKER_00`, ...). Mapping them to host
  and guest names would need UI (and probably an `upload_speakers`
  table). Worth it for show notes (M6) and shorts captions (M8).
- **No speaker-count hint.** pyannote accepts
  `num_speakers`/`min_speakers`/`max_speakers`, and podcasts usually
  know their host count, which could improve accuracy. That needs a job
  parameter, and jobs carry no parameters yet.
- **Overlapping speech is flattened.** Using
  `exclusive_speaker_diarization` gives one speaker per instant. The
  overlap-aware `speaker_diarization` output is discarded, and it could
  matter for crosstalk-heavy shorts (M8).
- **torchcodec is broken here** (`libnvrtc.so.13`). It is harmless
  because audio is passed preloaded, but a future torch/pyannote
  upgrade should re-check it. Also watch torch versions generally:
  **cu126 is the last torch build line with Pascal kernels**, so a GPU
  upgrade or a CPU move becomes necessary when torch cu126 stops being
  published for the versions pyannote needs.
- **GPU fallbacks are measured but not implemented:** model swapping
  between GPU and RAM (under 0.5 s per swap), and
  `embedding_batch_size=1` (1.8 GB peak). Use them if a future model
  squeezes the 3 GB.
- **A CUDA OOM poisons the worker process** (seen in M4). After
  ctranslate2 hits `out of memory`, every later GPU call in that
  process fails (`invalid device ordinal`) until the worker restarts.
  M4 removed the cause (torch cache release, 0008), but recovery
  belongs to M9 hardening. Candidates: detect CUDA errors in
  `process_job` and exit the worker so a supervisor restarts it, or run
  GPU work in a child process.


## 2026-09-27 — M5 planning (chapter generator): scope change in progress
Decided with the project owner during M5 planning. The formal record
goes into `docs/decisions/0009-…` once the research pass lands.
- **No Claude API for chaptering.** The owner wants everything to run
  locally, so spec §2.3/§5 ("Claude API", "Claude Sonnet") is being
  replaced for M5. **Still open:** whether M6 (show notes) also goes
  local. It is not decided, so don't assume it.
- **The owner chose to train their own models.** Two of them:
  - **Boundary model:** frozen multilingual sentence embeddings feeding
    a small sequence head, trained on VidChapters-7M.
  - **Title model:** a fine-tuned small multilingual seq2seq.
- **An unsupervised baseline stays** (TextTiling / change-point over the
  same embeddings). It is the fallback and the eval reference, the same
  pattern as M7 vs. M15.
- **Granularity:** few, broad chapters (~5–8 per hour).
- **Language:** English and Spanish (varies per episode), so all models
  must be multilingual.
- **Frontend is minimal:** `GET /uploads/{id}/chapters`, plus a chapter
  list with a "Copy for YouTube" button in Manage.
- **Estimate grows** from 3–4 days to ~2–3 weeks. The TF-IDF
  realignment step no longer applies, since boundaries are predicted on
  real segments. `docs/development-plan.md` M5 is to be updated.
- **Research pass** is running (`docs/research/m5-chapter-model.md`).
- **Model development is run as an experiment** (owner's request,
  2026-09-27). It lives in `experiments/m5-chapters/`, separate from
  the backend:
  - **Notebooks:** numbered `.ipynb` files (data → baseline →
    boundaries → titles → eval), committed **with outputs** so the git
    history shows how it developed.
  - **Tested code:** shared code goes in `lib/` with unit tests
    (metrics, boundary decoding).
  - **Tracking:** plain JSON per run (`runs/<date>-<name>/`, with
    config, metrics and plots) plus a dated log in the experiment's
    `README.md`. No MLflow or TensorBoard.
  - **Environment:** its own venv (`experiments/.venv`). The backend
    only gets the minimal inference deps; the production handler only
    *loads* an exported model.
  - **Open:** a second venv with torch cu126 costs ~4 GB against
    ~11 GB of free disk. Decide at design time whether to reuse the
    torch install or free up disk.
- **Planning closed (2026-09-27):**
  `docs/decisions/0009-chapter-generator-local-models.md`. Decided with
  the owner:
  - **Spanish:** cross-lingual transfer, measured on a Spanish-only
    test set.
  - **Fallback:** the trained boundary model ships only if it beats the
    baseline on validation F1@5s.
  - **Traceability:** a `chapters.model_version` column.
  - **Bulk experiment data** (datasets, embeddings, checkpoints) goes on
    the second drive, `/mnt/d` (NTFS, 801 GB free). venvs, Docker and
    the production bundle stay on the SSD.
- **Open, not decided:**
  - **Title fallback.** Nothing is defined if mT5-small's titles prove
    unusable. Raise it with the owner before building one.
    **Update 2026-09-28: being resolved.** 05 showed mT5's Spanish
    titles are unusable on the owner's episode.
    - **The owner approved** an experiment comparing local pretrained
      instruction LLMs (llama.cpp, CPU-sized), with mT5 as the
      fallback.
    - **Plan:** `experiments/m5-chapters/README.md`, "plan for 06".
    - **Outcome:** goes into `docs/decisions/0010-…`, amending 0009.
    - **Resolved 2026-09-28** (`docs/decisions/0010-chapter-titles-local-llm.md`):
      titles come from **Qwen3.5-4B** (GGUF, llama.cpp, CPU), with mT5
      as the per-chapter fallback.
      - The owner's blind review preferred qwen3.5-9b for all 6
        chapters, but it fails the CIDEr en and RAM gates. **Revisit
        the 9B** if the RAM budget changes.
      - The episode language comes from fastText LID in the chapters
        pipeline, because Whisper's detected language isn't persisted
        (see the M3 note above on `uploads.language`).
  - **Whether M6 (show notes) also goes local.** Spec §2.6 still says
    Claude API.
  - **Hierarchical chapters** (TreeSeg / ARC-Chapter style): out of
    scope for v1.

- **Worker memory after M5 (2026-10-13), open:** once a `chapters` job
  has run, the arq worker stays at ~6.8 GB RSS (1.6 GB idle; the chapter
  bundle stays loaded per process). On the 16 GB dev machine that leaves
  little room for later tools that load their own models (M7 relevance,
  M8 shorts). Options to weigh when those milestones are planned:
  unload the bundle after each job, a separate worker/queue for
  heavy-model tools, or accept it. Not decided.

- **torchcodec / torch mismatch (2026-10-13), open:** in the backend
  venv pyannote warns "torchcodec is not installed correctly"
  (`libtorchcodec_image.so: undefined symbol: torch_list_size`). No test
  fails and diarization works (we hand pyannote decoded audio), but
  pyannote's built-in file decoding would fail. Pin a torchcodec that
  matches the installed torch next time the backend deps are touched.

## 2026-10-16 — M6 planning (show notes)
Planned with the owner: `docs/decisions/0011-show-notes-local-llm.md`.
- **Resolved:** M6 goes local too. Show notes reuse the chapters
  bundle's Qwen3.5-4B (one shared `Llama` per worker process), so
  nothing in v1 calls an external LLM API.
- **Still open, not in M6:**
  - pushing show notes to YouTube via `videos.update` (spec §2.6). M6
    is copy/paste only.
  - speaker names in the notes. Diarization labels are anonymous, so
    they are not passed to the prompt (see the M4 note on an
    `upload_speakers` table).
  - the 9B for show notes too, if the RAM budget changes (0010).
- **M6 done (2026-10-16).** QA on upload 8 was approved by the owner.
  Noted for later, not built:
  - **Titles repeat chapter titles:** 2 of 5 titles were chapter
    titles verbatim, because the reduce prompt passes them in. A
    "don't reuse chapter titles" line would be a prompt v2.
  - **~16 min per 64-min episode** (estimated 6–8). Mostly map-step
    prefill on CPU. Candidates: fewer/larger chunks, or reusing the
    chapter titles step's work. Worth a look if M9 hardening finds it
    slow.
  - **No password reset command** in `app/cli.py` (came up during QA).

## 2026-10-16 — M7 planning (relevance v0)
Planned with the owner: `docs/decisions/0012-relevance-v0.md`.
- **For M15:** spec §2.4's encoder, `bge-base-en-v1.5`, is
  English-only, and episodes are en/es. Re-check the encoder choice
  then (M5's `multilingual-e5-small` is a candidate).
- **`model_versions` exists from M7**, for relevance only. Chapters and
  show notes keep text version columns. Unify them only if M15 or
  later actually needs it.

## 2026-10-16 — M7 QA findings (relevance v0)
See the QA run in `docs/decisions/0012-relevance-v0.md`.
- **Heuristic and LLM scores barely agree** (correlation −0.18 on
  upload 8). The owner approved the combined top 5. If M8's clip picks
  look off, try reweighting (more LLM) or dropping weak features; any
  change bumps the version label. M15 should treat the two components
  as separate, noisy pseudo-labels.
- **~17 min per 64-min episode** (estimated 5–10), like show notes.
  Relevance + show notes ≈ 33 min of CPU LLM per episode. A candidate
  for M9 hardening (larger batches, shorter prompts, or one shared pass).
- **The blind-moments protocol was skipped** and only one episode was
  reviewed. Repeat both when more episodes have transcripts (M8 QA is a
  natural place).

## 2026-10-16 — M8 (shorts generator)
Implemented per `docs/decisions/0013-shorts-generator.md`, with the crop
spike in `docs/research/m8-crop-spike.md`.
Deferred / noted, not built in M8:
- **Speaker-aware framing:** in a two-person shot, pick the face of the
  active speaker by mapping diarization speakers to faces. Episode 8
  has one person per shot, so it wasn't needed there.
- **Static two-shots and wide shots** (both hosts in frame) are
  untested. Try one such episode in M9 hardening.
- **NVENC encoding:** CPU libx264 already takes ~5 s per clip, so this
  is only worth it with much longer clips.
- **Per-clip titles/hooks** from the local LLM, and **publishing to
  YouTube** (`videos.insert` for Shorts): both open.
- **Clips over 60 s:** YouTube Shorts now allows up to 3 min. The cap
  is a constant (`MAX_S`) if the owner wants longer.
- **Caption styling:** the font, size, colours and position are fixed.
  Per-channel style presets could become job params.
- **The dashboard "Relevance curve" tile is still a dummy** (M7 didn't
  replace it). Swap it for a real aggregate when the dashboard is
  revisited (M9 or M11).
- **`jobs.params` is generic:** M4's speaker-count hint could use it.


## 2026-10-16 — M9 (hardening, season-1 run)
See `docs/decisions/0014-m9-performance.md` and `docs/research/m9-performance.md`.
- **Relevance quality (for M15):** the zero-shot 4B LLM discriminates
  weakly. Read greedily, it is unstable (CPU and GPU agree at Spearman
  ~0.74). Read as an expected value over the digits 0–9, it is stable
  (0.98) but picked logistics and credits on episode 8. M15's
  pseudo-labels should use the expected-value score
  (`experiments/m9-perf/llm_expected.py`), not greedy integers. Also
  try a bigger LLM if the RAM budget allows.
- **Auto-chaining tools after upload** (transcribe → … → shorts) is
  product scope; the M9 runner does it from outside. Consider it for a
  "process everything" button.
- **16 LLM layers on the GPU** would give ~25 % more LLM speed, if
  Whisper and pyannote could be unloaded between jobs. Model
  residency per job is an open question.
- **The `is_live` flip for relevance p1 → p2 was done by hand in the
  dev DB.** M15's live-switch mechanism should cover it.

## 2026-11-06 — M10 planning (market-analysis pipeline v0)
Plan in `docs/decisions/0015-market-pipeline-v0.md`. Left out of M10 on purpose
(open items for later milestones, not built now):
- **Analytics page wiring:** `frontend/src/app/analytics/` still uses
  `analytics-dummy-data.ts`. Swap the market chart to real
  `market_video_snapshots` data, which needs a read endpoint. That
  milestone is not assigned yet. It is also unclear what "trend score"
  should mean.
- **Channel-level stats:** `channels.list` subscriberCount/videoCount
  (§6.1) has no table in §4.2. A `market_channel_snapshots` table is
  needed if M13/M15 want channel size as a confounder.
- **New candidate channels:** discovering channels to add to the list
  (the §6.1 role of `search.list`) is not used in M10.
- **Snapshot retention/downsampling:** about 2.4M rows/year at about 25
  channels. Fine for now; revisit if the list grows.
- **Missed daily pull:** if the worker is down at 09:30 UTC, that day's
  snapshots for recent videos are lost (§4 of 0015). Consider a catch-up
  run on worker startup.

## 2026-11-07 — M10 implementation notes
- **New dev machine: RTX 5070 (Blackwell, sm_120).** The pinned
  `torch==2.8.0+cu126` ships kernels up to sm_90 only (checked:
  `torch.cuda.get_arch_list()`), so diarization (and anything else on
  torch CUDA) can't run on this GPU. The tests are unaffected (CPU). The
  pin was chosen for the GTX 1060 (cu128+ dropped Pascal, 0008 §2). Open:
  move to a cu128 build now that the 1060 is gone, and re-check
  ctranslate2 / llama.cpp CUDA builds and the M9 speed numbers? Not M10's
  scope; needs the owner's call.
- **`.env.example`** (resolved 2026-11-07: now `YOUTUBE_API_KEY=` + the `MARKET_*` settings; an empty key counts as unset) still had the placeholder `YOUTUBE_API_KEY=...`. A copy
  of that line would set the key to `...` (the tasks would then fail with
  HTTP 400 instead of skipping). It should be `YOUTUBE_API_KEY=` plus the
  three `MARKET_*` settings.
- **Seed list:** `app/market/seed_channels.json` holds the proposed §7 list.
  Two entries (Sastre y Maldonado, Lo Que Tú Digas) have no handle yet and
  are skipped by `market seed` until one is filled in.
- **La Cotorrisa** (seed list) is still unresolved, see 0015 "Seed and first pull". The
  owner needs to supply the official channel's handle or ID.
- **Chapter bundle missing on the new machine.** `CHAPTERS_MODEL_DIR` has no bundle here,
  so 6 integration tests (chapters, relevance, show notes) skip. The owner accepted this to
  close M10 (2026-11-07). Re-export it in the post-M10 GPU/models phase.
- **Test media are gitignored.** `scripts/build_test_fixtures.sh` rebuilds them from
  LibriSpeech, exactly as `ATTRIBUTION.md` describes. It was used on the new machine.

## 2026-11-07 — RTX 5070 hardware pass (docs/decisions/0016)
- **RTX 5070 / torch cu126 question (above): resolved.**
  - Moved to torch 2.8.0+cu128, ctranslate2 4.8.5 and llama.cpp built for sm_120.
  - GTX 1060 support is dropped.
- **Open, possible now that the GPU has room (not built):**
  - **Bigger chapter encoders:** multilingual-e5-base/large, bge-m3. Needs a retrain plus
    the 03/05 evaluation.
  - **mT5 recipe:** unfreeze the embeddings/LM head, use a larger batch.
  - **Training speed:** 03/04 run at 17–27% GPU because the batches are 1060-sized. A larger
    batch would cut the retrain from ~3 h.
  - **Show notes:** a larger `n_ctx` or one-shot generation (0011 rejected both for RAM).
  - **Shorts:** NVENC encoding (0013 left it out because the GPU was busy).
  - **LLM quantization:** Q6_K/Q8_0 for the 9B (downloaded, not evaluated). They need a
    "release Whisper/chapters before loading the LLM" hook to fit in VRAM.
- **torch cu130 would fix torchcodec** (pyannote's own decoder), but it means two CUDA majors
  in the venv next to ctranslate2's CUDA 12.
- **Bundles live on `/mnt/c` (9p).** After `release_all()`, each LLM job reloads the 5.7 GB
  GGUF over 9p (page cache helps). Moving `backend/models/` to ext4 would cut chapters job
  time. Measure first.
- **VRAM headroom is ~1.6 GB** with every model resident (10.6 of 12.2 GB). A heavy GPU app
  on the Windows desktop can push a job into OOM, which is retried as transient.
- ~~**Owner's blind review pending**~~ **Resolved 2026-11-08:** the owner chose the 9B for C3–C5
  (titles, relevance top-5, show notes). The 9B stays live, and M15's pseudo-labels use it (0016).
- **`backend/.env` still has the M9 values** and overrides the new defaults until the owner
  updates it (the agent can't edit `.env`).

## 2026-11-07 — M11 planning (OAuth + Analytics API v2)
Plan in `docs/decisions/0017-oauth-analytics-v2.md`. Left out of M11 on purpose:
- **Linking `analytics_videos` to `uploads`:** knowing which app upload became which published
  video (needed by M15 to correlate clip features with real retention). Manual link, title
  match, or `videos.insert` from the app are all open.
- **Encryption-key rotation** for `OAUTH_TOKEN_ENCRYPTION_KEY`: not built; a lost key means
  re-consent.
- **Retention by `audienceType`** (organic vs ads): M11 stores all traffic only.
- **Showing analytics data in the UI:** the Analytics page's charts stay on dummy data.
- **OAuth app still in "Testing" (open, deferred to the end of the project — 0017 amendment
  2026-11-08):** publishing needs a public homepage + privacy-policy URL on an authorized domain
  (plan: `<user>.github.io` with `index.html` + `privacy.html`, add that subdomain as an
  authorized domain, fill the Branding page, then Audience → Publish app; don't submit for
  verification). Until then, re-connect the channel from `/analytics` every ~7 days.
- **M11 follow-up (open): verify real Analytics rows** once the test channel has views
  (48–72 h lag). See 0017 "First live run". Until then, the live run only proved 0-row pulls.
- **Dev-env drift (open):** the WSL clone `~/projects/new_youtube_suite` (which holds the venv
  and `node_modules`) is still at M9. M11 was run from the Windows tree using that clone's venv
  and `node_modules`. Update the clone, or point the venv at the Windows tree, before relying on
  `scripts/dev.sh`.
- **Uvicorn access log prints the OAuth callback query** (single-use `code` and `state`).
  Low risk (the code is single-use, short-lived and already exchanged), but a log filter
  could redact it.

## 2026-11-08 — M12 planning/development (Reporting API, real CTR/impressions)
Plan in `docs/decisions/0018-reporting-api-reach.md`. Left out of M12 on purpose, or still open:
- **`channel_reach_combined_a1`** (traffic source / device / OS breakdown of impressions and
  CTR): not in M12, which covers basic only. Add it if the M13/M15 CTR model wants per-surface
  CTR (e.g. browse vs suggested). It would be a second job type on the same pipeline.
- **Verify on the first real CSV (open):**
  - the header spelling
  - the date format (`YYYY-MM-DD` vs `YYYYMMDD`; both accepted)
  - the CTR scale. `CTR_SCALE="auto"` reads a file with any value > 1 as percentages, so a
    percent-scale file whose CTRs are all ≤ 1 % would be misread as fractions. Pin
    `CTR_SCALE` once a real file with impressions has been seen (integration phase 2 prints
    it).
- ~~**Whether `yt-analytics.readonly` is enough for `jobs.create`**~~ **Resolved 2026-11-08:** yes; phase 1 created the job with the M11 grant (0018 "Live phase 1").
- **Reach rows are not linked to `analytics_videos` or `uploads`:** this is the same open
  linking question as M11, and M13/M15 will need it to pair (title, thumbnail) with real CTR.
- **Dev-env drift (M11 entry above): resolved 2026-11-08.**
  - The WSL clone was fast-forwarded to M11 (`git pull`). M12 is developed and tested there.
  - The gitignored test media were copied over from the Windows tree.
  - ~~**Still open:** `backend/.env` only in the Windows tree~~ **Resolved 2026-11-08:** the owner copied it to the WSL clone; the 4 diarization tests pass again.

## 2026-11-08 — M13 planning/development (CTR predictor v0, heuristic)
Plan in `docs/decisions/0019-ctr-heuristic-v0.md`; evidence in `docs/research/ctr-heuristic-evidence.md`.
Left out of M13 on purpose, or still open:
- **Calibrating the v0 weights against our own CTR (open, M15 territory):** once
  `analytics_reach_snapshots` has real CTR for published videos, check whether the four
  weighted features (and the zero-weight ones stored in `ctr_predictions.features`) track
  real CTR. This needs the same open link as M11/M12: which candidate became the published
  title/thumbnail of which video.
- **Standalone candidates** (`upload_id` NULL, no episode): deferred by the owner.
- **Recognised thumbnail text** (full OCR, e.g. text that repeats the title): the owner chose
  detection only.
- **Thumbnail candidate picker** (best frames from the episode): still deferred (v1-scope).
- **Dashboard "Avg. predicted CTR" tile:** **Removed 2026-11-08 with the other dummy tiles (0021); the note below still holds if it comes back:** stays dummy; an average of relative scores across
  episodes isn't meaningful anyway, so it needs a different metric before it's wired.
- **Manual QA (open follow-up; M13 was closed without it on 2026-11-08):** the owner ranks 3–5
  candidate sets on 2–3 real episodes and compares them with the heuristic. The findings go in
  0019. Frames are ready in `C:\Users\Pablo\Downloads\ctr-qa-frames\`.
- ~~**Dev env: `STORAGE_BASE_PATH` (open):** the uploaded episodes live in `/home/pablo/yts-data/uploads`,
  but the WSL clone's `backend/.env` resolves storage to `../uploads` (empty). The owner should set
  `STORAGE_BASE_PATH=/home/pablo/yts-data/uploads` before running `scripts/dev.sh` from the WSL clone.~~
  **Resolved 2026-11-08:** set in the WSL clone's `backend/.env` (backup `backend/.env.bak-2026-11-08`).
  All 446 storage keys in the DB (63 uploads, 63 audio, 315 clips, 5 CTR thumbnails) now resolve to
  existing files. The 5 thumbnails had been written to the old `../uploads` and were copied over.
  `scripts/start-wsl.cmd` launches the stack from Windows.

## 2026-11-08 — M14 planning/development (hook-strength scorer v0, heuristic)
Plan in `docs/decisions/0020-hook-heuristic-v0.md`.
Left out of M14 on purpose, or still open:
- **Training the LightGBM ranker (M15 territory):** the `hook_scores.feature_vector`s are its
  inputs. It needs a label: Shorts retention, or "viewed vs swiped away" per published short.
  That needs a link nobody has built yet, from a `clips` row to the YouTube Short it became
  (the same open linking question as M11–M13).
- **`start` (dead air, weight 0.30) never varies on this channel:** M8 cuts clips at segment
  starts, so every real clip had 0.0 s lead silence (dry run, 25 clips, 2026-11-08). Keep it
  for clips trimmed by hand, or move its weight once real data exists. This is for the
  manual QA to decide.
- **`address` mostly measures the hosts talking to each other:** "tú" between the two hosts
  counts as direct address even when it isn't aimed at the audience. Telling the two apart
  would need speaker/turn context (diarization exists) or an LLM. Open.
- **Zero-shot LLM hook judge** (as in M7 relevance): not built, because the plan scopes M14
  to heuristics. It could later serve as a pseudo-label source for the ranker.
- **Audio features** (energy, pitch, laughter): text-only v1 constraint (spec §2.8, §7).
- **Dashboard "Top hook-scoring clip" tile:** **Removed 2026-11-08 with the other dummy tiles (0021); the note below still holds if it comes back:** stays dummy. Hook scores are relative within
  one episode, so "top across episodes" needs a different metric before it's wired.
- **Manual QA (open follow-up; M14 was closed without it on 2026-11-08):** rank the clips of 2–3 episodes by their first
  15 s, then compare with the heuristic. The findings go in 0020.

## 2026-11-08 — Frontend UX + explainability pass (outside the milestone plan)
Plan in `docs/decisions/0021-frontend-ux-explainability.md`. Left out on purpose, or still open:
- **Owner review of the popup copy (open):** the `ToolDoc` text in `frontend/src/app/explain/tool-docs.ts`
  is what the TFM defends. Check every step, number and limitation against the decision records
  each doc cites in `source`.
- **Keep docs in sync (rule):** a change to a tool's pipeline updates its `ToolDoc` (steps, flow,
  model card) in the same change. When M15 trains a model, flip its `training.planned` and replace
  the planned flow with the real one.
- **Analytics page:** not redesigned; it still shows its preview market chart. It could reuse the
  ⓘ popups for `market` / `analytics` / `reporting`, which already exist.
- **Not built (ideas):** per-episode data lineage (which model version produced each stored result,
  shown in one table); a "compare episodes" view of relevance curves; the explanations in Spanish
  (the owner chose English for now).

## 2026-11-08 — Analytics & Dashboard data visualisation (outside the milestone plan)
Plan in `docs/decisions/0022-analytics-dashboards.md`. Left out on purpose, or still open:
- **Own-channel charts (open):** daily views, real CTR and retention render once the OAuth channel
  has views (same follow-up as M11/M12). Until then they show why they're empty.
- **View velocity (open):** needs ≥ 2 daily snapshots of the same videos (14-day window). Re-check
  after a few daily pulls; then consider per-video trajectories instead of the cross-sectional
  age curve.
- **Breakout-videos analysis:** offered (age-adjusted residual z-scores, top over/under-performers),
  not chosen by the owner. The residuals already exist in `app/insights/analysis.py`.
- **Linking for the own channel:** `uploads.youtube_video_id` also joins `analytics_videos` /
  `analytics_reach_snapshots` once the owner's own channel publishes episodes. This is the input
  M13/M15 need to pair candidates with real CTR (the 0019 open question).
- **Market timezone:** the publish heatmap uses Madrid time for every channel. Per-channel local
  time (US channels) would need a timezone per channel.
- **Precomputed Analytics (0023), resolved 2026-11-08:** the analyses moved out of the requests,
  into `insight_results`, refreshed hourly by the worker (no-op when the sources are unchanged).
  Still open: the growth payload (~150 KB) repeats channel names per point. A channel index
  would shrink it if it ever matters.
