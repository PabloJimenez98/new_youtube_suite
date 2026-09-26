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

