# YouTube Suite

## Status
Planning is complete. All four brainstorm phases are done: scope
(`docs/decisions/v1-scope.md`), research (`docs/research/bibliography.md`),
full specification (`docs/project-description.md`), and a milestone-based
development plan (`docs/development-plan.md`). **Milestones M0 (backend
& database skeleton), M1 (frontend skeleton & upload flow), M2
(async job pipeline), M3 (transcript/subtitle generator), M4
(speaker diarization), M5 (chapter generator), M6 (show notes & metadata generator), M7
(relevance-detection model v0), M8 (shorts generator), M9 (full-suite
hardening), M10 (market-analysis pipeline v0) and M11 (OAuth + Analytics
API v2) are done** — see `docs/decisions/` (0002–0020) for what was
decided. A hardware pass for the new RTX 5070 (outside the milestone
plan, decision 0016) is implemented; the owner's blind review chose the 9B
(2026-11-08). M11's follow-up (verify real Analytics rows) was closed by
the owner on 2026-11-14 (0017). The OAuth app stays in Testing until
the end of the project (0017 amendment). **M12 (Reporting API, real
CTR/impressions) is done** (0018): live phase 2 passed on 2026-11-14 (33
header-only reports, 0 failed). Open, non-blocking: pin `CTR_SCALE` once the
first non-empty reach file arrives. **M13 (CTR predictor v0, heuristic) is
done** (0019) and **M14 (hook-strength scorer v0, heuristic) is done** (0020).
Both were pulled ahead while M12 waited, and their manual QA was closed by the
owner on 2026-11-14. A frontend UX + explainability pass (outside the
milestone plan, decision 0021) is done; the owner approved the popup copy
(2026-11-14). A data-visualisation pass (Analytics/Dashboard charts,
`/insights` API, episode ↔ YouTube link, transcript viewer; also outside the
plan, decision 0022) is done, with its analyses precomputed by the worker into
`insight_results` (0023). **M15 (relevance BiLSTM + retraining pipeline) is
done** (0024, 2026-11-14). Iteration 1 failed the offline gate. Iteration 2,
`bilstm-e5s-20261114-i2` (exact heuristic + a BiLSTM for the LLM part), passed
it (0.652 vs 0.545) and the owner's blind A/B (3–2), and is **live**. Roll back
with `python -m app.cli models set-live relevance
heuristic-v0+Qwen3.5-9B-Q4_K_M+relevance-p3`. Open follow-up: the owner reviews
the rendered clips of uploads 5, 35 and 43. CTR and hook training are
follow-ups after v1, not M15 (owner, 2026-11-14). **All milestones M0–M15 are
done.**
Do not skip ahead in milestone order, and do not build a milestone's
scope beyond what that milestone specifies without discussing it first.

## Domain
YouTube Suite is a toolkit for producers of YouTube video podcasts:
upload an episode, run AI-powered tools against it (transcription,
diarization, chapters, shorts, show notes, CTR/hook scoring), with a
market-analysis pipeline feeding real performance data back into three
custom models over time. Full scope, architecture, data model, and
model strategy are in `docs/project-description.md` — treat that file
as the source of truth for "what are we building," not this one.

## Tech stack
Angular (frontend) + FastAPI (backend) + PostgreSQL (database).
Transcription via self-hosted `faster-whisper`; diarization via
`pyannote.audio`/WhisperX. Job-queue technology, blob storage, and the
ML training stack are resolved incrementally as their owning milestone
is reached (see `docs/development-plan.md`) — don't assume a specific
choice for those beyond what a milestone's planning sub-phase has
actually locked in.

## Where things go
- `docs/development-plan.md` — the milestone plan. Work happens
  milestone-by-milestone, in order, each with a planning → development
  → testing sub-phase (see `.claude/rules/development-phase.md`).
- `docs/brainstorm.md` — running log of ideas/open questions. Append,
  don't overwrite; mark resolved/rejected instead of deleting.
- `docs/research/` — findings from research tasks.
- `docs/decisions/` — dated decision records (what was decided, why,
  alternatives rejected). Add a new one whenever a milestone's planning
  sub-phase makes a real architectural choice (e.g. "chose arq over
  Celery for the job queue") — don't let those decisions live only in
  code comments or chat history.
- `docs/project-description.md` — full v1 specification (product,
  tools, architecture, data model, model strategy, data sources).

## Compact policy
When summarizing this conversation: preserve every open question and
every decision made so far, exactly as stated. Do not silently resolve
open questions in the summary — if something is undecided, keep it
flagged as undecided. Drop only fully abandoned/discarded tangents.

## Working rules for this phase
- Follow `docs/development-plan.md` milestone order; don't jump ahead
  or build ahead of the current milestone's scope.
- Each milestone: write a short planning note (design + test plan)
  before implementing, then implement, then write/run tests — including
  a regression check that earlier milestones still work.
- When a milestone's planning sub-phase makes a real architectural
  choice, record it in `docs/decisions/`.
- Ask clarifying questions before assuming scope; don't invent
  requirements to fill gaps.
- Flag when a topic is broad enough to warrant a dedicated research
  subagent rather than answering inline.
