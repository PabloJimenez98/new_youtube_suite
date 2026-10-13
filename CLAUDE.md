# YouTube Suite

## Status
Planning is complete. All four brainstorm phases are done: scope
(`docs/decisions/v1-scope.md`), research (`docs/research/bibliography.md`),
full specification (`docs/project-description.md`), and a milestone-based
development plan (`docs/development-plan.md`). **Milestones M0 (backend
& database skeleton), M1 (frontend skeleton & upload flow), M2
(async job pipeline), M3 (transcript/subtitle generator), M4
(speaker diarization) and M5 (chapter generator) are done** — see
`docs/decisions/` (0002–0010) for what was decided. The next step is
milestone **M6**.
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
