# Decision: Async Job Pipeline (M2)

**Date:** 2026-09-26
**Status:** Decided

## Context

M0/M1's app-shell pivot (`0004-full-app-shell-pivot.md`) already added the
`jobs` table and the endpoints to create/list job rows (`POST/GET
/uploads/{id}/jobs`, `GET /jobs`), plus per-tool "Run" buttons in the
Manage page. Nothing processes those rows — they sit at `status='queued'`
forever, by design, until M2. `docs/development-plan.md`'s M2 entry
leaves two things open: the job-queue technology, and how job progress
reaches the frontend.

## Decisions

### 1. Queue technology: `arq` over Redis

Chosen over Celery and over RQ.

- **vs. Celery:** Celery is a sync-first framework with its own broker
  abstraction, result backend, and worker-pool model — a lot of
  operational surface for a single-operator app with one worker
  process. The whole backend here (FastAPI, SQLAlchemy) is already
  asyncio-native; Celery's task functions are sync by default and
  bridging them back to `asyncpg`/async SQLAlchemy sessions means
  either a second sync DB engine or wrapping every call in
  `run_in_executor`. Not worth it at this scale.
- **vs. RQ:** RQ is simpler than Celery but still sync-only. Same
  bridging problem as Celery, just with less surrounding machinery.
- **arq**: asyncio-native (task functions are `async def`, run on the
  same event-loop model as the rest of the app), Redis-backed (one new
  `redis` service in `docker-compose.yml`, nothing else new to run),
  and its `WorkerSettings` class is the entire worker entrypoint — no
  separate result backend or exchange/queue config needed. This is the
  plan's own recommendation (`arq or RQ`) narrowed to `arq` specifically
  because of the async fit.

### 2. Job dispatch: one enqueue path, per-tool handler registry

Every `POST /uploads/{id}/jobs` call — regardless of `tool_type` —
enqueues the same arq task, `process_job(job_id)`. That task:
1. loads the `Job` row,
2. sets `status='running'`, `started_at=now()`,
3. looks up a handler function by `tool_type` in a registry
   (`app/worker/handlers/__init__.py`),
4. runs it,
5. sets `status='succeeded'`, `completed_at=now()` on success, or
   `status='failed'`, `completed_at=now()`, `error_message=str(exc)` on
   an unhandled exception from the handler.

M2 registers exactly one handler — `dummy` — used as the fallback for
all 8 `tool_type`s, since none of them have real processing yet. It
just `asyncio.sleep()`s for a configured duration and returns; there is
no results table yet for it to write into (that's each tool's own
milestone — M3 adds `transcript_segments`, etc.), so "placeholder
result" for M2 means *the job reaching `succeeded`*, not a persisted
payload.

**Why this shape matters for M3 onward:** each future milestone (M3
transcribe, M4 diarize, M5 chapters, …) registers its own real handler
for its `tool_type` in that same registry, replacing the dummy one for
that type only. The other 7 tool types keep running the dummy handler
until their own milestone lands. No changes to the enqueue path, the
state-machine code, or the frontend polling are needed as each new
handler is added — only a new file under `app/worker/handlers/`.

### 3. Progress reporting: polling (no websockets)

Per the plan's own recommendation. The Manage page (already fetching
uploads+jobs once on load per the M1 pivot) adds a fixed-interval
re-fetch (every 3s) while the page is active, stopped on
`ngOnDestroy`. No new endpoint needed — same `GET /jobs` call, just on
a timer instead of once.

### 4. State machine: unchanged

`queued → running → succeeded|failed` already exists exactly as
needed, via the CHECK constraint from `0003_create_jobs_table.py`. M2
adds the code that actually drives those transitions; no migration.

## Consequences

- New dependency: `arq` (+ its `redis` client dependency).
- New service in `docker-compose.yml`: `redis` (redis:7-alpine).
- New settings: `redis_url` (config.py + `.env.example`).
- New module: `app/worker/` (`settings.py` for `WorkerSettings`,
  `tasks.py` for `process_job`, `handlers/` for the per-tool-type
  registry, starting with `handlers/dummy.py`).
- `uploads.py`'s `create_job` enqueues the arq task right after commit.
- Running the worker locally is a separate process: `arq
  app.worker.settings.WorkerSettings` (documented in the root
  `README.md`).
- Integration tests assume Redis is reachable via `docker-compose`,
  the same assumption already made for Postgres — no skip-if-unavailable
  marker, consistent with the existing test suite's style.
