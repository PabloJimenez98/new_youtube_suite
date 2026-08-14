# YouTube Suite

A toolkit for producers of YouTube video podcasts. Upload an episode
and run a set of AI-powered tools against it — transcription, chapter
generation, short-clip extraction, show-notes drafting, and
pre-publish performance scoring (title/thumbnail CTR, clip "hook"
strength). A market-analysis component tracks a curated set of public
podcast channels plus one OAuth-connected YouTube channel, feeding
real performance data back into the platform's custom models over
time.

## Status

**Milestones M0 and M1 are done.** Planning was completed across four
brainstorm phases (scope → research → specification → development
plan) — see [Documentation](#documentation) below. Implementation
proceeds milestone-by-milestone from `docs/development-plan.md`:
- **M0** shipped a FastAPI backend, Postgres schema
  (`users`/`uploads`/`jobs`), single-user JWT auth, and a
  local-filesystem storage abstraction (decision record:
  [`docs/decisions/0002-backend-skeleton.md`](docs/decisions/0002-backend-skeleton.md)).
- **M1** shipped an Angular frontend skeleton — login screen, an
  upload form with progress reporting, and a bare uploads list — that
  calls M0's endpoints through the Angular dev-server's proxy
  (decision record:
  [`docs/decisions/0003-frontend-skeleton.md`](docs/decisions/0003-frontend-skeleton.md)).

Next up: **M2** (async job pipeline).

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | Angular |
| Backend | FastAPI (Python) |
| Database | PostgreSQL |
| Transcription | `faster-whisper` (`large-v3-turbo`) |
| Diarization | `pyannote.audio` 4.0.7 + WhisperX |
| Custom models | Relevance-detection, title/thumbnail CTR predictor, hook-strength scorer — see the model strategy in the project description |
| Blob storage | Local filesystem behind a `StorageBackend` interface (M0) |
| Job orchestration, ML training stack | Not yet decided — resolved incrementally per milestone in the development plan |

## Documentation

Planning and research live in `docs/`, produced across four phases:

| Phase | Output |
|---|---|
| 1 — Scope | [`docs/decisions/v1-scope.md`](docs/decisions/v1-scope.md) |
| 2 — Research | [`docs/research/bibliography.md`](docs/research/bibliography.md) (index of 5 research files) |
| 3 — Specification | [`docs/project-description.md`](docs/project-description.md) |
| 4 — Development plan | [`docs/development-plan.md`](docs/development-plan.md) |

Ongoing/running notes: [`docs/brainstorm.md`](docs/brainstorm.md).

## Getting started

### Backend

```bash
# 1. Local Postgres (+ optional pgAdmin at http://localhost:5050)
docker compose up -d

# 2. Backend env
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # adjust JWT_SECRET etc. for real use

# 3. Schema
alembic upgrade head

# 4. Seed the single operator account (no public signup endpoint)
python -m app.cli create-user you@example.com

# 5. Run
uvicorn app.main:app --reload
```

The API alone: `POST /auth/login` (form fields `username`/`password`)
for a token, `POST /uploads` (multipart, bearer token) to upload a
file, `GET /uploads` to list them.

To run the backend test suite, create the separate test database once
(`docker compose exec db psql -U youtube_suite -d youtube_suite -c
"CREATE DATABASE youtube_suite_test;"`), then `pytest` from `backend/`.

### Frontend

Requires Node 22+ and the Angular CLI (`npm install -g @angular/cli`).
With the backend running (above):

```bash
cd frontend
npm install
ng serve
```

Open `http://localhost:4200`, log in with the account you seeded, and
upload a file. The dev server proxies `/auth` and `/uploads` to the
backend at `127.0.0.1:8000` (see `frontend/proxy.conf.json`) — no CORS
setup needed. Run the frontend test suite with `ng test` (Vitest,
headless — no Chrome/browser install required).

## Development approach

Work proceeds milestone-by-milestone per the development plan. Each
milestone has three sub-phases: planning (design/test decisions locked
before coding), development, and testing (new tests plus a regression
check that nothing earlier broke). See `CLAUDE.md` and
`.claude/rules/development-phase.md` for how this is enforced.

## License

GPL-2.0 — see [`LICENSE`](LICENSE).
