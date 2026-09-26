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

**Milestones M0–M2 are done.** Planning was completed across four
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
- **M2** shipped the async job pipeline: an `arq` worker over Redis
  drives jobs through `queued → running → succeeded/failed` (every
  tool runs a dummy handler until its own milestone), and the Manage
  page polls job status every 3s (decision record:
  [`docs/decisions/0006-async-job-pipeline.md`](docs/decisions/0006-async-job-pipeline.md)).

- **M3** shipped the transcript/subtitle generator: ffmpeg audio
  extraction + faster-whisper `large-v3-turbo` (int8 on the GPU, ~14×
  real-time) persisting segment/word rows, and `.srt`/`.vtt` downloads
  from the Manage page (decision record:
  [`docs/decisions/0007-transcription-pipeline.md`](docs/decisions/0007-transcription-pipeline.md)).

- **M4** shipped speaker diarization: pyannote `community-1` on the
  GPU alongside Whisper (~19× real-time), word/segment speaker labels,
  a startup healthcheck that degrades only diarization, and
  speaker-labelled subtitles split at speaker changes (decision record:
  [`docs/decisions/0008-speaker-diarization.md`](docs/decisions/0008-speaker-diarization.md)).

Next up: **M5** (chapter generator).

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
| Job orchestration | `arq` worker over Redis (M2) |
| ML training stack | Not yet decided — resolved incrementally per milestone in the development plan |

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

### One command

Once the one-time setup below is done (venv, `.env`, operator account),
`scripts/dev.sh` starts everything in one terminal: docker services,
pending migrations, API, worker (with the CUDA libs on
`LD_LIBRARY_PATH`) and the Angular dev server, with prefixed output.
Ctrl+C stops the app processes; the containers keep running.

### Backend

```bash
# 1. Local Postgres + Redis (+ optional pgAdmin at http://localhost:5050)
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

# 5. Run the API and, in a second terminal, the job worker
uvicorn app.main:app --reload
export LD_LIBRARY_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cublas/lib:$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cudnn/lib"
arq app.worker.settings.WorkerSettings
```

Without the worker running, jobs created from the Manage page stay
`queued`.

**Transcription (M3)** needs the system `ffmpeg` binary
(`sudo apt install ffmpeg`). By default the worker runs faster-whisper
`large-v3-turbo` on the GPU in int8 (the first job downloads ~1.6 GB of
model into the Hugging Face cache). The CUDA 12 runtime comes from the
`nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip wheels, which is why the
worker needs the `LD_LIBRARY_PATH` line above. Without a usable GPU,
set `WHISPER_DEVICE=cpu` (slower, no other change). See
[`docs/decisions/0007-transcription-pipeline.md`](docs/decisions/0007-transcription-pipeline.md).

**Speaker diarization (M4)** uses pyannote's gated
`pyannote/speaker-diarization-community-1` pipeline. Set it up once:
1. Create a Hugging Face account and accept the model's conditions at
   https://huggingface.co/pyannote/speaker-diarization-community-1.
2. Create a *read* access token.
3. Put the token in `backend/.env` as `HF_TOKEN=...`.

At startup the worker runs a healthcheck: it diarizes a bundled
2-speaker clip. If that fails (missing token, license not accepted,
empty output), the worker still starts and the other tools keep
working, but `diarize` jobs fail with the reason. It shows as a tooltip
on the failed badge in Manage. torch comes from the PyTorch **cu126**
index (set at the top of `requirements.txt`), the last build that still
supports Pascal GPUs like the GTX 1060. See
[`docs/decisions/0008-speaker-diarization.md`](docs/decisions/0008-speaker-diarization.md).

The API alone: `POST /auth/login` (form fields `username`/`password`)
for a token, `POST /uploads` (multipart, bearer token) to upload a
file, `GET /uploads` to list them.

To run the backend test suite, create the separate test database once
(`docker compose exec db psql -U youtube_suite -d youtube_suite -c
"CREATE DATABASE youtube_suite_test;"`), then `pytest` from `backend/`. The tests also need Redis (from
`docker compose up -d`); they use logical DB 1 (`TEST_REDIS_URL`).
The transcription tests need `ffmpeg` and run Whisper `base` on CPU
(no GPU needed; ~145 MB model download on the first run). The
diarization tests run pyannote on CPU and need `HF_TOKEN` in
`backend/.env`. With both models on CPU, the full backend suite takes a
few minutes.

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

## Third-party models

- Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  running OpenAI Whisper `large-v3-turbo` (MIT).
- Speaker diarization: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
  (MIT) with the `pyannote/speaker-diarization-community-1` pipeline,
  whose weights are licensed **CC-BY-4.0**.
- Test/healthcheck audio: LibriSpeech (CC BY 4.0). See
  `backend/tests/fixtures/ATTRIBUTION.md` and
  `backend/app/assets/ATTRIBUTION.md`.

## Development approach

Work proceeds milestone-by-milestone per the development plan. Each
milestone has three sub-phases: planning (design/test decisions locked
before coding), development, and testing (new tests plus a regression
check that nothing earlier broke). See `CLAUDE.md` and
`.claude/rules/development-phase.md` for how this is enforced.

## License

GPL-2.0 — see [`LICENSE`](LICENSE).
