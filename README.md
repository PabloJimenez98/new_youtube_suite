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
(`sudo apt install ffmpeg`).
- **Default:** faster-whisper `large-v3` in int8_float16 on the GPU,
  16 VAD chunks at a time, greedy (`WHISPER_BATCH_SIZE=16`,
  `WHISPER_BEAM_SIZE=1`). On the RTX 5070 that's ~48× real time, with a
  6.5 GB peak while a job runs and ~2.2 GB resident afterwards. The first
  job downloads ~3 GB of model into the Hugging Face cache.
- **CUDA runtime:** the CUDA 12 runtime comes from the
  `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip wheels. That's why the
  worker needs the `LD_LIBRARY_PATH` line above.
- **Without a usable GPU:** set `WHISPER_DEVICE=cpu` and
  `WHISPER_COMPUTE_TYPE=int8` (slower, no other change).
- **Sequential decoding:** `WHISPER_BATCH_SIZE=0` and
  `WHISPER_BEAM_SIZE=5` restore M3's sequential decoding.

See
[`docs/decisions/0007-transcription-pipeline.md`](docs/decisions/0007-transcription-pipeline.md),
[`0014`](docs/decisions/0014-m9-performance.md) and
[`0016`](docs/decisions/0016-hardware-upgrade-rtx5070.md).

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
on the failed badge in Manage. torch comes from the PyTorch **cu128**
index (set at the top of `requirements.txt`), the first build with
Blackwell (RTX 50xx, sm_120) kernels. It dropped Pascal, so a GTX 10xx
no longer works. See
[`docs/decisions/0008-speaker-diarization.md`](docs/decisions/0008-speaker-diarization.md)
and [`0016`](docs/decisions/0016-hardware-upgrade-rtx5070.md).

**Chapter models (M5).** Chapters come from models trained locally in
the experiment [`experiments/m5-chapters/`](experiments/m5-chapters/README.md):
- a frozen multilingual sentence encoder feeding a BiLSTM (or the
  unsupervised TextTiling baseline) for boundaries;
- mT5-small for titles.

The weights are **not in git**. `experiments/m5-chapters/05_eval.ipynb`
exports a versioned bundle to `backend/models/chapters/<version>/`,
with `current` pointing at the one to serve.
- **Setup:** set `CHAPTERS_MODEL_DIR=models/chapters/current` in
  `backend/.env`.
- **Runtime:** the encoder, BiLSTM and mT5 run on `CHAPTERS_DEVICE`. The
  title LLM runs on the GPU like the other LLM tools (see "LLM on the
  GPU" below).
- **Without a bundle:** `chapters` jobs fail with a clear message, and
  the other tools are unaffected.

See [`docs/decisions/0009-chapter-generator-local-models.md`](docs/decisions/0009-chapter-generator-local-models.md).

**Show notes (M6).** Show notes (title suggestions, description, tags,
summary) come from the same local LLM that titles chapters: the GGUF
in the chapter bundle (Qwen3.5-9B Q4_K_M since the RTX 5070 pass), run
by llama.cpp fully on the GPU (`LLM_N_GPU_LAYERS=-1`).
- **Setup:** none beyond `CHAPTERS_MODEL_DIR`. The worker loads the
  GGUF once and shares it between `chapters` and `shownotes`.
- **Without a bundle with an LLM:** `shownotes` jobs fail with a clear
  message.
- **Editing:** the draft is edited and saved in Manage. Running show
  notes again replaces it, after a confirmation.

See [`docs/decisions/0011-show-notes-local-llm.md`](docs/decisions/0011-show-notes-local-llm.md).

**LLM on the GPU: llama-cpp-python with CUDA.** The three LLM tools
(chapter titles, show notes, relevance) share one GGUF.
- **Offload:** `LLM_N_GPU_LAYERS` (default `-1`) puts every layer on the
  GPU. On the RTX 5070 the 9B Q4_K_M uses ~5.5 GB and generates ~93 tok/s.
- **M9 behaviour:** `0` restores it: weights in RAM, prompt prefill on
  the GPU.
- **Release:** the worker releases the model after every LLM job, so
  Whisper and pyannote keep their room.
- **CPU build:** works too, just much slower.

To build the CUDA one without a system CUDA toolkit (RTX 50xx =
sm_120; use your GPU's compute capability):
```bash
B=~/.cache/yts-llama-cuda128 && mkdir -p $B/cuda $B/dl && cd $B
R=https://developer.download.nvidia.com/compute/cuda/redist
for p in cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz \
         cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz \
         cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz \
         libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz; do
  curl -fsSL -o dl/$(basename $p) $R/$p   # checksums: redistrib_12.8.1.json
  tar -xJf dl/$(basename $p) -C cuda --strip-components=1
done
ln -s lib cuda/lib64   # the tarballs use lib/, nvcc's link step looks in lib64/
PATH=$B/cuda/bin:$PATH CUDACXX=$B/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=on -DCUDAToolkit_ROOT=$B/cuda" \
  backend/.venv/bin/pip install --no-deps --force-reinstall --no-cache-dir llama-cpp-python==0.3.35
```
Notes on the build:
- Run the last command from the repo root. If `cmake` and `ninja` are
  missing, `pip install cmake ninja` into the venv first.
- At runtime the libraries come from the venv's NVIDIA wheels
  (`app/llm/cuda_libs.py`).

See [`docs/decisions/0014-m9-performance.md`](docs/decisions/0014-m9-performance.md)
and [`0016`](docs/decisions/0016-hardware-upgrade-rtx5070.md).

**Shorts (M8).** Shorts are cut from the relevance curve (run Relevance
first). The job picks up to five 20–60 s spans and frames each shot on
the detected face, or fits the whole frame over a blurred background
when there is none. It can burn in karaoke captions. Manage offers the
aspect ratio (9:16, 1:1, 4:5, 16:9) and a captions toggle before Run.
- **Setup:** download the YuNet face detector (~230 KB, MIT) once:
  ```bash
  mkdir -p backend/models/shorts
  curl -L -o backend/models/shorts/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```
  `SHORTS_FACE_MODEL_PATH` defaults to that path.
- **Runtime:** CPU only (OpenCV + ffmpeg/libx264), no GPU.
- **Without the model:** `shorts` jobs fail fast with a clear message.

See [`docs/decisions/0013-shorts-generator.md`](docs/decisions/0013-shorts-generator.md).

**Market pipeline (M10).** Daily public-stat snapshots of a curated list
of podcast channels, from the YouTube Data API v3. Backend and CLI only:
the Analytics page still shows dummy data.
- **Setup:** set `YOUTUBE_API_KEY` in `backend/.env` (a read-only key
  with "YouTube Data API v3" enabled in Google Cloud). Without it the
  market tasks log a skip; nothing else is affected.
- **Channels:** `python -m app.cli market add @handle` (or a channel URL
  or `UC…` ID) resolves the channel and backfills its whole catalogue
  through the uploads playlist. `market seed` loads
  `app/market/seed_channels.json`. Also `market list` and
  `market deactivate <id>` (stops pulling, keeps history).
- **Schedule:** the worker runs an RSS check every hour at :05 (new
  uploads get their first snapshot within the hour) and the daily pull at
  09:30 UTC: videos from the last 90 days every day, older ones weekly.
  `market pull` / `market rss` run them on demand.
- **Quota:** about 160 of the 10,000 daily units for ~25 channels. A soft
  cap (`MARKET_DAILY_UNIT_BUDGET`, default 5,000) and Google's own
  `quotaExceeded` both stop a run cleanly, keeping what it wrote.

See [`docs/decisions/0015-market-pipeline-v0.md`](docs/decisions/0015-market-pipeline-v0.md).

**Connected YouTube channel (M11).** One platform-level channel (yours),
connected with OAuth from the **Analytics** page. Every day the worker
pulls that channel's per-video engagement and audience-retention curves
from the YouTube Analytics API v2.

*One-time Google Cloud setup*, in the same project as `YOUTUBE_API_KEY`:
1. **APIs & Services → Library:** enable **YouTube Analytics API** and
   **YouTube Reporting API** (M12: real CTR/impressions; without it the
   reporting poll fails with HTTP 403 "has not been used in project … or it
   is disabled").
2. **Google Auth Platform:**
   - **Branding:** an app name and your email.
   - **Audience:** External. For now the app stays in **Testing**, with
     your Google account added under **Test users**. In Testing, refresh
     tokens expire after 7 days, so the card asks you to reconnect about
     weekly. Publishing (**Publish app** → *In production*, unverified)
     is deferred to the end of the project. It needs a public homepage
     URL and privacy-policy URL on an authorized domain set in
     **Branding** (see the 0017 amendment). After publishing, reconnect once.
   - **Data access:** add `…/auth/yt-analytics.readonly` and
     `…/auth/youtube.readonly`.
3. **Clients → Create client → Web application**, with the authorized
   redirect URI `http://localhost:4200/youtube/oauth/callback`.
4. **`backend/.env`:** set `GOOGLE_OAUTH_CLIENT_ID`,
   `GOOGLE_OAUTH_CLIENT_SECRET` and `OAUTH_TOKEN_ENCRYPTION_KEY`. Generate
   the key with:
   ```
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   If you lose the key, you have to reconnect.

*Connecting:* Analytics page → **Connect YouTube**. Approve both
permissions on Google's screen (it will show an "unverified app"
warning). You land back on the Analytics page with the channel shown.
The tokens are stored encrypted (Fernet) in `oauth_credentials` and are
never returned by the API.

*Schedule:* `analytics_daily_pull` runs at 10:15 UTC.
- It refreshes the video list, unlisted and private videos included.
- **Daily engagement** goes to `analytics_snapshots`, one row per video
  per *metric day*. Each run re-reads the last 7 days, because Analytics
  lags 48–72 h.
- **Retention curves** go to `analytics_retention_curve`, 100 points per
  video per pull: daily for videos from the last 90 days, weekly for
  older ones.
- `python -m app.cli analytics pull` runs it now; `analytics status`
  shows the connection and row counts.

*When access dies* (revoked, expired, password change):
- The connection goes to **needs reconsent**.
- The pull logs a WARNING and stops.
- The Analytics card shows **Reconnect** with Google's reason.

It is never silent. A network outage or a wrong client secret leaves the
status alone and logs an ERROR instead.

See [`docs/decisions/0017-oauth-analytics-v2.md`](docs/decisions/0017-oauth-analytics-v2.md).

**Real CTR and impressions (Reporting API, M12).** Thumbnail
impressions and CTR come only from the YouTube **Reporting API**, a
bulk-report system, not a live query. It uses the same connection and
needs no extra consent.

*Schedule:* `reporting_daily_poll` runs at 11:00 UTC.
- The first run creates a `channel_reach_basic_a1` job at Google, or
  adopts an existing one.
- From then on, YouTube posts **one CSV per day**, in Pacific time:
  - the first file arrives **within 48 h** of the job's creation;
  - 30 days of history follow within a few days.
- Each poll downloads the new files into `analytics_reach_snapshots`:
  one row per video per day, with CTR as a 0–1 fraction.
- When YouTube posts a corrected file for a day, it replaces that day's
  rows.
- Files stay downloadable for 60 days, so a worker that is down for
  weeks loses nothing.
- `python -m app.cli reporting poll` runs it now; `reporting status`
  shows the job, the report counts and the row counts.

The Analytics card shows a "Reach reports" line:
- "waiting for the first report"
- "up to <day>"
- failures and an expired job, in warning colour

See [`docs/decisions/0018-reporting-api-reach.md`](docs/decisions/0018-reporting-api-reach.md).

The API alone: `POST /auth/login` (form fields `username`/`password`)
for a token, `POST /uploads` (multipart, bearer token) to upload a
file, `GET /uploads` to list them.

The test media (`*.mp4` fixtures, the diarization healthcheck WAV) are
not in git. On a fresh clone, run `scripts/build_test_fixtures.sh` once.
It downloads LibriSpeech dev-clean and rebuilds them as described in the
`ATTRIBUTION.md` files.

To run the backend test suite, create the separate test database once
(`docker compose exec db psql -U youtube_suite -d youtube_suite -c
"CREATE DATABASE youtube_suite_test;"`), then `pytest` from `backend/`. The tests also need Redis (from
`docker compose up -d`); they use logical DB 1 (`TEST_REDIS_URL`).
The transcription tests need `ffmpeg` and run Whisper `base` on CPU
(no GPU needed; ~145 MB model download on the first run). The
diarization tests run pyannote on CPU and need `HF_TOKEN` in
`backend/.env`. With both models on CPU, the full backend suite takes a
few minutes. The market integration test (`-m youtube_api`) calls the
real Data API with `YOUTUBE_API_KEY` (a few units) and is skipped
without it. The OAuth/Analytics integration test is opt-in
(`pytest -m youtube_oauth`). It needs a channel connected from the app,
because it copies the dev DB's credential into the test DB, then forces
a real refresh and runs the pull twice. A channel with no views yet
correctly yields 0 analytics rows. The Reporting API integration test
(`pytest -m youtube_reporting -s`) has **two phases**, because of
YouTube's reporting lag:
- **Phase 1:** the first run creates or adopts the real reach job. It
  passes while no report exists yet and prints when to re-run.
- **Phase 2:** re-run it **≥ 48 h after the job was created**. It then
  checks the downloaded reports and rows, and prints the CSV header.

**GPU tests.** The fast suite always runs on CPU, whatever
`backend/.env` says. `pytest -m gpu` runs the opt-in GPU suite instead
(`tests/gpu/`). It loads the *production* configuration from `.env` plus
the defaults, and checks it on the real card:
- the CUDA stack;
- Whisper, pyannote, the LLM and chapters, each with a speed floor
  (catches a silent fallback to CPU or to system RAM) and a VRAM check;
- the worst-case set of resident models fitting in the 12 GB budget.

It needs the worker stopped, `HF_TOKEN` and a chapter bundle.

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
backend at `127.0.0.1:8000`, along with `/jobs` and `/clips` (see `frontend/proxy.conf.json`) — no CORS
setup needed. Run the frontend test suite with `ng test` (Vitest,
headless — no Chrome/browser install required).

## Third-party models

- Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  running OpenAI Whisper `large-v3-turbo` (MIT).
- Speaker diarization: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
  (MIT) with the `pyannote/speaker-diarization-community-1` pipeline,
  whose weights are licensed **CC-BY-4.0**.
- Chapters (M5), trained by us on
  [VidChapters-7M](https://antoyang.github.io/vidchapters.html) (code MIT;
  YouTube-derived data, rights with the original holders, used for
  training only, never redistributed) via the
  [Chapter-Llama](https://github.com/lucas-ventura/chapter-llama) JSON
  mirror (MIT), on top of `intfloat/multilingual-e5-small` (MIT) and
  `google/mt5-small` (Apache-2.0). Language ID in the experiment:
  fastText `lid.176` (CC BY-SA 3.0).
- Shorts face detection (M8): [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  `face_detection_yunet_2023mar.onnx` (MIT), run by OpenCV.
- Test face image: scikit-image's `astronaut` (NASA, public domain).
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
