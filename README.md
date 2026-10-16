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
set `WHISPER_DEVICE=cpu` (slower, no other change). Since M9 it decodes
4 VAD chunks at a time, greedily (`WHISPER_BATCH_SIZE=4`,
`WHISPER_BEAM_SIZE=1`), about 2× faster; `WHISPER_BATCH_SIZE=0` and
`WHISPER_BEAM_SIZE=5` restore M3's sequential decoding. See
[`docs/decisions/0007-transcription-pipeline.md`](docs/decisions/0007-transcription-pipeline.md)
and [`0014`](docs/decisions/0014-m9-performance.md).

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
- **Runtime:** inference runs on CPU and does not touch the GPU.
- **Without a bundle:** `chapters` jobs fail with a clear message, and
  the other tools are unaffected.

See [`docs/decisions/0009-chapter-generator-local-models.md`](docs/decisions/0009-chapter-generator-local-models.md).

**Show notes (M6).** Show notes (title suggestions, description, tags,
summary) come from the same local LLM that titles chapters: the
Qwen3.5-4B GGUF in the chapter bundle, run by llama.cpp (weights in RAM;
with the CUDA build below, the prompt prefill runs on the GPU).
- **Setup:** none beyond `CHAPTERS_MODEL_DIR`. The worker loads the
  GGUF once and shares it between `chapters` and `shownotes`.
- **Without a bundle with an LLM:** `shownotes` jobs fail with a clear
  message.
- **Editing:** the draft is edited and saved in Manage. Running show
  notes again replaces it, after a confirmation.

See [`docs/decisions/0011-show-notes-local-llm.md`](docs/decisions/0011-show-notes-local-llm.md).

**LLM speed (M9): llama-cpp-python with CUDA.** The LLM tools spend
most of their time reading the prompt. Built with CUDA, llama.cpp runs
that on the GPU (~13× faster here) while the weights stay in RAM. This
cuts show notes from ~16 to ~6 min and relevance from ~17 to ~9 min on
a 64-min episode. The CPU build works too, just slower. To build the
CUDA one without a system CUDA toolkit (GTX 1060 = sm_61):
```bash
B=~/.cache/yts-llama-cuda && mkdir -p $B/cuda && cd $B
R=https://developer.download.nvidia.com/compute/cuda/redist
for p in cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz \
         cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz \
         cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.6.77-archive.tar.xz \
         libcublas/linux-x86_64/libcublas-linux-x86_64-12.6.4.1-archive.tar.xz; do
  curl -sSL $R/$p | tar -xJ -C cuda --strip-components=1
done
PATH=$B/cuda/bin:$PATH CUDACXX=$B/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_NATIVE=on -DCUDAToolkit_ROOT=$B/cuda" \
  backend/.venv/bin/pip install --no-deps --force-reinstall --no-cache-dir llama-cpp-python==0.3.35
```
(Run the last command from the repo root; `pip install cmake ninja` into
the venv first if they are missing.) At runtime the libraries come from
the venv's NVIDIA wheels, with no `LD_LIBRARY_PATH` needed. See
[`docs/decisions/0014-m9-performance.md`](docs/decisions/0014-m9-performance.md).

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
