# Decision: Delivery Package for the Examiners

**Date:** 2026-11-22
**Status:** In progress.

## Context

- The project is finished (all milestones M0–M16 are done). The university's activity asks for a
  PDF with the "Materials and Methods" and "Results" sections, plus a justification of the
  implementation: a notebook or the source code. The format is agreed with the tutor.
- **The tutor asked for a delivery that the examiners can grade easily.** They must be able to see:
  - every process of the project, especially the data-science ones
  - the notebooks and the code
  - the application running
- **The application cannot be run by an examiner as it is.** It needs:
  - an NVIDIA GPU (developed on an RTX 5070) and WSL
  - about 13 GB of models, which are not in the repository
  - a Hugging Face token for pyannote
  - an OAuth app in Testing mode, which only the owner's account can use
- **Only two components have notebooks:** M5 (chapters, `experiments/m5-chapters/`) and M16 (the
  in-house LLM, `experiments/m16-own-llm/`). The rest of the data-science work lives in code,
  scripts, the dev database and decision records.

## Decisions (with the owner, 2026-11-22)

1. **Format: option "B + C".**
   - **B:** an examiner guide (`DELIVERY.md`, linked from the README). It has one row per
     process: report section → notebook → code → decision record → result.
   - **C:** new notebooks for the processes that had none.
   - Rejected:
     - a static site (Jupyter Book or MkDocs), judged not relevant
     - a hosted live demo, because of the cost, the risk and data exposure
2. **Everything goes in the public GitHub repository except the video.** The owner uploads the
   repository documents and the video (about 2 minutes, focused on the data-science processes) to
   the virtual classroom. The video is not published.
3. **No database data in the repository.** The notebooks run against the dev database and keep
   their outputs. **The outputs show aggregates only:**
   - metrics, summary tables and charts
   - no transcript text, no episode titles and no raw rows
   - episodes appear as numeric `upload_id`s
4. **The notebooks re-evaluate and do not retrain.** They recompute every reported number from the
   stored data and the saved models, calling the same backend code the app uses. Training is shown
   through the stored training logs (`metrics.json`, `model_versions`).
   - If a recomputed number differs from the report, the notebook says so; the number is not
     adjusted.
   - The rule of 2026-11-15 still applies: no invented results.
5. **Language: English**, like the report, the code and the existing notebooks.
6. **Docker Compose for the examiner, in two modes** (`docker/`).
   - **CPU mode** (`docker/compose.yml`) runs on any machine. It uses Whisper `small` and the
     project's trained chapter, title and relevance models. Show notes, and titles when chosen, use
     the in-house LLM.
   - **GPU mode** (`docker/compose.yml` + `docker/compose.gpu.yml`) is the deployed configuration of
     decision 0016, on an NVIDIA GPU through the NVIDIA Container Toolkit.
   - Implementation:
     - One backend image (`docker/backend.Dockerfile`, `VARIANT=cpu|gpu`). `llama-cpp-python` is
       compiled in a build stage: with CUDA for sm_75–sm_120 in GPU mode, portable AVX2 in CPU mode.
     - The frontend is built with nginx (`docker/frontend.Dockerfile`, `docker/nginx.conf`), which
       proxies the API paths of `frontend/proxy.conf.json`.
     - `fetch` is a one-off service that downloads the models into `docker/models`.
     - `init` runs on every `up`: migrations, the operator account, registration and live switch
       of the relevance BiLSTM, and the default LLM.
     - The project name is `yts-demo`, and Postgres and Redis get no host ports, so the stack never
       collides with the dev compose at the repository root, which is unchanged.
   - Plain Compose files rather than Compose *profiles*: the two modes differ in build arguments,
     environment and device reservations, which an override file expresses directly.
7. **Model distribution: a GitHub Release (`v1.0`) plus a download script**
   (`docker/fetch_models.py`).
   - **Release assets:** only the project's own trained weights, plus the public files their bundles
     need. A `SHA256SUMS` file is checked by the script.
     - `yts-chapters.tar` (1.7 GB): boundary BiLSTM, fine-tuned mT5-small titles, the e5 encoder
       copy, fastText LID, and the Qwen bundle's manifest
     - `yts-relevance.tar` (0.5 GB): the live BiLSTM `bilstm-e5s-20261114-i2` and the frozen split
     - `yts-own-llm.tar` (0.2 GB): `yts-own-110m-dpo`
     - `yts-extras.tar` (4 MB): the YuNet and PP-OCR detectors, the diarization healthcheck clip,
       and a demo clip (LibriSpeech, CC BY 4.0)
   - **Downloaded from their sources:** Whisper, pyannote (gated: the examiner's own Hugging Face
     token), xlm-emo-t, and, in GPU mode only, the Qwen3.5-9B GGUF, which is not the project's.
     Its default URL is `unsloth/Qwen3.5-9B-GGUF` on Hugging Face, the source recorded in
     `experiments/m5-chapters/README.md`. On 2026-11-22 its size and SHA-256 were checked to be
     identical to the deployed file.

## Delivered

- `DELIVERY.md`: the examiner guide. The README's outdated status section now points to it.
- `notebooks/01–08` plus the shared setup `notebooks/yts_nb.py`. Their outputs were scanned for
  private data (paths, episode titles, e-mails, long transcript text): none found.
- `docker/` and `.dockerignore`.

## Findings from the recomputation

Almost every reported number was reproduced exactly. The exceptions are stated in the notebooks and
in `DELIVERY.md`, and no number was adjusted:

- **M9 wall time:** the CSV gives 5.06 h, i.e. 0.33 × real time, against the 4.6 h and 0.30 ×
  recorded in 0014 and the report.
- **Shorts replay:** 21 of 40 student clips are near a teacher clip, against 22 recorded.
- **Relevance scoring time:** 0.25 s here, against 1.0 s recorded; the scores are identical.
- **Diarization:** the report's multi-speaker share was measured on the shorter segments of that
  time, so the notebook uses a word-level measure.

## Test of the examiner stack (2026-11-22, CPU mode)

The test ran on the development machine. That machine's Docker has no NVIDIA Container Toolkit, so
GPU mode was built but not run.

- **Setup:** `build`, `fetch` and `up` ran from scratch, with the release assets served locally.
  `init` then applied the migrations, created the account, registered `bilstm-e5s-20261114-i2` and
  made it live.
- **Smoke test** through nginx with the demo clip. With the default model setting, plus *Our model*
  for show notes, every tool succeeded:
  - transcription in 18 s
  - chapters: 3, with mT5 titles
  - relevance: 42 windows scored by the BiLSTM
  - show notes from `yts-own-110m-dpo`
  - shorts: 3 clips of 23–34 s
  - hook scores
  - The client-side routes (`/settings`, `/manage/1`) serve the app shell.
- **Two tools need input from the examiner:**
  - diarization needs `HF_TOKEN`; without it, the job fails cleanly
  - CTR needs candidates added in its panel
- **Fixes made during the test:**
  - The mT5 bundle's manifest now declares its LID file. Our own LLM, show notes and relevance
    need it.
  - The default LLM stays `qwen` in CPU mode, so that chapters fall back to mT5 titles and
    relevance uses the live BiLSTM.
  - The demo clip is encoded at 25 fps. At 1 fps the shorts renderer's duration check rejects the
    clips, which is not a real-episode case.
