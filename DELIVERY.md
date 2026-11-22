# YouTube Suite: Guide for the Examiners

This page is the entry point for grading the project. It shows where each process of the project
can be seen:

- the **report** (Materials and Methods, Results);
- the **notebooks** that recompute its numbers;
- the **code** that implements it;
- the **decision record** that explains why it was built that way.

The last section explains how to run the application.

| Deliverable | Where |
|---|---|
| Report (PDF) | [`mem/main.pdf`](mem/main.pdf) (LaTeX sources in [`mem/`](mem/)) |
| Source code, notebooks, decisions | this repository, tag `v1.0` |
| Trained models for the Docker stack | GitHub Release [`v1.0`](https://github.com/PabloJimenez98/new_youtube_suite/releases/tag/v1.0) |
| Video of the application (about 2 min) | uploaded separately to the virtual classroom |

## Reading order (about 15 minutes)

1. The report's chapter *Materials and Methods* (one section per component), then *Results*.
2. The **process map** below: one row per process, with its notebook and code.
3. One notebook end to end, for example
   [`notebooks/03_relevance_bilstm.ipynb`](notebooks/03_relevance_bilstm.ipynb). It covers the
   project's own trained model: its data split, training log, test evaluation recomputed from the
   saved weights, offline gate, blind A/B and promotion.
4. Optionally, run the application with Docker (last section), or watch the video.

## Process map

The project turns a podcast episode into transcripts, chapters, show notes, short clips and
scores. It also feeds market data back into its models. Every row is one process.

| # | Process | Report | Notebooks | Code | Decisions | Headline result |
|---|---|---|---|---|---|---|
| 1 | **Transcription and diarization** (Whisper, pyannote) | §3 *Transcription*, *Speaker Diarization*; §4 *Speech Processing* | [`notebooks/01_speech`](notebooks/01_speech.ipynb) | `backend/app/transcription/`, `backend/app/diarization/` | 0007, 0008, 0014, 0016 | `large-v3` int8_float16: 10.8 % WER vs. reference (old default 17.1 %); diarization batch 8 → 32 with DER 0 |
| 2 | **Chapter boundaries** (BiLSTM over e5 sentence embeddings vs. TextTiling) and **titles** (fine-tuned mT5, local LLM) | §3 *Chapter Generation*; §4 *Chapter Generation* | [`experiments/m5-chapters/01–06`](experiments/m5-chapters/) | `backend/app/chapters/`, `experiments/m5-chapters/lib/` | 0009, 0010 | English test F1@5s 17.89 vs. 8.54 for the baseline; Spanish 6.58 vs. 4.32 |
| 3 | **Show notes** (map-reduce, grammar-constrained JSON, validation rules) | §3 *Show-Notes Generation*; §4 *Show Notes* | [`notebooks/05_show_notes`](notebooks/05_show_notes.ipynb) | `backend/app/shownotes/` | 0011 | 63/63 drafts pass every rule; 21 % of title suggestions repeat a chapter title (known weakness) |
| 4 | **Relevance v0**: heuristic + zero-shot LLM teacher | §3 *Relevance Model*; §4 *Bootstrap version* | [`notebooks/02_relevance_teacher`](notebooks/02_relevance_teacher.ipynb) | `backend/app/relevance/` (`heuristics.py`, `llm_scoring.py`, `pipeline.py`) | 0012, 0014 | The two signals are uncorrelated (r = −0.009 over 19,348 windows) |
| 5 | **Relevance v1**: BiLSTM student distilled from the teacher | §3 *Relevance Model*, *Model Versioning*; §4 *Trained version* | [`notebooks/03_relevance_bilstm`](notebooks/03_relevance_bilstm.ipynb) | `backend/app/relevance/` (`bilstm.py`, `training.py`, `evaluation.py`, `retrain.py`) | 0024 | Test Spearman 0.652 vs. 0.545 for the heuristic: gate passed; blind A/B 3–2; live |
| 6 | **Shorts**: peak selection, face-tracking crop, captions | §3 *Shorts Generator*; §4 *Shorts* | [`notebooks/04_shorts`](notebooks/04_shorts.ipynb) | `backend/app/shorts/` | 0013, 0024 §8 | Replay on the test episodes: 5 clips each, 20–60 s; about half match the teacher's moments |
| 7 | **Market analysis**: YouTube Data/Analytics/Reporting APIs → age-adjusted residuals, clustering, drivers, keyword lift | §3 *Market-Analysis Pipeline*; §4 *Market Data and Analyses* | [`notebooks/06_market_analysis`](notebooks/06_market_analysis.ipynb) | `backend/app/market/`, `youtube_oauth/`, `analytics/`, `reporting/`, `insights/` | 0015, 0017, 0018, 0022, 0023 | 21 channels, 47,277 videos; k = 2 (silhouette 0.21); video length ρ = 0.30 |
| 8 | **CTR predictor and hook scorer** (transparent heuristics v0) | §3 *CTR Predictor and Hook-Strength Scorer*; §4 same | [`notebooks/07_ctr_hook`](notebooks/07_ctr_hook.ipynb) | `backend/app/ctr/`, `backend/app/hook/` | 0019, 0020 | Scores reproducible from stored features; framing and direct address carry the hook ranking |
| 9 | **In-house LLM** trained from scratch: tokenizer, ablations, 5B-token pretraining, SFT, DPO, GGUF export | §3 *In-House Language Model*; §4 same | [`experiments/m16-own-llm/00–08`](experiments/m16-own-llm/) (09–11: stage B, **not executed**) | `experiments/m16-own-llm/llm/`, `backend/app/llm/` | 0025 | 100.7M parameters, 0.963 bpb; titles gate passed, show notes (86 % valid) and relevance failed |
| 10 | **End-to-end performance** (old GPU vs. RTX 5070) | §4 *End-to-End Performance* | [`notebooks/08_end_to_end_performance`](notebooks/08_end_to_end_performance.ipynb) | `scripts/run_pipeline.py` | 0014, 0016 | All six tools at 0.12 × real time on the RTX 5070 |

Cross-cutting parts:

- **Model versioning and the live switch:** `backend/app/model_registry.py`, `python -m app.cli models …`
  (decision 0024).
- **User interface:** Angular, `frontend/src/app/`. The *How it works* page explains each model
  with measured charts (decision 0021).
- **Tests:**
  - `backend/tests/` (pytest; `-m gpu` for the GPU suite)
  - `frontend/src/**/*.spec.ts` (Vitest)
  - `experiments/*/tests/`
  - At the end of M16 the counts were 770 backend, 343 frontend and 48 experiment tests, all
    passing.
- **Why each choice was made:** [`docs/decisions/`](docs/decisions/), one dated record per
  architectural decision. The specification is
  [`docs/project-description.md`](docs/project-description.md); the plan is
  [`docs/development-plan.md`](docs/development-plan.md).

## About the notebooks

There are two kinds.

- **Experiment notebooks** (`experiments/m5-chapters/`, `experiments/m16-own-llm/`) are where the
  chapter models and the in-house LLM were developed: data preparation, baselines, training,
  evaluation and export. They keep the outputs of their original runs.
- **Delivery notebooks** (`notebooks/01–08`) cover the processes that were developed directly in
  the application (decision 0026).
  - They call the application's own code (`backend/app/…`) on the stored data and the saved
    models. They **recompute** each reported number instead of copying it.
  - They **do not retrain** anything. Training is shown through the training logs stored with each
    model.
  - A line such as `recomputed 0.6522 matches the report (0.6520)` compares a recomputed value with
    the report. Where a value differs, the notebook says so and explains why; values are never
    adjusted.
  - They show **aggregates only**: no transcript text, no episode titles, and episodes appear as
    numeric ids. The database of the operator's episodes is not distributed.
  - They are therefore meant to be **read with their saved outputs**. Re-running them needs the dev
    database and the local benchmark files, which are not distributed. The shared setup is in
    [`notebooks/yts_nb.py`](notebooks/yts_nb.py).

**Differences found while recomputing.** Almost every number was reproduced exactly. The few
differences are stated in the notebooks:

- **M9 season run, wall time (notebook 08).** The run's CSV gives 5.06 h for 15.1 h of audio
  (0.33 × real time), while the report and decision 0014 give 4.6 h (0.30 ×). The CSV value is the
  reproducible one. The 60 per-job values of the report's table are all reproduced.
- **Shorts replay (notebook 04).** 21 of 40 student clips are near a teacher clip, against the 22
  recorded. The recorded matching rule is not known.
- **Relevance scoring speed (notebook 03).** 0.25 s here, against the 1.0 s that the report gives
  for the same episode. The model and the scores are identical; the measurement conditions differ.
- **Multi-speaker segments (notebook 01).** The report's 7.1 % was measured on the shorter Whisper
  segments of that time. The notebook uses a measure that stays comparable as segments change.

## Run the application

The application runs as a Docker Compose stack: Postgres, Redis, the API, the job worker and the
web app. It starts empty, because no data of the operator's channel is distributed. A demo clip is
provided: `demo_conversation.mp4`, a 5-minute conversation between two LibriSpeech speakers
(CC BY 4.0).

| Mode | Needs | Models |
|---|---|---|
| **CPU** (any machine) | Docker, about 15 GB of disk (the backend image is 7.5 GB) | Whisper `small`; the project's trained chapter, title and relevance models; the in-house LLM for show notes |
| **GPU** (the deployed configuration) | NVIDIA GPU with ~12 GB VRAM and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), about 45 GB of disk | As deployed: Whisper `large-v3`, pyannote on the GPU, Qwen3.5-9B for titles, show notes and the relevance teacher |

```bash
git clone https://github.com/PabloJimenez98/new_youtube_suite.git && cd new_youtube_suite
cp docker/.env.example docker/.env      # set HF_TOKEN for diarization (see the file)

# CPU mode
docker compose -f docker/compose.yml build          # first build: 15–30 min
docker compose -f docker/compose.yml run --rm fetch # downloads ~2.5 GB of models
docker compose -f docker/compose.yml up -d

# GPU mode: the same three commands with both files (fetch also downloads the 5.7 GB Qwen GGUF)
docker compose -f docker/compose.yml -f docker/compose.gpu.yml build
docker compose -f docker/compose.yml -f docker/compose.gpu.yml run --rm fetch
docker compose -f docker/compose.yml -f docker/compose.gpu.yml up -d
```

Open http://localhost:4200 and log in with `examiner@example.com` / `examiner` (set in
`docker/.env`). Then:

1. **Library → Upload:** choose `docker/models/demo/demo_conversation.mp4` (or any episode).
2. **Manage → the episode:** run the tools in this order; each panel explains its result.
   - **Transcription** comes first. With `HF_TOKEN` set, **diarization** adds speaker labels.
     Without it, the diarization job reports that the model is unavailable and every other tool
     still works.
   - **Chapters** (trained boundary BiLSTM) and **Relevance** (trained BiLSTM) come next, with the
     default model setting.
   - **Show notes:** in CPU mode, pick *Our model* in the tool bar before running it. The default,
     Qwen, is not installed in CPU mode.
   - **Shorts**, then **Hook** (it scores the shorts).
   - **CTR** scores the title and thumbnail candidates that you add in its panel.
3. **How it works:** explains every model, with measured charts.
4. **Settings:** the default language model. In CPU mode, Qwen is not installed:
   - chapter titles come from the fine-tuned mT5;
   - relevance uses the trained BiLSTM, which needs no LLM;
   - *Our model* (the in-house LLM) can be chosen per run.

On the test machine in CPU mode, the whole demo clip went through every tool in about a minute:
3 chapters, 42 scored windows, a show-notes draft and 3 shorts. The demo clip is read speech, not a
podcast, so the content of the results is not meaningful. A real episode of 15–60 minutes shows
the tools as designed.

Stop the stack with `docker compose -f docker/compose.yml down`. Add `-v` to also delete the
database and the uploads. The models in `docker/models` are written by the container as root.

**Not available to examiners.** The market-analysis pages need a YouTube Data API key
(`YOUTUBE_API_KEY`) and a collection run:

```bash
docker compose -f docker/compose.yml exec api python -m app.cli market seed
docker compose -f docker/compose.yml exec api python -m app.cli market pull
docker compose -f docker/compose.yml exec api python -m app.cli insights refresh
```

The OAuth flows (Analytics and Reporting APIs) use a Google app in *Testing* mode that only the
owner's account can authorise. Their results are shown in notebook 06 and in the video.

**Development setup.** The stack used during development (WSL, a local venv, `scripts/dev.sh`)
and the test commands are described in [`README.md`](README.md).

## Data, licences and third-party models

- **Not distributed:**
  - the operator's episodes, transcripts and database
  - YouTube data collected through the APIs
  - the VidChapters-7M subset used for training (rights with the original holders)
- **Distributed in the v1.0 Release:** the project's own trained weights, plus the public files
  their bundles need:
  - the chapter boundary BiLSTM and the fine-tuned mT5-small titles
  - the relevance BiLSTM
  - `yts-own-110m-dpo`
  - inside the bundles, copies of the frozen `multilingual-e5-small` encoder (MIT) and fastText
    `lid.176` (CC BY-SA 3.0)
  - the YuNet (MIT) and PP-OCR (Apache-2.0) detectors
  - the LibriSpeech clips (CC BY 4.0)
- **Third-party models** are downloaded from their sources: Whisper, pyannote community-1,
  xlm-emo-t, spaCy and Qwen. See the README's *Third-party models* for their licences.
