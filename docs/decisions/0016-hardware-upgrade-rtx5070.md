# Decision: Hardware upgrade pass, GTX 1060 → RTX 5070 (before M11)

**Date:** 2026-11-07
**Status:** Implemented 2026-11-07 (see "Outcome" at the end). The owner's blind review (C3–C5) chose the 9B on all three, 2026-11-08. This is a pass outside the milestone plan, at the owner's request,
before M11. It follows the same planning → development → testing discipline.

## Context

- **Old machine:** GTX 1060 3 GB (sm_61), Ryzen 5 2600X 6C, 16 GB RAM.
  **New machine:** RTX 5070 12 GB (sm_120, Blackwell), Ryzen 7 7800X3D 8C/16T, 23 GB in WSL.
- About 30 decisions in 0007–0014 were sized to the old machine. The main ones:
  - Whisper turbo int8 at batch 4
  - torch `cu126`, which has no sm_120 kernels, so **the GPU does not work today**
  - pyannote batch 8
  - LLM weights in RAM (`n_gpu_layers=0`)
  - Qwen3.5-4B instead of the 9B the owner preferred in 0010's blind review
  - chapters inference on CPU
  - 6 LLM threads
- The M5 chapters bundle, the M9 episodes and the experiment data (the old `/mnt/d`) do not
  exist on this machine.

## Decided with the owner (2026-11-07)

- **Models:** upgrade the ones hardware held back, but change a default only after
  measuring quality against an anchor (the M9 setting re-measured on the new references).
- **Chapters:** retrain the **same M5 recipe** (e5-small + BiLSTM + mT5-small fallback) and
  run inference on the GPU. Bigger encoders go to brainstorm.
- **Episodes:** 5 from season 1 of *Sastre y Maldonado*.
- **References:**
  - 1x08 and the longest episode become the new references. M9's "upload 8" (a 64-min
    episode of unrecorded origin) is not recovered.
  - The ASR reference is large-v3 beam 5. Its bias towards large-v3 variants is noted
    explicitly; there is no hand-corrected set.
- **4B vs 9B:** the owner's blind review decides. CIDEr is informational only, as long as
  language accuracy and fallbacks don't regress. In 0010 the 9B failed the CIDEr-en gate
  (46.1 against 52.7); that gate is not applied here.
- **Tests:**
  - The fast CPU suite stays as it is, and is pinned explicitly to CPU so a GPU `.env`
    can't leak into it.
  - A new opt-in `-m gpu` suite loads the real production configuration.
- **GTX 1060 support is dropped.**

## Test data

- **Download:**
  - yt-dlp 2026.10.26 + yt-dlp-ejs 0.8.0, in its own venv (`~/.venvs/yt-dlp`), not in
    requirements. Node is the JS runtime.
  - Command: `yt-dlp --js-runtimes node:$(which node) -f "bv*[height<=720]+ba/b[height<=720]" -- <ids>`
  - Destination: `~/yts-data/podcasts/Sastre_y_Maldonado_T1_sample/`, WSL ext4. Local data,
    never committed.
- **Selection** from the "Temporada 1" playlist (`PLRIG0Ybczex0`, 28 videos), by duration
  percentile:

| Episode | ID | Duration | Video | Role |
|---|---|---|---|---|
| 1X09 | oSCkZ8yVz7M | 14:56 | h264 **480p** (the highest YouTube has) | shortest; low-resolution source for shorts |
| 1X08 | 8CPI8Ll9Hkg | 26:39 | av1 720p | reference; has an M9 row |
| 1X06 | znwuYC64UFI | 30:40 | av1 720p | p33 |
| 1X28 | VZ4NYtMNf9Q | 34:08 | h264 720p | p66 |
| 1X15 | L83BR6h-wdo | 43:17 | av1 720p | longest; reference |

All of them use Opus audio, about 2.5 h in total. All five have a row in
`experiments/m9-perf/season1_run.csv`.

- **M5 data:**
  - VidChapters' official splits, `train/val/test.json` from the authors' Drive folder,
    downloaded as `split_*.json`.
  - `docs/chapters.json` and `docs/asrs.json` from HF `lucas-ventura/chapter-llama`.
  - Destination: `~/yts-experiments/m5-chapters/raw/`.

## Plan (in order)

1. **CUDA stack**, with no default changes:
   - torch/torchaudio `2.8.0+cu128`: the same minor pyannote 4.0.7 was verified on. It adds
     sm_120 and stays on CUDA 12, like ctranslate2.
   - Re-pin the `nvidia-*` wheels to what torch declares.
   - Check ctranslate2 on sm_120: SASS/PTX, plus an fp16 transcription.
   - llama-cpp-python 0.3.35 rebuilt with the CUDA 12.8 toolkit and
     `CMAKE_CUDA_ARCHITECTURES=120`.
   - Full CPU regression.
2. **Baseline run A** with the M9 settings on the 5 episodes: transcribe + diarize first,
   the rest once the bundle exists. Compared episode by episode with M9.
3. **Chapters retrain**, same recipe on the GPU:
   - A run tag so the committed runs aren't overwritten.
   - Acceptance: BiLSTM F1@5s ±0.02 and still above TextTiling; mT5 CIDEr ±3.
   - Export bundle v2 with the 4B (M9-equivalent). Owner checkpoint **C2**.
4. **Whisper/pyannote:**
   - Matrix {turbo, large-v3} × {float16, int8_float16} × batch {4, 8, 16}, plus beam 5 for
     the finalists, measured against large-v3 references on 1x08/1X15.
   - Gates: WER ≤ anchor, drift p95 ≤ 0.5 s, fits in VRAM.
   - pyannote batch 32 vs 8.
5. **LLM:**
   - New setting `LLM_N_GPU_LAYERS`.
   - Candidates fully on the GPU: 4B Q4_K_M, 9B Q4_K_M/Q6_K, and 9B Q8_0 for quality only.
   - Titles: blind review **C3**.
   - Relevance: new version `relevance-p3`; Spearman stability plus a blind top-5 **C4**.
     The live flip in the dev DB is done by hand.
   - Show notes: same `n_ctx` and map-reduce; blind side-by-side **C5**.
6. **VRAM policy:**
   - Measure the real budget, with "Prefer No Sysmem Fallback" set while measuring.
   - Keep `gpu_lock`, `release_all()` and `empty_cache()`.
   - A pre-LLM release hook only if the measurements require it.
   - Chapters on cuda; shorts stay on CPU.
7. **`-m gpu` suite:**
   - stack check
   - Whisper, pyannote, LLM and chapters with the production configuration (VRAM and speed
     floors, which catch silent fallbacks)
   - worst-case co-residency
   - a real transcribe job
8. **Final run B:** compared with M9 and A; zero OOM retries; owner sample **C6**. Then
   regression (CPU + gpu + `ng test`) and docs.

Measurements: `docs/research/hw-upgrade-rtx5070.md`. Metrics: `experiments/hw-upgrade/`.

## Outcome (2026-11-07)

Measurements and tables are in `docs/research/hw-upgrade-rtx5070.md`.

### New defaults

| Setting | M9 (GTX 1060) | Now (RTX 5070) | Evidence |
|---|---|---|---|
| torch | 2.8.0+cu126 | **2.8.0+cu128** (cublas 12.8.4.1, cudnn 9.10.2.21) | sm_120 kernels; §1 |
| ctranslate2 | unpinned (4.8.2 verified) | **4.8.5 pinned** | float16/int8_float16/bf16 on sm_120; §1 |
| llama-cpp-python 0.3.35 | CUDA 12.6, arch 61 | **CUDA 12.8.1, arch 120** | §1, README |
| `WHISPER_MODEL` / `_COMPUTE_TYPE` / `_BATCH_SIZE` | large-v3-turbo / int8 / 4 | **large-v3 / int8_float16 / 16** | WER 17.1 → 10.8% vs the large-v3 reference, 48× real time; §4–5 |
| `DIARIZATION_EMBEDDING_BATCH_SIZE` | 8 | **32** | 22% faster, DER 0.0000 between the two; §5 |
| `CHAPTERS_DEVICE` | cpu | **cuda** | same boundaries as CPU; §8 |
| `LLM_N_GPU_LAYERS` (new) | 0 (hardcoded) | **-1** | §7 |
| `CHAPTERS_LLM_THREADS` | 6 | **8** | physical cores; irrelevant with full offload |
| Chapter bundle | `…20260928+qwen3.5-4b` (missing here) | **`bilstm-e5s-sentence-20261107+qwen3.5-9b`** | retrained, same recipe; §6 |
| Relevance version | `relevance-p2` (4B, CPU weights) | **`relevance-p3`** (9B, full offload), live | §9 |

### Decisions taken during implementation

- **LLM: Qwen3.5-9B Q4_K_M** (owner, 2026-11-07, mid-pass: "a slightly more powerful model
  should fit").
  - Fully offloaded it uses ~5.5 GB and generates ~93 tok/s.
  - **Why not Q6_K/Q8_0:** they don't fit next to the resident Whisper, pyannote and chapters
    without a release-before-load hook. That hook was not built.
  - The 4B bundle is kept for the blind review.
- **VRAM policy:** keep `gpu_lock`, `release_all()` after every LLM job and `empty_cache()`
  after diarization.
  - The worst case, with everything resident, peaks at **10,637 of 12,227 MiB** (desktop
    included). The test budget is 11,264 MiB.
  - No new release hook is needed.
  - ctranslate2 hands back its batch working memory (+6.5 GB peak, ~2.2 GB resident).
- **Tests:**
  - The fast suite pins every device and model setting to CPU through `CPU_PINS` /
    `FORCED_CPU_ENV` in `tests/conftest.py`, so it never follows a GPU `.env`.
  - The opt-in suite `pytest -m gpu` (`tests/gpu/`, 8 tests) undoes exactly those pins and
    checks the production configuration on the card. `addopts = -m "not gpu"` keeps it out of
    the default run.
- **M5 tooling:** a retrain no longer overwrites the committed runs.
  - `lib/runs.py`: `M5_RUN_TAG`, plus `M5_QA_UPLOAD_ID`, which replaces the hardcoded upload 8.
  - `tools/nb.py`: `M5_NB_DIR`.
  - `tools/run_nb.py`: runs a notebook with `experiments/m5-chapters` as the cwd.

### Deviations from the plan

- **The LLM half of run A was skipped.** It would have produced a misleading `relevance-p3`
  version with CPU weights.
- **Notebook 06 (title-LLM metrics) was not re-run on the 5070.** The v2 bundles carry the M5
  val rows for the same GGUFs, labelled as such. The owner's choice and blind review replace
  it.
- **mT5 val_en CIDEr improved by 8.6** (outside the ±3 acceptance band, but upwards). Read as
  run-to-run variance.

### Pending

- ~~**The owner's blind review**~~ **Done 2026-11-08:** the owner chose the **9B** for all of
  C3 (titles), C4 (relevance top-5) and C5 (show notes)
  (`experiments/hw-upgrade/blind/blind_sheet.md`). The 9B is already `current` and
  `relevance-p3` is already live, so nothing switches. The 4B bundle is no longer needed
  for the review.
- **`backend/.env`** still holds the M9 values. The agent can't edit it (permission rule);
  the owner updates it (lines in the handoff).
