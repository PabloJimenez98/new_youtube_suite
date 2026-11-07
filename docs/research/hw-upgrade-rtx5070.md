# Hardware upgrade: RTX 5070 measurements

The plan is in `docs/decisions/0016-hardware-upgrade-rtx5070.md`. This note holds the
measurements.

- **Machine:**
  - RTX 5070 12 GB (12,227 MiB), compute capability 12.0, Windows driver 617.14
  - Ryzen 7 7800X3D 8C/16T
  - 23 GB RAM in WSL2 (Ubuntu 24.04)
- **Previous machine (M9):** GTX 1060 3 GB, Ryzen 5 2600X, 16 GB.

## 1. CUDA stack (2026-11-07)

- **Before:**
  - `torch 2.8.0+cu126` reports the arch list `sm_50…sm_90`, with no sm_120.
  - Any torch kernel on the 5070 fails.
  - The pip freeze is saved locally at `~/yts-experiments/hw-upgrade/pip-freeze-cu126.txt`
    for rollback.
- **After:** `torch 2.8.0+cu128` / `torchaudio 2.8.0+cu128`.
  - torch declares `nvidia-cublas-cu12==12.8.4.1` (was 12.6.4.1) and
    `nvidia-cudnn-cu12==9.10.2.21` (unchanged). Both pins in `requirements.txt` follow it.
  - Arch list: `sm_70 … sm_90, sm_100, sm_120`.
  - An fp16 matmul and a conv1d on cuda work.
- **ctranslate2 4.8.5** (it was unpinned and arrived through faster-whisper; now pinned).
  - `get_supported_compute_types("cuda")` = bfloat16, float16, float32, int8,
    int8_bfloat16, int8_float16, int8_float32. On the 1060 it was only int8, int8_float32
    and float32.
  - Fixture clip (18.3 s) with turbo, beam 1, on cuda: float16, int8_float16 and int8 all
    give **WER 0.070**. The first run took 9.3 s (load and warm-up); the next ones 1.1 s and
    0.4 s.
  - The main risk in the plan (CT2 without sm_120 kernels) did not happen. No source build is
    needed.
- **pyannote 4.0.7 on cuda:** the startup healthcheck (A/B/A clip) passes in 18.9 s,
  including the model load.
- **llama-cpp-python 0.3.35, CUDA build:**
  - NVIDIA CUDA **12.8.1** redist tarballs, each with its SHA-256 checked against
    `redistrib_12.8.1.json`: nvcc 12.8.93, cudart 12.8.90, cccl 12.8.90, cublas 12.8.4.1.
  - `-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=on`.
  - **Gotcha:** the tarballs extract their libraries into `cuda/lib`, but nvcc's link step
    looks in `cuda/lib64`. CMake's compiler check then fails with
    `cannot find -lcudadevrt`. Fix: `ln -s lib cuda/lib64`.
  - At runtime `libllama.so` needs `libcudart.so.12`, which `app/llm/cuda_libs.preload()`
    loads from the venv's wheel, as before.

## 2. Test data

- **Episodes:** see 0016 "Test data".
  - 1X09 exists on YouTube only up to 480p; the others are AV1 or H.264 at 720p, with Opus
    audio.
  - yt-dlp needs a JS runtime and the EJS solver (`yt-dlp[default]` + `--js-runtimes node`).
    Without them some formats are missing and the download quietly drops to a lower
    resolution.
- **M5 raw data:**
  - `asrs.json` 22,410,063,039 B and `chapters.json` 2,081,069,414 B from HF, in about 6 min.
  - VidChapters' official `train/val/test.json` from the authors' Drive folder
    (12,010,140 / 123,000 / 123,000 B), via `gdown`.
- **Notebook 01 reproduces exactly** (`runs/2026-11-07-rtx5070-data`). Candidates (285,498),
  selection (11,587) and every per-split count and median are byte-identical to
  `runs/2026-09-27-data`.

## 3. Run A: M9 settings on the new machine (transcribe + diarize)

Settings: turbo int8, batch 4, beam 1, pyannote batch 8, `--parallel 2`, like M9. Job seconds
include GPU-lock waits in both runs. CSV: `experiments/hw-upgrade/runA_m9_settings.csv`,
compared with `experiments/hw-upgrade/compare_runs.py`.

| Episode | Transcribe M9 → A | Diarize M9 → A |
|---|---|---|
| 1X09 (14.9 min) | 67.3 → 19.8 s (3.4×) | 140.1 → 24.7 s (5.7×) |
| 1X08 (26.7 min) | 92.6 → 47.6 s (1.9×) | 88.8 → 27.4 s (3.2×) |
| 1X06 (30.7 min) | 86.5 → 38.7 s (2.2×) | 176.6 → 45.9 s (3.8×) |
| 1X28 (34.1 min) | 258.9 → 41.6 s (6.2×) | 296.9 → 71.6 s (4.1×) |
| 1X15 (43.3 min) | 205.7 → 52.0 s (4.0×) | 146.1 → 56.6 s (2.6×) |

The LLM half of run A (chapters/relevance/shownotes/shorts on M9 settings) was **not run**.
It would have created a misleading `relevance-p3` version with the weights in RAM, and run B
vs M9 already answers the question. This is a deviation from the plan.

## 4. Whisper variants (2026-11-07)

- **Excerpts:** four 5-min excerpts, 20.0 min in total, 3,659 reference words: 1x08 at
  5:20, 13:20 and 21:20, plus 1X15 at 19:10.
- **Reference:** `large-v3`, beam 5, sequential, float16, on the GPU (33–52 s per excerpt).
  Stored in `~/yts-experiments/hw-upgrade/asr/` (local, not in git).
- **Bias caveat:** the reference is large-v3, which favours large-v3 variants. There is no
  hand-corrected set (owner decision).
- **Commands:** `asr_variant.py` (the compute type is now a parameter, `model:type`) and
  `asr_compare.py`, which now also reports drift p95 over aligned words.
- All runs are beam 1, VAD on, `language=es`.

| Variant | WER | Drift p95 | Seconds (20 min of audio) |
|---|---|---|---|
| **anchor:** turbo int8 b4 (M9 default) | 17.08% | 0.39 s | 22 |
| turbo int8_float16 b4 / b8 / b16 | 17.08 / 17.11 / 17.00% | 0.39 s | 22 / 19 / 15 |
| turbo float16 b4 / b8 / b16 | 19.62 / 19.62 / 19.51% | 0.37 s | 20 / 19 / 16 |
| large-v3 float16 b4 / b8 / b16 | 14.21 / 14.18 / 14.16% | 0.30–0.31 s | 53 / 43 / 34 |
| **large-v3 int8_float16 b4 / b8 / b16** | 11.15 / 11.15 / **10.82%** | 0.33–0.34 s | 62 / 48 / **36** |

- **Anchor:** it lands at 17.1% on these episodes, where M9's set gave 13.9%. Different
  episodes and a different reference, so absolute numbers aren't comparable across the two
  notes; the ranking within this table is what counts.
- **Batch size:** it barely changes WER. It changes speed (b16 is the fastest everywhere).
  On the 1060 the batch was capped at 4 by VRAM.
- **large-v3 vs turbo:** large-v3 is clearly better.
  - int8_float16 b16 beats the anchor by **6.3 WER points**.
  - It beats even large-v3 float16, which is closer to the reference. So the gain is not
    only the reference's bias.
  - It runs at about 33× real time.
- **Gates** (WER ≤ anchor, drift p95 ≤ 0.5 s): every large-v3 variant passes. **Choice:
  large-v3 int8_float16, batch 16**, pending the VRAM check on a full episode (§5).

## 5. Full-episode VRAM and speed (1X15, 43.3 min)

- **Setup:** one config per process, VRAM sampled with `nvidia-smi` every 0.2 s, deltas over
  the process start.
- **Caveat:** measured while the M5 BiLSTM trained in the background. Its usage held at
  about 3 GB and the GPU sat at about 17%, so the deltas hold, but the times are slightly
  pessimistic.

| Config | Time | Real time | Weights | Peak |
|---|---|---|---|---|
| turbo int8 b4 (M9) | 47.8 s | 54× | +1,149 MiB | +1,776 MiB |
| **large-v3 int8_float16 b16** | **54.1 s** | **48×** | +1,985 MiB | **+6,451 MiB** |
| large-v3 int8_float16 b8 | 77.6 s | 33× | +1,985 MiB | +4,239 MiB |
| large-v3 float16 b16 | 50.6 s | 51× | +3,777 MiB | +8,141 MiB |

- **ctranslate2 hands its working memory back.** After a full transcription with the model
  still loaded, it uses +2,191 MiB (it was +1,985 MiB right after loading). So the +6.5 GB
  peak exists only while a transcription runs, and it's serialized by `gpu_lock`.
- **pyannote community-1:** `embedding_batch_size` 8 took 70.7 s with a peak of +980 MiB;
  **32 took 55.0 s** with a peak of +2,585 MiB.
  - Both found 5 speakers, with **DER 0.0000** between the two outputs.
  - **Choice: 32** (pyannote's default; the 1060 needed 8). `empty_cache()` hands the peak
    back.
- **Idle card:** about 980 MiB used by the Windows desktop.

## 6. M5 chapters retrain: same recipe, on the GPU (2026-11-07)

- **How it ran:** `M5_RUN_TAG=2026-11-07-rtx5070` and `M5_QA_UPLOAD_ID=5` (1x08 from run A).
  The notebooks were regenerated by `tools/make_0N.py` into `runs/<tag>-<kind>/` (`M5_NB_DIR`)
  and executed with the new `tools/run_nb.py`. The committed `runs/2026-09-2x-*` and the
  executed top-level notebooks are untouched. Notebook 02 was skipped (preliminary).
- **Times:**

| Notebook | RTX 5070 | GTX 1060 (M5) |
|---|---|---|
| 01 data | 4.7 min | ~4 min language pass + extraction |
| 03 boundaries (encode + 4 BiLSTM configs) | 2 h 04 min (encoding ~20 min) | encoding 54 min + 4 × 9–17 min |
| 04 mT5 titles | 54 min | 56 min |
| 05 eval + export | 7.4 min | — |

- **Why the BiLSTM and mT5 are no faster:** both are bound by one CPU core launching small
  kernels (BiLSTM: tiny model; mT5: batch 4 × 256 tokens, the 1060-era recipe). GPU
  utilization was 17–27%. Only the encoding got faster (2.7×). A larger batch would fix it,
  but that is a recipe change, so it goes to brainstorm.
- **Boundaries** (val, product setting):
  - BiLSTM `big` F1@5s 16.86 → **16.64** and tIoU-F1 18.62 → 18.37, both within ±2 points.
    Best epoch 8 → 10.
  - TextTiling 7.77 → 7.77 (it reproduces).
  - `bilstm_ships` stays **true**.
  - es_val (18 videos) shows the same pattern as M5 (TextTiling ahead there): BiLSTM 8.17 →
    7.42.
- **mT5 titles** (CIDEr, val, ground-truth boundaries):
  - en 67.9 → **76.5**. That's outside ±3 but upwards; it reads as run-to-run variance (GPU
    numerics, sample order), not a regression.
  - es 50.0 → **47.4**, within ±3.
  - lead-8 is unchanged.
- **Exported bundles** (`backend/models/chapters/`):
  - `bilstm-e5s-sentence-20261107+mt5s` (v1, 1.6 GB)
  - `…+qwen3.5-4b` (4.2 GB)
  - `…+qwen3.5-9b` (6.9 GB), which **`current` points at**
  - The title-LLM metrics attached to the v2 bundles are the M5 val rows for the same GGUF
    (GTX 1060, CPU inference), labelled as such. Notebook 06 was **not re-run** on the 5070,
    a deviation from the plan. The owner picked the 9B, and the blind review on real
    episodes is the comparison that counts.
- **The 6 bundle-dependent integration tests** run again on this machine.

## 7. LLM fully on the GPU

- **Smoke bench:** n_ctx 4096, ~3,000-token prefill, 200 generated tokens, `n_gpu_layers=-1`.

| GGUF | VRAM (weights + KV) | Prefill | Generation |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M | +3,441 MiB | 679 tok/s* | 105 tok/s |
| **Qwen3.5-9B Q4_K_M** | **+5,544 MiB** | **4,198 tok/s** | **93 tok/s** |

\* The 4B ran first and includes the one-off CUDA warm-up. M9 on the 1060: prefill 372 tok/s
(weights in RAM), generation on the CPU.

- **Choice (owner, 2026-11-07): Qwen3.5-9B Q4_K_M, fully offloaded.**
  - **Why not Q6_K/Q8_0:** they were downloaded, but at 7.5 and 9.5 GB they don't fit next to
    the resident Whisper + pyannote + chapters (see §8). They would need a release-before-load
    hook.
  - **Why not the CIDEr gate:** the M5 one (where the 9B failed) is not applied, by the
    owner's decision.
- **`LLM_N_GPU_LAYERS`** is a new setting (default `-1`), part of the `runtime._cache` key.
  `0` restores M9 behaviour.

## 8. VRAM policy: worst-case co-residency

`tests/gpu/test_gpu_models.py::test_worst_case_resident_set_fits_the_card` loads, all at once,
the production set:
- Whisper large-v3 int8_float16 b16
- pyannote b32
- the chapter bundle on cuda
- the 9B, fully on the GPU

It runs each one in turn, as the `gpu_lock` does.

- **Measured peak: 10,637 MiB** (Windows desktop included, ~1 GB) of 12,227 MiB. The test's
  budget is 11,264 MiB.
- **Kept:**
  - `gpu_lock`
  - `release_all()` after every LLM job (Whisper's +6.5 GB batch peak needs the LLM gone)
  - `empty_cache()` after diarization
- **Chapters inference moves to the GPU** (`CHAPTERS_DEVICE=cuda`). It gives the same
  boundaries as on CPU (`test_chapters_on_cuda_match_cpu`).
- **Headroom is about 1.6 GB.** A heavy GPU app on the Windows desktop (games, video editors)
  can push a job into CUDA OOM. That is treated as transient and retried (0014, `MAX_TRIES`).

## 9. Run B: final defaults, every tool, 5 episodes (2026-11-07)

- **Config:**
  - Whisper large-v3 int8_float16 b16, pyannote b32
  - chapters on cuda, bundle `…20261107+qwen3.5-9b`
  - `LLM_N_GPU_LAYERS=-1`, 8 threads, relevance `relevance-p3`
  - `--parallel 2`, `--rerun`
  - Passed as env overrides; `backend/.env` still had the M9 values.
- **Totals:**
  - **30/30 jobs succeeded, 0 failed**, and no OOM or retry in the worker log.
  - Wall time **0.31 h for 2.50 h of audio (0.12× real time)**. The M9 season run was
    0.30×, on a different, larger set.
  - CSV: `experiments/hw-upgrade/runB_final.csv`.

| Tool | 1X09 14.9 min | 1X08 26.6 | 1X06 30.7 | 1X28 34.1 | 1X15 43.3 |
|---|---|---|---|---|---|
| transcribe (M9 → B) | 67 → 24 s | 93 → 42 s | 87 → 51 s | 259 → 60 s | 206 → 51 s |
| diarize | 140 → 34 s | 89 → 68 s | 177 → 31 s | 297 → 66 s | 146 → 40 s |
| chapters | 45 → 57 s | 224 → 40 s | 43 → 64 s | 129 → 109 s | 233 → 67 s |
| relevance | 320 → 77 s | 363 → 54 s | 402 → 98 s | 352 → 127 s | 494 → 101 s |
| shownotes | 371 → 110 s | 177 → 52 s | 368 → 105 s | 231 → 74 s | 309 → 86 s |
| shorts | 38 → 28 s | 42 → 42 s | 49 → 46 s | 73 → 45 s | 54 → 41 s |

- **These are job seconds, waits included.** With `--parallel 2`, a job also waits for the
  other episode's GPU steps, which explains the noisy per-cell ratios.
- **Isolated** (blind-review script, nothing else running; 1x08 / 1X15):

| Model | Chapters | Show notes | Relevance |
|---|---|---|---|
| 9B | 37 / 37 s | 26 / 31 s | 28 / 46 s |
| 4B | 26 / 24 s | 20 / 25 s | 24 / 39 s |

- **Chapters is the only tool not faster everywhere.** The titles now come from the 9B on
  the GPU, so a chapters job includes loading a 5.7 GB GGUF from `/mnt/c` (9p) after
  `release_all()`, plus lock waits. It is still under 2 min per episode. Moving the bundles
  to ext4 goes to brainstorm.
- **Structural checks** (`experiments/hw-upgrade/structural_check.py`): every episode has
  - 3 speakers and 3–4 chapters with Spanish titles
  - **100% of the relevance windows with an LLM score** (no failed batch), under the live
    version `heuristic-v0+Qwen3.5-9B-Q4_K_M+relevance-p3`
  - show notes in Spanish
  - **5 shorts**, 9:16 H.264, 28–47 s (1X09 at 480×854 from its 480p source); 25 files on disk
- **Owner review material:** `experiments/hw-upgrade/blind/blind_sheet.md`. It covers 4B vs
  9B on 1x08 and 1X15: chapter titles, show notes and the relevance top-5, shuffled per
  section. The key is `blind_key.json`. These are checkpoints C3–C5, pending.
