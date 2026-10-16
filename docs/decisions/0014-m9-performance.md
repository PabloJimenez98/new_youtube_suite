# Decision: M9 Hardening — Faster Transcription, GPU Prefill for the LLM, Retries, Season Run

**Date:** 2026-10-16
**Status:** Implemented (M9 done, 2026-10-17).

## Context

- **The plan's M9** is the full-suite checkpoint: run real episodes
  through every tool, fix the integration bugs that appear, add basic
  retries, and polish viewers only where the run shows a need.
- **Decided with the owner (2026-10-16):**
  - **Data:** download season 1 of *Sastre y Maldonado* (28 episodes
    in the YouTube playlist, 15–43 min each, ~15 h in total), in 720p,
    to `D:` (`/mnt/d/podcasts/Sastre_y_Maldonado_T1/`). It is local
    test data only, never redistributed.
  - **Optimize first:** transcription, which the owner asked for, and
    the LLM steps, which the baseline showed take 70 % of the time.
  - **Run the whole season**, not 3–5 episodes.
  - **Failures during the run:** fix them, add a regression test,
    re-run the affected episodes. **Code:** commit at the end without
    closing M9; closing waits for the owner's review of a sample.
  - **Shorts:** 9:16 with captions.
- **All measurements** are in `docs/research/m9-performance.md`.

## Decisions

### 1. Transcription: batched greedy decoding

- **`WHISPER_BATCH_SIZE=4`, `WHISPER_BEAM_SIZE=1`** (new defaults).
  faster-whisper's `BatchedInferencePipeline` decodes 4 VAD chunks at
  once.
- **Measured:**
  - Whisper time on a 64-min episode: 343 → 165 s (2.1×)
  - VRAM peak 2.5 GB
  - batch 8 runs out of the 3 GB
- **Quality:** against a `large-v3` reference, the WER is 13.9 %, vs
  15.1 % for the M3 setting. The fast variant is no worse.
  Word-timestamp drift is p95 0.44 s.
- **`WHISPER_BATCH_SIZE=0`** restores M3's sequential decoding.
- **Rejected:**
  - batch 8 or more: out of memory
  - keeping beam 5: 16 % slower and no closer to the reference
  - `large-v3`: doesn't fit on the GPU next to the other models

### 2. LLM: llama.cpp built with CUDA; the prefill runs on the GPU

- **The bottleneck is the prompt prefill:** 28 tokens/s on the CPU.
  Thread count and batch size change nothing.
- **llama-cpp-python 0.3.35, built with CUDA for sm_61:**
  - built from NVIDIA's CUDA 12.6 redistributable tarballs, with no
    system toolkit or sudo (see README)
  - installed in the backend venv in place of the CPU build
- **`n_gpu_layers=0`:** the weights stay in RAM, and llama.cpp runs the
  large prompt batches on the GPU.
  - prefill 28 → 372 tokens/s; a relevance call 58 → 16 s
  - little VRAM, so it coexists with the resident Whisper and pyannote
- **`app/llm/cuda_libs.py`** preloads the venv's `libcudart`/`libcublas`
  (already present for faster-whisper and torch), so `import llama_cpp`
  works without `LD_LIBRARY_PATH`.
- **The LLM steps now take `gpu_lock`** (chapters titles, relevance,
  show notes), so their GPU use never overlaps Whisper's or pyannote's
  peaks.
- **Measured on episode 8:**
  - show notes 978 → 351 s
  - relevance 1010 → 478–554 s, with the two passes of §3
- **Rejected:**
  - 16 layers on the GPU: ~25 % faster still, but ~1.3 GB resident
    competing with Whisper
  - 24+ layers: don't fit
  - the prebuilt CUDA wheels: they stop at 0.3.19, before Qwen3.5

### 3. Relevance `relevance-p2`: two averaged passes

- **The GPU prefill changed M7's relevance LLM scores:** Spearman 0.53,
  and a different top 5.
- **Investigation** (research §4):
  - Greedy integers from this 4B model are **fragile**: CPU vs GPU
    agree at only 0.74 on the same batches. Asked for 20 numbers, the
    model writes smooth patterned runs.
  - Scoring each window by the **expected value over the digits 0–9**
    was stable (0.98), but on episode 8 its top picks included
    logistics and the closing credits.
- **Owner's choice:** keep M7's prompt and batching, and add a **second
  pass** with the blocks shifted by 7.5 s.
  - Every window gets one aligned score and one overlap-averaged
    score, and the LLM score is their mean.
  - Spearman against M7's CPU scores: 0.66 → 0.72.
- **Versioning:**
  - The version label is `heuristic-v0+Qwen3.5-4B-Q4_K_M+relevance-p2`.
  - p1's scores can't be reproduced with the GPU build, so p2 was set
    `is_live` by hand in the dev DB. The general live-switch mechanism
    is still M15's.
- **For M15:** the expected-value score is the better pseudo-label
  (reproducible, continuous). The experiment script is kept in
  `experiments/m9-perf/llm_expected.py`.

### 4. Retries and interrupted jobs

- **`process_job` retries only transient failures:**
  - a lost DB or Redis connection (`OperationalError`, invalidated
    connections, `ConnectionError`, `TimeoutError`)
  - CUDA out of memory
- **How:** the job goes back to `queued` with a "Retrying (attempt n/3
  failed)" message, and arq runs it again after 30 s × the attempt
  number (`MAX_TRIES = 3`).
- **Everything else fails at once:** a missing precondition, a missing
  model, bad input or a bug would only fail again.
- **At worker startup,** jobs still `running` (the single worker was
  stopped or crashed) are marked `failed`, with "Interrupted…", so the
  UI offers Run again.

### 5. Observability

- **Per-phase timings for transcription:** extraction, GPU wait,
  Whisper, saving the WAV, DB.
- **One line per LLM call:** prompt and completion tokens, seconds.
- **The worker now prints `app.*` INFO logs**; arq only configured its
  own logger before.

### 6. The season runner (`scripts/run_pipeline.py`)

- **A test harness, not a product feature.** It drives the real API:
  - uploads every video in a folder, skipping those already uploaded
    (by filename)
  - runs transcribe → diarize → chapters → relevance → shownotes →
    shorts in order
- **Resumable:** succeeded tools are skipped.
- **2 episodes in flight,** so CPU-only steps (shorts, uploads)
  overlap the other episode's GPU steps.
- **Output:** a CSV of every job's time and status.

## Alternatives rejected (beyond those above)

- **Fewer, larger LLM calls:** the prefill cost is per token, so
  there's no gain.
- **A pipeline-orchestration feature** (auto-chaining tools after
  upload): useful, but it is product scope that wasn't asked for.
  Noted in brainstorm.

## Season run (2026-10-16/17)

- **28/28 episodes** (15.1 h of audio) went through all six tools in
  4.6 h of wall time: 0.30× real time, ~10 min per 32-min episode.
- **Five integration bugs** surfaced and were fixed, each with a
  regression test. Details in research §5.
  1. **The LLM's resident ~1 GB of VRAM made batched Whisper run out of
     memory.** The LLM handlers now release the model after each job
     (`runtime.release_all`), and the retry policy recognises
     ctranslate2's and llama.cpp's OOM messages.
  2. **A closing fade split a clip into 1-frame shots,** and the shorts
     render failed (duplicate filter labels, an empty `sendcmd` file).
     Fix: a minimum shot length of 0.5 s, indexed labels, and a
     `sendcmd` file that is never empty. The season's shorts were
     re-rendered.
  3. **Batched Whisper's ~30 s segments made 30 s subtitle cues.** Cues
     are now split on word timestamps (≤ 84 characters, ≤ 7 s).
  4. **`uploads.duration_seconds` was never filled.** Transcription now
     sets it, and the existing uploads were backfilled.
  5. **Stopping the worker mid-render left orphan clip files.** The
     shorts cleanup now runs on cancellation too.
- **Owner's review (2026-10-17):** approved ("lo he revisado y está
  todo bien").

## Consequences

- **The backend venv needs the CUDA build of llama-cpp-python** for the
  speed-ups. The CPU build still works, just slower; the README
  documents both.
- **Whisper decoding is greedy now:** `WHISPER_BEAM_SIZE=5` and
  `WHISPER_BATCH_SIZE=0` give back M3's behaviour.
- **0012 §3 is amended by §3 here** (two passes), and the relevance
  version is now p2.
