# M9: where the time goes, and what was changed

**Date:** 2026-10-16
**Milestone:** M9 — Full-suite hardening (development sub-phase)
**Status:** Benchmarks complete. The season-1 run results are in §5.
Decisions: `docs/decisions/0014-m9-performance.md`.

Machine: Ryzen 5 2600X (6 cores / 12 threads), 16 GB RAM, GTX 1060 3 GB
(Pascal, sm_61), driver 580. Scripts in `experiments/m9-perf/`.

## 1. Baseline (M3–M8 settings, episode 8: 64 min, Spanish, 2 hosts)

| Tool | Job time | Share |
|---|---|---|
| Transcription | 392–411 s | 14 % |
| Diarization | 211–219 s | 7 % |
| Chapters | 261 s | 9 % |
| Show notes | 978 s | 33 % |
| Relevance | 1010 s | 35 % |
| Shorts | 45–55 s | 2 % |
| **Total (serial)** | **~48 min** | |

- **Inside transcription:**
  - ffmpeg audio extraction 18 s
  - Whisper decoding 343 s
  - model load 27 s on first use
- **Inside the LLM tools:** see §3. Prefill dominates.

## 2. Transcription (`bench_transcribe.py`, `asr_variant.py`, `asr_compare.py`)

Every variant is faster-whisper `large-v3-turbo`, int8, VAD on, with
word timestamps.

### Speed and VRAM (full episode 8)

| Variant | Whisper time | VRAM peak |
|---|---|---|
| sequential, beam 5 (M3) | 343 s | 1.8 GB |
| sequential, beam 1 | 246 s | 1.7 GB |
| batched ×2, beam 5 | 212 s | 1.9 GB |
| batched ×4, beam 5 | 192 s | 2.5 GB |
| **batched ×4, beam 1** | **165 s** | 2.5 GB |
| batched ×8 / ×16 | out of memory (3 GB) | — |

- A CUDA OOM inside faster-whisper leaves the device unusable for the
  rest of the process ("invalid device ordinal"). Every variant was
  re-run in its own process.

### Quality

- **No ground truth:** the season's videos only have YouTube's
  automatic captions.
- **Agreement with the current transcript:** every variant differs from
  the M3 transcript (sequential beam 5) by 13–16 % of words, so that
  comparison can't rank them.
- **Reference used instead:** `large-v3` (non-turbo, beam 5, on the CPU,
  since it does not fit the GPU next to anything) on three 5-min
  excerpts (10:00, 31:40, 55:00), 2711 words.

| Variant | WER vs large-v3 | sub | del | ins | Time (3 excerpts) |
|---|---|---|---|---|---|
| sequential, beam 5 (M3) | 15.1 % | 4.1 % | 6.8 % | 4.3 % | 79 s |
| sequential, beam 1 | 12.2 % | 3.7 % | 5.0 % | 3.5 % | 57 s |
| batched ×4, beam 5 | 13.6 % | 3.9 % | 7.6 % | 2.2 % | 45 s |
| **batched ×4, beam 1** | 13.9 % | 4.2 % | 7.2 % | 2.5 % | **39 s** |

- **What the differences are** (read by hand): mostly crosstalk and
  fast overlapping speech, where the models disagree on what was said.
  - The M3 setting invents more words (insertions).
  - The batched ones drop more, occasionally a whole short phrase said
    over another speaker.
  - None of the variants is clearly worse than the M3 setting.
- **Word timestamps:** drift against the M3 transcript on matched words
  is p50 0.01 s, p95 0.44 s, p99 1.1 s. That is fine for 15 s relevance
  windows, sentence snapping and captions.

## 3. LLM (`bench_llm.py`, Qwen3.5-4B Q4_K_M, llama.cpp 0.3.35)

One relevance batch of 20 blocks: 1244 prompt tokens, ~85 generated.

| Build / setting | Prefill | Generation | Call |
|---|---|---|---|
| CPU, 6 threads (M6/M7) | 28 tok/s (44 s) | 6.1 tok/s | 58 s |
| CPU, 4 threads | 20 tok/s | 5.7 tok/s | 79 s |
| CPU, 8 threads | 28 tok/s | 6.5 tok/s | 58 s |
| CPU, 12 threads | 28 tok/s | 4.8 tok/s | 62 s |
| CPU, 6 threads, n_batch 2048 | 28 tok/s | 6.3 tok/s | 59 s |
| **CUDA build, 0 layers on GPU** | **372 tok/s (3.3 s)** | 6.5 tok/s | **16 s** |
| CUDA build, 16 layers on GPU | 452 tok/s | 8.8 tok/s | 12 s |
| CUDA build, 24+ layers | context creation fails (VRAM) | — | — |

- **Threads and batch size change nothing:** the CPU is saturated at 6
  cores.
- **The prefill is the bottleneck.** Fewer, larger calls don't help,
  because the total number of prompt tokens is the same.
- **With the CUDA build at 0 layers,** the weights stay in RAM and
  llama.cpp offloads the large prompt-batch matmuls to the GPU. That
  gives 13× on the prefill at a small VRAM cost, which is why it was
  chosen over 16 layers (~1.3 GB resident, competing with Whisper).
- **Build:** llama-cpp-python 0.3.35 compiled with
  `-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61`, against NVIDIA's
  CUDA 12.6 redistributable tarballs (nvcc, cudart, cccl, cublas). No
  system CUDA toolkit or sudo is needed. The prebuilt cu12x wheels stop
  at 0.3.19, which predates Qwen3.5.

### Measured on the real worker (episode 8)

| Tool | M6/M7 (CPU) | M9 (CUDA prefill) |
|---|---|---|
| Show notes | 978 s | 351 s |
| Relevance, 1 pass (p1 method) | 1010 s | 237 s |
| Relevance, 2 passes (p2, adopted) | — | 478–554 s |

## 4. Relevance scores are fragile (the finding that changed relevance to p2)

Switching to the GPU prefill changed the relevance LLM scores far more
than numeric noise should:
- Spearman 0.53 against M7's CPU scores on episode 8
- a different top 5
- the intro jumped from ~0.33 to ~0.64

Investigation (`llm_consistency.py`, `llm_single.py`, `llm_expected.py`):

| Method | CPU vs GPU agreement | Notes |
|---|---|---|
| M7, 20 integers per call (greedy) | Spearman 0.74, mean \|Δ\| 7.4 pts | CPU 6 vs 8 threads: identical. The outputs are smooth patterned runs (45, 38, 30, 28, 25…), with an arbitrary level per batch. |
| One block per call, greedy integer | 72 % exact, Spearman 0.58 | Nearly binary (15 or 45): flips on tiny changes. 1.95 s per call on the GPU. |
| One block per call, **expected value over the digits 0–9** | **Spearman 0.98**, mean \|Δ\| 0.014 | Continuous and reproducible. 0.79 s per call. |

- **Expected value on episode 8** (a full run, 420 s): stable, but its
  top 5 included pure logistics ("aún no hemos acabado… ya me he
  perdido") and the closing credits/app plug. M7's approved top 5 looked
  better. The owner chose not to use it for picking shorts.
- **Conclusion:** the 4B model used zero-shot discriminates weakly.
  Read greedily it is unstable; read as an expectation it is stable but
  poorly calibrated. M7's risk note anticipated this low ceiling; the
  fix belongs to M15's trained model.
- **For M15:** the expected-value score is a much better
  *pseudo-label* candidate (reproducible, continuous) than greedy
  integers.
- **Adopted for M9 (owner's choice):** M7's method on the GPU, with **two
  passes averaged**: the second pass pairs the base blocks shifted by
  7.5 s (`relevance-p2`). On episode 8 its Spearman against M7's CPU
  scores rose from 0.66 to 0.72.

## 5. Season-1 run (2026-10-16/17)

- **Set:** 28 episodes of *Sastre y Maldonado* season 1 (the playlist
  lacks 1x19 and 1x21).
  - 720p (AV1 or H.264) with Opus audio
  - 15–43 min each, 907.8 min (15.1 h)
    in total, 32.4 min on average
  - all Spanish, 2 hosts plus guests
- **Harness:** `scripts/run_pipeline.py` against the real API and arq
  worker, 2 episodes in flight. All six tools ran on every episode, with
  9:16 shorts and captions. Per-job CSV:
  `experiments/m9-perf/season1_run.csv`.
- **Result:** **28/28 episodes completed every tool**, after the fixes
  below.
  - Wall time 4.6 h, plus ~10 min of fix-and-resume, so ~9.9 min per
    episode: **0.30× real time** for the whole suite.
  - Every episode came out with:
    - ~56 transcript segments and 3–4 speakers
    - 3–5 chapters
    - relevance on every window (no LLM batch failed)
    - show notes in Spanish
    - 5 shorts of 30–53 s on average per episode
- **Job times** (started → completed; they include waits for the
  shared GPU lock, because 2 episodes are in flight):

  | Tool | Median | p90 | Max |
  |---|---|---|---|
  | transcribe | 155 s | 259 s | 308 s |
  | diarize | 161 s | 306 s | 344 s |
  | chapters | 110 s | 251 s | 275 s |
  | relevance | 353 s | 463 s | 494 s |
  | shownotes | 295 s | 413 s | 441 s |
  | shorts | 62 s | 79 s | 101 s |

  - Pure Whisper time: median 81 s, max 106 s per episode (from the
    worker's phase timings).

### Bugs found and fixed (each with a regression test)

1. **CUDA OOM in transcription (1x03).**
   - **Cause:** with the CUDA build, a resident llama.cpp context holds
     ~1 GB of VRAM. Whisper (~1.1 GB) + pyannote + the LLM left 2.15 GB
     in use, and batched Whisper needs +1.1 GB.
   - **Fix:** `runtime.release_all()` closes the LLM at the end of every
     LLM job; reloading takes seconds. The retry policy now also
     recognises ctranslate2's "CUDA failed with error out of memory"
     and llama.cpp's "Failed to create llama_context".
   - **Tests:** `tests/test_llm_runtime.py`, `test_relevance_handler.py`,
     `test_worker.py`.
2. **Shorts render failed on a clip with several no-face stretches
   (1x30).**
   - **Cause:** the `fit` branches reused the filtergraph labels
     `[bg][fg][b][f]`. Fix: the labels are indexed per run.
   - **Underneath it:** a closing fade crossed the cut threshold on
     every other frame and made 16 shots of 1–4 frames. A crop run that
     short had no track sample, so its `sendcmd` file was empty and
     ffmpeg rejected it.
   - **Fix:** a minimum shot length of 0.5 s (a fade counts as one cut),
     and a `sendcmd` file always has at least one command.
   - **Tests:** `test_shorts_framing.py`, `test_shorts_render.py`.
   - The season's shorts were re-rendered with the fix.
3. **Subtitles with ~28 s cues** (a regression from batched Whisper).
   - **Cause:** batched decoding returns one segment per ~30 s VAD
     chunk, and subtitles had one cue per segment.
   - **Fix:** cues over 84 characters or 7 s are split on word
     timestamps, preferably after punctuation. 1x01 went from ~56 cues
     to 495 (3.1 s on average).
   - **Tests:** `test_subtitles.py`.
4. **`uploads.duration_seconds` was never set**, by any milestone.
   - The dashboard showed every video as "missing duration".
   - **Fix:** transcription sets it from the extracted WAV's header; 29
     existing uploads were backfilled from their stored WAVs.
   - **Test:** `test_transcribe_handler.py`.
5. **Orphan clip files after stopping the worker mid-render.**
   - **Cause:** arq cancels the job (`CancelledError`, a
     `BaseException`), so the shorts handler's `except Exception`
     cleanup never ran. Two files were left over from the stop during
     fix 1; they were deleted.
   - **Fix:** the cleanup catches `BaseException`.
   - **Test:** `test_shorts_handler.py`.
- **After the shorts re-render** (16 min for 28 episodes, no failures):
  145 clips (28 episodes × 5, plus episode 8) and exactly 145 files on
  disk.
