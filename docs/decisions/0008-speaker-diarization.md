# Decision: Speaker Diarization (M4)

**Date:** 2026-09-26
**Status:** Decided — implemented and verified (M4 done 2026-09-26)

## Context

M3 (`0007-transcription-pipeline.md`) persists transcript segments and
words with an empty `speaker_label`, and keeps the 16 kHz mono WAV in
storage (`uploads.audio_storage_path`) for reuse here. M4 registers a
real handler for `tool_type='diarize'`: pyannote `community-1` produces
speaker turns, those turns are assigned to words and segments, and the
result is written to `speaker_label`. The development plan also asks
for a startup healthcheck, because a misconfigured Hugging Face
license or token fails silently.

The hardware question (GPU vs. CPU on a 3 GB GTX 1060 already hosting
Whisper) was settled by a measured spike:
[`docs/research/m4-diarization-spike.md`](../research/m4-diarization-spike.md).

Decided with the project owner during planning:
- The HF read token lives in `backend/.env` as `HF_TOKEN`, with the
  `community-1` conditions accepted on Hugging Face.
- Subtitles show speakers when labels exist (§6).
- A failing healthcheck **degrades only `diarize`**. The worker still
  starts and the other 7 tools are unaffected (§5).

## Decisions

### 1. GPU, co-resident with Whisper, `embedding_batch_size=8`

Spike results, measured in one process as in the real worker:

| Setup | Speed | Peak GPU (total, incl. desktop) |
|---|---|---|
| pyannote alone, batch 32 (default) | 17.1× real-time | 2695 MiB |
| **pyannote with Whisper resident, batch 8** | **17.4× real-time** | **2326 MiB** |
| pyannote with Whisper resident, batch 1 | 12.6× real-time | 1810 MiB |
| CPU | < 0.84× real-time (aborted) | — |

Choice: keep **both models resident** and set pyannote's
`embedding_batch_size` to 8. This costs no speed and leaves ~750 MiB
of headroom on 3 GB.

- **Shared GPU lock.** The `asyncio.Lock` introduced in M3's
  `transcribe.py` moves to `app/worker/gpu.py` and guards both
  handlers, so two inferences never run on the card at once and their
  peaks never stack.
- **Loading.** The pipeline is a lazy per-process singleton, like
  `whisper.get_model()`.
- **Settings:**
  - `hf_token` (secret)
  - `diarization_device` (`cuda`)
  - `diarization_embedding_batch_size` (8)
  - `diarization_healthcheck` (True)

Alternatives rejected:
- **CPU:** more than 71 min per hour of audio.
- **Sequential GPU↔RAM model swapping:** measured under 0.5 s per swap,
  but unnecessary at batch 8. It stays documented as the fallback if
  headroom ever runs short.
- **pyannote `3.1`:** lower VRAM, but older, and the spec chose
  `community-1`.

Not measured by the spike: peak VRAM on a full-length episode.
Embedding batches are fixed-size, so the peak should not grow with
duration. The manual check fills this in:

#### Measured (manual full-episode check, 2026-09-26)

**Setup:** the owner's ~64.3 min, 2-host podcast episode, run through
the dev stack on the GTX 1060.

**Time:** the job ran for **3 min 31 s**, taken from the job row's
`started_at`/`completed_at`. That is **~18× real-time**, matching the
10-minute spike (17×). The global clustering step did not visibly
scale worse on the full episode.

**Owner's review of the VTT:** mostly right, but it gets confused in
fast back-and-forth exchanges.

**Diagnosis** (read-only query on the dev DB):
- **254 of 3563 segments (7.1%)** contain words from more than one
  speaker. That is 1018 of 11 341 words (9%).
- The word-level labels in those segments are stored correctly. Each
  subtitle cue, however, shows only the segment's majority speaker, so
  a quick interjection inside a Whisper segment renders under the wrong
  speaker. **Much of the visible confusion is therefore a cue-rendering
  limitation, not a diarization error.** Follow-up in "Post-review
  notes" below.
- The episode got **4 distinct speaker labels** rather than 2. To be
  checked with the owner: guests or clips in the episode, or
  over-clustering.

**Peak VRAM on a full episode, measured after the OOM fix below.** The
run used the real backend code in production order, one process, with
the GPU otherwise idle (desktop ~510 MiB):

| Phase | Time | Peak process | Peak GPU total |
|---|---|---|---|
| Startup healthcheck | 4.6 s | 768 MiB (138 MiB after the cache release) | 1278 MiB |
| Transcribe full episode | 350 s (11.0× real-time) | 1458 MiB | 2016 MiB |
| **Diarize full episode, Whisper resident** | 199 s (**19.4× real-time**) | 1848 MiB | **2398 / 3072 MiB** |

- **Headroom:** worst-case peak leaves ~670 MiB free. A busier desktop
  (Firefox playing previews measured ~690 MiB) still leaves ~0.5 GB.
- **Duration:** the full-episode peak is in line with the 10-minute
  spike (2326 MiB), confirming that pyannote's VRAM doesn't scale with
  audio length.
- **Output:** the diarization again found 4 speakers (1963 turns).

### 2. Dependencies: torch cu126, CUDA wheel pins realigned

- Adds `torch==2.8.0+cu126`, `torchaudio==2.8.0+cu126` (via
  `--extra-index-url https://download.pytorch.org/whl/cu126`),
  `pyannote.audio==4.0.7` and `soundfile`.
- **Why cu126:** the cu128/cu129 builds of torch 2.8 dropped Pascal.
  In the cu126 build the arch list includes `sm_60`, which the sm_61
  card runs, and GPU ops were verified.
- **M3's CUDA wheel pins move down to torch's:**
  `nvidia-cudnn-cu12==9.10.2.21` and `nvidia-cublas-cu12==12.6.4.1`.
  Spike check D1 verified faster-whisper / ctranslate2 4.8.2 on these
  versions: same output, same speed. 0007 is annotated accordingly.
- **torchcodec does not load here** (`libnvrtc.so.13`: it expects CUDA
  13 libraries). pyannote therefore never decodes files itself. The
  stored WAV is read with `soundfile` and passed preloaded as
  `{"waveform", "sample_rate"}`.
- **No WhisperX.** Its dependency pins conflict with faster-whisper
  1.2.1 and pyannote 4. This diverges from spec §2.2 and §5, which name
  WhisperX's `assign_word_speakers`. The ~50-line assignment step is
  reimplemented instead (§3).
- **Attribution.** The `community-1` weights are CC-BY-4.0 and are
  credited in the README.

### 3. Speaker assignment (`app/diarization/assign.py`, pure functions)

Input is the **`exclusive_speaker_diarization`** output: pyannote 4's
overlap-free variant, with at most one speaker at any instant.

- **Word:** takes the turn with the **largest time overlap**. A word
  that overlaps no turn (it falls in a gap) takes the **nearest** turn
  to its midpoint. Words without timestamps keep `NULL` (spec §2.2
  step 4).
- **Segment:** takes the **majority** speaker among its words. A tie
  goes to the first labelled word's speaker. A segment with no labelled
  words keeps `NULL` (spec §2.2 step 3).
- **Labels:** renamed by order of first appearance, so the first person
  to speak is `SPEAKER_00`. pyannote's raw label numbering is
  arbitrary.
- Turns are sorted and searched with bisection, so the cost is not
  O(words × turns).

### 4. `diarize` handler

1. If the startup healthcheck failed, fail immediately with its message.
2. **Preconditions:** a `succeeded` transcribe job, persisted segments,
   and `audio_storage_path`. Otherwise fail with an explicit
   "Run Transcript first" error.
3. Read the WAV through `LocalStorageBackend.path_for`, as in M3.
4. Run pyannote in `asyncio.to_thread` under the shared GPU lock.
5. Assign speakers, then write in **one transaction**. All of the
   upload's labels are cleared first, so a re-run is idempotent.

Re-running `transcribe` replaces the rows, which drops speaker labels;
`diarize` must then be run again. This is noted in brainstorm, since
automatic chaining is out of M4's scope.

### 5. Startup healthcheck

- **What it runs:** at `WorkerSettings.on_startup`, the worker loads the
  pipeline and runs it on a bundled clip,
  `app/assets/diarization_healthcheck.wav`. The clip is synthetic, has
  two known speakers, and doubles as a test fixture.
- **Pass condition:** non-empty output **and** at least 2 speakers.
  This catches the silent failure modes (wrong license or token, empty
  output) regardless of their cause.
- **On failure:** the error is logged at ERROR and stored in module
  state. `diarize` jobs fail fast with that message, and the Manage
  page shows it as a tooltip (§7). The other tools keep working, as
  chosen by the owner.
- **Cost:** a few seconds at startup. It also preloads the pipeline.

### 6. Subtitles show speakers

When a segment has a `speaker_label`:
- SRT cues are prefixed with `[SPEAKER_00] `.
- VTT cues use the standard voice span, `<v SPEAKER_00>…`.

Without diarization the output is byte-for-byte what M3 produced. The
endpoint status rules are unchanged.

**Per-speaker cue splitting.** This was added after the owner's
full-episode review; see "Post-review notes".
- **The problem:** Whisper segments often contain a quick interjection
  from the other speaker, and one cue per segment shows only the
  majority label.
- **The change:** a segment whose word labels change speaker is split
  into one cue per speaker run. Each cue starts at its run's first
  timed word, and its text is that run's words.
- **Noise guard:** runs shorter than 2 words (`MIN_RUN_WORDS`) are
  folded into a neighbour, so a single mislabelled word never becomes
  its own cue.
- **Unlabelled words:** words with no label (untimed) stay in the run
  they sit in.
- **Unchanged cases:** segments with one speaker, or with no word
  labels, still render exactly as before.
- **Scope:** rendering only. The DB and pyannote are untouched.

### 7. Frontend (minimal)

- **Speakers "Run" button:** disabled until a transcript exists (the
  existing `hasTranscript()`), with a "Needs a transcript" title.
- **Failure reasons:** the job status badge shows `error_message` as a
  tooltip on failed jobs. Until now the UI never showed why any job
  failed, and the degraded-healthcheck path depends on that being
  visible.

## Test plan

**Fixture**
- A ~35 s synthetic A/B/A "conversation" of two LibriSpeech dev-clean
  speakers (one female, one male; CC BY 4.0).
- Known turn boundaries in JSON, plus an attribution file.
- Used as the healthcheck clip. An MP4 version drives the full
  pipeline.

**Unit tests**
- Word assignment: inside a turn, straddling two turns (largest overlap
  wins), in a gap (nearest wins), no timestamps (`NULL`), no turns at
  all.
- Segment majority, tie-break and all-`NULL`.
- Relabelling by first appearance.
- SRT/VTT with and without labels.

**Handler tests (mocked pyannote)**
- Labels are persisted.
- A re-run is idempotent.
- Clean failures when the transcript or the audio is missing.
- Fast failure when the healthcheck failed.

**Healthcheck tests**
- Real pyannote on CPU with the clip: non-empty output, 2 speakers.
- Degraded path with a fake pipeline that returns empty output or
  raises.

**Integration test**
- The MP4 fixture goes through real `transcribe` (Whisper `base`, CPU)
  and then real `diarize` (pyannote, CPU).
- **At least 90% of words must be correctly labelled** against the
  known boundaries, under the best label permutation, and every
  segment must be labelled.

**Regression**
- The full M0–M3 suite.
- The real-arq-worker test chains `transcribe`, then `diarize`.

**Manual check**
- The owner's ~64 min episode on the GPU: record time and peak VRAM in
  §1, and check that the VTT voice tags are coherent.

## Consequences

- **New dependencies:** torch/torchaudio cu126 (large, ~4 GB with the
  CUDA wheels), pyannote.audio 4.0.7 and soundfile. The CUDA wheel pins
  are realigned to torch's.
- **New modules:** `app/diarization/` (`assign`, `pipeline`),
  `app/worker/gpu.py`, `app/worker/handlers/diarize.py` and
  `app/assets/`.
- **Existing handlers:** `diarize` no longer uses the dummy handler.
  The other 6 tool types still do.
- **Worker startup:** a few seconds slower because of the healthcheck.
  It can be disabled via `DIARIZATION_HEALTHCHECK=false`.

## Post-review notes

- **Owner's verdict on the full episode:** diarization is right for the
  most part and only gets confused in fast exchanges. Nothing else
  needs to change unless there is a cheap, high-impact fix.
- **That fix was per-speaker cue splitting (§6).** The diagnosis
  showed that 7.1% of segments mix speakers at word level while their
  cue showed a single label. It is a rendering-only change, approved by
  the owner within M4's scope.
- **What remains after it:** boundary blur that pyannote itself
  produces on rapid turns, and single-word interjections (folded by
  design). These are accepted.
- **4 speakers on a 2-host episode is correct.** The owner confirmed
  the episode has more voices (guests, clips or ads), so this is not
  over-clustering.
- **Bug found by the manual check: CUDA OOM in the next transcription.**
  With the fixed worker order (the startup healthcheck runs pyannote
  first), the worker process held **1120 MiB after the healthcheck**,
  even though pyannote's weights are only tens of MB. The difference
  was torch's caching allocator keeping inference memory reserved.
  - **What failed:** a following `transcribe` job loaded Whisper
    (process at 2112 MiB) and OOMed during decoding, at 2978/3072 MiB
    total. The next job then failed with
    `cudaErrorInvalidDevice: invalid device ordinal`, because a
    ctranslate2 context is unusable after a CUDA OOM, so the worker
    needs a restart.
  - **Why the spike missed it:** it loaded Whisper *first*, and check D
    called `torch.cuda.empty_cache()` by hand. That call never made it
    into production code.
  - **Fix:** `pipeline.diarize` now calls `torch.cuda.empty_cache()` in
    a `finally` after every run, including the healthcheck. A
    regression test covers both success and failure.
  - **Verification.** Run in production order on the full episode:
    healthcheck, then Whisper transcribing, then pyannote diarizing
    with Whisper resident. It passes; see the table in §1.
  - **Process note.** Two intermediate verification runs also OOMed.
    Both overlapped with a restarted dev worker competing for the same
    GPU, so they were contamination, not a second bug. A step-by-step
    instrumented re-run on an idle GPU showed the cache release was
    the whole fix. Whisper alone on the new cuDNN 9.10 / cuBLAS 12.6
    libraries was also checked separately (peak 1384 MiB process,
    11.2× real-time), which rules out the library downgrade.
    **Lesson:** GPU measurements need the GPU to themselves; say so
    explicitly before starting one.
- **Final manual check through the UI passed (2026-09-26).** With the
  fix, the owner re-ran the episode from the Manage page: transcribe
  took 6 min 32 s, diarize 3 min 38 s, and both were verified correct.
- **Cue-splitting impact on that episode:** 254 of 3563 segments mix
  speakers at word level, but splitting adds only **50 cues**
  (3563 → 3613). Most mixes are single-word minority runs, folded by
  design (`MIN_RUN_WORDS = 2`). Those are either genuine one-word
  interjections or diarization noise, which can't be told apart
  without ground truth. Tunable if it proves too conservative.
- **Tests:** final count is backend 119 passed, frontend 95 passed.
  Suites were run directly, because the `test-runner` agent still
  isn't loaded in this session.

