# Decision: Transcription Pipeline (M3)

**Date:** 2026-09-26
**Status:** Decided — implemented and verified (M3 done 2026-09-26)

## Context

M3 replaces the M2 dummy handler for `tool_type='transcribe'` with real
processing: ffmpeg audio extraction → faster-whisper (`large-v3-turbo`,
`word_timestamps=True`) → `transcript_segments`/`transcript_words` rows,
plus `.srt`/`.vtt` download endpoints. `docs/development-plan.md`'s M3
entry flags GPU vs. CPU deployment as a real decision. Several smaller
choices (intermediate-audio storage, handler DB access, endpoint
semantics, subtitle caching) also had to be made. The enqueue path and
the frontend polling from `0006-async-job-pipeline.md` stay as they are.

Decided with the project owner during planning:
- v1 production target is **this machine** (self-hosted).
- Podcast language is **auto-detected** per episode (`language=None`).
- The intermediate WAV is **persisted in the StorageBackend**, with only
  its key stored in the DB. The transcript itself is stored in the DB.
- Handlers receive the DB session (registry signature change, §4).

## Decisions

### 1. Deployment target: GPU (GTX 1060 3GB), `compute_type="int8"`

Hardware: NVIDIA GTX 1060 3GB (Pascal, compute capability 6.1, ~2.5 GB
VRAM free with the desktop running), Ryzen 5 2600X (6c/12t), 16 GB RAM.

- CTranslate2 on CC 6.1 only supports **int8 (as int8_float32) and
  float32** ([quantization docs](https://opennmt.net/CTranslate2/quantization.html)).
  `float16`/`int8_float16` need CC ≥ 7.0, so they are not options here.
- `large-v3-turbo` in int8 needs ~1.5–1.6 GB of VRAM, which fits in
  ~2.5 GB. float32 (~3 GB) does not fit.
- `BatchedInferencePipeline` is **not used**: its memory spikes are
  incompatible with 3 GB (faster-whisper issue #1257). beam_size stays
  at 5 by default, since research shows it barely affects memory.
- Pascal throughput is **unverified**. The estimate is 2–4× real-time.
  The first development step is a spike that measures real-time
  factor (RTF) and peak VRAM on a real episode. Results go in the
  "Measured" section below.
- **CPU int8 is the fallback, switched by config only**
  (`WHISPER_DEVICE=cpu`). No code path differs.
- CUDA runtime comes from pip wheels (`nvidia-cublas-cu12`,
  `nvidia-cudnn-cu12==9.*`), per the faster-whisper README.
  `LD_LIBRARY_PATH` setup is documented in the root README.

New settings: `whisper_model` (`large-v3-turbo`), `whisper_device`
(`cuda`), `whisper_compute_type` (`int8`), `whisper_beam_size` (5),
`whisper_vad_filter` (True, to cut hallucinations on silence/music
beds), `worker_job_timeout_seconds` (10800).

**Tests use `base` on CPU int8.** This keeps them independent of the
CUDA setup and fast (~145 MB, downloaded once).

Alternatives rejected:
- **CPU as the primary target**: it leaves an available GPU idle for
  the heaviest job in the app.
- **float32 on GPU**: does not fit in the available VRAM.
- **A smaller production model (`small`/`medium`)**: turbo in int8
  already fits, and its accuracy is the reason it was picked in
  `project-description.md` §5.

#### Measured (development spike, 2026-09-26)
Setup:
- faster-whisper 1.2.1, ctranslate2 4.8.2, driver 580.
- `ctranslate2.get_supported_compute_types("cuda")` returned
  `{int8, int8_float32, float32}`, which confirms the CC 6.1 limits.

Run:
- `large-v3-turbo`, int8, `beam_size=5`, `word_timestamps=True`,
  `vad_filter=True`.
- Input: 10 min of English TV-drama speech with a music bed. This is a
  harder case than a podcast.
- **Result: 40.7 s, i.e. ~14.7× real-time** (a 1 h episode takes about
  4 min).
- **Peak GPU memory was 2074 MiB total**, of which ~520 MiB is the
  desktop, so the model uses ~1.55 GB. That leaves ~1 GB of headroom.
  The status bar reads 2074/3072 MiB.

The first model load includes a one-time ~1.6 GB download into the
Hugging Face cache.

### 2. Schema

Migration `0004` creates the two tables from `project-description.md` §4.

**`transcript_segments`**
- `upload_id` FK → `uploads`, `ON DELETE CASCADE`, indexed.
- `segment_index`.
- `start_seconds`, `end_seconds`.
- `text`.
- `avg_logprob`.
- `no_speech_prob`.
- `speaker_label` (NULL until M4).
- `UNIQUE(upload_id, segment_index)`.

**`transcript_words`**
- `segment_id` FK → `transcript_segments`, `ON DELETE CASCADE`, indexed.
- `word_index`.
- `word`.
- `start_seconds`, `end_seconds` (both NULL-able).
- `probability`.
- `speaker_label`.
- `UNIQUE(segment_id, word_index)`.

Migration `0005` adds `uploads.audio_storage_path TEXT NULL`.

Not stored, and noted in `brainstorm.md` in case M5/M6 need them:
- Whisper's `compression_ratio`, which the spec's table omits too.
- The detected language, which the schema has no column for.

### 3. Audio extraction and storage

- ffmpeg is run as a subprocess (`asyncio.create_subprocess_exec`):
  `-vn -ac 1 -ar 16000 -c:a pcm_s16le`. On a non-zero exit, the stderr
  tail is raised and ends up in `job.error_message`.
- Input is read through `LocalStorageBackend.path_for`, the same
  local-only escape hatch the streaming endpoint uses (0005). An S3
  backend would need a download-to-temp step first.
- Output is written into a worker `TemporaryDirectory` and transcribed
  from there. It is then saved to storage as `{upload_id}_audio.wav`,
  with the key recorded in `uploads.audio_storage_path`. The key is
  flat because `_resolve` flattens paths. The temp dir is always
  removed, on success or failure.
- On re-transcription, the same key is overwritten.
- `DELETE /uploads/{id}` also deletes the audio key. Transcript rows go
  with the upload through `CASCADE`.
- The WAV is persisted because M4 (diarization) consumes the same audio.

Alternatives rejected:
- **Decoding in-process with PyAV** (`faster_whisper.decode_audio`):
  the plan specifies ffmpeg, M8 needs a system ffmpeg anyway, and it
  would leave no WAV artifact for M4.
- **A temp-only WAV**: rejected by the owner in favour of reuse.

### 4. Handler: registry signature gets the DB session

`Handler = Callable[[Job, AsyncSession], Awaitable[None]]`.
`process_job` passes its session (a one-line change in `tasks.py`), and
`dummy.run` ignores it. This keeps 0006's test seam, where the session
factory comes from `ctx`. The rejected alternative, having handlers
open their own session, would bypass that seam.

`transcribe.run`:
1. Loads the upload.
2. Extracts the audio.
3. Runs Whisper in `asyncio.to_thread`. The lazy `segments` generator
   is consumed *inside* the thread, because that is where the real
   work happens.
4. Replaces the upload's segments and words in one transaction
   (delete, then bulk insert). This makes re-runs idempotent, and
   readers see either the old or the new transcript, never a mix.

The model is a lazily-loaded per-process singleton. An `asyncio.Lock`
serializes transcriptions, because arq allows 10 concurrent jobs and
two models would not fit in 3 GB. `WorkerSettings.job_timeout` is
raised from arq's 300 s default, which would kill any real episode.

### 5. `.srt` / `.vtt` endpoints: on demand, no blob cache

Routes: `GET /uploads/{id}/transcript.srt` and
`GET /uploads/{id}/transcript.vtt`, with Bearer auth.

| Case | Response |
|---|---|
| Any `succeeded` transcribe job exists | **200**, serves the current rows (an empty but valid file if there are 0 segments) |
| No `succeeded` job, but one is `queued`/`running` | **409** |
| No job, only `failed` jobs, or unknown upload | **404** |

During a re-run, the previous transcript keeps being served. Content
types are `application/x-subrip` and `text/vtt` (UTF-8), sent with
`Content-Disposition: attachment`.

**Divergence from `project-description.md` §2.1 step 5**, which says to
cache the files in blob storage. Generation is O(segments) and cheap,
and a cache would need invalidating on every re-transcription and
again when M4 adds speaker labels.

Format:
- One cue per segment, as the spec says. Word-level re-cueing is noted
  in brainstorm.
- Timestamps are rounded to whole milliseconds before splitting into
  hours, minutes, seconds and milliseconds, so `,1000` never appears.
- Negative timestamps are clamped to 0, and `end` is clamped so it is
  never before `start`.
- Blank lines inside a cue's text are collapsed, since they would end
  the cue early.
- Empty cues are skipped.
- VTT escapes `&`, `<` and `>`, which also neutralizes `-->`.

### 6. Frontend: download buttons only

The 0004 pivot left **no transcript screen and no transcript dummy
data** to swap. The only transcript UI is Manage's per-tool
"Transcript" cell. M3 adds `.srt`/`.vtt` download buttons there, shown
once a transcribe job has succeeded.

Downloads go through `HttpClient` with `responseType: 'blob'`, so the
auth interceptor supplies the token. This avoids 0005's
query-param-token workaround, which exists only for `<video src>`.

A transcript viewer/editor and progress display are out of M3's scope
(noted in brainstorm).

## Test plan

**Fixture**
- LibriSpeech dev-clean utterance `1272-128104-0009`, 18.3 s (openslr.org/12, **CC BY 4.0**, via the small `hf-internal-testing/librispeech_asr_dummy` subset rather than the 346 MB test-clean archive),
  muxed into a tiny MP4.
- Its reference transcript, taken from `*.trans.txt`.
- An attribution file, committed alongside both.

**Unit tests**
- SRT/VTT formatting on mock segments.
- Timestamp edge cases: carry, over 1 h, negatives.
- Escaping, blank-line collapse, empty input.

**API tests**
- 200, 409 and 404 as specified in §5, plus 401 without a token.

**Handler tests (mocked Whisper, real ffmpeg)**
- Segments and words are persisted in order.
- A re-run replaces rows instead of duplicating them.
- The WAV is stored and its key recorded.
- No temp files are left behind.
- A corrupt input ends in `failed`, with the ffmpeg error in
  `error_message`.

**Integration test**
- The real pipeline runs on the fixture with `base` on CPU.
- Accuracy: **WER ≤ 0.15** (jiwer, lowercased, punctuation stripped).
- Timestamps are monotonic, words fall inside their segments, and
  everything is within the audio duration.

**M2 regression**
- `test_worker.py`'s dummy-path tests move to a tool type that is still
  a dummy (`hook`). `transcribe` is real now, so they would fail on a
  fake storage key.
- A new real-arq-worker test uploads the fixture through the API and
  asserts `queued → running → succeeded` plus persisted segments.

**Measured accuracy:** the fixture's WER is **0.070** with both `base`
(CPU) and `large-v3-turbo` (GPU). The residual errors are normalization
differences such as "Michael Angelo" vs. "Michelangelo". The 0.15
threshold therefore has margin.

**Manual check**
- One real episode on the GTX 1060, recording RTF and VRAM in §1.
- `.srt`/`.vtt` downloaded from the Manage page.

## Consequences

- New runtime dependencies:
  - `faster-whisper`
  - CUDA 12 wheels: `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`
  - ffmpeg as a **system dependency**
- New dev dependency: `jiwer`.
- New modules: `app/transcription/` (audio, whisper, subtitles),
  `app/worker/handlers/transcribe.py`, `app/routers/transcripts.py`,
  `app/models/transcript.py`.
- `transcribe` no longer uses the dummy handler. The other 7 tool
  types still do.

## Post-implementation notes

- **Manual end-to-end check passed (2026-09-26).** The owner ran a real
  YouTube podcast episode through the Manage page. The transcript is
  correct and both `.srt` and `.vtt` download.
- **Bug found by that check, and fixed.** Non-ASCII titles (here a
  fullwidth `：`) crashed the download endpoints with
  `UnicodeEncodeError`, because header values must be latin-1.
  `Content-Disposition` now sends an ASCII `filename` fallback plus the
  real name in RFC 6266 `filename*=UTF-8''…`. A regression test covers
  it.
- **`test-runner` wasn't loaded in this session.** Its agent definition
  post-dated the session, so the suites were run directly with compact
  output. Final count: backend 78 passed, frontend 85 passed.
- **Dev convenience:** `scripts/dev.sh` starts the whole local stack
  in one terminal. It is tooling, not milestone scope.

