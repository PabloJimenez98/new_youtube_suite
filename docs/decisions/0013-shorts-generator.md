# Decision: Shorts Generator, Mechanical Layer (M8)

**Date:** 2026-10-16
**Status:** Implemented (M8 done, 2026-10-16). The
changes made during implementation are marked **(impl.)**.

## Context

- **Spec §2.5** describes the shorts tool: peak-finding on the
  relevance curve, boundary expansion, sentence snap, duration rules and
  dedup, an ffmpeg cut, subject detection, a smoothed crop, optional
  caption burn-in, and `clips` rows.
- **The plan's M8** asks for these as **separately testable units**,
  and warns that crop quality is the most likely "looks worse than
  expected" surprise.
- **The input curve is M7's v0** (0012), with a low quality ceiling.
  M8 has to tolerate it.
- **The crop spike** (`docs/research/m8-crop-spike.md`) measured
  YuNet and histogram shot cuts on episode 8. Those numbers set the
  thresholds below.
- **Decided with the owner (2026-10-16):**
  - **Job parameters:** a new `jobs.params` column. Manage offers an
    aspect-ratio select (default 9:16) and a captions checkbox.
  - **Up to 5 clips per run, 20–60 s each.**
  - **No face in a shot:** fit the whole frame over a blurred
    background.
  - **Karaoke captions:** the spoken word is highlighted.

## Decisions

### 1. Segment selection (`app/shorts/selection.py`, pure)

- **Input:** the relevance windows (start, end, score) and the
  transcript words.
- **Smoothing:** a 3-window moving average, to calm v0's noise.
- **Peaks:** `scipy.signal.find_peaks` on the smoothed curve, with
  prominence ≥ 0.05 and peaks ≥ 60 s apart.
- **Expansion:** outward from the peak while the smoothed curve stays
  above a floor, capped at 60 s.
  - **(impl.)** The floor sits 70 % of the way from the episode's
    **median** to the peak, not at 70 % of the peak's height. On
    episode 8 (mean 0.46, peaks ~0.78) the plain 70 % let all five
    clips run to the 60 s cap; with the median-relative floor they came
    out 31–53 s.
- **Sentence snap:**
  - the start moves back to the nearest sentence start (a word after
    `.?!`, or a segment start)
  - the end moves forward to the nearest sentence end
  - under 20 s → extended one sentence at a time, alternating sides
  - over 60 s → cut at the last sentence end ≤ 60 s, else at the hard
    60 s limit in a gap between words
- **Dedup:** a candidate overlapping a better one by > 30 % of the
  shorter span is dropped. The best 5 are kept, ranked by peak score.

### 2. Subject detection and framing (`app/shorts/framing.py`)

- **Detector:** YuNet (`cv2.FaceDetectorYN` from
  `opencv-python-headless`; the ~230 KB opencv_zoo ONNX model, MIT).
  - CPU only. It competes with neither the GPU (M3/M4) nor the shared
    LLM's RAM.
  - **The model path comes from the `SHORTS_FACE_MODEL_PATH` setting**;
    the README has the download line. Without the model the job fails
    fast.
- **Sampling:** PyAV decodes the span at a 320 px width.
  - Every frame gets an H-S histogram for cut detection.
  - Every 5th frame (5 fps at 25 fps) goes through YuNet, with score
    threshold 0.6.
- **Shots:** a Bhattacharyya distance > **0.25** between consecutive
  frames is a cut. In the spike, shots stayed ≤ 0.12 and cuts were
  0.37–0.48.
- **Subject:**
  - faces are scored `area × confidence`, halved when the face touches
    the frame edge (usually a partly visible foreground person)
  - within a shot, the candidate nearest the previous centre wins ties
    within 20 %
- **Smoothing:** an EMA (α 0.3) on the crop centre, **reset at every
  cut**, with a dead zone: no move under 5 % of the crop width.
- **Mode per shot:**
  - **`crop`** when ≥ 50 % of the shot's samples have a face: a
    full-height window of the target ratio, centred on the face and
    clamped to the frame
  - **`fit`** otherwise: the frame scaled to the output width over a
    blurred, cropped copy of itself
  - A target as wide as or wider than the source (16:9 on 16:9) is
    always `fit`, which then needs no padding.
- **Track:** `[{t, x, y, w, h}]` at 5 fps, in source pixels, stored as
  `crop_bounding_box_track` (spec §4). The shot list and modes are
  stored in `framing`.

### 3. Karaoke captions (`app/shorts/captions.py`, pure)

- **Cues** come from the transcript words inside the clip, with times
  relative to the clip:
  - 3–6 words and ≤ 28 characters per cue
  - break after `.?!,;:`
  - a pause over 0.7 s starts a new cue
- **ASS:**
  - DejaVu Sans Bold, white with a black outline, in the lower third,
    sized relative to the output height
  - **(impl.)** one `Dialogue` per word, each showing the whole cue
    with only that word in yellow (`{\c&H00FFFF&}`), from the word's
    start to the next word's start. `{\k}` tags were planned, but they
    fill cumulatively (every word already said stays coloured), and the
    owner asked for the spoken word to be highlighted.
- Untimed words inherit the previous word's time, as in M7.
- `{`, `}` and `\` in the text are replaced by `(`, `)` and `/`.
  ASS has no escape for override braces.

### 4. Render (`app/shorts/render.py`)

- **One ffmpeg pass per clip:** input `-ss`/`-to`, the framing filter
  graph, `ass=` when captions are on, and libx264 `veryfast` CRF 20 +
  AAC 128k + `+faststart`.
  - It always re-encodes: the source is AV1/Opus, and cropping needs a
    re-encode anyway.
  - **NVENC is left out of v1:** the GPU is M3/M4's, and CPU is
    simpler.
- **Moving crop:** `sendcmd` carries one `crop x/y` command per output
  frame, interpolated from the 5 fps track.
- **Mixed shots:** each run of same-mode shots is a `trim` + `setpts`
  branch with its own filter, joined with `concat`. The audio is taken
  whole from the input.
- **Output size (impl.):** the output's **short side** equals the
  source's short side, capped at 1080, and the long side follows the
  ratio (even numbers). A 720p source gives 9:16 at 720×1280, 1:1 at
  720×720, 4:5 at 720×900 and 16:9 at 1280×720. The 405 px crop is
  upscaled ~1.8×, but not to 1080×1920.
- **Files:** stored through `LocalStorageBackend` under
  `{uuid}_short_u{upload}_{rank}.mp4` (a flat key, as `_resolve`
  requires).
- **Check:** `ffprobe` after each render (duration ±0.5 s, size, video
  and audio streams). A bad file fails the job.

### 5. Job parameters (migration `0009`)

- **`jobs.params`:** JSONB, nullable.
- **Validation in `JobCreate`, by `tool_type`:**
  - `shorts` → `ShortsParams(aspect_ratio: "9:16" | "1:1" | "4:5" | "16:9" = "9:16", captions: bool = True)`
  - every other tool → params must be null (422 otherwise)
- **`JobOut`:** exposes `params`.
- **Re-runs** keep using the job's own params.

### 6. `clips` table (migration `0009`)

- **The spec §4 columns:**
  - `upload_id` (FK, CASCADE)
  - `start_seconds`, `end_seconds`
  - `aspect_ratio` (CHECK on the 4 values)
  - `caption_burned_in`
  - `storage_path`
  - `crop_bounding_box_track` JSONB
- **Added:**
  - `rank`
  - `score` (peak height)
  - `peak_window_index`
  - `model_version_id` (FK to the relevance version used; this is
    §2.5's "source RelevanceScore window references")
  - `framing` JSONB
  - `created_at`
- unique (`upload_id`, `rank`)

### 7. Handler and API

- **Handler** `app/worker/handlers/shorts.py`:
  - needs relevance scores, otherwise "Run Relevance first"
  - selection → for each span: framing, ASS, render, in
    `asyncio.to_thread`, without the GPU lock
  - `progress_pct` advances per clip
  - **a re-run replaces** the rows in one transaction, and deletes the
    old files only after the new ones are written
- **Deleting an upload** deletes its clip files too.
- **A curve with no peak** fails the job with a clear message rather
  than succeeding with no clips.
- **API:**
  - `GET /uploads/{id}/clips` (404 for a missing upload or when no
    `shorts` job has succeeded; 409 while one is queued or running, as
    chapters)
  - **(impl.)** `GET /clips/count` for the dashboard tile; the dev proxy
    forwards `/clips`
  - `GET /clips/{id}/file`: Range streaming with query-param auth
    (0005). The Range helpers move to `app/routers/media.py` so the
    uploads and clips routers share them. `?download=true` adds a
    `Content-Disposition` for the download link.

### 8. Frontend (Manage)

- **The Shorts tool row:**
  - an aspect-ratio select and a "Captions" checkbox next to Run
  - Run is disabled without relevance scores
- **A shorts panel:**
  - a vertical `<video>` per clip, with time range, score and a
    download link
  - "Show in episode", which seeks the row's player (M7's `seekTo`)
- **Dashboard:** "Shorts generated" shows the real count instead of the
  dummy (the 0004 pivot flag).

## Alternatives rejected

- **YOLOv8 / ultralytics** (Autocrop-vertical, openshorts):
  - AGPL-3.0, which this project would inherit
  - heavier
  - a person box frames worse than a face for talking heads
  These projects stay references for the algorithm.
- **MediaPipe:** another large dependency for the same face boxes.
- **A static crop per shot:** simpler, but subjects lean and gesture,
  and a smoothed track costs little more.
- **Python per-frame cropping** (OpenCV writing frames to an ffmpeg
  pipe): flexible, but slower, and more code than `sendcmd`.
- **Fixed 9:16 with captions always on:** less work, but the spec makes
  both user choices.
- **Simple phrase captions:** the owner chose karaoke.

## Test plan

- **Unit, pure:**
  - **selection:** an isolated peak; peaks < 60 s apart; prominence
    below the threshold; 70 % expansion; snap in both directions;
    extension to 20 s; the 60 s cut without a sentence end; overlap
    dedup; the cap of 5; a flat or empty curve; peaks at the episode
    edges
  - **framing (fake detections):** cut detection; face choice with the
    edge penalty; EMA reset at cuts; the dead zone; clamping; crop vs.
    fit by coverage; geometry for each ratio; sendcmd interpolation
  - **captions:** cue grouping (size, punctuation, pause); relative
    times; `\k` tags summing to the cue duration; escaping; untimed
    words; accents
- **Render integration** (real ffmpeg, a synthetic `testsrc` + sine
  fixture): 9:16 crop with captions; fit; mixed crop/fit; 16:9. Each
  checks duration, size and the audio stream.
- **Detector integration** (skips without the model): YuNet finds the
  face in `tests/fixtures/face.jpg` (scikit-image's public-domain
  `astronaut`), and analysis + render follow it on a synthetic
  face-then-cut video. The M3/M4 fixtures have no faces.
- **Handler:**
  - rows with the version FK
  - a re-run replaces rows and files
  - "Run Relevance first"
  - a missing model fails fast
  - params reach the pipeline
  - an upload delete removes the files
- **API:** job params (valid; 422 for an unknown ratio or params on
  another tool); clips 200/404/409; the file with Range 206 and auth.
- **Migration:** `0009` up and down.
- **Frontend (vitest):** params are sent; Run is disabled without
  relevance; the gallery shows the videos; "Show in episode" seeks; the
  dashboard count.
- **Regression:** the full backend, frontend and experiment suites.
- **Manual QA on episode 8:**
  - 9:16 with captions, and 1:1 without
  - time and worker RSS recorded
  - the owner judges the framing, cuts, captions and clip choice
  - the plan asks for 2–3 episodes, but only episode 8 has relevance
    scores

## QA run (2026-10-16)

- **Setup:** real API + arq worker on the owner's 64-min episode
  (upload 8: 720p AV1/Opus, multi-camera). Jobs were created through
  `POST /uploads/8/jobs`; a 21:9 request got a 422 as designed.
- **Clips (both runs pick the same spans):**

  | Rank | Source | Length | Peak | Shots |
  |---|---|---|---|---|
  | 1 | 33:59 | 31.1 s | 0.78 | 10 |
  | 2 | 31:21 | 52.7 s | 0.68 | 14 |
  | 3 | 57:37 | 53.2 s | 0.67 | 6 |
  | 4 | 40:14 | 38.4 s | 0.66 | 7 |
  | 5 | 56:21 | 31.4 s | 0.64 | 6 |

  - Every shot was framed in `crop` mode (faces in every shot); `fit`
    was exercised only by the tests.
- **Time and memory:**
  - 9:16 with captions: **55 s** for all five clips
  - 1:1 without captions: **45 s**
  - worker RSS 1.8 GB (no LLM loaded)
  - analysis ~4–7 s per clip, encode ~4–5 s per clip
- **Bug found and fixed (impl.):** in clip 4 a blurred foreground hand
  was detected as a face. It scored 0.6–0.7 with three times the real
  face's area, 12 px in from the frame edge, so the 1 px edge penalty
  missed it. The crop swung between hand and speaker.
  - Fix: faces are weighted by `area × confidence⁴`, and the edge band
    is 3 % of the width. A regression test uses the measured boxes.
  - After the fix the shot holds on the speaker.
- **Re-runs** removed the previous run's files: only the five current
  clips are left on disk.
- **Owner's review (2026-10-16):** approved ("veo todo correcto"):
  framing, cuts, captions and clip choice.
- **Not done:** 2–3 episodes. Only upload 8 has relevance scores.

## Consequences

- **`jobs.params` is generic.** Later tools can use it (e.g. M4's
  speaker-count hint, noted in brainstorm).
- **New dependency:** `opencv-python-headless`, and a model file
  outside git under `backend/models/shorts/`.
- **Speaker-aware framing** (picking the active speaker's face from
  diarization) is not in v1. It is noted in brainstorm.
