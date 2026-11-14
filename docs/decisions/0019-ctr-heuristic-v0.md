# Decision: Title/thumbnail CTR predictor v0, heuristic only (M13)

**Date:** 2026-11-08
**Status:** Approved 2026-11-08 as proposed.

## Context

- **What the plan asks for in M13:**
  - a **heuristic-only** v0: no trained fusion MLP, because there are no labelled
    (title, thumbnail, real-CTR) pairs yet
  - an engineered feature set and a weighted-combination scorer
  - feature extractors:
    - title: length, word count, numbers, question
    - thumbnail: face count and size via a lightweight detector
    - thumbnail: text area via OCR
  - persistence in `ctr_predictions`
  - testing: unit tests per extractor, and manual QA of the rankings against intuition
- **Scheduling:** M13 starts while M12 waits for its live phase 2 (≥ 2026-11-10 12:38 UTC). The
  plan allows it: "technically independent of M10–M12 for the v0 heuristic version — can be
  pulled earlier if useful for scheduling". The owner confirmed on 2026-11-08. M12 stays
  `In progress` until phase 2.
- **Decided with the owner (2026-11-08):**
  1. **Input:** a per-video panel on the Manage page.
     - Titles are pre-filled from the M6 show-notes `title_suggestions`; the owner can edit or
       add them.
     - Thumbnails are uploaded by the owner.
     - The existing `ctr` job scores the candidates.
     - Standalone candidates (`upload_id` NULL) are deferred.
  2. **Text area:** text **detection** only, with no recognition. It uses OpenCV's DB detector
     and the opencv_zoo PP-OCRv3 model, so there is no new pip dependency.
  3. **Weights:** set after a research pass. The findings are in
     `docs/research/ctr-heuristic-evidence.md`.

### What the research changes

There is **no measured CTR evidence** for any feature, and most sources measure views or
breakouts. So:

- Only features with at least a platform reason or a consistent weak signal get a
  **non-zero weight**:
  - title length
  - face presence/count
  - face size
  - text coverage
- Every other feature is **computed, stored and shown, with weight 0**:
  - digits
  - `?`, `!`, brackets
  - ALL-CAPS words
  - "you"
  - word count
  - brightness, contrast, saturation
  - text-box count

  The values stay in the stored feature vector, so M15 (or a later manual calibration against
  `analytics_reach_snapshots`) can learn their weights from our own CTR, instead of us
  guessing a direction now.
- The score is **relative within one upload** (spec §2.7), which also removes video-level
  confounders (topic, channel size). It is not a CTR %.
- **Expected limitation:** the owner's candidates will often sit in the same length band, so
  title-only differences will often tie. The panel says so, rather than inventing a
  difference.

## Decisions

### 1. Features (`app/ctr/features.py`, pure functions)

**Title** (no model):

| Feature | Definition | Weight |
|---|---|---|
| `title_chars` | `len(title.strip())` | scored (below) |
| `title_words` | whitespace tokens | 0 |
| `has_digit` | any `\d` | 0 |
| `has_question` | contains `?` or `¿` | 0 |
| `has_exclamation` | contains `!` or `¡` | 0 |
| `has_brackets` | `()` `[]` `{}` pair | 0 |
| `caps_words` | words of ≥ 2 letters, all upper-case | 0 |
| `second_person` | `you`/`your` (EN) or `tú`/`tu`/`te`/`usted` (ES), whole words | 0 |

**Thumbnail:** the image is decoded with OpenCV and resized to fit 1280 × 720.

| Feature | Definition | Weight |
|---|---|---|
| `face_count` | YuNet faces with score ≥ 0.6 (the M8 detector and `SHORTS_FACE_MODEL_PATH`) | scored |
| `largest_face_area` | the biggest face box ÷ image area | scored |
| `text_area` | the union area of DB text boxes ÷ image area | scored |
| `text_boxes` | the number of text boxes | 0 |
| `brightness` | mean of the V channel (HSV), 0–1 | 0 |
| `contrast` | standard deviation of luminance, 0–1 | 0 |
| `saturation` | mean of the S channel, 0–1 | 0 |
| `aspect_ratio` | width ÷ height | 0 (shown with a warning if it is not ≈ 16:9) |

**Text detector:**
- `cv2.dnn.TextDetectionModel_DB` with `text_detection_en_ppocrv3_2023may.onnx` (2.3 MB,
  Apache-2.0, opencv_zoo).
- The parameters are the opencv_zoo demo's: input 736 × 736, binary threshold 0.3, polygon
  threshold 0.5, max candidates 200, unclip ratio 2.0, mean `(123.675, 116.28, 103.53)`,
  scale `1/255/(0.229, 0.224, 0.225)`.
- The model path comes from a new setting, `CTR_TEXT_MODEL_PATH`, defaulting to
  `models/ctr/text_detection_en_ppocrv3_2023may.onnx`. The README gets a one-line download step,
  as for YuNet.
- A missing model makes the job fail clearly ("set CTR_TEXT_MODEL_PATH …"), as shorts does.

### 2. Scorer (`app/ctr/scorer.py`, pure, version `heuristic-v0`)

Each scored feature maps to a sub-score in [0, 1] through a piecewise-linear shape. The total is
the weighted sum × 100.

| Sub-score | Weight | Shape (all priors; see the research note) |
|---|---|---|
| title length | **0.35** | 0 at ≤ 5 chars; rises to 1.0 at 40; 1.0 for **40–60**; falls to 0.4 at 100 (past truncation, not rejected). Titles over 100 chars are rejected at input (YouTube's cap). |
| faces | **0.30** | 0 faces → 0.2; 1 → 0.9; **2 → 1.0** (two-host podcast prior); 3+ → 0.9 |
| face size | **0.20** | no face → 0; rises to 1.0 at **10 %** of the frame; 1.0 for 10–35 %; falls to 0.6 at 60 %+ |
| text coverage | **0.15** | 0 % → 0.7 (neutral); **2–20 % → 1.0**; falls to 0.3 at 50 %+ |

- **Title-only candidate** (no thumbnail yet):
  - the score uses the title sub-score alone, rescaled to 0–100;
  - it is flagged `partial`, and the panel ranks partial candidates in their own group, because
    they can't be compared fairly with full ones.
- **Explainability:** each row stores every feature and every sub-score, and the panel shows
  them: "faces 1/1.0 · text 4 % · 52 chars".
- **The weights live in code**, in one table. They are also copied into the model version's
  `architecture_notes`, so a stored score can be traced to the formula that produced it.

### 3. Data model: `ctr_predictions` (spec §4.1, plus additions)

| Column | Type | Notes |
|---|---|---|
| id | PK | |
| upload_id | FK → uploads, ON DELETE CASCADE | nullable as in the spec, but **required by the M13 API**; standalone is deferred |
| candidate_title | TEXT | 1–100 chars |
| candidate_thumbnail_path | TEXT NULL | storage key; NULL = title-only |
| predicted_score | FLOAT NULL | 0–100; NULL until scored, or after an edit |
| features | JSONB NULL | **addition**: raw features + sub-scores, as `hook_scores.feature_vector` has |
| partial | BOOLEAN | **addition**: scored without a thumbnail |
| model_version_id | FK → model_versions NULL | |
| source | TEXT (`manual`/`show_notes`) | **addition**: where the title came from |
| scored_at, created_at, updated_at | TIMESTAMPTZ | |

- **Candidates exist before they are scored.** The row is both the candidate and its latest
  prediction. Editing the title or the thumbnail clears `predicted_score`, `features` and
  `scored_at`, so a stale score is never shown.
- **History:** a re-run overwrites the score. Keeping every past prediction isn't needed for
  v0, and M15 recomputes features anyway.
- **Model version:** `model_versions` gets (`ctr`, `heuristic-v0`, `is_live=true`) via the M7
  get-or-create helper, with the weight table in `architecture_notes`.

### 4. API and job

- **Candidate CRUD**, under the uploads router, app auth:
  - `GET /uploads/{id}/ctr-candidates`: the list, ranked by score (full first, then partial,
    then unscored)
  - `POST /uploads/{id}/ctr-candidates`: multipart; `title` is required and `thumbnail` is
    optional
  - `PATCH /ctr-candidates/{cid}`: change the title and/or replace the thumbnail
  - `DELETE /ctr-candidates/{cid}`
  - `GET /ctr-candidates/{cid}/thumbnail`: serves the stored image
  - `POST /uploads/{id}/ctr-candidates/from-show-notes`: creates title-only candidates from the
    latest draft's `title_suggestions`, skipping titles already present
- **Thumbnail validation:**
  - JPEG, PNG or WebP, sniffed from the bytes, not the extension;
  - ≤ 2 MB, YouTube's own thumbnail limit;
  - must decode with OpenCV;
  - otherwise 422 with the reason.
- **At most 10 candidates per upload:** a comparison UI, not a gallery.
- **Job:**
  - `POST /uploads/{id}/jobs {tool_type: "ctr"}` with no params, replacing the dummy handler;
  - **(re)scores every candidate of the upload**: the scoring is deterministic and fast, at
    about 100 ms per image on CPU;
  - no candidates → the job fails with "add candidates first";
  - an unreadable thumbnail fails that candidate's score (`features.error`), not the job.
- **CPU only.** It doesn't touch the GPU lock (decision 0016) and adds no LLM.

### 5. Frontend: a "CTR" panel in the Manage video row

- **Location:** next to the relevance, shorts and show-notes panels, opened from the existing
  `ctr` tool button.
- **Content:**
  - the candidate list (thumbnail preview, editable title, character count against 100, delete)
  - "Add candidate" and "Add titles from show notes"
  - "Score" (runs the job and polls like the other tools)
  - the ranking: the score as a bar, its rank, the partial/unscored badges, and an expandable
    breakdown of the sub-scores plus the zero-weight features
- **Wording:** the score is labelled **"relative score (heuristic v0), not a CTR %"**.
- **Dashboard tile:** the "Avg. predicted CTR" tile stays dummy. Wiring it is out of scope.

### 6. Code layout

- `app/ctr/`: `features.py`, `detectors.py` (YuNet + DB wrappers, lazily loaded),
  `scorer.py`, `pipeline.py`
- `app/models/ctr.py`
- migration `0013_create_ctr_predictions.py`
- `app/routers/ctr.py`
- `app/worker/handlers/ctr.py`
- `frontend/src/app/manage/ctr-panel/`

## Test plan

**Unit (default suite):**

- **Title features:** a table of cases:
  - empty and whitespace
  - digits, `?`/`¿`, `!`/`¡`, brackets
  - caps words (not single letters, not "I")
  - second person in EN and ES, whole words only ("young" doesn't count)
  - emoji and accents counted as characters
  - 100-char limit
- **Scorer shapes:**
  - each piecewise function at its breakpoints and in between
  - monotonic where intended
  - weights sum to 1
  - a title-only candidate → partial, scored from the title alone
  - ranking order: full → partial → unscored
  - a deterministic tie stays a tie (no hidden tiebreak)
- **Image features** (fixtures, no network):
  - `tests/fixtures/face.jpg` (M8) → `face_count ≥ 1` and a plausible face area
  - a synthetic blank image → 0 faces, 0 text
  - a synthetic image with large `cv2.putText` text → `text_area` above the threshold
  - brightness/contrast/saturation on solid-colour images
  - the aspect ratio
  - These skip, with a reason, without the ONNX models, as M8's detector tests do.
- **Validation:** wrong type (sniffed), > 2 MB, an undecodable file, an 11th candidate,
  a title over 100 chars → 422.
- **Endpoints:**
  - CRUD, with auth required
  - an edit clears the score
  - `from-show-notes` dedupes and handles no draft (404)
  - the thumbnail is served
  - deleting an upload cascades to its candidates and thumbnail files
- **Handler:**
  - scores every candidate and stores `features`, `model_version_id` and `scored_at`
  - a re-run is idempotent
  - no candidates → a failed job with the message
  - one bad image → that candidate errors and the others score
  - a missing model path → a clear failure
- **Migration 0013:** upgrade/downgrade, the FK cascade.
- **Frontend:**
  - the panel's empty, list, scoring and ranked states
  - partial/unscored badges
  - the breakdown
  - the title counter
  - "from show notes"
  - the service calls

**Manual QA (owner, the plan's item):** on 2–3 real episodes, enter 3–5 candidate
(title, thumbnail) sets, rank them yourself first, then compare with the heuristic's order.
Note where it disagrees and why. The breakdown makes that easy. The findings go in this record;
they may adjust the shapes, but not the scope.

**Regression:** the full CPU suite + `ng test`.

## Alternatives rejected

- **Directional weights for digits, questions, caps and "you":** no CTR evidence either way.
  Guessing would bake in folklore; logging them with weight 0 keeps the option open (research
  note).
- **Full OCR (RapidOCR):** the owner chose detection only. Recognised text (for example,
  overlap with the title) is a later addition.
- **A standalone page (`upload_id` NULL):** deferred by the owner.
- **A synchronous scoring endpoint:** fast enough, but the Manage page already runs every tool
  as a job with progress and errors. Consistency wins.
- **A separate `ctr_candidates` table:** one row as candidate + latest prediction is enough
  for v0. The spec's `ctr_predictions` already carries the candidate columns.
- **SigLIP-2 / BGE embeddings now:** they are the trained version's inputs, which is M15's
  territory per the plan.

## Implementation notes (2026-11-08)

### Deviations and additions to the plan above

- **The candidates panel is always shown** in the Manage row, because candidates exist
  before any job runs.
  - The row's existing CTR **Run** is the "Score" button. It is blocked with "Add
    title/thumbnail candidates first" while the panel holds none.
  - The panel refetches when a newer `ctr` job succeeds.
- **Thumbnails in `<img>`:** `GET /ctr-candidates/{id}/thumbnail` accepts `?token=`, like the
  video and clip streams (decision 0005), because an `<img>` can't send a header.
  `/ctr-candidates` is added to the dev proxy.
- **`get_or_create_version`** (M7, `handlers/relevance.py`) now takes `model_name`, defaulting
  to `relevance`. The CTR handler uses `model_name="ctr"`.
- **Detector models are checked only when a thumbnail needs them.** Title-only candidates
  score even if the ONNX files are missing.
- **`PATCH` with an unchanged title keeps the score.** Only a real change clears it. There is
  also `remove_thumbnail=true`, to turn a candidate back into title-only.
- **Titles are whitespace-normalised** (runs of spaces collapse) before the 100-char check.
  The same applies to show-notes suggestions: duplicates are matched case-insensitively, and
  over-long suggestions are skipped.
- **Ties share a rank** in the API (`rank`), as planned: "a tie is a tie".
- **Unrelated fix:** the M12 models (`reporting_reports.job_id`,
  `analytics_reach_snapshots.source_report_id`) now declare `BigInteger` on their FKs, matching
  migration 0012 and the other models. This changes no schema.

### Tests

- **`tests/test_ctr.py` (76):**
  - title-feature table
  - scorer shapes at breakpoints
  - the weighted sum
  - partial rescaling
  - zero-weight features leave the score unchanged
  - pure image stats and box union
  - validation (sniffed type, 2 MB, undecodable)
  - the real detectors on a blank image, big text, the M8 face fixture and down-scaling
    (these skip without the models)
  - endpoints: auth, CRUD, the 10-candidate cap, edits clearing the score, `from-show-notes`
    dedupe, ranking groups and ties, upload deletion removing thumbnail files
  - handler: no candidates, title-only without models, missing model, real scoring with a
    broken thumbnail isolated, idempotent re-run
- **`test_0013_…` in `tests/test_migrations.py`:** the FK cascade and the `source` CHECK.
- **Frontend:**
  - `manage/ctr-panel/ctr-panel.spec.ts` (12): the service, the panel states, the breakdown,
    add/edit/delete, from-show-notes, the refetch on a new job, the 10-candidate cap
  - two CTR cases in `video-row.spec.ts`
- **Regression (2026-11-08, WSL clone, with `.env` and the models):**
  - backend: 611 passed, 0 failed (10 opt-in deselected)
  - frontend: 169/169
- **Manual QA:** not done at closing (see below).

## Milestone closed (2026-11-08)

The owner closed M13 on the automated test results. The plan's manual QA (comparing the
heuristic's ranking with the owner's own ranking on real episodes) is a follow-up (**closed 2026-11-14**: the owner confirmed the manual QA),
not a blocker:

- Candidate frames for it were extracted outside the repo, in
  `C:\Users\Pablo\Downloads\ctr-qa-frames\` (1X09, 1X06, 1X28). They include the published
  thumbnail, a one-face close-up, a two-face frame (1X06 only), a neutral frame, and a
  synthetic heavy-text variant.
- Most sampled frames had exactly one face: the show is cut as single-camera close-ups. The
  face-count dimension will vary less than expected on this channel.
- **Things to check:** whether the studio's neon sign and the SER logo count as "text on
  thumbnail". In 1X09 the published thumbnail exists only at 480×360 (4:3), so the aspect
  warning shows.
- Findings go here, and may adjust the shapes.
