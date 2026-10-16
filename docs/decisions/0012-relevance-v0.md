# Decision: Relevance Model v0, Heuristics + Batched Local LLM (M7)

**Date:** 2026-10-16
**Status:** Implemented (M7 done, 2026-10-16).

## Context

- **Spec §2.4** describes the relevance model: a trained BiLSTM over
  frozen embeddings, scoring 10–20 s transcript windows in `[0, 1]`.
- **The plan's M7** ships only the **bootstrap** version: heuristic
  rules plus a zero-shot LLM pass. The trained model is M15's, and it
  learns from these scores as pseudo-labels.
  - The quality ceiling is low **by design** (M7 risk note), and M8
    must tolerate it.
- **What "relevant" means here:** a moment that works as a
  **standalone short**: interesting, self-contained, engaging (funny,
  insightful, controversial, emotional). That is what M8 needs.
- **Decided with the owner (2026-10-16):**
  - **Local LLM, batched:** the shared Qwen3.5-4B (0011 §1). One call
    per window would take over an hour per episode on CPU.
  - **`model_versions` now:** the spec §4 table is created in M7, and
    `relevance_scores` references it.
  - **UI:** a relevance curve plus the top 5 moments in Manage, which
    seek the video on click. This goes beyond the plan's M7 bullet,
    approved by the owner, to make the manual review practical.

## Decisions

### 1. Windows (`app/relevance/windows.py`, pure)

- **Base blocks:** 7.5 s, built by walking word timestamps, not
  Whisper's segments.
  - An untimed word joins the block of the previous timed word.
  - Silent gaps give empty blocks.
  - A segment without words is spread over its own span as one atom.
- **Windows:** 15 s (2 consecutive blocks) with a 7.5 s hop, so
  **50 % overlap**, inside the spec's 10–20 s range.
  - ~510 windows for a 64-min episode.
  - The last window may be shorter, ending at the episode's duration.
- **Speakers:** each block keeps the `speaker_label` of its words, when
  diarization has run.

### 2. Heuristics (`app/relevance/heuristics.py`)

Each feature is in `[0, 1]`, with English and Spanish word lists. The
language comes from fastText LID, as in 0010/0011.

| Feature | How | Weight |
|---|---|---|
| Topic anchor | density of the episode's top-30 keywords (TF-IDF across windows, en/es stopwords removed) | 0.35 |
| Question → answer | a sentence ending in `?` (Whisper punctuates questions in en and es) followed by ≥ 8 words, in the window or the next one | 0.25 |
| Speaker changes | speaker changes per minute, saturating at 6/min | 0.20 |
| Fillers | `1 −` the filler ratio (`eh`, `o sea`, `um`, `you know`…), saturating at 15 % | 0.20 |

- **Without diarization:** the speaker feature is left out and the
  other weights are renormalized.
- **A window with < 5 words** scores 0.
- **Topic anchor scaling:** divided by the episode's maximum, so the
  densest window is 1.

### 3. Batched zero-shot LLM (`app/relevance/llm_scoring.py`)

- **Unit:** **non-overlapping 15 s blocks** (~256 per 64 min),
  **20 per call** (~13 calls).
- **Output:** grammar-constrained JSON `{"scores": [int × N]}`. The
  grammar enforces N and integers; the 0–100 range is not enforced, so
  out-of-range values are clamped locally.
- **Prompt:** in the episode language. Numbered blocks, each with its
  chapter title when chapters exist. It asks how well each block would
  work as a standalone short.
- **Windows:** a 15 s window with a 7.5 s hop overlaps at most two LLM
  blocks; its LLM score is their overlap-weighted mean, scaled to
  `[0, 1]`.
- **Retry/repair (same policy as 0011 §4):**
  - an invalid JSON or a wrong count → retried: greedy, then sampled
    with seeds 1 and 2
  - a batch that fails all 3 attempts → its windows get
    `llm_score = NULL` and use the heuristic alone (degraded, not
    failed)
  - **every** batch failing → the job fails
- **Runtime:** the shared Llama (0011 §1). `LlamaChat` moves from
  `app/shownotes/pipeline.py` to `app/llm/chat.py` so both tools use
  it. Model and LID come from the chapters bundle; without one the job
  fails fast.
- **Expected time:** ~5–10 min per 64-min episode, to be measured.
  **Measured (2026-10-16, QA run):** 1010 s (~17 min), about twice the
  estimate. See "QA run" below.

### 4. Combination

- `score = 0.4 · heuristic + 0.6 · llm`, or the heuristic alone when
  `llm_score` is NULL.
- The weights are versioned constants: a change after the manual
  review bumps the version label.

### 5. Schema (migration `0008`)

- **`model_versions`** (spec §4):
  - `id`, `model_name` (relevance/ctr/hook), `version_label`,
    `architecture_notes`, `training_data_reference` JSONB, `metrics`
    JSONB, `artifact_path` (nullable), `is_live`, `trained_at`
    (nullable), `created_at`
  - unique (`model_name`, `version_label`)
  - a partial unique index: **one `is_live` row per `model_name`**
- **M7's row:** the handler get-or-creates `relevance` /
  `heuristic-v0+<gguf stem>+relevance-p1`.
  - It is created live when no relevance version is live yet.
  - `architecture_notes` / `training_data_reference` record the
    heuristics, weights and prompt version.
  - **Switching the live version (rollback, A/B) stays M15's.**
- **`relevance_scores`:**
  - `id`, `upload_id` (FK, CASCADE), `window_index`,
    `window_start_seconds`, `window_end_seconds`, `score`,
    `heuristic_score`, `llm_score` (nullable), `model_version_id` (FK)
  - unique (`upload_id`, `window_index`)
  - **The components are stored apart** because M15 needs them as
    pseudo-labels.
- **Chapters and show notes** keep their text `model_version` /
  `llm_model_used` columns: they aren't relevance/ctr/hook models.

### 6. Handler and API

- **Handler** `app/worker/handlers/relevance.py`, the chapters pattern:
  - needs a succeeded `transcribe` job and segments, otherwise
    "Run Transcript first"
  - uses diarization and chapters when they exist
  - CPU in `asyncio.to_thread`, no GPU lock
  - replaces the upload's rows in one transaction
- **`GET /uploads/{id}/relevance`** (404/409 as chapters) returns:
  - `version`: the model version's label (not `model_version`, a
    prefix pydantic reserves)
  - `windows`: `start`, `end`, `score`
  - `top_moments`: the 5 best windows at least 60 s apart, with their
    text
  - This is **display only**. Real peak-finding is M8's.

### 7. Frontend (Manage)

- `needsTranscript` covers `relevance`.
- **A relevance panel:**
  - an SVG curve on a fixed 0–1 scale with a time axis
  - the 5 top moments with timestamp, score and trimmed text
  - a click seeks the row's `<video>` to that moment
  - the model version label

## Alternatives rejected

- **One LLM call per window:** ~500 calls, over an hour per episode on
  CPU.
- **LLM only on the heuristic top-k:** faster, but the LLM could never
  rescue a moment the heuristics miss.
- **Heuristics only:** seconds per episode, but a very low ceiling, and
  M15 would get no LLM pseudo-labels. It also departs from the plan.
- **A text `model_version` column** (as in M5/M6): simpler, but M15
  needs `is_live`, and the owner preferred to have the table now.
- **No UI:** respects the bullet literally, but makes the manual review
  much harder.

## Test plan

- **Unit:**
  - windows: overlap/hop, untimed words, gaps, an episode shorter than
    one window
  - each heuristic with en and es cases; renormalization without
    diarization; combination; empty windows
  - LLM scoring with a fake: batches of 20, blocks → windows mean,
    invalid JSON or wrong count → retry, an exhausted batch → NULL and
    degraded, every batch failing → error
  - `model_versions`: get-or-create is idempotent; the partial index
    rejects a second live row
- **Handler (fake pipeline):** rows persisted with the version FK; a
  re-run replaces them; "Run Transcript first"; a missing bundle fails
  fast; diarization labels reach the pipeline.
- **Migration:** `0008` up/down.
- **API:** 200 (top moments ≥ 60 s apart, ordered), 404, 409, auth.
- **Integration (skip-if-absent bundle):** the real LLM scores a batch
  from the M3 fixture and returns N valid integers.
- **Frontend (vitest):** curve renders; moments listed; a click seeks
  the video; Run disabled without a transcript.
- **Regression:** full backend, frontend and experiment suites.
- **Manual review protocol:** before seeing results, the owner notes
  ~5 moments of episode 8 they would clip. Then we measure how many
  fall in the curve's top 20 %, and the owner judges the top 5 moments.
  The plan asks for 2–3 episodes, but only episode 8 has a transcript.

## QA run (2026-10-16)

- **Setup:** a real `relevance` job (job 13) on the owner's 64-min
  Spanish episode (upload 8, diarized, 6 chapters), through the arq
  worker.
- **Result:**
  - succeeded in 1010 s (~13 LLM calls at ~75 s each), 0 batch retries,
    0 NULL `llm_score`
  - worker RSS 5.9 GB (the same shared Llama as show notes)
  - 512 windows; score mean 0.461, min 0.134, max 0.813
  - `model_versions`: `relevance` /
    `heuristic-v0+Qwen3.5-4B-Q4_K_M+relevance-p1`, live
- **Top 5 moments:** 34:07 (0.81), 57:52 (0.73), 31:52 (0.72),
  40:37 (0.67), 56:30 (0.66).
- **Owner's review (2026-10-16):** the top 5 moments are correct;
  approved.
- **Findings:**
  - **Heuristic vs. LLM correlation across windows: −0.18.** The two
    signals barely agree. Not tuned in v0; worth revisiting in M8 if
    its clip picks look off, and relevant for M15's pseudo-labels.
  - **Not done:** the owner's blind "~5 moments first" step (the review
    went straight to the top 5), so the top-20 % hit rate was not
    measured; and a second/third episode, since upload 8 is the only
    one with a transcript.

## Consequences

- **Spec §2.4 inference steps 2–4** (BGE + BiLSTM) are not in v0. They
  arrive in M15.
- **`model_versions` exists from M7.** M15 adds new versions and the
  live switch on top of it.
- **For M15:** `bge-base-en-v1.5` is English-only and episodes are
  en/es (noted in `brainstorm.md`).
