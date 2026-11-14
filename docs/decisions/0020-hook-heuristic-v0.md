# Decision: Hook-strength scorer v0, heuristic only (M14)

**Date:** 2026-11-08
**Status:** Approved 2026-11-08 as proposed.

## Context

- **What the plan asks for in M14:**
  - a **heuristic-only** v0, because LightGBM needs labelled data that doesn't exist yet
  - the full engineered feature list from spec §2.8, as the implementation checklist
  - feature extractors:
    - question and exclamation counts
    - superlative and numeric-claim detection
    - direct address and imperatives
    - a sentiment/emotion classifier
    - pacing from word timestamps
    - keyword overlap with the title
  - a weighted heuristic scorer, with persistence in `hook_scores`
  - testing: unit tests per extractor, and manual QA on real clips from M8/M9
- **This is the bootstrap version, not the final model.** The LightGBM pairwise ranker from
  spec §2.8 step 3 is the fast-follow, in M15 territory. It will train on the stored
  `feature_vector`s once retention or performance data exists for the clips.
- **Scheduling:** M14 starts while M12 waits for its live phase 2 (≥ 2026-11-10 12:38 UTC).
  - M14 depends only on M8 (`clips`) and M3 (transcripts).
  - The owner confirmed on 2026-11-08.
  - M12 stays `In progress` until its phase 2 runs.
- **Decided with the owner (2026-11-08):**
  1. **Emotion:** use the multilingual `MilaNLProc/xlm-emo-t` (XLM-T, 19 languages including
     ES and EN, labels joy/anger/fear/sadness, MIT code) instead of the spec's English-only
     `j-hartmann/emotion-english-distilroberta-base`, because the episodes are Spanish.
  2. **NLP:** add **spaCy** with `es_core_news_sm` and `en_core_web_sm`, used for sentence
     splitting, POS/morphology (imperatives, superlatives) and NER.
  3. **Trigger:** the existing `hook` tool is its own job. It scores every clip of the upload,
     and the scores show on the clip cards of the Shorts panel. This is the CTR pattern.

### Evidence

- **No measured evidence links any engineered feature to retention.** The M14 prior-art pass
  (`docs/research/ctr-hook-prediction-prior-art.md` §2) found no paper, dataset or model for
  scoring an opening's retention. Retention curves are proprietary platform data.
- A 2026-11-08 search for newer evidence on Shorts openings turned up only creator and vendor
  blogs:
  - **The metric is real:** YouTube Studio's "viewed vs swiped away" is the share of Shorts
    impressions that were watched rather than swiped away at once.
  - **Everything else is unverified:** YouTube publishes no benchmarks. The thresholds and
    figures in circulation (question hooks, "pain-point" openings +23 %) trace back to unnamed
    studies.
  - **Sources:** [bytecap retention benchmarks](https://www.bytecap.io/research/youtube-shorts-retention-benchmarks),
    [ContentStudio Shorts analytics](https://contentstudio.io/blog/youtube-shorts-analytics-guide),
    [1of10 Shorts analytics](https://1of10.com/blog/youtube-shorts-analytics/).
  - **Consistent claim:** the opening 1–3 s decide the swipe. That is a platform-mechanics
    reason (the swipe happens before the content can pay off), not measured evidence.

### Consequences (same rule as M13, decision 0019)

- A feature gets a **non-zero weight** only when it has a platform reason or a consistent
  weak signal:
  - **dead air at the start**, which is the platform-mechanics reason;
  - **delivery pace**;
  - **an opening question or curiosity gap**;
  - **direct address**.
  The last three are consistent creator claims, but unmeasured.
- **Every other §2.8 feature is computed, stored and shown, with weight 0.** That covers
  readability, sentence stats, exclamations, superlatives, numbers, NER, emotion and title
  overlap. M15's LightGBM (or a manual calibration) learns their weights from our own data,
  instead of us guessing a direction now.
- **Emotion gets weight 0 for a model-specific reason too:** xlm-emo-t has **no neutral
  class**, so its maximum probability is high even for flat speech. That makes it a poor
  arousal proxy as the spec intends it. The full distribution is stored.
- **The score is relative within one upload**, comparing that upload's clips. It is not a
  retention %. The panel says so.

## Decisions

### 1. Opening window (`app/hook/window.py`, pure)

- **Window:** the words whose start falls in `[clip.start, min(clip.start + 15 s, clip.end))`
  (spec §2.8 step 1), from `TranscriptWord` rows. A segment without words counts as one word
  at its start.
- **First sentence:** the first spaCy sentence of the window text. Whisper's punctuation
  drives spaCy's sentence splits.
- **Language:** detected once per upload, `es` or `en`, from the stopword ratio over the
  transcript.
  - The stopword lists are M7's (`app/relevance/heuristics.STOPWORDS`), so no new model is
    needed.
  - Another language falls back to `en`, and the feature vector records `language` and
    `language_fallback`.

### 2. Features (`app/hook/features.py`; spaCy and emotion wrapped in `app/hook/nlp.py`)

| Feature | Definition | Weight |
|---|---|---|
| `lead_silence_s` | seconds from `clip.start` to the first word's start (no word in the window → 15) | scored |
| `words_per_second` | window words ÷ the window's length (`min(15, clip length)`) | scored |
| `first_is_question` | the first sentence ends in `?`, or contains `¿` | scored (framing) |
| `curiosity_connectives` | phrase matches in the window: EN "but", "until", "the reason why", "turns out", "here's why"…; ES "pero", "hasta que", "la razón por la que", "resulta que", "lo que nadie"… | scored (framing) |
| `first_unresolved_demonstratives` | in the first sentence, demonstrative pronouns with no noun head (spaCy `PRON` + `PronType=Dem`, e.g. "this", "eso", "esto") | scored (framing) |
| `second_person` | 2nd-person pronouns and verbs (spaCy `Person=2`, plus the EN "you/your") | scored (direct address) |
| `imperatives` | verbs with `Mood=Imp` (ES); EN: a sentence-initial base-form verb (`VB`) with no subject | scored (direct address) |
| `words`, `sentences`, `avg_sentence_words` | counts | 0 |
| `readability` | Flesch reading ease (EN) / Fernández-Huerta (ES), with a vowel-group syllable count | 0 |
| `questions`, `exclamations` | `?` / `!` counts (`¿`/`¡` are not counted twice) | 0 |
| `superlatives` | EN `Degree=Sup` or "most/least + ADJ"; ES "el/la/los/las más/menos + ADJ" and `-ísimo` forms | 0 |
| `numerics` | spaCy `NUM` tokens plus digit tokens (listicle framing, e.g. "3 ways") | 0 |
| `entity_density` | spaCy entities ÷ words | 0 |
| `emotion` | xlm-emo-t probabilities `{joy, anger, fear, sadness}`, plus `emotion_max` | 0 (see Context) |
| `title_overlap` | Jaccard overlap of content words (stopwords removed) between the window and the episode title | 0 |
| `title_source` | `show_notes` (the latest draft's first title suggestion) or `filename` (stem, `_`/`-` → spaces). `uploads` has no title column. | — |

### 3. Scorer (`app/hook/scorer.py`, pure, version `heuristic-v0`)

Each scored dimension maps to a sub-score in [0, 1]. The total is the weighted sum × 100. The
shapes are priors.

| Sub-score | Weight | Shape |
|---|---|---|
| `start` (dead air) | **0.30** | lead silence ≤ 0.3 s → 1.0; falls linearly to 0 at 2.0 s |
| `pace` | **0.25** | words/s: 0 at ≤ 1.0; rises to 1.0 at 2.3; 1.0 up to 3.8; falls to 0.6 at 5.0+. Conversational ES/EN lies at roughly 2.5–3.5. |
| `framing` | **0.25** | the first sentence is a question → 1.0; otherwise a curiosity connective or an unresolved demonstrative in the first sentence → 0.7; otherwise 0.3 |
| `address` | **0.20** | `second_person + imperatives`: 0 → 0.3, 1 → 0.8, ≥ 2 → 1.0 |

- **Ties stay ties.** The ranking adds no hidden tiebreak, as in M13.
- **A clip with no words in its window** scores `start` 0 and `pace` 0. It is still scored,
  not an error.
- **Explainability:** the `feature_vector` stores every feature, every sub-score and the
  weights. The panel shows the sub-scores plus the zero-weight features.
- **The weights live in code, in one table.** They are copied into the model version's
  `architecture_notes`.

### 4. Data model: `hook_scores` (spec §4.1, plus additions); migration `0014`

| Column | Type | Notes |
|---|---|---|
| id | BIGINT PK | |
| clip_id | FK → clips ON DELETE CASCADE, **UNIQUE** | one row per clip: the latest score |
| score | FLOAT | 0–100 |
| feature_vector | JSONB | features, sub-scores, weights, language, title source |
| model_version_id | FK → model_versions | |
| scored_at | TIMESTAMPTZ | **addition** |

- **A shorts re-run** replaces the `clips` rows, so their hook scores cascade away. The new
  clips show as "unscored" until the hook job runs again. No stale score is ever shown.
- **History:** a re-run overwrites the row (upsert), as in M13.
- **Model version:** `model_versions` gets (`hook`, `heuristic-v0`, `is_live=true`) through
  `get_or_create_version(..., model_name="hook")`.

### 5. API and job

- **No new endpoint.** `ClipOut` (`GET /uploads/{id}/clips`) gains `hook: {score, rank,
  features, scored_at} | null`, so the Shorts panel needs one call.
  - `rank` is computed among the scored clips of the upload, and ties share a rank.
- **Job:** `POST /uploads/{id}/jobs {tool_type: "hook"}`, with no params, replacing the dummy
  handler.
  - It scores every clip of the upload.
  - No clips → the job fails with "No shorts yet -- run Shorts first".
  - A missing spaCy model or emotion model → the job fails with the install step.
- **CPU only.** It doesn't take the GPU lock (decision 0016).
  - The models load lazily, once per worker process.
  - xlm-emo-t (XLM-R base, about 1.1 GB) comes from the HF cache. A new setting,
    `HOOK_EMOTION_MODEL`, holds the model name or a local path.
- **The M2 regression tests used `hook` as "the dummy tool".** With `hook` real, every
  `tool_type` has a real handler.
  - Those tests (`test_worker.py`, `test_jobs.py`, `test_market_worker.py`) now patch
    `HANDLERS["hook"]` back to `dummy.run`, through one fixture.
  - `dummy.py` stays, as the M2 state-machine test double.

### 6. Frontend

- **Shorts panel:** each clip card gets a hook badge (score + rank, or "unscored") and an
  expandable breakdown: the four sub-scores, then the zero-weight features with their values.
  - **Wording:** the badge is labelled **"hook score (heuristic v0), relative to this
    episode's clips, not a retention %"**.
- **Video row:** the `hook` **Run** button is blocked with "Run Shorts first" until the
  upload has clips. The panel refetches clips when a newer `hook` job succeeds.
- **Dashboard tile:** "Top hook-scoring clip" stays dummy. Wiring it is out of scope, like
  the CTR tile in M13.

### 7. Dependencies

- **`requirements.txt`:**
  - `spacy==3.8.18` (adds only new packages: thinc, blis and their siblings; no upgrades,
    checked with `pip install --dry-run`)
  - `es_core_news_sm` and `en_core_web_sm` 3.8.0, pinned as GitHub-release wheel URLs
- **README:** one line about the first-run download of xlm-emo-t.

## Test plan

**Unit (default suite):**

- **Window:**
  - words sliced at `start` / `start + 15` / `clip.end`
  - a segment without words
  - an empty window
  - language detection on ES/EN text, and the fallback for other text
- **Features:** a table of ES and EN sentences per extractor:
  - questions (`¿…?`, a trailing `?`, a question that isn't the first sentence)
  - exclamations (`¡…!` counted once)
  - superlatives (EN "the best", "most important"; ES "el más …", "buenísimo")
  - numerics ("3 formas", "three ways", "2026")
  - second person (EN "you"; ES "tú", "sabes", "te", "usted")
  - imperatives (ES "mira", "escucha"; EN "Listen to this")
  - curiosity connectives (whole phrases only)
  - unresolved demonstratives ("Esto es lo que…" counts; "este libro" doesn't)
  - entity density
  - readability on a known-score sentence
  - title overlap, with stopwords removed
  - lead silence and pace from synthetic timestamps
  - These use the real spaCy models.
- **Emotion:** the real xlm-emo-t on a clearly joyful and a clearly angry ES sentence →
  that label is the argmax, and the probabilities sum to ≈1.
- **Scorer:**
  - every shape at its breakpoints and in between
  - weights sum to 1
  - zero-weight features leave the score unchanged
  - an empty window scores without an error
  - ties stay ties
- **Handler:**
  - scores every clip
  - stores the score, `feature_vector`, `model_version_id` and `scored_at`
  - a re-run is idempotent (upsert, no duplicate rows)
  - no clips → a failed job with the message
  - a missing model → a clear failure
  - the title source is show notes when a draft exists, otherwise the filename
- **API:**
  - `ClipOut.hook` is null before a run, filled after
  - ranks are shared on ties
  - deleting the clips (a shorts re-run) cascades the hook scores
- **Migration 0014:** upgrade/downgrade, the UNIQUE constraint on `clip_id`, the FK cascade.
- **Frontend:**
  - badge states (unscored, scored, rank)
  - the breakdown
  - the Run button blocked without clips
  - the refetch after a hook job

**Manual QA (owner, the plan's item):** on 2–3 real episodes with shorts from M8/M9:

1. Rank each episode's clips by how well their first 15 s hook you, before looking at the
   scores.
2. Compare your ranking with the heuristic's order, and note where it disagrees and why.
   The breakdown makes that easy.

The findings go in this record. They may adjust the shapes, but not the scope.

**Regression:** the full CPU suite + `ng test`, through the test-runner.

## Alternatives rejected

- **Directional weights for exclamations, numbers, superlatives, NER, readability and title
  overlap:** there is no evidence either way. They are logged with weight 0, as in M13.
- **Emotion max-probability as a scored arousal proxy:** xlm-emo-t has no neutral class
  (see Context).
- **A zero-shot LLM hook score, as in the relevance model:** the plan scopes M14 to heuristics,
  and spec §5 notes the hook model has no bootstrap pseudo-label source. An LLM judge would
  be scope expansion; it is noted in `brainstorm.md`.
- **Audio features** (energy, pitch variance): prior-art flag 4 says audio matters, but the
  text-only v1 constraint (spec §2.8, §7) holds. This is a known gap for after v1.
- **Auto-chaining after the shorts job:** the owner chose its own job.
- **A separate `/hook-scores` endpoint:** embedding the score in `ClipOut` is one call and
  one source of truth for the panel.

## Implementation notes (2026-11-08)

### Deviations and additions to the plan above

- **The framing curiosity count is first-sentence only.** The §2 table lists
  `curiosity_connectives` over the whole window. Both are stored:
  - `first_curiosity` (first sentence) feeds `framing`, as §3 says;
  - `curiosity_connectives` (whole window) has weight 0.
- **ES imperatives:** the small model mis-tags some clause-initial imperatives (for example
  "escucha" as NOUN). A short lexicon (mira, escucha, oye, fíjate, imagina, dime…) also counts
  them, but only at a sentence start or right after punctuation, so "él piensa" isn't counted.
  An imperative isn't also counted as second person.
- **`Te`:** the ES small model tags it `Person=3`. The second-person lexicon catches it.
- **Emotion, no neutral class, confirmed on the real model:** "El martes fuimos a la tienda y
  compramos pan." scores joy 0.99. This is why the emotion feature stays at weight 0.
- **The M2 dummy tool:** the conftest fixture `dummy_hook` patches `HANDLERS["hook"]` back to
  `dummy.run`. `test_worker.py` applies it module-wide (`pytestmark`), and
  `test_market_worker.py` applies it to its real-worker test. `test_jobs.py` only enqueues, so
  it needs no change.
- **Model loading:** the emotion model loads once per worker process, about 40 s on the first
  download and then a few seconds. spaCy and the emotion model are loaded before any clip is
  scored, so a missing model fails the job without partial writes.
- **`ClipOut.hook.rank`:** 1 + the number of strictly higher scores. Ties share a rank.

### Dry run on real data (2026-11-08, read-only, dev DB: 5 episodes × 5 clips)

- **Scores ranged from 68.5 to 100.**
- **`start` was 1.0 for every clip:** M8 cuts at segment starts, so no clip opens on dead air.
  On this channel the 0.30 weight separates nothing. Noted in `brainstorm.md`; the manual QA
  decides whether to move it.
- **Pace was mostly in the 2.3–3.8 band.** A few clips with interjections fell below it, and
  one fast clip (4.2 w/s) fell slightly.
- **In practice, `framing` and `address` carry the ranking.**
  - The opening questions were detected correctly, for example "¿Tú quieres morir por la
    puerta grande?" → 100.
  - **`address` mostly counts the hosts talking to each other ("tú").** This is a known
    limitation (`brainstorm.md`).

### Tests

- **`tests/test_hook.py` (59):**
  - the window slice and the empty window
  - language detection (ES, EN, fallback)
  - every extractor on ES/EN sentences with the real spaCy models: questions, `¿…?` counted
    once, whole-phrase curiosity, demonstrative pronouns vs determiners, imperatives and second
    person, superlatives vs comparatives, numerics, syllables and readability, title overlap,
    end-to-end extract, the empty window
  - the real xlm-emo-t on a joy and an anger sentence
  - scorer shapes at their breakpoints, weights summing to 1, the weighted sum, zero-weight
    invariance, the empty window
  - handler: no clips, a missing spaCy model, a missing emotion model, scoring 3 clips
    (question > dead air > empty), the idempotent upsert with one model version, the title
    source (show notes vs filename)
  - API: `hook` null, then ranked with ties; the clip-delete cascade
- **`test_0014_…` in `tests/test_migrations.py`:** UNIQUE `clip_id`, the cascade, the downgrade.
- **Updated:**
  - `test_clips_api.py`: `"hook": None`
  - `test_worker.py` / `test_market_worker.py`: `dummy_hook`
- **Frontend:**
  - `shorts-panel.spec.ts` (+3): the unscored state, score and rank with the
    "not a retention %" wording, the breakdown
  - `video-row.spec.ts` (+2): Run blocked with "Run Shorts first", the refetch after a newer
    hook job
- **Regression (2026-11-08, WSL clone):**
  - backend: **672 passed**, 0 failed (10 opt-in deselected; 611 before)
  - frontend: **174/174** (169 before)
  - `ng build`: OK
- **Manual QA:** pending. It is the owner's item, above.

## Milestone closed (2026-11-08)

The owner closed M14 on the automated test results, as for M13. The plan's manual QA is an
follow-up (**closed 2026-11-14**: the owner confirmed the manual QA), not a blocker: compare the heuristic's clip ranking with the owner's own
ranking on 2–3 real episodes.

- **Ready to use:** the 5 episodes in the dev DB already have shorts. Run the `hook` tool on
  them from the Manage page.
- **Things to check:**
  - whether `start` (always 1.0 here) should keep its 0.30 weight;
  - how much host-to-host "tú" inflates `address`.
- **Where the findings go:** this record. They may adjust the shapes.
