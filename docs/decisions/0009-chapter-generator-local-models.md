# Decision: Chapter Generator with Locally Trained Models (M5)

**Date:** 2026-09-27
**Status:** Decided. Planning is done; the experiment and development
are next.

## Context

Spec §2.3/§5 and `docs/development-plan.md` M5 plan chaptering as a
Claude API call that returns boundary phrases and titles. A TF-IDF
lexical-realignment step then snaps those boundaries back onto
transcript timestamps.

During M5 planning the project owner chose instead to run chaptering
**entirely locally, with models we train ourselves**. The model work is
to be done **as an experiment**, with its own notebooks, so the owner
can follow how the models were developed and evaluated.

Research and measurements behind this document:
[`docs/research/m5-chapter-model.md`](../research/m5-chapter-model.md).

Decided with the owner during planning:
- **Local only.** There is no Claude API in M5. Whether M6 (show notes)
  also goes local is **still open**.
- **Two models, both trained by us:**
  - a **boundary** model
  - a **title** model
- **An unsupervised baseline is kept** as the fallback and as the
  evaluation reference.
- **Granularity:** "few, broad" chapters.
- **Languages:** English and Spanish, varying per episode.
- **Spanish** is scarce in the training data (~0.85–1.7k videos of at
  least 15 min). We rely on cross-lingual transfer and measure it on a
  Spanish-only test set.
- **Fallback rule:** the trained boundary model goes to production only
  if it beats the baseline on validation. Otherwise the baseline ships.
  M5 closes either way, with the outcome documented.
- **Traceability:** add a `chapters.model_version` column, which the
  spec §4 does not have.
- **Frontend is minimal:** a chapter list plus "Copy for YouTube".

## Decisions

### 1. Training data: VidChapters-7M, via the Chapter-Llama JSON mirror

**Source:** HF `lucas-ventura/chapter-llama`, MIT. It holds VidChapters'
Whisper ASR (sentence-level, with start/end) and user chapters.
- **Why the mirror:** the original `asr.pkl` is one 19.3 GB pickle and
  cannot be loaded in 16 GB of RAM.
- **Selection:**
  1. Stream `chapters.json` (2 GB) with `ijson`.
  2. Filter by duration (15–120 min), chapter count (≥ 3) and language.
  3. Stream `asrs.json` (22 GB, over HTTP) and keep only the selected
     videos. The full ASR is never stored.
- **Subset:**
  - ~8–10k English videos, spread over the 15–30 / 30–60 / 60–120 min
    buckets.
  - **Every usable Spanish video**, with language confirmed on the ASR
    text as well as the titles.
  - VidChapters' official train/val/test split.
  - A **Spanish-only test set** (~200–300 videos) held out.
- **Storage: bulk data goes on the second drive** (the owner's request).
  - **The drive:** a 931 GB NTFS disk, mounted with `ntfs3`, rw,
    `uid=1000`, with ~801 GB free.
  - **Measured speed:** ~196 MB/s for a 512 MB direct-I/O file, and
    writes work.
  - **What goes there:** the downloaded `chapters.json`, the selected
    ASR, the embeddings, intermediate datasets and training
    checkpoints. That is the heavy, write-once/read-sequentially data,
    which NTFS handles fine.
  - **Estimate:** ASR for 10k videos is well under 1 GB. Embeddings for
    ~2–3 M units × 384 dims in float16 come to ~2 GB. Checkpoints are
    ~1.2 GB each for mT5.
  - **What stays on the SSD:**
    - **venvs:** NTFS permissions, exec bits and symlinks are
      unreliable for them
    - **Docker:** overlay2 needs ext4/xfs
    - **notebooks, `lib/`, `runs/`:** small, and tracked in git
    - **the exported production bundle** (`backend/models/`): it is
      loaded by the worker
  - **Path:** the experiment reads its data root from
    `YTS_EXPERIMENT_DATA`, set in `experiments/.env`. It points to a
    stable mount point, `/mnt/d/yts-experiments/m5-chapters`. The
    current udisks automount path (`/media/pablo/<UUID>`) only exists
    after a desktop login, so it is not used.
  - **Setup (owner, needs sudo):** add an `/etc/fstab` entry so `/mnt/d`
    is always mounted:
    `UUID=DE04FBD304FBACA1 /mnt/d ntfs3 rw,uid=1000,gid=1000,nofail,x-systemd.automount 0 0`.
    `nofail` keeps boot working if the drive is absent.
  - **Leave alone:** the drive already has `youtube_suite*` folders from
    an earlier project. The experiment uses its own
    `yts-experiments/` folder and never touches them.

**Licence.** VidChapters code and the mirror are MIT. The data is
YouTube-derived, and the rights stay with the original holders.
- **Fine:** training models for this single-operator product.
- **Not allowed:** we never redistribute the data.
- **Revisit:** if the product or its weights are ever distributed.

### 2. One text unit for training and inference

- **The mismatch:**
  - VidChapters ASR comes in ~5 s sentences.
  - Our Whisper segments are **~0.8 s median** (3563 segments in the
    64-min dev episode).
- **Fix:** both sides are normalised to the same unit before encoding.
  Two options are compared in the experiment (notebook 02/03):
  1. **sentences** rebuilt from word timestamps and punctuation, with a
     duration cap
  2. **fixed time blocks** of ~5–10 s
- **Implementation:** the winner becomes a pure function in the backend
  (`app/chapters/units.py`) with unit tests. The experiment's `lib/`
  uses the same logic.
- **Snapping:** a predicted boundary always lands on a real transcript
  timestamp (a unit start, which is a segment/word start).

### 3. Boundary model: frozen multilingual encoder + BiLSTM tagger

- **Encoder:**
  - Frozen `intfloat/multilingual-e5-small` (118 M, MIT) or
    `paraphrase-multilingual-MiniLM-L12-v2` (118 M, Apache-2.0).
  - The experiment picks one: both encode a 64-min episode in ~7 s on
    this CPU (measured).
  - `multilingual-e5-base` is ~4× slower and is only tried if the small
    ones clearly underperform.
- **Head:** a BiLSTM over the unit-embedding sequence produces a
  per-unit "chapter starts here" probability.
  - **Class imbalance:** weighted BCE or focal loss.
  - **CRF:** kept only if it measurably helps.
  - **Size:** a few hundred K parameters.
  - **Training:** on the GPU with the worker stopped.
- **Decoding:** YouTube's rules and the owner's granularity are applied
  here, not left to the model:
  - peak picking with a minimum chapter length (≥ 60 s)
  - a target of ~5–8 chapters per hour
  - the first chapter forced to 0:00
  - at least 3 chapters, and every chapter at least 10 s (YouTube)
- **Inference:** on CPU, in seconds.

### 4. Baseline: unsupervised, over the same embeddings

- **Method:** TextTiling-style, using the cosine dissimilarity between
  adjacent windows of unit embeddings, with the **same decoding** as
  §3.
- **No training.** It is the eval reference in every notebook, and the
  production fallback (rule in §7).

### 5. Title model: mT5-small with frozen embeddings

- **Model:** `google/mt5-small` (300 M, Apache-2.0).
- **Training data:** (chapter transcript text, truncated to 512
  tokens) → (user chapter title), taken from the same subset. That
  includes ~15k Spanish pairs.
- **Titles are lightly cleaned first:** strip speaker prefixes and stray
  quotes, and drop empty or non-textual titles.
- **Memory (measured on the 1060):**
  - Full fine-tuning **does not fit** in 3 GB, not even at batch 1.
  - Freezing the shared embedding and the LM head (~256 M of the
    300 M params) fits at batch ≤ 4 × 512 tokens, peaking at
    1.5–2.5 GB.
  - Speed: ~0.35 s/step, ~75 min per epoch for 50k pairs.
- **Settings:** Adafactor, gradient checkpointing.
- **Inference:** on CPU, one title per chapter.
- **Not decided yet:** a fallback for titles if mT5's output proves
  unusable. It is flagged in brainstorm and will be raised with the
  owner before any fallback is built.

### 6. The experiment: `experiments/m5-chapters/`

- **Layout:**

  ```
  README.md            dated log: hypotheses, each run, results, conclusions
  01_data.ipynb        selection/streaming of VidChapters, statistics
  02_baseline.ipynb    units + unsupervised baseline + metrics
  03_boundaries.ipynb  BiLSTM training, comparison vs. baseline
  04_titles.ipynb      mT5-small fine-tune, sample outputs
  05_eval.ipynb        final eval (test EN, test ES, owner's 64-min episode) + export
  lib/                 shared code (data, units, metrics, decoding, models) + pytest tests
  runs/<date>-<name>/  config.json, metrics.json, plots (in git)
  data/, checkpoints/  gitignored
  ```

- **Notebooks** are committed **with outputs**.
- **Environment:** its own venv, `experiments/.venv`.
  - A `.pth` file points it at `backend/.venv`'s site-packages, so it
    reuses torch 2.8 cu126 (verified: 139 MB instead of ~4 GB).
  - Experiment-only packages (jupyter, ijson, ftlangdetect, …) are
    listed in `experiments/requirements.txt`.
- **Metrics:** boundary metrics reuse Chapter-Llama's definitions (MIT):
  - **F1@5s** (primary) and F1@3s
  - tIoU-averaged F1 over 0.5–0.95, and mean tIoU
  - Pk / WindowDiff
  - chapter-count error
  - Titles: CIDEr/BLEU with ground-truth boundaries, plus a manual read.
- **Export:** `05_eval` writes a versioned bundle to
  `backend/models/chapters/<version>/`, which is gitignored. The bundle
  holds:
  - the encoder name
  - the head `state_dict`
  - the mT5 checkpoint
  - the unit and decoding config
  - `metrics.json`
  - a pointer to its run

### 7. Production integration (backend)

- **Handler:** `tool_type='chapters'` gets a real handler,
  `app/worker/handlers/chapters.py`. It replaces the dummy handler for
  that type only.
- **Preconditions:** a succeeded `transcribe` job and persisted
  segments. Otherwise it fails with "Run Transcript first", as `diarize`
  does.
- **Steps:**
  1. Load segments and words.
  2. Build units (§2).
  3. Encode and predict boundaries on CPU.
  4. Decode (§3).
  5. Generate one title per chapter.
  6. Replace the upload's chapter rows in **one transaction**, so a
     re-run is idempotent.
- **Threading:**
  - All inference runs on CPU inside `asyncio.to_thread`.
  - **The GPU lock is not taken:** no GPU is used, so it doesn't
    contend with Whisper or pyannote.
  - Models load lazily as per-process singletons, like Whisper and
    pyannote.
- **Model selection:** `chapters_model_dir` points at an exported
  bundle. The bundle says whether its boundary model is `bilstm` or
  `baseline`, which is the §7 fallback rule, decided in `05_eval` on
  validation F1@5s.
  - With no bundle configured, `chapters` jobs fail fast with a clear
    message. The other tools are unaffected.
- **Schema:** migration `0006` creates `chapters`:
  - `id`, `upload_id` (FK, CASCADE), `order_index`, `start_seconds`,
    `title`
  - **`model_version` TEXT**, e.g. `bilstm-e5s-2026-10-05+mt5s-2026-10-08`
    or `baseline-tiling+mt5s-…`
  - unique (`upload_id`, `order_index`)
- **API:** `GET /uploads/{id}/chapters` returns JSON (`start_seconds`,
  `title`, `model_version`). Status rules mirror the subtitle endpoints
  (0007 §5):
  - **404:** no chapters job has ever succeeded
  - **409:** no success yet, but a job is queued or running
- **Frontend (Manage):**
  - When a chapters job has succeeded, the "Chapters" row shows the
    list, formatted `MM:SS`, or `H:MM:SS` over 1 h.
  - It has a "Copy for YouTube" button that copies `00:00 Title` lines.
  - "Run" stays disabled until a transcript exists, extending
    `needsTranscript`.
- **New backend deps:** `transformers` and `sentencepiece`, reusing the
  torch already present.

### 8. What this changes relative to the spec and plan

- **Spec §2.3 steps 2–3** (LLM call + TF-IDF realignment) are replaced
  by §§2–5 here. The **TF-IDF realignment step is dropped**: it exists
  to repair LLM paraphrasing, and here boundaries are predicted
  directly on real transcript units.
- **Spec §5** "Claude API for LLM-based chaptering" no longer applies
  to chapters.
- **Spec §4 `chapters`** gains `model_version`.
- **`development-plan.md` M5** is rewritten to match. The estimate goes
  from 3–4 days to **~2–3 weeks**, and a risk note is added.

## Test plan

**Experiment (`experiments/m5-chapters/lib/`, pytest)**
- Metrics against hand-computed cases: F1@Ns, tIoU-F1, Pk/WindowDiff.
  Where applicable, a cross-check against Chapter-Llama's implementation
  on the same inputs.
- Unit building: segments/words → units, including edge cases (empty
  text, untimed words, a single huge segment).
- Decoding: on synthetic probability curves, check minimum length, the
  target count, first chapter at 0, ≥ 3 chapters, and determinism.
- Baseline: on a synthetic 3-topic sequence it finds the 2 true
  boundaries.

**Experiment (notebooks)**
- **Numbers reported:**
  - on validation, for model selection
  - once on test EN and test ES (Spanish-only, held out)
  - trained model vs. baseline, every time
- **Manual QA:** the owner's 64-min episode, with both boundary sets and
  titles shown side by side.

**Backend (pytest)**
- **Unit tests:**
  - `units.py`
  - decoding + YouTube constraints
  - "Copy for YouTube" timestamp formatting, if done server-side
- **Handler tests:**
  - A **tiny fake bundle** (random-init head + stub encoder/title
    model) persists rows with `model_version`.
  - A re-run is idempotent.
  - "Run Transcript first" failure.
  - A missing bundle → fast, explicit failure.
- **Migration:** up/down for `0006`, joining the existing migration
  smoke test.
- **API:** `GET /uploads/{id}/chapters` returns 200/404/409 and
  requires auth.
- **Integration:** the real exported bundle on CPU over the M3 fixture
  transcript. It must produce ≥ 1 chapter, the first at 0.0, with
  non-empty titles and increasing starts.
  - This test is skipped if no bundle is exported, as the only
    exception to the suite's no-skip style.
  - **Why the exception:** the weights are gitignored and are not
    reproducible in CI.

**Frontend (vitest)**
- The chapter list renders from the service.
- "Copy for YouTube" writes the expected text.
- "Run" is disabled without a transcript.

**Regression**
- The full backend suite (M0–M4) and the full frontend suite.
- The real-arq-worker chain test extended with `chapters`.

## Consequences

- **New top-level directory:** `experiments/`, with its own venv and
  requirements. The data and checkpoints are gitignored.
- **New backend module** `app/chapters/` (`units`, `decoding`,
  `baseline`, `model`), plus a handler, migration `0006`, a router, and
  settings (`chapters_model_dir`, `chapters_device=cpu`).
- **The backend gains** `transformers` and `sentencepiece`.
- **Model weights are not in git.** A fresh clone needs the experiment
  re-run or the bundle copied in. This is documented in the README.
- **Workflow:** GPU training requires stopping the worker (0008 lesson:
  GPU measurements and runs need the GPU to themselves).
