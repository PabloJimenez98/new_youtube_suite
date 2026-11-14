# Decision: Relevance BiLSTM v1, Pseudo-Label Store and Live Switch (M15)

**Date:** 2026-11-14
**Status:** Done (2026-11-14). i1 failed the offline gate; i2 passed it and the owner's blind A/B (3–2), and is live. See "Results" and "Promotion".

## Context

- **Spec §2.4** describes the relevance model: frozen window embeddings → BiLSTM → MLP + sigmoid,
  trained by distillation from heuristic + zero-shot LLM pseudo-labels, with a
  temporal-smoothness regularizer. M7 (0012) shipped only the bootstrap. The live version is
  `heuristic-v0+Qwen3.5-9B-Q4_K_M+relevance-p3` (0014 §3, 0016 §9).
- **The plan's M15** asks for these pieces:
  - pseudo-label generation and a BiLSTM training script
  - `model_versions` artifact storage and a live-model swap with a **rollback path**
  - offline evaluation on a held-out set
  - a manual A/B against the heuristic before `is_live` flips
  - a regression check that M8 (shorts) stays sane
  - **M7 stays live until the trained model demonstrably beats it.**
- **Data in the dev DB (read-only check, 2026-11-14):**
  - **63 episodes** with p3 scores: 40.4 h in total, 15–56 min each, 19,348 windows, all with
    an `llm_score`.
  - **Languages:** 61 episodes in Spanish, 2 in English (`show_notes_drafts.language`).
  - **The two teacher signals don't agree.** Over all windows, the Pearson correlation between
    the heuristic and the LLM scores is **−0.009**; per episode it ranges from −0.24 to 0.27.
    This confirms the M7 finding (0012) at scale.
  - **`analytics_retention_curve`: 0 rows.** Retention-derived labels (spec §2.4 training
    step 3) are not possible yet.
  - **Installed:** torch `2.8.0+cu128` with CUDA available, `transformers` 5.17, and
    `multilingual-e5-small`, which is already in the chapters bundle (0009).
- **Decided with the owner (2026-11-14):**
  - **Promotion criterion:** a blind top-5 A/B on held-out episodes, as C4 in 0016, plus an
    offline gate (§7).
  - **Pseudo-labels:** the stored p3 scores, learned as **two heads** (heuristic and LLM),
    combined with p3's 0.4/0.6 weights (§3).
  - **Definition of done:** the pipeline, a tested swap and rollback, and **two complete
    train → evaluate cycles**, documented. M15 closes whether or not the model is promoted;
    M7's version stays live if it doesn't win (§9).

## Decisions

### 1. Encoder: `multilingual-e5-small`, not `bge-base-en-v1.5` (a deviation from the spec)

- **Why:** `bge-base-en-v1.5` is English-only, and 61 of the 63 episodes are Spanish. This was
  flagged in `brainstorm.md` at M7.
- **The pick:** `intfloat/multilingual-e5-small` (118 M, MIT), frozen.
  - It is already vendored for chapters (0009 §3), so there is no new download or license.
  - Use the same input prefix and pooling as `app/chapters/pipeline.py`, so there is one
    embedding convention in the repo.
- **Embedding cache:** the encoder is frozen, so window embeddings are computed once per upload.
  They are stored as `.npy` files keyed by encoder and transcript hash, outside the repo, under
  the storage base path.
- **Iteration lever:** `multilingual-e5-base` (§9).

### 2. Pseudo-label store: a new table, `relevance_pseudo_labels` (migration `0017`)

- **Problem:**
  - `relevance_scores` has one row set per upload (unique `upload_id, window_index`), and a
    handler re-run **replaces** it.
  - Once a trained model is live, re-running relevance on an episode would delete that
    episode's teacher labels.
  - Training on rows written by the student would be self-distillation, and it would happen
    silently.
- **Table** `relevance_pseudo_labels`:
  - `id`, `upload_id` (FK, CASCADE), `window_index`, `window_start_seconds`,
    `window_end_seconds`, `heuristic_score`, `llm_score` (nullable),
    `teacher_version_id` (FK → `model_versions`), `created_at`
  - unique (`upload_id`, `window_index`, `teacher_version_id`)
  - Several teachers per upload can coexist. That leaves room for the expected-value LLM labels
    of §9.
- **Backfill in the migration:** copy every `relevance_scores` row whose version has no
  `artifact_path` (a teacher, not a trained model). Today that is all 19,348 rows.
- **Writers:**
  - The teacher path of the relevance handler (§6) now writes **both** tables, so new episodes
    keep feeding the store while M7 is live.
  - The pseudo-label CLI (§5) writes **only** this table.
- **Reader:** the training dataset builder. It never reads `relevance_scores`.

### 3. Model and loss (spec §2.4, two heads)

- **Input:** the same 15 s / 7.5 s-hop windows as M7 (`app/relevance/windows.py`), so window
  indices line up with the pseudo-labels exactly. Each window becomes one e5 embedding (384-d).
- **Architecture:**
  - Linear 384 → 128, then dropout 0.3.
  - A 1-layer BiLSTM, hidden size 128 per direction (spec range 128–256).
  - Two heads, each FC-64 → ReLU → FC-1 → sigmoid, giving `ĥ` (heuristic) and `l̂` (LLM).
- **Score:** `ŝ = 0.4·ĥ + 0.6·l̂`, the same constants as p3 (0012 §4), stored in the bundle
  config.
  - **Why two heads:** the two teacher signals are uncorrelated, so a single target blurs two
    different notions of relevance.
  - With separate heads, each one can be evaluated, and the combination weights can be changed
    later without retraining.
- **Loss:**
  - The training loss is `Huber(ĥ, h) + Huber(l̂, l) + λ · mean((ŝ_t − ŝ_{t−1})²)`.
  - Huber δ = 0.1 is robust to label noise (spec).
  - Padding is masked. Windows with a NULL `llm_score` drop out of the LLM term only.
  - λ ∈ {0, 0.1, 0.5}, chosen on the validation split.
- **Training:**
  - Adam with lr 1e-3, one episode per sequence, batches of 8 episodes (padded).
  - Early stopping on the validation Spearman of `ŝ`, with patience 10.
  - **3 seeds.** The reported metric is the mean ± sd, and the bundle keeps the median seed.
- **No 45-min super-chunking (a deviation from spec §2.4 inference step 3):**
  - The longest episode is 56 min (~450 windows × 384 floats), which is trivial in memory.
  - The whole sequence is scored in one pass.
  - Super-chunking was a GPU-memory measure for much longer inputs, and it would add stitching
    artefacts for no gain. Revisit only for episodes over ~3 h.

### 4. Split (fixed, by episode)

- **Train 48, val 7, test 8 episodes.**
  - Stratified so that 1 of the 2 English episodes is in test.
  - Seeded, and written once to `models/relevance/split.json` (upload ids), so every iteration
    uses the same test set.
- **The blind A/B episodes** are 5 of the 8 test episodes. They are picked before training, and
  the owner is not told which ones they are.
- **New episodes uploaded after the split** go to train in later retraining runs. The test set
  stays frozen across M15's two iterations.

### 5. Pipeline: CLI, not a worker job

Training is offline and run by hand, so it is a CLI next to `market` / `analytics` /
`reporting`, not an arq task. As in 0009, GPU training needs the worker to be stopped.

| Command | What it does |
|---|---|
| `relevance pseudo-label [--missing]` | runs the teacher (the live heuristic version, or `--teacher <label>`) on uploads with a transcript but no labels from that teacher, and writes only to `relevance_pseudo_labels` |
| `relevance train --label <label> [--lambda …] [--seeds 3]` | builds the dataset (pseudo-labels + cached embeddings + `split.json`), trains, writes a bundle, and registers a **non-live** `model_versions` row with metrics |
| `relevance eval <label>` | the offline metrics of §7 on the test split, for the new version and the baselines; stored in `model_versions.metrics` |
| `relevance ab-sheet <label>` | writes the blind A/B sheet (top-5 moments per test episode for both versions, in random order) outside the repo, with a separate answer key |
| `models list [--name relevance]` | lists versions, live flags and their key metrics |
| `models set-live <name> <label>` | in one transaction: unsets the current live row, sets the new one, and logs both labels. **Rollback is `set-live` back to the previous label.** |

- **Bundle:**
  - Saved to `backend/models/relevance/<label>/` (the chapters-bundle pattern): `weights.pt`,
    `config.json` (encoder, prefix, dims, λ, combination weights, window spec, seed) and
    `metrics.json`.
  - `artifact_path` points at the bundle directory.
  - Version labels look like `bilstm-e5s-20261115-i1`.
- **`model_versions` gets no migration:**
  - A row with an `artifact_path` is a trained model, and its `config.json` names the
    architecture.
  - A row without one is a teacher/heuristic.
  - `training_data_reference` records the teacher label, the split hash, the episode counts and
    the embedding encoder.

### 6. Inference: the handler dispatches on the live version

- **Live version without `artifact_path` (today):** unchanged, the heuristic + LLM path. It
  also writes `relevance_pseudo_labels` (§2).
- **Live version with `artifact_path` (BiLSTM):**
  - windows → cached or fresh e5 embeddings → BiLSTM.
  - It writes `relevance_scores` with `score = ŝ`, `heuristic_score = ĥ`, `llm_score = l̂`.
    These columns hold the components of the score, and the version FK says which model
    produced them.
  - It doesn't load the LLM and doesn't write pseudo-labels.
  - The GPU lock is held only for the encoder.
- **Expected effect:** relevance per episode goes from several minutes of LLM time to seconds.
  This will be measured, and it is reported but is not a promotion criterion.
- **Switching versions doesn't rescore existing episodes.** Their scores and clips keep the
  version FK they were made with. A user re-run, or the M9 season runner, rescores with the
  live version.
- **M8, the API and the frontend need no change:** they read `relevance_scores.score` and the
  version label.
- **The relevance `ToolDoc` (0021 rule)** is updated in the same change: `training.planned`
  becomes `false`, the real training flow replaces the planned one, and the model card names
  the live version.

### 7. Evaluation and promotion

- **Offline metrics, on the test split, per episode, then averaged:**
  - the Spearman of `ŝ` vs. the p3 combined score, and of each head vs. its own target
  - top-20 % overlap with p3 (precision)
  - top-5 moment overlap (60 s apart, the 0012 §6 rule)
  - smoothness, as the mean |Δŝ|
- **Baselines:**
  - the heuristic alone (cheap, computed exactly)
  - a constant (the training mean)
- **Offline gate:** the BiLSTM's mean Spearman vs. p3 must be **higher than the heuristic
  alone's**.
  - The heuristic is part of p3, so it already agrees partly with it.
  - Beating it means the BiLSTM learned the LLM signal from text. Failing it means the BiLSTM
    adds nothing over seconds of heuristic.
- **Blind A/B (the owner, about 1 h), as C4 in 0016:**
  - On the 5 A/B test episodes, the owner sees two top-5 lists per episode, labelled A and B,
    with timestamps and text, in random order.
  - The owner picks the better list or calls it a tie.
- **Promotion rule:** the offline gate passes **and** the BiLSTM is preferred or tied on
  **≥ 3 of 5** episodes. Only then does `models set-live relevance <label>` run.
- **After promotion:** the shorts regression check of §8 runs. If it fails, it's
  `set-live` back to p3.

### 8. Shorts regression (plan's Testing bullet)

On 2 or 3 test episodes rescored with the promoted BiLSTM, a shorts job must succeed with
these properties:

- the usual number of clips
- clip lengths inside M8's bounds
- no clip in the first or last 30 s unless the curve really peaks there
- clips no closer together than M8's minimum separation

The owner looks at the clips. Also, the BiLSTM's curves must not be degenerate. A curve that
is nearly flat or saturated would give M8's peak finder nothing to work with, so the
prominence statistics are logged.

### 9. Iterations (two cycles budgeted)

- **i1:** the defaults above.
- **i2:** chosen from i1's failure analysis. The levers, in order of cost:
  1. **Score combination:** compute the heuristic **exactly** at inference (it is cheap) and
     use the BiLSTM only for the LLM head. No retraining is needed, only a new version label
     and an eval.
  2. **Capacity and regularization:** hidden 256, a λ sweep, 2 BiLSTM layers.
  3. **Encoder:** `multilingual-e5-base`, which needs a new embedding cache.
  4. **Expected-value LLM labels:** rescore the 63 episodes with the 9B, reading the expected
     value over the digits (0014 §3, `experiments/m9-perf/llm_expected.py`), stored as a second
     teacher in `relevance_pseudo_labels`.
     - **Costs ~8–9 h of GPU.** The owner confirms before it runs.
- **Each cycle** gets a full eval, and an A/B if the offline gate passes, recorded below in a
  "Results" section.
- **Two failed cycles still close M15:** the result is documented, M7 stays live, and the next
  levers go to `brainstorm.md`.

## Alternatives rejected

- **`bge-base-en-v1.5` (the spec):** English-only, with 97 % Spanish data.
- **Training straight from `relevance_scores`:** a live trained model would overwrite its own
  training targets (§2).
- **One combined target:** it averages two uncorrelated signals. The owner chose two heads.
- **Offline metrics as the only gate:** a distilled student can at best match its teacher on
  the teacher's labels, so that can't show it beats the heuristic. The plan also asks for a
  manual A/B.
- **A scheduled retraining worker job (spec §6.4):** the plan's M15 entry asks for a training
  script and a swap. A periodic loop is useful only with retention and CTR labels, which don't
  exist yet (0 retention rows). The CLI can be scheduled later.
- **Retention-derived labels (spec §2.4 step 3):** there is no data. They stay a fast-follow
  once the channel has views.
- **Super-chunking:** see §3.
- **Training the CTR and hook models here:** outside the plan's M15 entry, which covers the
  relevance model only. See the ToolDoc note in `brainstorm.md`.

## Test plan

- **Unit (default CPU suite):**
  - **Dataset builder:** pseudo-labels align with windows by index; uploads without labels are
    skipped; rows from trained versions are never read; `split.json` is deterministic and
    stratified.
  - **Model:** output shapes; padding masked; both heads in [0, 1].
  - **Loss:** the smoothness term is 0 on a constant curve; a NULL LLM label drops out of the
    LLM term only.
  - **Training:** overfits a tiny synthetic set (loss < ε); a seed reproduces exactly on CPU.
  - **Bundle:** save/load round-trip gives identical outputs; a missing or corrupt bundle fails
    fast with a clear error.
  - **Eval:** each metric on hand-built curves, including the top-5 rule at 60 s; the gate
    logic.
  - **`models set-live`:** exactly one live row after a switch; rollback restores the previous
    one; an unknown label fails without changing anything.
- **Handler (fake pipeline and fake BiLSTM):**
  - A live heuristic writes both tables.
  - A live BiLSTM writes `relevance_scores` only, never calls the LLM, and tags the rows with
    its version.
  - A re-run under a BiLSTM leaves the pseudo-labels untouched.
- **CLI:** `pseudo-label --missing` skips labelled uploads and writes only the store; `train`
  registers a non-live row; `ab-sheet` writes a sheet and a separate key.
- **Migration `0017`:** up/down; the backfill copies teacher rows only; the unique constraint.
- **Integration (skip-if-absent bundle or encoder):** the real e5 + bundle score the M3 fixture
  transcript, with the expected window count and finite scores in [0, 1].
- **Manual:** the blind A/B of §7 and the shorts check of §8.
- **Regression:** the full backend CPU suite, `ng test`, and the experiments suite. The GPU
  suite runs only if the handler's GPU path changes.

## Implementation notes (2026-11-14)

- **Code:**
  - `app/relevance/{bilstm,dataset,training,evaluation,retrain}.py`
  - `app/model_registry.py`
  - migration `0017`
  - the handler dispatch in `app/worker/handlers/relevance.py`
  - CLI `relevance …` / `models …`
- **Additions to the plan above:**
  - **`relevance ab-score`** parses the owner's filled answer table against the key and stores
    the A/B verdict in `model_versions.metrics.ab`.
  - **`relevance status`** shows both halves of the promotion rule.
  - **`models set-live`** refuses a trained relevance version whose stored gate or A/B didn't
    pass, unless given `--force`. Switching to a teacher (the rollback) is never refused.
  - **A missing or broken bundle** fails `set-live` before anything changes, and fails a
    relevance job with a message that names the rollback command.
- **Dev DB:** migrated to `0017`. The backfill copied the 19,348 p3 windows.
- **ToolDocs (0021 rule):**
  - Relevance `training.planned` is now `false`, with the real flow.
  - The CTR, hook and M15 texts now say that CTR and hook training are follow-ups after v1 and
    not part of M15 (owner, 2026-11-14, option 3).
- **Unrelated fix:** `tests/test_reporting.py::test_task_runs_and_logs_one_line` had expired.
  The fake's clock was fixed at 2026-11-08 while the task reads the real clock, so after 48 h it
  logged the lag warning instead of "WAITING". The test now sets the fake's clock to the real
  time.

## Results

### i1 — `bilstm-e5s-20261114-i1` (defaults of §3)

- **Data:**
  - The split was written to `models/relevance/split.json`: 48 train, 7 val, 8 test, 5 A/B.
  - Train-target stats: mean 0.553, std 0.133. The heuristic component's std is 0.179 and the
    LLM's is 0.189.
- **Training:** 2 min 37 s on the RTX 5070, including the first embedding pass over 63 episodes.
  - The best mean validation Spearman was at λ = 0; λ barely mattered.
  - The seeds varied a lot (0.39 / 0.40 / 0.50). The two lower seeds early-stopped at epochs 5
    and 9, while seed 2 ran to epoch 60.
  - The median seed (seed 1, val 0.401) was kept.
- **Test (8 episodes):**

| | Spearman vs p3 | top-20 % precision | top-5 overlap | std of the curve |
|---|---|---|---|---|
| BiLSTM i1 | 0.396 | 0.390 | 0.125 | **0.003** |
| heuristic alone | **0.545** | 0.478 | 0.275 | 0.175 |
| constant | 0.000 | 0.198 | 0.025 | 0 |

  Head Spearman: heuristic head 0.496, LLM head 0.380.
- **Offline gate: FAILED.** No A/B was run.
- **Failure analysis:**
  - The curve collapsed to almost the mean (std 0.003), so its ranking is weak and M8's peak
    finder would get nothing to work with.
  - The cause looks like optimisation, not capacity: patience 10 on a noisy validation
    Spearman stops most seeds within the first 10 epochs.
- **Diagnosis (train/val only, never the test split; `patience 40`, 3 seeds each):**
  - With patience 40 the collapse disappears. Validation Spearman is 0.50 on all 3 seeds and
    the curve std is 0.07–0.08.
  - lr 3e-3, Huber δ = 1.0 and hidden 256 change nothing (±0.01).
  - Even so, the heads-only score stays below the heuristic alone on validation (0.50 vs 0.533).
    The LLM head learns only ~0.36 Spearman from text.
  - Replacing the heuristic head with the **exact heuristic** gives 0.623–0.629 on validation.

### i2 — `bilstm-e5s-20261114-i2` (levers 1 + 2 of §9)

- **Changes:**
  - **`combine_mode = "exact_heuristic"`:**
    - The heuristic is computed exactly at inference, with the teacher's own code and LID rule.
      The fastText LID file is copied into the bundle.
    - The BiLSTM provides only the LLM component.
    - Validation selects on that same combination.
  - **`patience` 10 → 40,** now the default in `TrainConfig`.
  - **The same model, loss and split as i1.** The heuristic head is still trained but unused in
    this mode.
- **Training:** 8 min 22 s. λ = 0 was best again (λ = 0.5 is slightly worse), and the 3 seeds
  agree within 0.002. The median seed (seed 2) was kept, with validation Spearman 0.627.
- **Test (8 episodes):**

| | Spearman vs p3 | top-20 % precision | top-5 overlap | std of the curve |
|---|---|---|---|---|
| **BiLSTM i2** | **0.652** | **0.575** | **0.375** | 0.086 |
| heuristic alone | 0.545 | 0.478 | 0.275 | 0.175 |
| constant | 0.000 | 0.198 | 0.025 | 0 |

  LLM-head Spearman vs the teacher's LLM component: 0.408.
- **Offline gate: PASSED** (0.652 > 0.545).
- **Inference speed:** a 27-min test episode (213 windows) takes 1.0 s to score (encoder +
  BiLSTM + heuristic, GPU, no embedding cache) plus a one-off bundle load of 2.0 s. The same
  episode takes minutes of LLM time with p3.
- **Blind A/B (the owner, 2026-11-14): PASSED,** with 3 wins, 0 ties and 2 losses. That is
  exactly the ≥ 3 of 5 rule.
  - The owner's answers were, in sheet order: B, B, A, B, A.
  - The filled sheet is at `~/yts-data/m15-ab/bilstm-e5s-20261114-i2/ab_sheet.md` and the blank
    one at `ab_sheet.blank.md`.
  - The verdict is stored in `model_versions.metrics.ab`, and `relevance status` reports the
    version as promotable.
  - It is a narrow margin. One more loss would have failed it.
- **Shorts check (§8), offline:**
  - M8's `select_spans` ran on i2's scores and on p3's for all 8 test episodes. Nothing was
    rendered or written.
  - For i2: 5 clips per episode (40 in total, the same as p3), every length within 20–60 s, no
    clip in the first or last 30 s (p3 has one), and a minimum gap between clips of 31–239 s.
  - The curves are not degenerate: 14–27 peaks after M8's smoothing and a curve std of
    0.076–0.110.
  - 22 of the 40 i2 clips are within 30 s of a p3 clip; the rest are different moments.
  - The owner's look at rendered clips happens after promotion, on 2–3 test episodes.
- **What i2 means for the spec:**
  - With `exact_heuristic`, spec §2.4's "BiLSTM produces the score" holds only for the LLM half.
    The heuristic half stays rule-based, as in v0, but costs seconds and no LLM.
  - This is the trade-off the data supports: the heuristic is cheap and exact, and the text
    model can only approximate it.

### Regression (2026-11-14, WSL clone)

- **Backend:** 745 passed, 10 deselected (the opt-in markers). This includes the M15 suites,
  the 0017 migration test, and the real-encoder integration test.
- **Frontend:** `ng test` 312/312.
- **GPU suite:** not run. The GPU path changed only by adding the BiLSTM branch, which the
  integration test and the real i1/i2 runs exercised on the GPU.

## Promotion and milestone closed (2026-11-14)

- **Live switch:** the owner chose option 1.
  - `models set-live relevance bilstm-e5s-20261114-i2` passed its promotion check and switched
    `heuristic-v0+Qwen3.5-9B-Q4_K_M+relevance-p3` → `bilstm-e5s-20261114-i2`.
  - **Rollback:** `python -m app.cli models set-live relevance heuristic-v0+Qwen3.5-9B-Q4_K_M+relevance-p3`.
- **Rendered shorts check:** the real handlers (`process_job`, worker stopped) ran on test
  uploads 5, 35 and 43. Upload 43 was the one with the least overlap with p3's clips.
  - Each relevance job succeeded in 0.4–3.1 s, through the BiLSTM path. Each shorts job
    succeeded in 31–45 s with 5 clips of 21–59 s (clips 316–330).
  - All 19,348 pseudo-labels (63 uploads) are intact.
  - `relevance_scores`: 3 uploads now come from i2 and 60 still come from p3. They aren't
    rescored automatically (§6).
- **Open follow-up (the owner, non-blocking):** watch the rendered clips of uploads 5, 35 and 43
  in Manage. If they disappoint, roll back (see above) and re-run those three episodes.
- **M15 is done:** the pipeline is built and tested, two train → evaluate cycles are
  documented, and i2 is promoted.
