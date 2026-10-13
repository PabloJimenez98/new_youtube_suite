# Experiment M5: local chapter models

Development log of the chapter generator's models. This log explains
**how** the models got to where they are. The **what** and the **why**
are in:
- [`docs/decisions/0009-chapter-generator-local-models.md`](../../docs/decisions/0009-chapter-generator-local-models.md)
  (design and plan)
- [`docs/research/m5-chapter-model.md`](../../docs/research/m5-chapter-model.md)
  (verified research and measurements)

## Goal

Generate YouTube chapters (start + title) for long-form podcasts in
English and Spanish, 100% locally:
1. **Boundaries:** a frozen multilingual encoder over text units feeds a
   BiLSTM that scores "a chapter starts here".
   - **Baseline:** TextTiling over the same embeddings.
   - **Shipping rule:** the BiLSTM only goes to production if it beats
     the baseline on validation F1@5s.
2. **Titles:** mT5-small fine-tuned with its embeddings frozen.

## Structure

| Path | Contents |
|---|---|
| `01_data.ipynb` | Selecting and streaming the VidChapters subset (en + es), statistics |
| `02_baseline.ipynb` | Text units + TextTiling baseline + metrics |
| `03_boundaries.ipynb` | BiLSTM training, comparison against the baseline |
| `04_titles.ipynb` | mT5-small fine-tune, sample titles |
| `05_eval.ipynb` | Final eval (test EN, test ES, the owner's episode) and export of the bundle to `backend/models/chapters/` |
| `lib/` | Shared, tested code: `metrics`, `units`, `decoding`, `baseline` (plus data and models as they come) |
| `tests/` | `pytest` over `lib/` |
| `runs/<date>-<name>/` | `config.json`, `metrics.json`, plots per run (in git) |
| `$YTS_EXPERIMENT_DATA/m5-chapters/` | Heavy data, embeddings and checkpoints (on the second drive, **not** in git) |

Notebooks are committed **with their outputs**, so the history shows
what each run produced.

## Environment

```bash
# one-time setup (from the repo root)
python3 -m venv experiments/.venv
# reuse the backend's torch 2.8 cu126 instead of installing a second copy (~4 GB)
echo "$PWD/backend/.venv/lib/python3.12/site-packages" \
  > "$(experiments/.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/zz_backend_site.pth"
experiments/.venv/bin/pip install -r experiments/requirements.txt
cp experiments/.env.example experiments/.env    # adjust YTS_EXPERIMENT_DATA

# tests
cd experiments/m5-chapters && ../.venv/bin/python -m pytest

# notebooks (or open the .ipynb in VS Code and pick the experiments/.venv kernel)
experiments/.venv/bin/jupyter lab
```

**Before training on the GPU**, stop the worker. GPU runs and
measurements need the card to themselves (lesson from 0008).

## Log

Newest entries at the bottom. Each entry: date, what was tried, result,
conclusion.

### 2026-09-27: setup and verification

- **Research verified against primary sources.** Several claims from
  the first subagent report were corrected; see §7 of the research
  document.
- **Venv:** `experiments/.venv` reuses the backend's torch through a
  `.pth` file. It takes 343 MB with transformers 5.17 + jupyter.
  CUDA is OK.
- **Measurements on this machine** (details in the research document):
  - **Encoders on CPU**, the owner's 64-min episode (713 units of ~5 s):
    - e5-small: 6.9 s
    - MiniLM-L12: 6.5 s
    - e5-base: 28.9 s
  - **mT5-small training on the 1060:**
    - Full fine-tuning: **OOM** even at batch 1.
    - Embeddings + LM head frozen: fits at batch 4 × 512 tokens
      (2.4 GB peak), ~0.35 s/step.
  - **Spanish in VidChapters:** only ~850–1.7k videos of at least
    15 min, against ~290k in English. So we go with cross-lingual
    transfer plus a Spanish-only test set.
- **`lib/` implemented with 29 tests:**
  - `metrics`: F1@Ns, tIoU-F1, mean tIoU, Pk, WindowDiff, count error
  - `units`: sentence / block
  - `decoding`: peaks + YouTube rules
  - `baseline`: embedding TextTiling
- **Metrics cross-checked against Chapter-Llama's implementation**
  (`tools/results/metrics_distance.py`, `metrics_overlap.py`) on 300
  random synthetic videos. The results are **identical**:

  | Metric | Chapter-Llama | ours |
  |---|---|---|
  | P/R @3s | 1.77 / 1.60 | 1.77 / 1.60 |
  | P/R @5s | 2.76 / 2.74 | 2.76 / 2.74 |
  | P/R @30s | 16.42 / 15.35 | 16.42 / 15.35 |
  | tIoU-F1 | 12.09 | 12.09 |
  | mean tIoU | 50.62 | 50.62 |

  So our numbers are comparable with those published by Chapter-Llama
  and VidChapters.
- **Next:** `01_data.ipynb`. It needs the second drive mounted at
  `/mnt/d` (fstab entry, run by the owner with sudo).

### 2026-09-27: preliminary baseline (`02_baseline_prelim.ipynb`)

**Setup.**
- **Data:** Chapter-Llama's `sml300_val` subset: 300 English videos,
  median 20 min, max 60 min.
- **Why this subset:** it let us start while `asrs.json` (22 GB)
  downloads. The official run of notebook 02 will use our own `val` +
  `es_val`.
- **GPU encoding:** ~500 units/s. After sorting by length it went up to
  **~1050/s** (`lib/encode.py`, with a test).

**Unit ceiling.** This is F1 after snapping the references onto unit
starts, so no model can do better with that unit:

| unit | units/video | s/unit | ceiling F1@3s | ceiling F1@5s |
|---|---|---|---|---|
| sentence (2–15 s) | 144 | 6.0 | **72.8** | **83.3** |
| block5 | 116 | 8.1 | 67.8 | 83.2 |
| block10 | 76 | 12.7 | 47.2 | 70.8 |

**TextTiling baseline vs. random** (mean over configs; best config in
brackets):

| setting | random F1@5s | baseline F1@5s | best |
|---|---|---|---|
| product (~6/h) | 4.2–4.8 | ~10.8 | 12.4 (e5-small, block10, 120 s) |
| oracle count | 7.2–8.0 | ~15.8 | 17.3 (minilm, block5, 60 s) |

**Conclusions.**
- **The baseline beats random by ~2.5×, but is weak in absolute terms.**
  There is plenty of room for the trained model.
- **The encoders are equivalent:** e5-small 10.75 vs MiniLM 10.85 in
  product. **We pick e5-small** (MIT licence).
- **Unit: we pick `sentence`.** It has the highest ceiling, and the
  baseline differences between units were marginal (~0.7 points).
- **⚠️ The creators chapter densely:** 8.7 chapters per video in
  videos of ~20–60 min. With "few and broad" (~6/h) we predict ~3.4 per
  video, so recall is capped by design. That is why we always report
  the oracle count alongside the product setting. The shipping rule
  still uses the product setting, which is what ships.

### 2026-09-27: code for 03 and 04 ready

- **`lib/`:** `store` (memmap float16 embeddings), `build_store`,
  `model` (BiLSTM + side features: relative position and similarity to
  the previous unit), `train` (soft targets, `pos_weight`, selection by
  val F1@5s after decoding), `titles` (cleaning, pairs, Adafactor with
  frozen embeddings, beam search, BLEU/CIDEr).
- **Checks:**
  - **62 tests** pass, including one where the BiLSTM learns synthetic
    topic changes (F1@5s > 90).
  - **Titles smoke test on the GPU:** 100 batches in 21 s, peak
    **1.8 GB**. The loop works; the output is still `<extra_id_0>`
    after 25 steps, which is expected. mT5's sentinels are now
    suppressed at generation time.

### 2026-09-27: backend integration and 05 ready (waiting on data)

- **05_eval + `lib/export.py`:** exports the bundle
  (`manifest.json`, `encoder/`, `boundary.pt`, `titles/`) to
  `backend/models/chapters/<version>` and updates `current`. The bundle
  `kind` follows the shipping rule decided on validation in 03.
- **Backend:**
  - `app/chapters/`: `units`, `decoding`, `baseline` and `model` are
    copies of `lib/`. A test fails if they drift.
  - `pipeline.Chapterer` (CPU)
  - the `chapters` handler, migration `0006` (+ `model_version`), and
    `GET /uploads/{id}/chapters`
- **Frontend:** the chapter list in Manage plus "Copy for YouTube".
  "Run" is disabled without a transcript.
- **Regression:**
  - backend: **143 passed, 2 skipped** (the 2 real-bundle integration
    tests, pending the export)
  - frontend: **109 passed**, and the production build is OK
  - experiment: **63 passed**
- **Waiting on:** `asrs.json`, 10.5 of 22.4 GB downloaded at 12:26.

### 2026-09-27: data (`01_data.ipynb`)

**First run: selection by title language.**
- 1 443 "Spanish" videos were selected.
- **1 328 of them had an English transcript.** fastText on short titles
  misfires (e.g. an English makeup video detected as `es`).
- Conversely, **half of the real Spanish-speech videos have English
  titles**, so title detection would never have found them.

**Fix: language decided on the transcript for all 291 704 candidates**
(1 streaming pass over `asrs.json`, ~4 min, cached).

**Final data** (15–120 min, ≥ 3 chapters, chapters of at least 10 s,
usable transcript):

| lang | split | videos | hours | median chapters | median s/sentence |
|---|---|---|---|---|---|
| en | train | 9 000 | 7 207 | 9 | 5.0 |
| en | val | 900 | 716 | 8 | 5.0 |
| en | test | 1 480 | 1 173 | 8 | 5.1 |
| es | train | 112 | 60 | 8 | 13.9 |
| es | es_val | 18 | 11 | 9 | 8.8 |
| es | es_test | 77 | 47 | 9 | 11.5 |

**Findings.**
- **⚠️ Only 207 videos in VidChapters-7M have Spanish speech** and fit
  the criteria (135 channels), against 274k in English. That is far
  below the ~0.85–1.7k estimated from titles (research §1.3, now
  corrected).
  - The Spanish split changes to 40/20/40 (train/val/test) by channel;
    a 25 % test would have been ~50 videos.
  - The channel hash is lumpy, so `es_val` ended up with only 18 videos
    and its numbers will be noisy. Spanish conclusions will be drawn
    mainly on `es_test` (77), in 05.
- **Spanish sentences are 2–3× longer** (9–14 s vs 5 s). This is
  probably WhisperX's sentence splitting on Spanish. Our `sentence`
  units cap at 15 s but can't split a longer VidChapters "sentence".
  Worth keeping in mind when interpreting the Spanish results.
- **Titles:** only ~112 training videos (~900 pairs) are in Spanish.
  They are oversampled in 04 so that mT5 doesn't learn to answer only in
  English. **Risk:** Spanish titles may come out poor; see 04/05.

### 2026-09-27: boundaries (`03_boundaries.ipynb`)

**Encoding** (e5-small, `sentence` units, GPU):
- 9 112 train videos → 3.59 M units in 54 min
- val: 900 videos, 350k units
- ~9.7 boundaries per video, ~40 units per boundary

**Training.** 4 configs × 10 epochs, ~9 min each (`big` 17 min).

| config | best val F1@5s (product) | oracle F1@5s |
|---|---|---|
| default (proj 256, h 128, pos_weight 10) | 16.24 | 22.14 |
| pos5 | 16.36 | 21.87 |
| pos20 | 16.60 | 21.75 |
| **big** (proj 384, h 256), epoch 8 | **16.86** | **22.00** |

The differences between configs are small (±0.6). The class weight
barely matters, and the larger model adds a little.

**Validation, BiLSTM `big` vs the TextTiling baseline** (same units and
decoding):

| split | setting | method | F1@5s | F1@3s | F1@30s | tIoU-F1 | Pk ↓ |
|---|---|---|---|---|---|---|---|
| val (en, 900) | product | baseline | 7.77 | 5.36 | 20.85 | 12.52 | 0.423 |
| val (en, 900) | product | **BiLSTM** | **16.86** | **13.20** | **31.63** | **18.63** | **0.361** |
| val (en) | oracle | baseline | 11.53 | 7.95 | 34.23 | 20.98 | 0.420 |
| val (en) | oracle | **BiLSTM** | **22.00** | **16.82** | **44.85** | **30.42** | **0.335** |
| es_val (18) | product | baseline | **10.31** | **5.65** | 20.17 | 11.99 | 0.401 |
| es_val (18) | product | BiLSTM | 8.17 | 5.26 | **22.33** | **12.39** | **0.347** |
| es_val | oracle | baseline | **15.74** | **9.15** | 38.07 | 23.37 | 0.349 |
| es_val | oracle | BiLSTM | 10.98 | 5.66 | **39.10** | **25.76** | **0.334** |

**Shipping rule** (val F1@5s, product): **BiLSTM 16.86 vs baseline
7.77, so the BiLSTM ships.**

**Reading.**
- **English:** the trained model **more than doubles** the baseline on
  every metric.
- **Spanish (18 videos, very noisy):**
  - **mixed.** The BiLSTM loses on precise placement (F1@5s/3s) but
    wins on coarse placement (F1@30s, tIoU-F1, Pk).
  - **consistent with** Spanish units being 2–3× longer (see 01):
    coarse units make ±5 s hard for the model, which was trained on
    ~5 s units.
  - **with 18 videos, nothing is decided here.** It is measured once on
    `es_test` (77) in 05.
  - **if the BiLSTM also loses in Spanish there,** the owner decides
    whether it's acceptable or whether the language route changes.
    That route is not decided and is not built.

### 2026-09-28: titles (`04_titles.ipynb`)

**Setup.**
- **Pairs:** chapter ASR text → chapter title, from `train`. English
  capped at 60 000 of 86 704; all 845 Spanish pairs **oversampled ×8**
  so mT5 keeps answering in Spanish. 66 760 pairs in total.
- **Training:** 1 epoch, batch 4 × grad-accum 4, lr 1e-3 (Adafactor),
  256 input tokens, embeddings + LM head frozen (44 M trainable
  params). **56 min** on the 1060.
- **Checkpoint selection:** best CIDEr on validation (500 en + 117 es,
  ground-truth boundaries), evaluated every 2 000 steps:

  | step | B1 | B4 | CIDEr |
  |---|---|---|---|
  | 2000 | 14.02 | 4.91 | 61.78 |
  | 4000 | 15.05 | 5.18 | 63.68 |
  | 4173 (end) | 14.05 | 5.01 | **64.72** |

**Validation, mT5-small vs `lead-8`** (the chapter's first 8 words):

| set | method | B1 | B4 | CIDEr |
|---|---|---|---|---|
| val_en (500) | **mT5-small** | **14.38** | **4.79** | **67.88** |
| val_en | lead-8 | 8.04 | 2.40 | 30.22 |
| val_es (117) | **mT5-small** | **12.31** | **6.11** | **49.99** |
| val_es | lead-8 | 7.88 | 2.13 | 32.75 |

**Reading.**
- **mT5 roughly doubles the extractive reference in English** and wins
  clearly in Spanish too.
- **CIDEr is still rising at the end of the epoch.** A second epoch may
  help; not tried yet.
- **Qualitatively:** many titles are reasonable short topic phrases
  ("What's going on with Vision Fund 2"). Two recurring failure modes:
  - **Repetition:** e.g. "EC2: ¿Qué es EC2? ¿Qué es EC2? ¿Qué es EC2…".
    A generation-time guard (`no_repeat_ngram_size`) would likely fix
    it; not applied, pending the owner.
  - **Lyrics / music videos** in the Spanish set produce lyric lines
    as titles. That's not the podcast domain, so it matters less.
- **Title fallback is still open** (see `docs/brainstorm.md`); the
  owner's episode QA in 05 is the real check.

### 2026-09-28: final eval and export (`05_eval.ipynb`)

**Encoding:** `test` 1 480 videos (~575k units), `es_test` 77 videos.

**Boundaries on test** (evaluated once; the shipping decision was
already taken on validation in 03):

| split | setting | method | F1@5s | F1@3s | F1@30s | tIoU-F1 | Pk ↓ |
|---|---|---|---|---|---|---|---|
| test (en, 1 480) | product | baseline | 8.54 | 6.52 | 21.29 | 12.74 | 0.423 |
| test (en) | product | **BiLSTM** | **17.89** | **14.14** | **32.79** | **18.49** | **0.366** |
| test (en) | oracle | baseline | 12.89 | 9.66 | 34.91 | 21.96 | 0.413 |
| test (en) | oracle | **BiLSTM** | **23.23** | **18.22** | **46.13** | **31.57** | **0.333** |
| es_test (77) | product | baseline | 4.32 | 1.87 | 18.92 | 11.90 | 0.417 |
| es_test | product | **BiLSTM** | **6.58** | **4.99** | **23.02** | **14.84** | **0.366** |
| es_test | oracle | baseline | 8.74 | 5.21 | 35.97 | 20.82 | 0.405 |
| es_test | oracle | **BiLSTM** | **11.19** | **7.51** | **39.88** | **24.47** | **0.342** |

**By duration (product, F1@5s baseline → BiLSTM):**
- test: 15–30 min 10.0 → 17.5; 30–60 min 8.0 → 17.6; 60–120 min
  7.0 → 17.1. The BiLSTM doesn't degrade on long episodes; the
  baseline does.
- es_test: small buckets (44 / 21 / 12 videos), noisy: 5.7 → 5.9,
  2.9 → 2.3, 1.6 → 14.0.

**Titles on test** (ground-truth boundaries, up to 800 pairs):

| set | method | B1 | B4 | CIDEr |
|---|---|---|---|---|
| test (800) | **mT5-small** | **18.30** | **8.83** | **82.09** |
| test | lead-8 | 8.98 | 3.60 | 36.86 |
| es_test (620) | mT5-small | 9.57 | 5.46 | 49.94 |
| es_test | lead-8 | 8.18 | 4.42 | 44.17 |

**Owner's episode** (upload 8, 64 min, Spanish), 6 chapters each:

| baseline | BiLSTM |
|---|---|
| 0:00:00 Chapter 5: „Advantages Of the SER | 0:00:00 Chapter 5: „Advantages Of the SER |
| 0:04:35 Chapter 5: „Cutting Calvos | 0:24:50 Pregunta sobre los jueces |
| 0:08:05 Pablo: "Cerrado", "Cerrado", … | 0:29:49 Teresa Perales Serrat - María Gray |
| 0:28:44 Case antes de Sastre y Maldonado | 0:32:57 Romero Tronco |
| 0:31:42 Mourinho 2 y 2 a.C | 0:37:20 Cuando nos siguen a casa |
| 0:48:46 Asamblea General de las Naciones Unidas | 0:45:50 Chapter 5: „Against Ceuta's Competencias |

**Reading.**
- **Boundaries, English:** the test results confirm validation. The
  BiLSTM is ~2× the baseline on every metric and every duration.
- **Boundaries, Spanish:** on `es_test` (77 videos, 4× `es_val`) the
  **BiLSTM wins on every metric**, including F1@5s (6.58 vs 4.32). The
  `es_val` loss in 03 looks like noise from 18 videos. Absolute
  Spanish numbers are still low (long units, little data).
- **Titles, English:** strong on test (CIDEr 82 vs 37).
- **⚠️ Titles, Spanish: weak.** On `es_test` mT5 barely beats `lead-8`
  (CIDEr 49.9 vs 44.2). **On the owner's episode the titles are not
  usable:** English titles on Spanish content ("Chapter 5: „Advantages
  Of the SER"), repetition ("Cerrado" ×8), and invented names/facts
  ("Mourinho 2 y 2 a.C", "Asamblea General de las Naciones Unidas").
  This is exactly the case the open **title-fallback** question
  covers. Not decided; raised with the owner.
- **Boundaries on the episode:** the BiLSTM puts its first cut at
  24:50, so the first chapter is ~25 min. With no ground truth, the
  owner judges whether the cuts match the episode's topics.

**Export:** `backend/models/chapters/bilstm-e5s-sentence-20260928+mt5s`
(2.8 GB, `current` → it). `kind = bilstm`, per the shipping rule.

### 2026-09-28: plan for 06, titles with a local pretrained LLM

**Why.** The Spanish titles from 05 are not usable, and more mT5
training can't fix that: there are only 845 Spanish pairs, and
VidChapters has no more. **The owner approved replacing
"titles from a model we train" with an experiment comparing local
pretrained instruction LLMs**, as long as it's documented. The
boundaries (BiLSTM) don't change. The outcome goes into a new decision
record (`0010`) that amends 0009.

**Hardware limits** (they define the candidates):
- Ryzen 5 2600X (6c/12t, AVX2, no AVX-512)
- 16 GB RAM (~11.8 GB free with the desktop running)
- GTX 1060, 3 GB
- **In production the `chapters` job runs in the same worker as Whisper
  and pyannote, which hold the GPU.** So the model is sized and timed
  **CPU-only**. The GPU is used only to speed up the offline evaluation.
- **Budget:** GGUF Q4_K_M, at most ~5 GB RAM (KV cache included). One
  7–8B model is included as an upper bound, to see whether size pays.
  The owner accepts slow generation. As a soft limit, a 64-min episode
  should take ≤ 15 min of titles on CPU.

**Runtime:** llama.cpp via `llama-cpp-python`.
- **Why:** in-process in the worker like Whisper, with no extra
  service. Models are GGUF files that can go inside the bundle. It
  offloads partially to the GPU for the eval.
- **Alternative rejected for now:** Ollama. It's easier to switch
  models with, but it's one more daemon in production.
- Install details (CPU vs CUDA build on Pascal) are verified in the
  first step.

**Candidates** (instruction-tuned, multilingual with Spanish; the
shortlist is re-checked for newer releases in the same size class
before downloading):

| model | size | why |
|---|---|---|
| ~~Qwen3-1.7B~~ | ~1.1 GB Q4 | the smallest that could work; speed floor |
| ~~Qwen3-4B~~ | ~2.5 GB | strong multilingual at 4B (thinking mode off) |
| ~~Gemma 3 4B-it~~ | ~2.5 GB | strong multilingual at 4B |
| ~~Llama 3.2 3B Instruct~~ | ~2 GB | Spanish officially supported |
| ~~Qwen3-8B~~ *or* Salamandra-7B-instruct (BSC, Spanish-focused) | ~5 GB | upper bound |

**Revised shortlist (2026-09-28), after re-checking for newer
releases.** Qwen 3.5 and Gemma 4 supersede Qwen3 and Gemma 3.
Llama 3.2 is dropped as superseded. All files are Q4_K_M, stored in
`$YTS_EXPERIMENT_DATA/m5-chapters/llm/`:

| model | GGUF (repo) | file | role |
|---|---|---|---|
| Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF` | 1.28 GB | speed floor |
| Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF` | 2.74 GB | 4B class |
| Gemma 4 E2B-it | `unsloth/gemma-4-E2B-it-GGUF` | 3.11 GB | 2B-effective class |
| Gemma 4 E4B-it | `unsloth/gemma-4-E4B-it-GGUF` | 4.98 GB | 4B-effective class (8B params with per-layer embeddings) |
| Salamandra-7B-instruct-2606 (BSC, Spanish-focused, Apache-2.0) | `mradermacher/salamandra-7b-instruct-2606-GGUF` | 4.85 GB | upper bound, Spanish specialist |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF` | 5.68 GB | upper bound, replaces Qwen3-8B |

**Runtime update.**
- **Production target:** `llama-cpp-python` 0.3.35, built CPU-only
  from source.
- **Why not CUDA:** the prebuilt CUDA wheels stop at 0.3.19, which is
  likely too old for these architectures, and the machine has no
  `nvcc`.
- **GPU for the offline eval only:** llama.cpp's official Vulkan
  release binaries, if the eval turns out too slow on CPU.

**Method** (`lib/llm_titles.py`, tested, plus `06_titles_llm.ipynb`):
- **One call per chapter.** Input: the chapter text, capped at ~1 500
  tokens (head + evenly spaced samples if longer), the episode language
  (detected on the transcript, as in 01), and the titles already given
  (to avoid repeats).
- **The prompt asks for** one short title (≤ 8 words), same language,
  no quotes, no invented names. Greedy decoding with
  `max_tokens ≈ 24`.
- **Post-processing** (unit-tested): strip quotes/prefixes such as
  "Title:" or "Capítulo 3:", collapse repeated n-grams, cap the length,
  and check the language. If the title fails, retry once, then fall
  back to mT5.
- The **same prompt for every model**, so we compare models, not
  prompts. One prompt iteration is allowed, on `val` only.

**Evaluation:**
1. **Automatic,** same fixed sample for every method (150 `test` + 150
   `es_test` pairs, GT boundaries): B1/B4/CIDEr, **language accuracy**
   (fastText on the title), repetition rate, mean length.
   mT5 and `lead-8` are re-scored on the same sample.
   - ⚠️ CIDEr rewards copying the creator's words. A good paraphrase can
     score low, so it's a sanity check, not the judge.
2. **Speed/memory,** CPU-only: s/chapter, peak RSS, and the total for
   the owner's 64-min episode.
3. **Owner's blind review:** all methods' titles for the episode's 6
   BiLSTM chapters, shuffled and unlabeled.

**Selection rule** (**confirmed by the owner, 2026-09-28**; both 7–8B
models are tried, Qwen3-8B and Salamandra-7B; downloads go to
`/mnt/d`):
- **Rule:** the owner's blind review picks the winner among the models
  that pass these gates:
  - language accuracy ≥ 95 % on `es_test`
  - CIDEr not far below mT5 on `test`
  - within the CPU time limit
- **If none passes:** mT5 stays, with the repetition and language
  guards added.

### 2026-09-28: 06 running overnight (CPU), setup notes

**Started** 11:53. The owner chose to let it run overnight on CPU
rather than move the evaluation to the GPU.
- **Estimate:** ~11.5 h in total. The 7–9B models take most of it.
- **Trace:**
  - `runs/2026-09-28-titles-llm/progress.log`: rows per model every
    10 min, written by `trace.sh`, a detached logger
  - `nbconvert.log`: copied there when the notebook exits
  - `gen_<model>.jsonl`: every generation, with its raw outputs and
    timings, written as it happens so a restart resumes

**Deviations from the plan, and why:**
- **The gates use validation, not test.** 05 had already looked at
  test, and selecting among 6 models on test would be selection on test.
  - The sample is 150 `val` (seed 2) + all 117 `es_val` pairs.
  - Test is run once at the end, for the winner and mT5 only.
- **The LLMs get the whole chapter**, capped by `sample_text` at 900
  words (head + 4 spread windows). mT5 keeps its 2000-char input, as
  shipped.
- **Model files download in `MODELS` order,** and the notebook waits
  for each file. Evaluation overlaps with the ~2 h download (~3 MB/s),
  and `load_s` excludes the wait.

**Implementation findings:**
- **Prompt rendering:**
  - The chat template embedded in each GGUF is rendered with jinja
    (`enable_thinking=False`), not llama-cpp-python's guessed format.
  - For Qwen 3.5 this produces an empty `<think></think>` block, so
    the model answers directly.
  - Gemma 4 renders `<bos><|turn>system…<|turn>model`.
- **BOS token:** Qwen's GGUF declares no BOS, and llama.cpp then
  reports a default id (`','`). `LlamaTitler` only uses a BOS the file
  declares (`tokenizer.ggml.bos_token_id` + `add_bos_token`).
- **Build:** `llama-cpp-python` 0.3.35 was built CPU-only
  (`GGML_NATIVE`, AVX2).
- **RSS is process-wide.** It includes torch and the notebook, not
  only the model: ~3.9 GB with Qwen3.5-2B loaded. The model's own
  footprint is roughly its GGUF size plus the KV cache.

**First numbers (validation sample, CPU, 6 threads):**
- **Qwen3.5-2B:** 8.6 s per chapter on average (38 min for 267).
  - 0 failures, 1 retry.
  - Every Spanish title is in Spanish and on topic, e.g.
    "Solucionar atasco de papel en HP ENVY 5540" or
    "Todos somos diferentes".
- **Qwen3.5-4B:** 22.8 s per chapter, so ~2.6× the 2B.

### 2026-09-28: 06 results and the owner's pick

**The runs.** Runs 1–4 died, none from a bug.
- **Runs 1–3 were killed by systemd-oomd** together with the VS Code
  scope. `setsid nohup` doesn't leave the editor's cgroup.
- **Run 4 had its own systemd `--user` unit, with `MemoryMax=11G`.**
  The cap made the GGUF page cache thrash, and oomd killed it on
  memory pressure.
- **Run 5 ran without the cap** and completed (22:14–22:44). Section 4
  now caches each model's episode titles in `ep_<model>.json`.
- **Launching:** long runs go through
  `systemd-run --user --unit=… --same-dir` and no memory cap.

**Validation** (150 `val` + 117 `es_val`, CPU, 6 threads):

| method | CIDEr en | CIDEr es | B1 en | B1 es | words en/es | episode, 6 ch. |
|---|---|---|---|---|---|---|
| mt5-small | 65.9 | 50.0 | 12.9 | 12.3 | 5.6 / 4.8 | n/a |
| lead-8 | 31.9 | 32.8 | 7.3 | 7.9 | 8 / 8 | n/a |
| qwen3.5-2b | 57.8 | 47.6 | 16.6 | 12.0 | 4.7 / 4.9 | 1.9 min |
| qwen3.5-4b | 55.9 | 44.5 | 13.8 | 10.5 | 7.3 / 8.5 | 4.6 min |
| gemma4-e2b | 47.7 | 27.9 | 13.4 | 7.7 | 6.5 / 8.5 | 2.5 min |
| gemma4-e4b | 49.3 | 53.8 | 13.2 | 11.8 | 7.0 / 7.4 | 4.7 min |
| salamandra-7b | 51.6 | 47.2 | 12.4 | 12.2 | 3.8 / 4.9 | 5.1 min |
| qwen3.5-9b | 46.1 | 54.5 | 12.8 | 11.1 | 7.1 / 7.1 | 6.9 min |

- **Every LLM** got 100 % language accuracy, 0 raw repeats and 0
  fallbacks. mT5 got 90.6 % language accuracy on `es_val`, with 6–7 %
  repeats.
- **Gates:** only **qwen3.5-2b** and **qwen3.5-4b** pass all of them.
  The others fail CIDEr en ≥ 0.8 × mT5 (52.7).
- **Episode times** are real, measured on the 6 BiLSTM chapters. They
  are above the median-based estimate, because the episode's chapters
  hit the 900-word cap.
- **Peak RSS** in the notebook process was ~10.4 GB with the 9B.

**Owner's blind review** (`blind_sheet.md`; key in `blind_key.json`):
- The owner picked **qwen3.5-9b's** title in all 6 chapters.
- mT5 won none.

**Decision** (owner, 2026-09-28):
- **Qwen3.5-4B**, following the confirmed rule: the pick is among the
  gate-passers, with no exception made for the 9B.
- Recorded in `docs/decisions/0010-chapter-titles-local-llm.md`.

**Test** (06 section 5):
- The winner vs mT5, once, on 05's test draws cut to the first 150 of
  each split.
- The full sample would take ~10 h on CPU, so the owner chose the
  reduced one.

### 2026-09-28 23:05: handoff (where M5 stands, what's next)

**Running overnight, detached:**
- **The test run:** systemd unit `m5-06-run7`, section 5 of 06,
  qwen3.5-4b vs mT5 on 150 + 150 pairs. It takes ~2 h and resumes from
  `test_qwen3.5-4b.jsonl` if restarted.
  - Check: `systemctl --user status m5-06-run7`,
    `wc -l runs/2026-09-28-titles-llm/test_qwen3.5-4b.jsonl` (300 when
    done), and `progress.log` (the tracer, unit `m5-06-trace7`).
  - Done when `runs/2026-09-28-titles-llm/test_metrics.json` exists and
    the notebook has saved with outputs.
  - If it died: `journalctl --user -u m5-06-run7`, then relaunch with
    the same `systemd-run --user --unit=… --same-dir …
    jupyter nbconvert --execute --inplace 06_titles_llm.ipynb`
    (no MemoryMax). It resumes from the caches.
- **The backend build of `llama-cpp-python` 0.3.35** (CPU,
  `CMAKE_ARGS="-DGGML_NATIVE=on"`) was started from the Claude session
  and may not have survived it.
  - Check: `backend/.venv/bin/python -c "import llama_cpp"`
  - If it fails, re-run
    `CMAKE_ARGS="-DGGML_NATIVE=on" backend/.venv/bin/pip install llama-cpp-python==0.3.35`.
  - `fasttext-numpy2-wheel` is already installed in the backend venv.

**Done (not yet committed):**
- **Decision:** `docs/decisions/0010-chapter-titles-local-llm.md`.
- **Backend:**
  - `app/chapters/llm_titles.py` (copy of the lib module) and
    `app/chapters/text.py` (`clean_title`, `LanguageId`)
  - `pipeline.py`: bundle v2, with LLM titles in order, previous
    titles passed in, fastText episode language, and per-chapter mT5
    fallback
  - `config.chapters_llm_threads`, `.env.example`, `requirements.txt`
- **Backend tests:** `tests/test_chapters_pipeline.py` has drift tests
  for `llm_titles` and `clean_title`, and fake-LLM tests. **16/16
  pass.**
- **Experiment:** `lib/export.py:add_llm_titles`, which derives the v2
  bundle from v1, plus its test. **22/22 pass**
  (`test_export` + `test_llm_titles`).
- **Docs:** this README (06 results) and `docs/brainstorm.md`
  (resolved title fallback, 9B revisit note).

**Next steps, in order:**
1. **Check the test:** read `test_metrics.json` and fill the test row
   into 0010 and this README.
2. **Export bundle v2:**
   - Add a section 6 cell to 06 that calls `add_llm_titles`, or run it
     once and log it here.
   - Arguments:
     - source `backend/models/chapters/current` (v1,
       `bilstm-e5s-sentence-20260928+mt5s`)
     - version `bilstm-e5s-sentence-20260928+qwen3.5-4b`
     - GGUF `/mnt/d/yts-experiments/m5-chapters/llm/Qwen3.5-4B-Q4_K_M.gguf`
     - lid `/mnt/d/yts-experiments/models/lid.176.ftz`
     - `llm_cfg={n_ctx: 4096, max_words: 900, max_tokens: 32, prompt_version: 'v1'}`
     - the test metrics
   - This moves `current`.
   - **Don't edit `06_titles_llm.ipynb` while an nbconvert run is
     active:** `--inplace` overwrites the file when it finishes.
3. **`backend/.env` has no `CHAPTERS_MODEL_DIR`**, so
   `test_chapters_integration.py` is probably skipped. Set it (see
   `.env.example`) and add an integration assertion: model_version ends
   with `+qwen3.5-4b`, and titles are non-empty.
4. **Manual check:** run a `chapters` job on upload 8 through the
   worker, and check the titles and the worker's RSS (budget ~5 GB for
   the LLM).
5. **Full regression:** the backend and frontend suites through the
   `test-runner` agent. Then mark M5 Done in
   `docs/development-plan.md`, update the M5 entry (titles are now
   LLM + mT5 fallback, per 0010), and update the CLAUDE.md status line.

### 2026-10-13: 06 test run, bundle v2 export, production check

- **Test run:** run 7 had died silently (unit gone, no journal) at
  16/300 rows. Relaunched as `m5-06-run8` (same `systemd-run` recipe);
  it resumed from `test_qwen3.5-4b.jsonl` and finished in ~1.5 h.
  Results in `runs/2026-09-28-titles-llm/test_metrics.json`, and as a
  table in `docs/decisions/0010-chapter-titles-local-llm.md`:
  - CIDEr en 76.9 vs mT5 90.5 (0.85×, gate ≥ 0.8 holds)
  - CIDEr es 34.2 vs mT5 49.6 (no gate; same direction as validation)
  - lang ok es 100 % vs mT5 88 %; repeats 0 %; fallback 0 %
- **Bundle v2:** exported once with `lib.export.add_llm_titles` (the
  arguments listed in the handoff above, metrics = `test_metrics.json`,
  run = `runs/2026-09-28-titles-llm`) to
  `backend/models/chapters/bilstm-e5s-sentence-20260928+qwen3.5-4b`
  (5.3 GB). `current` now points at it; v1 is kept next to it.
- **Backend:** `llama-cpp-python` 0.3.35 rebuilt in `backend/.venv`
  (CPU, `-DGGML_NATIVE=on`). `backend/.env` now sets
  `CHAPTERS_MODEL_DIR`, so `test_chapters_integration.py` runs (3/3,
  including the new `…+qwen3.5-4b` assertion).
- **Production check:** a `chapters` job on upload 8 via the arq worker
  (job 11) succeeded in ~4.3 min with the same 6 titles as
  `ep_qwen3.5-4b.json`. Worker RSS: 1.6 GB idle → 6.8 GB after.
