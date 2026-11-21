# Decision: Own LLM — From-Scratch Pretraining, Post-Training and a Qwen / Own Switch (M16)

**Date:** 2026-11-14
**Status:** Planning done (2026-11-14). Development in progress (A0 since 2026-11-14; see "Implementation notes").

## Context

- **The plan's M16** (added 2026-11-14) asks for:
  - our own transformer, trained from random initialisation
  - post-training
  - an app option to use Qwen or our model for the three LLM tasks
  - every experiment recorded in notebooks
- **Owner constraints:**
  - **No pretrained weights anywhere.** The tokenizer is ours too.
  - **Data:** post-training may use any data it needs, including Qwen3.5-9B outputs (output
    distillation, never weights) and open instruction/preference datasets.
  - **Coverage:** the model covers chapter titles (0010), show notes (0011) and the LLM part of
    relevance (0012/0024).
- **Measured (`experiments/m16-own-llm/00_system_and_gpu_benchmark.ipynb`, run 2):**
  - **Hardware:** RTX 5070 11.9 GiB (sm_120), 7800X3D, 23 GiB RAM in WSL, 836 GiB free on ext4.
  - **GEMM peak:** ~62 TFLOPS bf16, ~123 TFLOPS fp8.
  - **Llama-style decoder throughput** (V=32k, T=2048, fused AdamW, fp32 states):
    - ~110M: **60.6k tok/s** (B=8, 8.5 GiB)
    - ~330M: **18.6k tok/s** (B=2, 8.7 GiB)
    - ~730M: does not fit
  - **Run 1 was invalid.** The Windows driver silently spilled VRAM into system RAM (MFU 3–5 %),
    so the allocator must always be capped.
- **How the app uses the LLM today:**
  - One GGUF (Qwen3.5-9B Q4_K_M) is shared by the three tools through `app/llm/runtime.py`. It
    comes from the chapter bundle's manifest (`titles.llm`) and is fully offloaded to the GPU,
    with `n_ctx` 4096.
  - **Titles:** one call per chapter (≤ 900 words of transcript), at most 32 new tokens, greedy
    then one sampled retry, then the mT5 fallback.
  - **Show notes:** map-reduce. The map input is ≤ 2,600 tokens plus 256 out. The reduce input
    is ≤ 2,400 tokens plus 900 out, as grammar-constrained JSON (`DRAFT_JSON_SCHEMA`), with three
    attempts.
  - **Relevance p3:** batches of 20 blocks, grammar-constrained JSON `{"scores": [int × 20]}`,
    two passes.
    - The live relevance version is the BiLSTM `bilstm-e5s-20261114-i2`, which uses **no LLM at
      inference**.
    - The LLM is used there only as the teacher (`relevance pseudo-label`) and in the zero-shot
      version `heuristic-v0+<gguf>+relevance-p3`.
  - The chat prompt is rendered from the GGUF's own jinja template (`render_prompt`), and JSON is
    enforced by a llama.cpp grammar. A small model gets valid JSON for free from the grammar.
- **Local data:**
  - Sastre y Maldonado T1–T2: 63 episodes with p3 labels, about 40 h, about 0.6M tokens of
    transcript.
  - The M5 VidChapters splits (English).
  - 22 curated channels in `backend/app/market/seed_channels.json` (11 Spanish, 11 English).

## Decided with the owner (2026-11-14)

- **Staging (1a):**
  - **Stage A:** an end-to-end pilot at ~110M. It covers pretraining, post-training, GGUF and the
    app selector, and validates the whole chain cheaply.
  - **Stage B:** the main model, ~330M on 20–30B tokens.
- **Selector (2a):** a global default in Settings, overridable on each run of a tool. Every
  result records which model produced it.
- **Task data (3a):** download and transcribe about 300–500 h from the 22 M10 channels. Qwen3.5-9B
  is the teacher for the task data.
- **Pretraining languages (4a):** 70 % Spanish, 30 % English.
- **Pause/resume must be simple:**
  - one command, or Ctrl+C, to stop
  - one command to continue, exactly where it stopped
  - nothing lost but a few minutes on a crash
- **Dense vs hybrid (earlier, 2026-11-14):** dense is the main line. The hybrid is an ablation
  only (§3).

## Decisions

### 1. Staging and owner checkpoints

| Step | What | GPU time (est.) | Owner checkpoint |
|---|---|---|---|
| A0 | Tokenizer, pretraining data download and sharding, task-corpus download and transcription, Qwen teacher outputs | ~2–3 days (Whisper ~11 h for 500 h; teacher ~1–2 days) | — |
| A1 | Ablations at ~110M, 1B tokens each (§3) | ~8 runs × ~4.6 h ≈ 1.5 days | — |
| A2 | Stage A pretraining: ~110M, 5B tokens, best ablation recipe | ~1 day | — |
| A3 | Stage A post-training, GGUF export, app selector, eval | ~1 day | **C1:** the owner tries the A model in the app (functional, not quality) |
| B0 | Fix the stage B size and budget from the A1 scaling and the A2/A3 lessons | — | **C2:** the owner approves the wall-clock budget before launch |
| B1 | Stage B pretraining: ~330M, 20–30B tokens (default 25B, ~15.5 days at 18.6k tok/s), then a 4k context extension | ~2–3 weeks | — |
| B2 | Stage B post-training and eval | ~2–3 days | **C3:** blind review per task (titles, show notes, relevance) |

M16 closes after C3, whether or not the own model becomes the default for any tool (§11).
Pauses add calendar time, not GPU time.

### 2. Architecture baseline: a dense decoder that llama.cpp already runs

- **Graph:** pre-norm decoder-only.
  - RMSNorm, RoPE (θ = 100,000 from the start, so the 4k extension needs no RoPE change)
  - SwiGLU MLP (hidden ≈ 8/3·d, rounded to 64)
  - grouped-query attention, no biases, tied input/output embeddings
  - attention via `F.scaled_dot_product_attention` (flash), as measured in notebook 00
- **Sizes:**
  - Stage A: d768 / L12 / 12 heads (4 KV heads).
  - Stage B: d1024 / L24 / 16 heads (4 KV heads). Final shape from §3 depth vs width.
- **Export constraint:** the graph must map 1:1 onto a llama.cpp architecture.
  - It is `llama` by default.
  - If QK-norm wins its ablation, the export target is `qwen3`, which llama.cpp supports with
    per-head q/k RMSNorm.
  - Anything that maps onto neither (ReLU², value embeddings, the hybrid) needs §10's PyTorch
    serving path, and is only adopted with the owner's agreement.
- **Precision:** bf16 autocast with fp32 master weights. FP8 training is not used, because it
  adds risk and the measured bf16 MFU is already about 78–86 %.
- **Written by us in PyTorch** (`llm/model.py`). No `transformers` model classes, and no trl.

### 3. Ablations at ~110M (A1, notebook 02)

- **Protocol:** each run trains on the same 1B-token prefix of the stage A mix with the same
  seed.
  - **Metric:** bits per byte on a fixed held-out set: Spanish web, English web and our
    transcripts, reported separately. Bits per byte is tokenizer-independent and comparable
    across runs.
  - **Throughput** is recorded too.
  - **Rule:** a change is adopted if it improves combined bpb by ≥ 0.01 at ≤ 10 % throughput
    cost, or if it improves bpb at no cost.
- **Candidates:**
  1. Baseline (§2) with AdamW.
  2. Muon (2-D hidden matrices) + AdamW (embeddings and norms).
  3. QK-norm.
  4. ReLU² instead of SwiGLU at the same parameter count. It doesn't export to llama.cpp, so it
     needs a big win.
  5. Deeper and thinner vs wider and shallower, at equal parameters.
  6. The learning-rate schedule: warmup-stable-decay (trapezoidal) vs cosine. WSD also lets a
     paused run extend its budget without a restart.
  7. Hybrid: Gated DeltaNet + full attention 3:1. Only if its Triton kernels run correctly on
     sm_120; failing that is recorded in the notebook as a result.
- **Scaling points:** three extra runs at 0.5B / 1B / 2B tokens on the winning recipe give the
  loss-vs-tokens curve used at B0 to size stage B.

### 4. Tokenizer

- **Type:** byte-level BPE, **32,768** vocab, trained with HF `tokenizers` (a library, not
  pretrained weights).
  - **Training data:** ~2 GB sampled from the 70/30 mix plus task-corpus transcripts.
  - **Pre-tokenizer:** the Llama-3 split regex, so llama.cpp's existing `llama-bpe`
    pre-tokenizer reproduces it exactly. This is checked by the parity test.
- **Special tokens:** `<|endoftext|>` (document separator and EOS), `<|im_start|>` and
  `<|im_end|>` (a ChatML-style template), and `<|pad|>`.
- **Chat template:**
  - A jinja ChatML template stored in the GGUF metadata, so the existing `render_prompt` works
    unchanged.
  - It ignores `enable_thinking`, because the model has no thinking mode.
  - The same template formats the SFT data.
- **Acceptance (notebook 01):**
  - fertility (tokens per word) on Spanish and English held-out text, compared with Qwen's
    tokenizer as a reference
  - lossless round-trip
  - accented characters and ñ inside words handled without fragmentation

### 5. Pretraining data (A0, notebook 01)

- **Sources:**
  - FineWeb-2 `spa_Latn` for the 70 % Spanish share.
  - FineWeb-Edu for the 30 % English share.
  - Notebook 01 checks for a quality-filtered Spanish derivative and its licence before
    downloading. Licences are recorded in the notebook.
- **Volume:** stage B at 25B tokens needs about 17.5B Spanish and 7.5B English tokens. Tokenised
  shards are uint16 (`.bin` plus an index), about 50 GB.
  - Raw parquet is deleted after tokenising, once the shard checksums are recorded.
  - Location: `~/yts-experiments/m16-llm/data/` on WSL ext4.
- **Decontamination:** every evaluation set (§9) is n-gram-checked against the shards, and hits
  are removed. Task-corpus transcripts do not enter pretraining; they are post-training data
  only.
- **Packing:** documents are concatenated with `<|endoftext|>` into fixed 2,048-token rows, and
  attention is not masked across documents. That's standard at this scale; per-document masking
  is not worth its complexity.

### 6. Task corpus and teacher outputs (A0, notebook 03)

- **Download:**
  - yt-dlp with audio only, in the existing venv (0016).
  - Up to about 20 h per channel from the 22 seed channels, choosing the most recent long-form
    videos (≥ 15 min, no shorts), about 300–500 h in total.
  - Location: `~/yts-data/podcasts/m16-corpus/`, never committed.
  - Not all of the English channels need full volume, since 30 % of the corpus is English.
- **Processing:** a standalone script in `experiments/m16-own-llm/` calls the backend's library
  functions directly. It uses no worker jobs and no dev-DB rows, so the app's data stays clean.
  1. Transcription with the production Whisper settings (0016).
  2. Chapters (the BiLSTM boundaries plus **Qwen titles**).
  3. Show notes (the **Qwen** map-reduce, keeping the map and reduce calls separately as training
     examples).
  4. Relevance p3 window scores from **Qwen**.
- **Output:** JSONL records of `(task, messages, teacher output, metadata)`. Diarization is
  skipped, because no task uses speakers.
- **Split by episode, frozen in `split.json`:**
  - **Test:** two whole channels (one Spanish, one English) and the 0024 test episodes of Sastre
    y Maldonado.
  - **Val:** 5 % of the remaining episodes.
  - Nothing from test enters any training stage.
- **The teacher only runs while training is paused or finished.** It needs the GPU.

### 7. Training infrastructure and pause/resume

- **One CLI for every stage** (pretrain, extension, SFT, DPO): `python -m llm.train
  {start <config>|pause|resume|status}`, run in WSL. A thin Windows wrapper
  (`scripts/llm-train.cmd pause|resume|status`, next to the existing WSL launcher) makes it a
  double-click or one PowerShell command.
- **`start` and `resume`:** launch the trainer detached (`setsid`, logs in the run dir), so
  closing the terminal does not stop it.
- **`pause`, Ctrl+C or SIGTERM:**
  - The trainer finishes the current optimizer step, writes a checkpoint and exits.
  - VRAM is freed by the process exit, typically within seconds.
- **`resume`** (no arguments) continues the most recent run from its latest checkpoint, with
  bit-identical state:
  - model and optimizer
  - schedule step and tokens seen
  - data-loader position (shard, offset)
  - the Python, NumPy, torch and CUDA RNG states
  - a config hash; resume refuses on a mismatch
- **Autosave every 15 min.** Checkpoints are written atomically (temp file, then rename).
  - The last 2 are kept, plus a permanent one every 10 % of the budget.
  - A power cut, Windows update or WSL shutdown loses at most 15 min.
- **`status`:**
  - running or paused
  - step, tokens seen and % of the budget
  - tokens/s, ETA and the last train/val loss
  - the path to the latest sample generations
- **Guards:**
  - `torch.cuda.set_per_process_memory_fraction(0.92)` always (notebook 00, run 1).
  - `start` and `resume` refuse to launch if less than 9.5 GiB of VRAM is free. That means the
    app worker or a game is using the GPU, and the message says so. Training and app jobs never
    share the card.
- **Validation and samples:** every 250M tokens the trainer writes the held-out bpb plus a few
  fixed-prompt generations to the log. Notebook 04 plots the log while the run is going.
- **The owner's note:** Windows sleep must be off during long runs. The trainer cannot prevent
  it, and the autosave bounds the loss.

### 8. Pretraining budget and context

- **Stage A:** 5B tokens at T=2,048.
- **Stage B:**
  - Default 25B tokens at T=2,048, then a **context-extension phase at T=4,096** for the last
    ~5 % of tokens, which matches the app's `n_ctx` of 4096.
  - Final size and tokens are fixed at B0 within 20–30B, from the A1 scaling curve.
  - With the WSD schedule, the decay phase starts only when the owner confirms the budget is
    final. A run can be extended while it is still in the stable phase.

### 9. Post-training (notebooks 05–06)

Every stage uses the ChatML template, with the loss only on assistant tokens.

1. **General SFT:** an es/en mix from open instruction datasets.
   - Candidates to verify in notebook 05: `HuggingFaceTB/smoltalk`, `CohereForAI/aya_dataset`
     (Spanish part) and `allenai/tulu-3-sft-mixture`.
   - Licences are recorded, and the target is about 70 % Spanish.
   - It is filtered to sequences of 4,096 tokens or fewer.
2. **Task SFT:** the §6 teacher outputs for all three tasks, in the exact prompt format the app
   sends, plus VidChapters titles (the M5 train split).
   - 10–20 % general-SFT replay limits forgetting.
3. **DPO:** pairs per task.
   - **Chosen:** the teacher output, or the best of n own-model samples under the §9.4
     verifiers.
   - **Rejected:** own-model samples that fail a verifier or lose to the teacher. English
     general preference data (e.g. `HuggingFaceH4/ultrafeedback_binarized`) may be mixed in if
     it helps on val.
4. **GRPO (optional, only if B2 has time):** with verifiable rewards:
   - schema-valid JSON before repair
   - correct language (the bundle's fastText LID)
   - title ≤ 8 words that is not already used
   - exactly 20 integers for relevance
- **Loops:** SFT, DPO and GRPO are written by us on our model class (`llm/post.py`). trl assumes
  `transformers` models.

### 10. Evaluation, gates and serving

- **Base model (pretraining):** held-out bpb (es / en / transcripts) and fixed-prompt samples,
  for monitoring only.
- **Gates on the frozen test split.** "Selectable" requires the functional gate. "Default for a
  tool" requires that plus winning C3.

| Task | Functional gate (selectable) | Quality comparison (informs C3) |
|---|---|---|
| Titles | ≤ 10 % of chapters fall back to mT5; ≥ 95 % in the right language | CIDEr vs the M5 references (VidChapters, en) for own / Qwen / mT5; CIDEr vs Qwen's titles as silver references (es) |
| Show notes | ≥ 95 % of episodes produce a valid draft within the 3 reduce attempts; ≥ 95 % right language | blind side-by-side with Qwen on 5 held-out episodes |
| Relevance (zero-shot version) | 100 % valid batches; Spearman with Qwen p3 ≥ 0.8 × Qwen's own pass-to-pass Spearman (measured first, in notebook 07) | blind top-5 as C4 in 0016 |

- **Export (notebook 08):** `llm/export_gguf.py` writes an f16 GGUF (~0.7 GB for 330M) with the
  `gguf` Python package.
  - It stores the architecture metadata, tokenizer, merges, `llama-bpe` pre-tokenizer, special
    token ids and the chat template.
  - No quantisation: at this size f16 fits easily and keeps parity tests exact.
- **Parity:** llama.cpp vs PyTorch must give identical token ids and close next-token logits on
  fixed prompts (§Test plan).
- **PyTorch serving path:** only if an ablation that llama.cpp can't run wins and the owner
  accepts it. Otherwise it is not built.

### 11. App integration

- **Model store:** `backend/models/llm/<version>/` holds `model.gguf` and `manifest.json` (label,
  `n_ctx`, training reference, eval summary). `current` points at the active own model, the same
  pattern as `models/chapters/current`. All of it is gitignored.
- **New setting `LLM_OWN_DIR`:** unset or missing means the "own" option is hidden in the UI,
  and the API rejects it with a clear message.
- **Resolution:** a new `app/llm/registry.py` provides `resolve_llm(choice)` → `(gguf path,
  n_ctx, label)`.
  - `qwen` resolves to the chapter bundle's GGUF exactly as today.
  - The generator and scorer caches in the show-notes, relevance and chapters modules are keyed
    by the choice.
  - Titles: the chapter pipeline uses the resolved LLM instead of `titles.llm` when the choice is
    `own`. The mT5 fallback and the LID are unchanged.
- **Choice precedence:** `jobs.params["llm"]` (per run), then the global setting, then `qwen`.
  - The global setting lives in a new single-row-per-key `app_settings` table (migration `0018`,
    key `llm_default`), with `GET`/`PUT /settings/llm`.
  - Qwen stays the global default until a tool wins C3.
- **Frontend:**
  - A Settings control for the global default.
  - A small "Model: Qwen / Own" selector on the launch of chapters, show notes and relevance,
    defaulting to the global value.
  - Results show the model label.
- **Traceability:** `chapters.model_version`, `show_notes_drafts.model_name` and the relevance
  version label already embed the GGUF stem. Own-model outputs are identifiable with no schema
  change.
- **Relevance nuance:**
  - The live version is the BiLSTM, which runs no LLM. For relevance, choosing **Own** runs the
    zero-shot version `heuristic-v0+<own gguf>+relevance-p3`; choosing **Qwen** keeps today's
    behaviour (the live version).
  - The `relevance pseudo-label` CLI gets `--llm qwen|own`, so the own model can also serve as
    the BiLSTM's teacher.
  - Retraining the BiLSTM on own-model labels is **not** in M16's scope (noted in brainstorm).
- **VRAM:** a 330M f16 model plus a 4k KV cache is well under Qwen's ~5.5 GB, so there is no
  change to the 0016 policy (`gpu_lock`, `release_all()`).

### 12. Layout and notebooks

- **Code:** `experiments/m16-own-llm/`.
  - `llm/`: `model.py`, `tokenizer.py`, `data.py`, `train.py`, `post.py`, `export_gguf.py`
  - `configs/*.yaml` and `tests/`
  - the backend integration in `backend/app/llm/`
- **Execution:** code is edited in the Windows checkout and run from WSL with
  `experiments/.venv`.
  - Data, shards, checkpoints and logs live on ext4 under `~/yts-experiments/m16-llm/`.
  - New experiment dependencies: `datasets`/`pyarrow`, `huggingface_hub`, `gguf`, `pyyaml`.
- **Notebooks**, numbered. Every experiment is recorded, including failed runs, which are kept
  and annotated as run 1 of 00 was.
  - 00 hardware benchmark (done)
  - 01 tokenizer and pretraining data
  - 02 ablations
  - 03 task corpus and teacher
  - 04 pretraining monitor (stage A, then B)
  - 05 SFT
  - 06 DPO / GRPO
  - 07 evaluation and gates
  - 08 GGUF export and parity
- **Long runs** are CLI processes, not notebook cells. The notebooks launch short experiments and
  analyse the logs of long ones.

## Alternatives rejected

- **Fine-tuning Qwen or any pretrained base (QLoRA etc.):** it would give much better quality,
  but the owner explicitly wants no pretrained weights.
- **Going directly to ~330M (no stage A):** a broken link in the chain (tokenizer pre-tokenizer,
  GGUF export, chat template) would surface only after weeks of GPU.
- **~500–700M:** needs 8-bit or Muon-only states plus activation checkpointing, and 6–10 weeks of
  GPU for pretraining alone.
- **Hybrid linear attention as the main line:** its gain is mostly long-context efficiency we
  don't need at 4k, it is weaker at exact recall (which extractive tasks need), it can't be
  served by llama.cpp, and its sm_120 kernels are unverified. It is kept as ablation 7.
- **FP8 training:** about 2× GEMM peak, but more numerical risk, and bf16 already reaches about
  80 % MFU.
- **trl / `transformers` trainers:** they assume HF model classes. Our own loops are short and
  keep the model ours.
- **Running the teacher through worker jobs into the dev DB:** that would mix hundreds of
  foreign episodes into the app's library, insights and analytics.
- **Global-only or per-tool-only selector:** the owner chose global plus per run (2a), which
  allows side-by-side comparisons.
- **Pretraining at T=4,096 throughout:** about 20 % more compute for the 95 % of tokens that
  don't need it.

## Test plan

**Fast CPU suite, `experiments/m16-own-llm/tests/`.** Tiny models (d64, L2), no GPU.

- **Tokenizer:**
  - Lossless round-trip on es/en text with accents, ñ, emoji and numbers.
  - Each special token is a single id.
  - The chat template renders SFT examples and app prompts identically.
- **Data:**
  - The shard writer and reader round-trip.
  - The loader is deterministic for a given (seed, position).
  - Packing inserts `<|endoftext|>` between documents.
  - The decontamination filter removes a planted eval n-gram.
- **Resume (the owner's key requirement):**
  - Train N steps uninterrupted vs N/2, pause, then resume N/2. Weights, optimizer state and the
    loss sequence must be identical (bitwise on CPU).
  - The `PAUSE` file and SIGTERM both give a clean exit with a checkpoint.
  - Resume with a changed config refuses.
  - A checkpoint interrupted mid-write (simulated) leaves the previous one usable.
- **Post-training:**
  - The SFT loss mask covers only assistant tokens.
  - On a toy pair, the DPO loss decreases when the chosen log-probability rises.
  - The GRPO verifiers (JSON schema, language, title rules, 20 integers) have accept and reject
    cases.
- **Export:** a random tiny model is exported to GGUF and loaded with `llama_cpp` on CPU.
  - Token ids are identical to our tokenizer on 50 es/en strings.
  - The final-position logits match PyTorch within a tolerance (max abs difference ≤ 1e-2 in
    f16).
  - The embedded chat template renders through `render_prompt`.

**Backend fast suite**, added to `backend/tests`, CPU-pinned as usual.

- `resolve_llm`: `qwen` gives the bundle's GGUF, `own` gives `LLM_OWN_DIR/current`, and `own`
  without a model gives a clear error.
- Choice precedence: the job param over the global setting over `qwen`.
- `GET`/`PUT /settings/llm`, plus the migration `0018` up and down.
- The chapters, show-notes and relevance handlers pass the choice through to a fake `ChatLLM`.
  The version label contains the own GGUF stem.
- `relevance pseudo-label --llm own` reaches the own model.
- **Selector set to `qwen` is unchanged:** the existing fakes and fixtures still give the same
  outputs.

**Frontend (`ng test`):**
- The Settings control loads and saves the default.
- The per-run selector defaults to the global value and is hidden when the own model is
  unavailable.
- Results show the model label.

**GPU suite (`pytest -m gpu`):**
- The own GGUF loads fully offloaded.
- It produces a title, a schema-valid show-notes draft and a valid 20-score relevance batch on
  the fixture episode.
- Its peak VRAM is within the 0016 budget.
- Co-residency with Whisper and pyannote is unchanged.

**Evaluation (notebook 07):** the §10 gates for stage A (C1 is functional only) and stage B (C3).

**Manual:**
- **Pause/resume during a real stage A run:** pause, play a game or watch video, resume. Check
  that the loss curve continues and `status` is right. Done by the owner at least once before
  stage B.
- **C1:** use the A model in the app.
- **C3:** blind reviews.

**Regression:** the backend fast suite, `-m gpu` and `ng test` with the selector on `qwen`, plus
a greedy Qwen run on a fixed episode, comparing titles and relevance scores before and after the
integration.

## Risks and open items

- **Quality gap:** ~330M on ~25B tokens is far below Qwen3.5-9B. Show notes (long input,
  structured output) is the most likely to stay on Qwen; that outcome is acceptable (§1, closes
  at C3).
- **Teacher time competes with training for the GPU.** A0 is scheduled before any long run, and
  the stage B teacher top-ups (if needed) go in pauses.
- **Dataset availability and licences** (§5, §9) are verified in notebooks 01 and 05 before
  download. If a candidate is gone or its licence doesn't fit, it is replaced and the change is
  recorded in this file.
- **llama.cpp tokenizer parity** for a custom BPE is the main export risk. It is caught at stage
  A by the export tests, which is the reason stage A exists.

## Implementation notes (A0, 2026-11-14)

- **Spanish source: `epfml/FineWeb2-HQ` `spa_Latn`, not plain FineWeb-2.**
  - It is the top-10 % quality subset of FineWeb-2, ODC-By and ungated, the same licence as
    FineWeb-2.
  - Size: 504 files, about 115k documents each. Each file holds about 0.39 GB of text, roughly
    200 GB in all, so the 95 GB target is about half of it.
  - The files carry an embedding column (1.1 GB per file). `prepare_data fetch` keeps only `text`
    (about 160 MB of zstd parquet per file) and deletes the download.
  - English source: FineWeb-Edu `sample/10BT`, all 14 files (about 48 GB of text).
- **Measured rates:**
  - HF download: about 1.6 files/min (es).
  - yt-dlp: about 0.6 MB/s per stream, so `-N 4` concurrent fragments are used (about 3 MB/s).
  - Corpus processing: **about 3.3 s per minute of audio**. For a 112-min episode that is Whisper
    141 s, chapters 17 s, show notes 78 s and relevance 134 s, with about 80 recorded Qwen calls.
    About 22 h of GPU for the full corpus.
- **Task corpus:** 374 h selected from 20 channels (231 h es, 143 h en).
  - La Cotorrisa is excluded: its `ref` was already unresolved in M10 (0015).
  - Sastre y Maldonado T1–T2 (58 local files) are added from `~/yts-data`.
  - **Frozen split** (`m16-corpus/split.json`):
    - **Test:** the whole channels `poco-se-habla` (es) and `lenny-s-podcast` (en), plus the 8 test
      episodes of 0024 (upload ids mapped to YouTube ids, read-only, in
      `~/yts-data/podcasts/sym_split_0024.json`).
    - **Val:** the 7 val episodes of 0024, plus 5 % of the rest by id hash.
- **Teacher capture:** `llm.corpus process` runs the app's own pipelines (WSL clone, production
  settings) and records every LLM call through a wrapper around `ChatLLM.complete`, plus a patch on
  `LlamaTitler.title`.
  - Each call is stored as `{task, messages, output, json_schema, max_tokens}` in `calls.jsonl`, so
    task SFT uses the exact prompts the app sends.
- **Muon** (`llm/optim.py`) is implemented for ablation 2: Newton-Schulz on 2-D hidden matrices,
  AdamW on the tied embedding and the norms. Each parameter group carries a `base_lr` scaled by the
  WSD factor.
- **Cleanup** (owner, 2026-11-14): intermediates are deleted as soon as they are consumed (TRAINING.md
  §5b).
  - Ablation runs keep only their last checkpoint.
  - The WSL vhdx lives on C: and needs sparse mode or compaction to give the space back.
- **WSL background processes** (`setsid nohup`) survive with no terminal open, because the distro
  runs systemd. This was verified with a 3-minute idle check.
- **Owner documentation:** `experiments/m16-own-llm/TRAINING.md` (Spanish).

## Implementation notes (A0 + early A3 code, 2026-11-14 evening)

- **Network incident (2026-11-14, about 16:45–19:00).**
  - **Cause:** sustained parallel downloads filled the IPv4 NAT state of the ISP's CGNAT. The first
    hop after the router is `10.0.5.242`, a private address. New IPv4 TCP connections then failed for
    the whole PC while ICMP and IPv6 kept working, which is what the owner saw as slow internet.
  - **Main contributor:** `hf_xet`, which opens dozens of connections per file. Not WSL: restarting
    WSL only seemed to help because the NAT state expired in the meantime.
  - **Changes:**
    - WSL now uses mirrored networking (`C:\Users\Pablo\.wslconfig`), which gives WSL native IPv6.
    - HF downloads use plain HTTP with one connection (`HF_HUB_DISABLE_XET=1`, `--workers 1`). The
      HF CDN is IPv4-only.
    - yt-dlp runs with `--ip 4 --fragments 1` behind `tools/download_guard.sh`, which stops the
      download if IPv4 fails for 3 consecutive minutes.
    - WSL's mirrored IPv6 later proved flaky, so IPv6-only downloading is not relied on.
  - **Backend tests** must run with `HF_HUB_OFFLINE=1` while IPv4 is unreliable:
    `test_chapters_pipeline` calls `from_pretrained`, which otherwise hangs.
- **Spanish text paused at 37.8 GB** (128 files, **8.24B tokens**). With English (**10.56B
  tokens**) that is 18.8B, enough for A1 and stage A. The rest of the 95 GB target (for stage B,
  about 21B es tokens) resumes later with a single connection, preferably at night.
- **Post-training code done** (0025 §9), all reusing the trainer, so it inherits exact
  pause/resume, checkpoints and the queue:
  - `llm/sft_data.py`: ChatML rendering tokenized in one piece, exactly as the app tokenizes
    prompts, with an assistant-only mask from character offsets.
  - `llm/task_data.py`: teacher calls → SFT examples, keeping only usable outputs.
  - `llm/dpo.py`: pairs, reference log-probs and the loss.
  - Trainer support: `data.kind: sft|dpo`, `objective: dpo`, `init_from`.
- **App integration done in the backend** (0025 §11):
  - `app/llm/registry.py`, `app/settings_store.py`.
  - `app_settings` table (migration `0018`), `GET`/`PUT /settings/llm`.
  - Job params `{"llm": "qwen"|"own"}` for chapters, shownotes and relevance, resolved and stored at
    creation (the request's choice, else the global default).
  - **Handlers:** Qwen keeps the exact pre-M16 calls. A relevance run with own runs the zero-shot
    version over the own GGUF (even when the BiLSTM is live), never writes pseudo-labels and never
    becomes live.
  - `relevance pseudo-label --llm qwen|own`.
- **Tests:**
  - experiments: 38 (resume parity incl. Muon, SFT and mid-step pauses; queue; GGUF parity; SFT/DPO
    runs).
  - backend: `test_llm_choice.py` (17) and `test_llm_own_gguf.py` (4), the latter loading a real
    exported GGUF fixture through `LlamaChat`/`LlamaTitler`, plus migration `0018`.
  - Backend fast suite: 761 passed, 5 skipped (baseline before the change: 740 + 5).
  - `-m gpu` and `ng test` are still pending.
- **Still open:**
  - ~~The frontend selector. Where the global default lives is an owner question: there is no
    Settings screen.~~ Resolved 2026-11-15: Settings screen + per-tool picker (see below).
  - Whether to filter relevance batches with a strong position bias from task SFT (notebook 03).

## Implementation notes (A0 completed + A1 started, 2026-11-15)

- **A1 launched** (11:39): the 8 ablation configs run as one queue (`a1-baseline`, `a1b-lr2e-3`,
  `a2-muon`, `a3-qknorm`, `a4-relu2`, `a5-deep`, `a5b-wide`, `a6-cosine`), ~4.4 h each at ~68.6k tok/s.
  `a1-baseline` finished with val bpb 1.0743. The hybrid (§3.7) and the scaling points (§3) have no
  config yet; the scaling runs need the winning recipe first.
- **Downloads done:**
  - Corpus audio: 284 of 309 selected videos. The other 25 are members-only or HTTP 403 and cannot be
    downloaded; not replaced (owner may ask). 111 episodes wait for the GPU.
  - Spanish text for stage B: complete, 95.3 GB in 333 files (`tools/fetch_guard.sh`: one connection,
    IPv4, stops after 3 min without IPv4 and relaunches). Not tokenized yet: appending shards while A1
    reads the index is avoided, even though the loader reads shards in order.
- **`tools/after_ablations.sh`** waits until all 8 ablations are `done` and no trainer is alive, then
  runs `llm.corpus process` and `prepare_data tokenize --source es`. A manual pause never triggers it,
  and it never starts stage A (the recipe is chosen from notebook 02 with the owner).
- **`tools/m16_watch.sh`** (`scripts\llm-watch.cmd`): read-only live view of all of the above.
- **General post-training data (§9 steps 1 and 3), prepared ahead of A3** (owner, 2026-11-15):
  `llm/general_sft.py`, notebook 05.
  - **Chosen:** `HuggingFaceTB/smoltalk2` multilingual no_think, Spanish only by LID (Apache-2.0);
    `CohereLabs/aya_dataset` es+en (Apache-2.0); `Iker/OpenHermes-2.5-Spanish` (Apache-2.0, machine
    translation of GPT-4 outputs); `HuggingFaceTB/smol-smoltalk` (Apache-2.0); and for DPO
    `HuggingFaceH4/ultrafeedback_binarized` (MIT).
  - **Rejected:** `allenai/tulu-3-sft-mixture` (built for 8B+, maths/code heavy; its multilingual part
    is Aya, taken directly) and the full `smoltalk` (smol-smoltalk is its small-model subset).
  - **Mix** (`data/sft/general/mix.json`): 150M target tokens, 70.0 % Spanish. Native Spanish data is
    only ~22M target tokens, so the translated OpenHermes carries the Spanish share; it is the first
    thing to down-weight if A3 shows translationese.

## Amendment: M16 ends after stage A (owner, 2026-11-15)

**Why.** The owner may not be able to keep the PC training for the ~3 weeks stage B needs. Faking
the missing stages (filling notebooks with invented results, or passing off a downloaded model as
ours) was considered and **rejected**: the results must stay real. Instead the scope is cut to what
can actually be run.

**Decided:**
- **In scope:** A1 → A2 → A3 → **C1**. The ~110M stage A model is M16's deliverable: our tokenizer,
  our pretraining from random initialisation, our post-training (§9 steps 1–3), GGUF export and the
  app selector. M16 closes after C1 and the regression pass.
- **Out of scope, left as future work:** B0, B1, B2 and checkpoints C2 and C3. The full stage B
  run is described as beyond the project's scope in the write-up, with the cost measured here
  (~15.6 days of GPU for 25B tokens at 18.6k tok/s, then ~2.5 days of post-training).
- **Stage B is still written, not run:** `configs/stage_b.yaml` and its notebooks (stage B
  pretraining monitor, sizing from the A1 scaling runs, post-training and eval) are written once A3
  has settled the code they reuse. Each one starts with a banner saying it was **not executed** and
  holds no outputs. Nothing in them is presented as a result.
- **The A1 scaling runs (§3) stay**: they are cheap (~14 h) and they are the evidence the stage B
  sizing would rest on.
- **Selector (owner, 2026-11-15):** a minimal **Settings** screen for the global default, plus a
  Qwen/own choice when chapters, show notes or relevance are launched (§11, unchanged).
- **Selectability:** the own model is selectable in the app once exported, as the owner's explicit
  choice. The §10 functional gates are still measured on the frozen test split and reported as
  measured; failing one does not hide the option, but it is shown in the notebook and to the owner
  at C1. **Qwen stays the global default**, since without C3 no tool can switch its default.
- **Data:** the extra Spanish text fetched for stage B (205 files, 22.9 GB of zstd text, not
  tokenized) was **deleted** (owner). The manifest now lists only the 128 tokenized files, so a
  future `fetch` downloads them again (~2 h). `tools/after_ablations.sh` no longer tokenizes.
- **Expected quality:** a 110M model on 5B tokens will be well below Qwen3.5-9B. Titles and the
  grammar-constrained relevance scores are its best chances; show notes the weakest. All of it is
  reported as measured.
- **Frontend selector done (2026-11-15),** closing the open item from the 2026-11-14 notes:
  - `/settings` (new nav entry "Settings"): the global default, with our model disabled and marked
    "not installed" while `LLM_OWN_DIR` has no export. Saving shows the backend's 422 message.
  - The chapters, show-notes and relevance tool bars get a Qwen / own picker, preset to the default
    and overridable per tool. It is **hidden while our model is not installed**, and then runs send
    no `llm` param, so the app behaves exactly as before M16 until A3 exports a model.
  - A chip next to a run's status shows which model it used (`job.params.llm`).
  - Tests: `ng test` 324 passed (40 files; new: `llm-settings.spec.ts`, `settings.spec.ts`, 4 picker
    cases in the workspace spec, the six-link shell spec). `ng build` passes. The backend was not
    changed in this step.

## A1 result and the stage A recipe (2026-11-17)

Picked by Claude under the owner's authorisation of 2026-11-16: choose the recipe mechanically by
the §3 rule and launch the scaling runs, then A2, without waiting. The full table is in notebook 02.

- **Measured:** 8 runs at 1B tokens each. Baseline bpb is 1.0743 (es 0.9886, en 1.1601) at 68.8k
  tok/s.
  - lr 2e-3: −0.0164 bpb, no cost.
  - Muon: −0.0241 bpb, −2.2 % throughput.
  - QK-norm: −0.0091 bpb, −1.7 %.
  - wide (d1024/L7): −0.0038 bpb.
  - ReLU²: +0.0010 bpb.
  - deep (d576/L21): +0.0026 bpb, −11.3 %.
  - cosine: +0.0305 bpb.
- **Adopted: AdamW lr 2e-3.** It is the only change that passes the rule. Every candidate ran at
  lr 1e-3, so each win is checked against a1b as well.
- **Muon not adopted:** it beats a1b by only 0.0076 bpb, with a throughput cost. This is the
  closest call of A1. Muon together with a higher AdamW LR was not measured. The owner can still
  switch it before A2 starts.
- **Nothing needs the owner's agreement:** ReLU² did not win, and the hybrid was not run (no
  config).
- **Gap:** the transcripts were not reported as a separate held-out set (§3 asks for it). The eval
  covers es and en web only.
- **Recipe for A2:** `configs/stage_a.yaml`, the §2 baseline with lr 0.002. It exports to llama.cpp
  as `llama`.
- **Scaling points:** `configs/make_scaling.py` → `configs/scaling/` holds 0.5B and 2B. The 1B point
  is `ablation-a1b-lr2e-3`, whose config is identical except for its name. Re-running it would cost
  ~4.4 h for a bit-identical result.
- **Queue:** scaling-0.5b → scaling-2b → stage_a. It starts once the corpus processing that
  `tools/after_ablations.sh` started after A1 has finished.
- `configs/make_ablations.py` derives from `stage_a.yaml`, so regenerating it now would give lr 2e-3
  ablations. The committed `configs/ablations/*.yaml` are what was run.

## A2 and A3 results (2026-11-18 / 19)

Run by Claude under the owner's instruction of 2026-11-18 to carry A3 on autonomously up to C1.
The run-by-run log is in `experiments/m16-own-llm/NIGHT_LOG.md`, and the measured results are in
notebooks 06–08.

- **A2:** `stage-a-110m`, 5B tokens. Val bpb **0.9629** (es 0.8978, en 1.0279). The scaling points
  were 0.5B: 1.1450, 1B: 1.0579 and 2B: 1.0012.
- **Task data rebuilt** from all 323 processed episodes. The 2026-11-14 build had only 56.
  - The split is 259 train / 15 val / 49 test episodes.
  - Train has 13,498 examples: relevance 7,933, show-notes map 3,346, show-notes reduce 260 and
    titles 1,959.
- **Post-training (§9 steps 1–3), configs in `configs/a3/`:**
  - **General SFT** (`a3-sft-general`): 1 epoch of the notebook 05 mix, 244M tokens at T=4096, lr
    3e-4. Val SFT loss 1.534 → 1.259. This is also the model's first training at the app's n_ctx;
    A2 ran at 2048.
  - **Task SFT, general replay set by loss tokens:** about 15 % of the *loss* tokens (0.24M of
    1.54M). Task prompts are long transcripts with short answers, so 15 % of the rows would have been
    69 % of the loss.
  - **Task SFT, one epoch:** `a3-sft-tasks-e1`, lr 1.5e-4, val 1.437. A 3-epoch run
    (`a3-sft-tasks`) overfit, with its minimum 1.453 at about 1 epoch and 1.495 at the end, and was
    discarded.
  - **DPO pairs** (`llm/dpo_pairs.py`): the teacher's answer is chosen and a temperature-0.7 sample
    of the SFT model is rejected, but only when that sample fails a §9.4 verifier.
    - Pairs: 1,509 train / 123 val. The reduce prompts were sampled 4 times, since there are only
      260 of them.
    - Verifier failures: relevance Spearman < 0.3 vs the teacher, titles over 8 words, and drafts
      `parse_draft` rejects.
    - With no judge, samples that pass a verifier are not used. Ultrafeedback was not mixed in.
  - **DPO** (`a3-dpo`): 2 epochs, beta 0.1, lr 2e-6, 32 pairs per step. Val DPO loss 0.692 → 0.472,
    preference accuracy 0.85. The first launch hit CUDA OOM at 2 pairs per micro-batch; the run uses 1.
- **Gates (§10)** are measured by `llm/evaluate.py`, which runs the app's own pipelines on the 49
  frozen test episodes. For relevance, Qwen's chapter titles are the context for both models.

  | gate | own-sft | own-dpo | threshold |
  |---|---|---|---|
  | titles: mT5 fallback | 0 % | 0 % | ≤ 10 % |
  | titles: right language | 99.6 % | 98.4 % | ≥ 95 % |
  | show notes: valid within 3 attempts | 57 % | 86 % | ≥ 95 % |
  | relevance: valid batches | 100 % | 100 % | 100 % |
  | relevance: Spearman vs Qwen p3 | 0.211 | −0.073 | ≥ 0.8 × 0.490 (Qwen pass-to-pass) |

- **Reading.**
  - **Titles** pass for both models.
  - **Show notes** improved most with DPO but still fail the gate. The remaining failures are too
    few distinct tags or titles, and some truncated JSON.
  - **Relevance** fails for both models. Their scores are smooth, content-independent shapes (ramps,
    arches): the shape of Qwen's score lists was learnt, but not its dependence on the content. DPO
    on "disagrees with the teacher" pairs moved the shape without fixing it.
  - This contradicts the expectation above that grammar-constrained relevance would be the 110M
    model's best chance. Titles were.
- **Installed:** `yts-own-110m-dpo` (f16 GGUF, 203 MB) at `backend/models/llm/current`.
  `yts-own-110m-sft` is kept alongside it. The PyTorch vs llama.cpp parity on the installed model
  is max |Δlogit| 0.015, with the same top-5 (notebook 08).
  - The model is selectable despite the failed gates (amendment). **Qwen stays the global default.**
- **App clone:** the app runs from the WSL clone (`scripts/start-wsl.cmd`). It was at `43d64d0`
  without M16, so the uncommitted M16 diff and the new files were applied to it, with no commit.
  Its empty `frontend/node_modules` was restored with `npm ci`.
- **Regression:**
  - backend 770 passed
  - frontend 324 passed, and `ng build` OK
  - experiments 48 passed (new: `tests/test_evaluate.py`)
- **Open for the owner:**
  - C1, trying the model in the app.
  - Whether a relevance follow-up is worth it (e.g. a per-block prompt instead of 20 blocks per
    call), which would be outside M16's scope.
  - ~~Deleting the consumed SFT checkpoints, ~8 GB.~~ Done 2026-11-19 with the owner's approval
    (`NIGHT_LOG.md`): `runs/a3-sft-tasks` and the `ckpt/` folders of the A3 runs, ~10 GB. The
    `final_model.pt` files and the GGUFs stay.
- **Stage B written, not executed (amendment):**
  - `configs/stage_b.yaml`: d1024/L24/16 heads, 294.7M parameters by our count, 25B tokens, 4k
    extension over the last 5 %, lr 1.5e-3 as a guess for B0 to confirm.
  - Notebooks `09_stage_b_sizing`, `10_stage_b_pretraining` and `11_stage_b_post_training_eval`,
    generated by `tools/make_stage_b_notebooks.py`. Each starts with a NOT EXECUTED banner and has
    no outputs. The config was only checked to load and build, not trained.

## C1 and the explainability pass (2026-11-21)

- **C1 passed (owner, 2026-11-21).** The owner ran the checklist in the app: Settings shows both
  models with Qwen as default; the per-tool picker appears on chapters, show notes and relevance;
  runs with our model finish and are labelled with it; Qwen runs are unchanged.
- **Bug found by C1:** `frontend/proxy.conf.json` did not proxy `/settings/llm`, so in dev
  `ng serve` answered it with `index.html`: Settings said "Couldn't load the settings" and the
  picker stayed hidden (it needs `own_available`). Only `/settings/llm` is proxied, since
  `/settings` is also the page route. Unit tests mock HTTP and could not catch it.
- **Explainability pass, at the owner's request** (in the style of 0021/0022, outside the
  milestone plan):
  - How it works has a section "Our own language model (M16)" (`#own-llm`): key numbers, a
    per-task verdict, the architecture, the training stages, and 13 charts (parameters, tokens per
    stage, pretraining loss / bpb / learning rate, ablations, scaling, the four post-training
    curves, the gates), each with a table view, plus a gate table and a comparison with Qwen.
  - **Every number comes from the measured runs.** `tools/make_frontend_data.py` reads the runs'
    `metrics.jsonl`, `eval/*/gates.json` and `configs/stage_a.yaml` and generates
    `frontend/src/app/how-it-works/own-llm-data.ts`. The exact parameter count is 100,682,496
    (the "110M" name is the size class of the first benchmark, run without GQA).
  - A doc `ownllm` (popup), nine glossary terms, and the model in the system diagram and the
    model-status table. Manage shows an ⓘ next to each LLM picker and a one-line gate verdict
    when our model is chosen; Settings links to the section.
  - Stale facts fixed along the way: the relevance doc and status table still said "heuristic v0,
    BiLSTM next" (it has been live since M15), and the CTR/hook rows said "M15" instead of "after
    v1".
