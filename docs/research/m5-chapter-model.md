# Research: training our own chapter models (M5)

**Date:** 2026-09-27
**Status:** Verification pass complete (2026-09-27). The experiment will add to it.

## Context

During M5 planning the project owner chose to run chaptering **locally
with models we train ourselves** instead of calling the Claude API
(spec §2.3/§5). The model work is run as an experiment in
`experiments/m5-chapters/` (see `docs/brainstorm.md`, 2026-09-27).

- **Two models:**
  1. **Boundaries:** chapter starts over a frozen multilingual sentence
     encoder.
  2. **Titles:** a small fine-tuned multilingual seq2seq.
- **Unsupervised baseline:** TextTiling-style over the same
  embeddings. It serves as the fallback and the eval reference.
- **Languages:** English and Spanish.
- **Granularity:** "few, broad" chapters.
- **Hardware:**
  - CPU: Ryzen 5 2600X (6c/12t, AVX2)
  - RAM: 16 GB (~10 GB free)
  - GPU: GTX 1060 3 GB, Pascal sm_61, torch 2.8 cu126
  - Disk: 111 GB SSD with ~54 GB free, plus a 931 GB NTFS drive (not
    mounted)
  - Inference shares the GPU with Whisper and pyannote, which leave
    ~670 MiB free (0008), so **inference runs on CPU**. Training may use
    the full GPU with the worker stopped.

A first pass came from the `research-analyst` subagent. **Several of its
claims turned out to be wrong or unsourced.** This document keeps only
what was checked against primary sources. Corrections are listed in
§7.

## 1. Data: VidChapters-7M

**Verified facts** (paper arXiv 2309.13952, NeurIPS 2023 D&B; project
page; repo `antoyang/VidChapters`):

- **Size:** 817K user-chaptered YouTube videos and 7M chapters. On
  average a video has 8.3 chapters and lasts 1354 s (~23 min).
  Adjacent chapter starts are 142 s apart, and a title averages 5.4
  words.
- **Language:** 93.9% of videos have **English ASR** and 92.9% have
  English chapter titles. 13 languages appear in more than 1K videos.
  **The paper gives no Spanish count** (measured below, §1.3).
- **Speech:** 97.3% of videos have ASR (Whisper large-v2, aligned with
  WhisperX, split into sentences).
- **License:** code is MIT. The page adds: "Copyright and all rights
  therein are maintained by the authors or by other copyright holders".
  The data is YouTube-derived, and there is a removal-request form.
  - **Implication:** fine for training a model used internally. We do
    not redistribute the data. The exported model weights are a
    derivative, so if the product is ever distributed or sold, revisit
    this.

### 1.1 Download options (sizes measured with HTTP HEAD / HF API)

| Source | File | Size | Format |
|---|---|---|---|
| Original (INRIA) | `asr.pkl` | **19.3 GB** (19 299 783 429 B) | **one** pickle (protocol 3): `{video_id: {"text": [...], "start": [...], "end": [...]}}` |
| Original (Drive) | `chapters.pkl` | ~1.9 GB | pickle |
| Original (Drive) | `chapters_dvc_{train,val,test}.json` | train ~401 MB | titles + timestamps per video |
| Original (Drive) | `train.json` / `val.json` / `test.json` | 12 MB / 123 KB / 123 KB | video-ID lists (official split) |
| **HF `lucas-ventura/chapter-llama`** (MIT) | `docs/asrs.json` | 22.4 GB | the same ASR as JSON |
| HF `lucas-ventura/chapter-llama` | `docs/chapters.json` | 2.08 GB | `{vid: {duration, title, description, channel_id, view_count, chapters: {"<sec>": "<title>"}}}` |
| HF `lucas-ventura/chapter-llama` | `subset_data/{asrs,chapters}_<subset>.json` | 1–437 MB | ready-made subsets (below) |

- **Consequence: never load `asr.pkl` / `asrs.json` whole.** A 19–22 GB
  pickle or JSON does not fit in 16 GB of RAM.
- **What works instead:**
  - The HF server supports HTTP range requests (checked: `206 Partial
    Content`), so the full `asrs.json` can be **streamed through
    `ijson`, keeping only the selected videos**.
  - Nothing large has to be stored on disk.
- **Subsets** published by the Chapter-Llama authors, with the ASR
  already split per subset:
  - `sml1k_train` (1k: short+medium+long)
  - `sml10k_train` (10k; ASR 258 MB)
  - `s10k-2_train`
  - Validation: `s100_val`, `m100_val`, `l100_val`, `sml300_val`
  - Test: `s_test` (<15 min), `m_test` (15–30 min), `l_test`
    (30–60 min)
- **The subsets are effectively English-only.** The HF card declares
  `language: en`. A Spanish stopword check found 0 Spanish-looking
  videos in `l100_val` and `sml1k_train`.
- **They also cap at 60 min.** `sml1k_train` has a median of 19 min and
  7 chapters; `l100_val` has a median of 43 min, 8 chapters and ~474
  sentences.
- **Spanish training data therefore has to be mined from the full
  files** (streamed, see above).

### 1.2 Sample record (verified)

```
chapters: {"duration": 2971.2, "title": "...", "chapters": {"56": "Deep thoughts on ...", "300": "...", ...}}
asr:      {"text": [" All praises are due to Allah.", ...], "start": [...], "end": [...]}
```

The first chapter does not always start at 0 (here it starts at 56 s).
Titles are free-form: some carry speaker prefixes ("Dan: …") or
unbalanced quotes, so they need light cleaning for title training.

### 1.3 Spanish volume

**Measured, 2026-09-27.**
- **Method:** stream the full `chapters.json` (2.08 GB, 816 130 videos)
  with `ijson`, then run fastText language-ID (`ftlangdetect`, lid.176)
  on the video title + chapter titles. Wall time 1 min 37 s.
- **Counted with confidence > 0.5:**

| Language | < 15 min | 15–30 | 30–60 | 60–120 | > 120 min |
|---|---|---|---|---|---|
| en | 407 571 | 154 704 | 79 166 | 48 730 | 11 589 |
| **es** | 3 166 | **506** | **203** | **117** | 28 |
| other / low-confidence | 65 840 | 23 370 | 13 533 | 6 228 | 1 379 |

- **Counted at any confidence**, videos of at least 15 min: en 331 819,
  **es 1 736** (882 of them with confidence ≤ 0.5), de 1 131, hi 775,
  fr 723. Spanish videos of at least 20 min have a median of 10
  chapters.

**Conclusions.**
- **Spanish is scarce.** There are roughly **850–1 700 Spanish videos
  of at least 15 min**, about 0.5% of the long videos.
- **English is plentiful:** ~290k English videos of at least 15 min,
  far more than the several thousand after which Chapter-Llama reports
  diminishing returns.
- **Boundaries:** train mostly on English and rely on the frozen
  multilingual encoder for cross-lingual transfer. Include every usable
  Spanish video in training, and **hold out a Spanish-only test set**
  (~200–300 videos) to measure transfer honestly.
- **Titles:** ~1.7k Spanish videos × ~10 chapters gives ~15k Spanish
  (text → title) pairs. That is usable for fine-tuning alongside the
  English pairs.
- **⚠️ Superseded by `01_data` (2026-09-27): these title-based counts
  are wrong.** Running language ID on the **transcript** of all 291 704
  candidates finds only **207 usable Spanish-speech videos** of 15–120
  min (135 channels). Title-based ID has many false positives, and half
  of the real Spanish-speech videos have English titles. See the
  experiment log (`experiments/m5-chapters/README.md`).
- **Caveat:** this language ID is on titles, not on speech. Some videos
  have English titles over Spanish speech, or the reverse. The final
  filter must also check the ASR text, which gets streamed from
  `asrs.json` when the selected transcripts are pulled.

## 2. Unit mismatch between training data and our transcripts (important)

- **VidChapters ASR is sentence-level:** ~5.4 s per sentence (`l100_val`:
  474 sentences in ~43 min).
- **Our own transcript is much finer-grained.** The owner's 64-min
  episode has **3563 Whisper segments, median 0.8 s** (read-only query
  on the dev DB), because of VAD and word timestamps.
- **Consequence:** a model trained on sentences and applied to
  sub-second segments would see a different input distribution. **Both
  sides must be normalised to the same unit.**
- **Options for the design:**
  - Re-split both sides into sentences. We keep word timestamps (M3),
    so we can split on punctuation.
  - Fixed time blocks (e.g. ~10–15 s windows) on both sides. These
    are also cheaper to run the sequence model over.

## 3. Prior work and what numbers to expect

- **VidChapters paper, speech-only chapter generation (test).**
  SODA, CIDEr and METEOR measure boundaries and titles together;
  R@0.5 and P@0.5 are recall and precision at tIoU ≥ 0.5.
  - Text tiling + random titles: SODA 0.4, CIDEr 0.8, METEOR 0.7,
    **R@0.5 23.6 / P@0.5 26.0**
  - Vid2Seq speech-only (C4+HTM): SODA 10.5, CIDEr 50.7, METEOR 8.7,
    **R@0.5 46.5 / P@0.5 40.4**
- **Titles with ground-truth boundaries** (Vid2Seq speech-only):
  CIDEr 120.5, B1 23.5, B4 11.0.
- **Chapter-Llama** (CVPR 2025, arXiv 2504.00072; code MIT, Llama-3.1-8B
  fine-tuned). Its **F1** is segment-level P/R averaged over tIoU
  thresholds 0.5–0.95, so it is strict.
  - Vid2Seq: F1 26.7 overall (short 33.4 / medium 19.0 / long 16.7).
  - Chapter-Llama **ASR only** (short): F1 38.5, tIoU 68.1.
  - ASR + frame captions: F1 45.3 overall.
  - Training data: ~20k videos. They report diminishing returns
    **above several thousand training videos**.
- **Their code also implements boundary-distance metrics:** Recall,
  Precision and F1 at 3 s and at 5 s (`tools/results/metrics_distance.py`).
- **What this means for us:**
  - An 8B LLM is the state of the art. A frozen-encoder + BiLSTM head
    will sit **well below** it.
  - **No published number exists** for our exact setup, so there is no
    target to promise.
  - The realistic, honest bar: **beat the unsupervised baseline by a
    clear margin**, on the same validation data and the same metrics.
- **Metrics we will report:**
  - Boundaries:
    - F1@5s and F1@3s (Chapter-Llama's definition)
    - Chapter-Llama F1 (tIoU 0.5–0.95) and mean tIoU, for
      comparability
    - Pk / WindowDiff, as classic segmentation metrics
    - chapter-count error
  - Titles, with ground-truth boundaries: CIDEr / BLEU, plus a manual
    read.

## 4. Encoder candidates (checked on the HF API)

| Model | Params | License | Notes |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | 118 M | MIT | Needs a `"query: "` / `"passage: "` prefix |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 M | Apache-2.0 | Classic sentence-transformer |
| `intfloat/multilingual-e5-base` | 278 M | MIT | Larger; slower on CPU |
| `Alibaba-NLP/gte-multilingual-base` | 305 M | Apache-2.0 | Larger |

The subagent's CPU throughput figures (e.g. "2604 docs/s") are
**unsourced**. So we measured on this CPU instead.

**Measured, 2026-09-27.**
- **Setup:** 6 torch threads, batch 64, max 128 tokens.
- **Input:** the owner's 64-min episode, with its 3563 Whisper segments
  grouped 5 at a time into 713 units of roughly 5 s each, which is
  about VidChapters' sentence size.
- **Results:**
  - `multilingual-e5-small`: 713 units in **6.9 s** (104 units/s)
  - `paraphrase-multilingual-MiniLM-L12-v2`: 713 units in **6.5 s**
    (109 units/s)
  - `multilingual-e5-base`: 713 units in **28.9 s** (25 units/s), ~4×
    slower than the small models

**Implications.**
- **Inference:** encoding a whole episode on CPU takes seconds, so
  there is no need for the GPU.
- **Training-data encoding:** ~5k videos × ~300 sentences = 1.5 M
  sentences would take **~4 h on CPU**. It should be faster on the GPU
  with the worker stopped; not measured yet.

## 5. Title model candidates

- **`google/mt5-small`:** Apache-2.0, ~300 M params, of which ~250 M
  are the 250k-token vocabulary's embedding and LM head.
- **Memory:** fp32 weights (~1.2 GB) plus gradients (~1.2 GB) already
  roughly fill a 3 GB Pascal card before activations. **The subagent's
  "1.6–1.9 GB, feasible" estimate is unverified and looks optimistic.**
- **Mitigations to test in a short memory spike:**
  - trim the vocabulary to en+es tokens, which shrinks the model a lot
    (a common technique for mT5)
  - freeze the embeddings
  - use Adafactor with gradient checkpointing
  - LoRA
- **Measured on this GTX 1060, 2026-09-27** (script in the session
  scratchpad; the numbers are what counts):
  - **Setup:** GPU otherwise idle (desktop ~580 MiB). `google/mt5-small`
    has 300 176 768 params. Adafactor, input 512 tokens, target 16
    tokens. Each configuration ran in its own process.

  | Batch | Input | Grad. checkpointing | Embeddings + LM head frozen | Result |
  |---|---|---|---|---|
  | 1 | 512 | no | no | **OOM** |
  | 1 | 512 | yes | no | **OOM** |
  | 1 | 512 | yes | **yes** | OK: peak 1500 MiB allocated / 1574 reserved, 0.36 s/step |
  | 4 | 512 | yes | **yes** | OK: peak 2373 / 2548 MiB, 0.35 s/step |
  | 8 | 256 | yes | yes | **OOM** |

  - **Conclusion:** full fine-tuning of mT5-small **does not fit** on
    3 GB.
  - **What fits:** freezing the shared embedding and the LM head (the
    250k-vocab matrices, ~256 M of the 300 M params), leaving ~44 M
    trainable. That runs at batch ≤ 4 and 512 input tokens, at
    ~0.35 s/step. At that rate, 50k pairs is ~12.5k steps/epoch, about
    **75 min per epoch**.
  - **Untested alternative:** vocabulary trimming to en+es would also
    allow training the embeddings.
- **Venv approach verified at the same time.** A venv whose `.pth`
  file points at `backend/.venv`'s site-packages reuses torch 2.8 cu126
  (CUDA available). Adding transformers 5.17 + sentencepiece cost only
  **139 MB**, instead of a ~4 GB second torch install.
- **Other candidates:**
  - `google/mt5-base` (580 M): does not fit.
  - `Qwen/Qwen2.5-0.5B-Instruct` (494 M, Apache-2.0): multilingual, but
    fp32 training does not fit, and Pascal has no fast fp16/bf16.
- **Training data:** (chapter transcript text → title) pairs from the
  same VidChapters subset.

## 6. Reusable code

| Code | License | Use |
|---|---|---|
| `lucas-ventura/chapter-llama` | MIT | Data loaders and the **evaluation metrics** (tIoU F1, F1@3s/5s, SODA) |
| `antoyang/VidChapters` | MIT | Data format, `dvc_eval` (SODA/CIDEr) |
| TreeSeg (`AugmendTech/treeseg`) | see repo | Reference for the unsupervised baseline |

## 7. Corrections to the first-pass subagent report

| Claim | Status |
|---|---|
| "GTX 1060 needs PyTorch < 2.0 / CUDA 11.1" | **Wrong.** torch 2.8 cu126 has run on this card since M4 (0008 §2). |
| "Annotations/ASR 50–100 GB" | **Wrong.** The ASR is 19.3 GB (pickle) / 22.4 GB (JSON), and the subsets are 1–437 MB. |
| "80K–150K en+es videos over 20 min" | **Wrong in composition.** English has ~290k videos of at least 15 min; Spanish only ~0.85–1.7k (§1.3). |
| "Expected F1@±5s 0.62–0.68", "BLEU1 0.50–0.58" | **Unsourced**, and no comparable published result exists (§3). Not used as targets. |
| "Chapter-Llama text-only F1@30s ~0.72 / ~0.61" | **Wrong metric.** Chapter-Llama ASR-only is F1 38.5 on the tIoU-averaged metric (short videos). |
| "mT5-small fp32 on 3 GB: 1.6–1.9 GB" | **Wrong for full fine-tuning** (OOM at batch 1). True only with embeddings/LM head frozen (1.5 GB at batch 1). Measured (§5). |
| "e5-small 2604 docs/s on CPU" | **Wrong by ~25×** for our input: ~104 units/s measured (§4). Still fast enough per episode. |
