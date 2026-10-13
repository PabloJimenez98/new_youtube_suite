# Decision: Chapter Titles from a Local Pretrained LLM (M5, amends 0009)

**Date:** 2026-09-28
**Status:** Decided. The model is chosen.
- **The one-time test run:** done 2026-10-13 (run 8; run 7 died
  silently at 16/300 rows and resumed from the cache). Results below.
- **Bundle v2:** exported 2026-10-13 as
  `bilstm-e5s-sentence-20260928+qwen3.5-4b`; `current` points at it.

## Context

- **Why mT5 was dropped:** 0009 §5 planned the chapter titles with
  mT5-small, fine-tuned by us. In 05 its Spanish titles for the owner's
  episode were not usable, and more training can't fix that: there are
  only 845 Spanish pairs, and VidChapters has no more.
- **What the owner approved:** comparing local pretrained instruction
  LLMs instead, documented as experiment 06. The plan, candidates and
  selection rule are in `experiments/m5-chapters/README.md` ("plan for
  06").
- **Boundaries don't change:** they stay with the BiLSTM (0009 §3).

## What was measured (06, validation sample)

- **Setup:**
  - 150 `val` (en) + 117 `es_val` pairs, with ground-truth boundaries
  - CPU, 6 threads, GGUF Q4_K_M, the same prompt for every model
- **Results:** every LLM got 100 % language accuracy, 0 repeats and
  0 fallbacks. mT5 got 90.6 % language accuracy in Spanish.

| method | CIDEr en | CIDEr es | gates | owner's episode (6 ch.) |
|---|---|---|---|---|
| mt5-small | 65.9 | 50.0 | ✗ (lang es) | n/a |
| qwen3.5-2b | 57.8 | 47.6 | ✓ | 1.9 min |
| **qwen3.5-4b** | 55.9 | 44.5 | ✓ | 4.6 min |
| gemma4-e4b | 49.3 | 53.8 | ✗ CIDEr en | 4.7 min |
| salamandra-7b | 51.6 | 47.2 | ✗ CIDEr en | 5.1 min |
| qwen3.5-9b | 46.1 | 54.5 | ✗ CIDEr en, RAM | 6.9 min |
| gemma4-e2b | 47.7 | 27.9 | ✗ CIDEr en | 2.5 min |

- **Gates:**
  - language accuracy ≥ 95 % in Spanish
  - CIDEr en ≥ 0.8 × mT5's
  - ≤ 15 min per episode on CPU
  - within the ~5 GB RAM budget
- **Owner's blind review** (6 chapters, 7 unlabeled candidates each):
  the owner picked **qwen3.5-9b's** title for all 6.

## Test result (run once, after the choice)

- **Sample:** 05's test draws (`Random(1)`), first 150 of each split.

| method | set | CIDEr | BLEU-1 | BLEU-4 | lang ok | repeats | words |
|---|---|---|---|---|---|---|---|
| mt5-small | test (en) | 90.5 | 18.9 | 7.9 | 100 % | 5.3 % | 5.7 |
| **qwen3.5-4b** | test (en) | 76.9 | 18.2 | 5.4 | 100 % | 0 % | 7.3 |
| mt5-small | es_test | 49.6 | 9.8 | 5.6 | 88.0 % | 6.7 % | 5.1 |
| **qwen3.5-4b** | es_test | 34.2 | 9.5 | 2.8 | 100 % | 0 % | 8.8 |

- **Gates hold on test:** CIDEr en is 0.85 × mT5's (gate ≥ 0.8), and
  Spanish language accuracy is 100 % (mT5: 88 %, which fails the gate).
  LLM fallback rate 0 %. ~14.9 s per chapter on CPU.
- **Caveat:** Qwen's CIDEr es is lower than mT5's (34.2 vs 49.6), the
  same direction as on validation. Its titles are longer (8.8 vs 5.1
  words), which CIDEr penalises against short reference titles. No gate
  covers CIDEr es; the choice rests on the gates plus the owner's
  episode, where mT5's Spanish titles were unusable.
- **Production check (2026-10-13):** a real `chapters` job on the
  owner's 64-min episode (upload 8) through the arq worker gave the
  same 6 titles as 06's episode run, with `model_version`
  `…+qwen3.5-4b`, in ~4.3 min. Worker RSS went from 1.6 GB idle to
  6.8 GB after the job (+5.2 GB: encoder, BiLSTM, mT5, GGUF + KV cache).

## Decisions

### 1. Title model: Qwen3.5-4B-Instruct, GGUF Q4_K_M, via llama.cpp

- **Choice:** the owner chose **qwen3.5-4b** (2026-09-28).
- **Why not qwen3.5-9b:** the owner preferred its titles, but it fails
  two gates. Its CIDEr en is 46.1, below the 52.7 minimum, and its RSS
  in the notebook process was ~10.4 GB, against the ~5 GB budget. On
  this 16 GB machine it also triggered systemd-oomd kills during the
  eval.
- **Following the confirmed rule:** the owner applied the selection
  rule as confirmed ("pick among the models that pass the gates")
  rather than make an exception.
- **qwen3.5-2b vs 4b:** both pass. The owner chose 4b directly, without
  a second blind round.
- **Rejected:** qwen3.5-9b (above), gemma4-e4b, salamandra-7b and
  gemma4-e2b (all fail the CIDEr en gate), and keeping mT5 alone.
- **Runtime:** `llama-cpp-python` 0.3.35, built CPU-only (AVX2), in
  process in the worker.
- **Prompt and post-processing:** exactly as evaluated
  (`experiments/m5-chapters/lib/llm_titles.py`, prompt v1):
  - one call per chapter
  - the chapter text capped at 900 words (head + 4 spread windows)
  - the titles already given are passed in, so they don't repeat
  - greedy decoding, then one sampled retry
- **Fallback:** mT5 stays in the bundle. When the LLM gives no usable
  title after the retry, that chapter gets mT5's title. On validation
  the fallback rate was 0 %.

### 2. Episode language: fastText LID over the transcript

- **Why it's needed:** the prompt is written in the episode's language
  (en/es), and the language check rejects confidently wrong-language
  titles.
- **The gap:** the backend doesn't persist Whisper's detected language.
- **Choice:** the chapters pipeline detects the language on the
  transcript text with fastText `lid.176.ftz`, the same model and
  function as in 06. The model ships inside the bundle.
- **Other languages:** anything other than `es` gets the English prompt,
  as in 06.
- **Alternative rejected:** persisting Whisper's language on the
  transcript. It needs a migration plus changes to M3's handler, and
  existing transcripts would have none. It's noted in `brainstorm.md`
  for later.
- **License:** `lid.176` is CC BY-SA 3.0. The model file is only
  redistributed inside our own gitignored bundle.

### 3. Bundle v2 layout

- **How it's built:** `lib/export.py:add_llm_titles` derives it from
  the exported v1 bundle. The boundary model and mT5 are copied as they
  are, without re-running 05.

The bundle is exported to `backend/models/chapters/<version>/`, as in
0009 §6.

```
manifest.json    + titles.llm {file, n_ctx, n_threads, max_words, max_tokens, prompt_version}
                 + titles.lid {file}
encoder/  boundary.pt  titles/   unchanged (titles/ = mT5, now the fallback)
llm/Qwen3.5-4B-Q4_K_M.gguf       (2.7 GB, copied)
lid.176.ftz
```

- **Version string:** `bilstm-e5s-sentence-20260928+qwen3.5-4b`.
- **Backward compatible:** a manifest without `titles.llm` still runs
  mT5 only.

### 4. Backend changes

- **New `app/chapters/llm_titles.py`:** a copy of the experiment module.
  A drift test guards it, like the other copies. Its only difference is
  the `clean_title` import.
- **New `app/chapters/text.py`:**
  - `clean_title`, copied from the experiment and drift-tested
  - `LanguageId`, the experiment's class without the download step,
    since the model file ships in the bundle
- **`Chapterer` changes:**
  - loads the LLM and LID once per process, when the manifest has them
  - titles the chapters in order, passing the previous titles
  - falls back to mT5 per chapter
- **New settings:** `CHAPTERS_LLM_THREADS` (default 6).
- **New deps:** `llama-cpp-python==0.3.35` (CPU build) and
  `fasttext-numpy2-wheel==0.9.2`.

## Test plan

- **Experiment:**
  - `tests/test_llm_titles.py`, which already exists
  - the one-time test run: qwen3.5-4b vs mT5 on 05's test draws
    (`Random(1)`), **cut to the first 150 of each split**. The full
    ≤ 800 + 800 would take ~10 h on CPU; the owner chose the reduced
    sample. It runs in `06` section 5, with results in
    `runs/2026-09-28-titles-llm/test_metrics.json`.
- **Backend, unit tests:**
  - drift tests for `llm_titles.py` and `text.py`
  - `Chapterer` with a fake LLM titler: previous titles passed in
    order, per-chapter fallback to mT5 when the LLM returns None, and
    language from LID
  - a bundle without `titles.llm` still works (mT5 only)
- **Backend, integration:** the real bundle, which is skip-if-absent
  as in 0009. It must return non-empty titles and model version
  `…+qwen3.5-4b`.
- **Regression:** the full backend and frontend suites.

## Consequences

- **Memory:** the worker holds ~3.5 GB more (GGUF + KV cache) once
  chapters has run.
- **Speed:** titling costs ~5 min per 64-min episode on CPU (was
  seconds with mT5).
- **Bundle size:** it grows by ~2.7 GB.
- **What stays open:** the preference for 9B's titles is recorded. If
  the RAM budget changes (more RAM, or the GPU freed for this job),
  re-open the choice with the 9B.
