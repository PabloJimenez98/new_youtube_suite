# Decision: Show Notes from the Local LLM (M6)

**Date:** 2026-10-16
**Status:** Implemented (M6 done, 2026-10-16).

## Context

- **Spec §2.6/§5** plan show notes as a Claude API call returning
  structured JSON (`title_suggestions`, `description`, `tags`,
  `summary`).
- **M5 went local** (0009, 0010), and whether M6 follows was left open.
- **Decided with the owner (2026-10-16):**
  - **Local LLM:** the same Qwen3.5-4B GGUF that titles chapters. No
    Claude API.
  - **Evaluation:** manual QA only. There is no reference dataset for
    show notes, so automatic metrics would say little, and 4B was
    already validated for the language and RAM gates in 0010.
  - **Editing:** one editable draft per upload. A re-run replaces it,
    after a warning in the UI.
  - **Description:** the LLM's text plus a deterministic
    `00:00 Title` chapter block, when chapters exist.
- **Still open, not in M6:**
  - pushing show notes to YouTube via `videos.update` (spec §2.6): M6
    is copy/paste only
  - speaker names: diarization is anonymous, so speakers are not passed
    to the prompt

## Decisions

### 1. Model and a shared LLM runtime

- **Model source:** show notes read `titles.llm.file` and
  `titles.lid.file` from the chapters bundle manifest
  (`CHAPTERS_MODEL_DIR`).
  - No new model or setting.
  - No bundle, or a bundle without `titles.llm` → `shownotes` jobs fail
    fast with a clear message, as `chapters` does.
- **One `Llama` per process:** `app/llm/runtime.py` holds
  `get_llama(path, n_ctx, n_threads)`, cached by `(path, n_ctx)`, plus a
  global `llm_lock`.
  - **Why:** without it the worker would hold the 2.7 GB GGUF (plus
    KV cache) twice once both tools have run.
  - **Why the lock:** arq runs up to `max_jobs=10` jobs at once, so
    `chapters` and `shownotes` can overlap, and a `Llama` is not
    thread-safe.
- **`LlamaTitler` gains an optional `llm=` kwarg** in the experiment's
  `lib/llm_titles.py` and its backend copy, so the drift test stays
  green. `Chapterer` passes the shared instance and holds `llm_lock`
  around each `title()` call.
- **Context:** `n_ctx=4096`, as for chapters, so both share one
  instance. The pipeline is sized to fit it.
- **Thinking:** off (`enable_thinking=False`), as in 0010.
- **Language:** fastText `LanguageId` on the transcript. `es` gets the
  Spanish prompt; anything else gets the English one (0010 §2).
- **Show notes alone** load only the GGUF and LID, not the encoder or
  mT5.

### 2. Map-reduce pipeline (`app/shownotes/`)

1. **Chunks** (pure function):
   - one chunk per chapter when chapters exist, split further above
     1600 words
   - otherwise 1600-word blocks cut on segment boundaries
   - a transcript that fits the reduce budget (≤ 2400 tokens, ~1600
     words) skips the map step entirely
   - any text still over its budget is cut to fit (head + spread
     windows, `sample_text` from 0010)
2. **Map:** per chunk, a 3–5 bullet summary (≤ ~150 tokens) in the
   episode language, greedy.
3. **Reduce:** summaries + chapter titles (if any) → **one call whose
   output is constrained by a JSON grammar** built from the §3 schema
   (`LlamaGrammar.from_json_schema`).
   - If the summaries don't fit the prompt budget, they are first
     summarized again in groups (one extra level).
4. **Validate and normalize** (§3), compose the description with the
   chapter block, persist.

- **Expected time:** ~6–8 min for the owner's 64-min episode on CPU
  (chapters measured ~15 s per call with ~1200 prompt tokens).
  **Measured (2026-10-16, QA run):** 978 s (~16 min), about twice the
  estimate, valid on the first reduce attempt. See "QA run" below.

### 3. Structured-output schema

| Field | Validation | Normalization |
|---|---|---|
| `title_suggestions` | 3–5 non-empty strings, each ≤ 100 chars (YouTube) | trim, case-insensitive dedup, `clean_title` (quotes, enumerators) |
| `description` | ≥ 1 char; LLM text ≤ 1500 chars; with the chapter block ≤ 5000 (YouTube) | trim; cut at the last sentence end when over 1500 |
| `tags` | 5–15 tags, total ≤ 500 chars (YouTube) | lowercase, strip `#` and commas, dedup; drop trailing tags until the total fits |
| `summary` | ≤ 600 chars, non-empty | trim; cut at the last sentence end when over |

- **`PATCH` edits** are validated against YouTube's limits (422
  otherwise): ≤ 5 titles of ≤ 100 chars, description ≤ 5000, tags
  ≤ 15 and ≤ 500 chars in total, summary ≤ 600, and no `<` `>` in the
  description or tags.
  - **The minimums apply to generation only** (≥ 3 titles, ≥ 5 tags).
    An edit may keep a single title or drop every tag.
- **Tag length** counts the way YouTube does: tags are joined by
  commas, and a tag with a space is quoted (+2).

### 4. Retry/repair policy

- **Repair locally first:** §3 normalization fixes what doesn't need
  the model again (lengths, duplicates, excess tags).
- **Hard failures:**
  - output that doesn't parse as JSON
  - missing fields
  - fewer titles or tags than the minimum after normalization
  - a description fastText confidently puts in the other language
- **On a hard failure the reduce call is retried:**
  - attempt 1: greedy
  - attempts 2–3: temperature 0.7, seeds 1 and 2
  - each failed attempt is logged with its reason
- **After 3 failures:** the job fails with
  `"Show notes: model returned no valid draft after 3 attempts"`, and
  the previous draft is left untouched.
- **Map step:** an empty summary is retried once (sampled). If it is
  still empty, the head of the chunk stands in for it. That degrades
  quality but doesn't fail the job.

### 5. Schema, handler, API

- **Migration `0007`** creates `show_notes_drafts`:
  - `id`, `upload_id` (FK, CASCADE, **unique**), `title_suggestions`
    JSONB, `description` TEXT, `tags` TEXT[], `summary` TEXT,
    `language` TEXT, `llm_model_used` TEXT, `created_at`, `updated_at`
  - `llm_model_used`: the GGUF's stem plus the prompt version, e.g.
    `Qwen3.5-4B-Q4_K_M+shownotes-v1`
  - **vs. spec §4:** adds `language`, the timestamps and the unique
    constraint
- **Handler** `app/worker/handlers/shownotes.py`, the `chapters`
  pattern:
  - needs a succeeded `transcribe` job and segments, otherwise
    "Run Transcript first"
  - reads chapters when they exist
  - CPU inference in `asyncio.to_thread`, no GPU lock
  - replaces the draft in one transaction
- **API:**
  - `GET /uploads/{id}/shownotes`: 404 when no `shownotes` job ever
    succeeded, 409 when none has but one is queued/running (0009 §7
    rules)
  - `PATCH /uploads/{id}/shownotes`: partial edit, validated, bumps
    `updated_at`
  - both require auth

### 6. Frontend (Manage)

- `needsTranscript` covers `shownotes`.
- **Re-run warning:** when a draft exists, "Run" asks for
  confirmation, since it overwrites edits.
- **"Show notes" panel:**
  - editable title suggestions, each with a copy button
  - description textarea with copy
  - tags (comma-separated) with copy
  - summary
  - Save (`PATCH`)
  - `llm_model_used` shown, like `model_version` for chapters

## Alternatives rejected

- **Claude API (spec default):** better quality, one call, ~30 s. But
  the transcript leaves the machine, it needs a key and per-episode
  cost, and the owner wants local.
- **Both, behind a config switch:** more code and tests for an adapter
  nobody would use now. Adding one later only means swapping out the
  `app/shownotes` pipeline's LLM calls.
- **Storing generated and edited versions separately:** protects edits
  across re-runs, but needs more columns and merge logic for a
  single-operator tool. The UI warning covers the risk.
- **A mini-experiment (2B/4B/9B blind review):** ~3–5 extra days, with
  no reference data to measure against. The 9B already fails the RAM
  gate (0010).
- **A larger `n_ctx` for one-shot generation:** a 64-min episode is
  ~14k tokens. A second instance or a bigger KV cache costs GBs of RAM
  on a 16 GB machine that is already tight (brainstorm, 2026-10-13).

## Test plan

- **Unit:**
  - schema validation/normalization: limits, dedup, tag pruning to 500,
    title/description truncation
  - chunking: by chapter, long-chapter splitting, fixed blocks, a short
    transcript with no map step
  - description composition with the chapter block (with and without
    hours)
  - **retry path with a fake LLM:** malformed then valid → succeeds on
    attempt 2; wrong language → retried; 3 failures → error; empty map
    summary → degraded fallback
  - shared runtime: the same path loads once (fake loader)
- **Handler (fake LLM):** persists a row with `llm_model_used`; a re-run
  replaces it; "Run Transcript first"; a missing bundle fails fast;
  chapters are used when present.
- **Migration:** `0007` up/down in the migration smoke test.
- **API:** GET 200/404/409 + auth; PATCH 200/422/404 + auth.
- **Integration (skip-if-absent bundle, as 0009):** the real Qwen on
  the M3 fixture transcript produces a schema-valid draft.
- **Frontend (vitest):** panel renders; edit + Save sends `PATCH`; copy
  buttons write the expected text; Run disabled without a transcript;
  re-run asks for confirmation when a draft exists.
- **Regression:** full backend and frontend suites, including the M5
  drift tests.
- **Manual QA:** the owner's 64-min episode (upload 8) and one more,
  recording time, worker RSS, language and quality.

## QA run (2026-10-16)

- **Setup:** a real `shownotes` job (job 12) on the owner's 64-min
  Spanish episode (upload 8, 6 chapters), through the arq worker.
- **Result:**
  - succeeded in 978 s, with no reduce retries
  - language `es`
  - worker RSS 5.9 GB after the job (1.6 GB idle)
  - 5 titles, 15 tags, a ~110-word description plus the chapter block,
    and a 2-sentence summary
- **Owner's review (2026-10-16):** approved as is. Observations noted:
  - 2 of the 5 titles repeat chapter titles verbatim, because the
    reduce prompt passes them in
  - names are only as good as the ASR (e.g. "Lasarrió")
  - ~16 min is twice the estimate
- **Not done:** a second episode. Upload 8 is the only one with a
  transcript.

## Consequences

- **Spec §2.6 step 2 and §5** "Claude API for show notes" no longer
  apply. Nothing in v1 calls an external LLM API now.
- **Spec §4 `show_notes_drafts`** gains `language`, timestamps and a
  unique `upload_id`.
- **Coupling:** show notes depend on the chapters bundle for their
  model files. If the two tools ever need different models, add a
  `SHOWNOTES_MODEL_PATH` setting then.
- **Estimate:** 2–3 days → ~3–4 days (map-reduce and the shared
  runtime).
