# YouTube Suite — Project Description (v1)

**Date:** 2026-08-13
**Status:** Phase 3 output — full specification, based on `docs/decisions/v1-scope.md` (Phase 1) and `docs/research/bibliography.md` (Phase 2). Expanded with verified technical specs and architecture proposals.

> **Correction vs. the first draft of this document**: the original Phase 3 draft assumed the OAuth-connected channel's YouTube Analytics API would directly expose real CTR/impressions data. Verification found this is **not** true for the live-query Analytics API (`reports.query`) — real thumbnail-impression/CTR data only became available (Jan 2026) via the separate **YouTube Reporting API**, an async bulk-job/CSV system, not a live query endpoint. This changes the market-analysis architecture (§6) and CTR predictor's data story (§5) — flagged inline below, not buried.

## 1. Product overview

YouTube Suite is a toolkit for producers of YouTube video podcasts. A
user uploads a long-form podcast video and runs a set of AI-powered
tools against it: transcription, chapter generation, short-clip
extraction, show-notes drafting, and pre-publish performance scoring
(title/thumbnail CTR, clip "hook" strength). A separate market-analysis
component tracks a curated set of public podcast channels plus one
OAuth-connected YouTube channel, feeding real performance data back
into the platform's custom models over time.

**Problem it solves**: turning a raw podcast recording into everything
YouTube expects (subtitles, chapters, shorts, metadata) is manual and
repetitive today, and deciding *which* moments are worth clipping or
*whether* a title/thumbnail will perform is largely guesswork. The
platform automates the repetitive production work and progressively
replaces the guesswork with models trained on the operator's own
accumulated performance data.

**For whom**: v1 is single-operator, not multi-tenant SaaS — built for
the project owner's own podcast production workflow. Architecture
choices below keep the door open for multi-tenancy later, but it is
not built now.

## 2. In-scope tools/features (v1) — detailed

Each tool below specifies exact input data, exact output data, and a
step-by-step account of how it works internally.

### 2.1 Transcript/subtitle generator

- **Input**: an uploaded video file's audio track, extracted via ffmpeg to 16kHz mono WAV (Whisper's expected input format).
- **Output**: persisted `TranscriptSegment` and `TranscriptWord` rows (§4), plus on-demand/cached downloadable `.srt` and `.vtt` files.
- **How, step by step**:
  1. On upload, extract the audio track from the video container to 16kHz mono WAV via ffmpeg.
  2. Run **faster-whisper** (`large-v3-turbo` model, CTranslate2 backend) over the audio with `word_timestamps=True`.
  3. Whisper returns a list of segment objects, each with `id, start, end, text, avg_logprob, compression_ratio, no_speech_prob, words[]` — where each word object has `start, end, word, probability`.
  4. Persist one `TranscriptSegment` row per segment and one `TranscriptWord` row per word, linked to its parent segment.
  5. Generate `.srt` (index + `HH:MM:SS,mmm --> HH:MM:SS,mmm` + text per segment) and `.vtt` (`WEBVTT` header + `HH:MM:SS.mmm` timestamps) directly from the persisted segments; cache as files in blob storage.
  6. Mark the transcription `Job` row `succeeded`, unlocking downstream jobs (diarization, chapters, relevance scoring) that depend on the transcript.
- **English-only for v1** — no translation/multi-language handling.

### 2.2 Speaker diarization

- **Input**: the same 16kHz mono WAV audio + the already-persisted transcript (segments and words with timestamps).
- **Output**: `speaker_label` populated on existing `TranscriptWord` and `TranscriptSegment` rows — no new top-level entity.
- **How, step by step**:
  1. Run **pyannote.audio 4.0.7**'s `speaker-diarization-community-1` pipeline (Hugging Face–gated, license acceptance required) over the audio, returning a `pyannote.core.Annotation` — iterated as `(turn.start, turn.end, speaker_label)` triples, turn-level only (no word granularity at this layer).
  2. Skip WhisperX's own forced-alignment pass (redundant — Whisper's native `word_timestamps=True` already produced word-level timing in §2.1); go straight to WhisperX's `assign_word_speakers()` function, which maps each `TranscriptWord` to the speaker turn whose `[start, end]` interval contains the word's timestamp (nearest-overlap rule for words straddling two turns).
  3. Update each `TranscriptWord.speaker_label`; derive each `TranscriptSegment.speaker_label` as the majority speaker among its words (first word's speaker as tiebreak).
  4. Words with characters missing from the alignment vocabulary (e.g. certain numerals/currency symbols) keep `speaker_label = NULL` — a known WhisperX limitation, not a bug to fix.
- **Version gotcha** (Phase-2-research-confirmed): WhisperX 3.8.6 defaults to the `community-1` pipeline, not the older `speaker-diarization-3.1` most tutorials reference — using the wrong Hugging Face license acceptance fails **silently** (empty speaker labels, no exception). Worth a startup healthcheck that asserts non-empty diarization output on a known test clip.

### 2.3 Chapter/timestamp generator

- **Input**: persisted `TranscriptSegment` rows (timestamped transcript text).
- **Output**: ordered `Chapter` rows (`start_time`, `title`).
- **How, step by step**:
  1. Concatenate transcript segments into one document, retaining a mapping from character offset back to originating segment timestamps.
  2. Run topic segmentation: send the transcript (chunked if it exceeds the LLM's context window) to the chosen LLM API with a prompt requesting chapter breakpoints + titles.
  3. Because the LLM operates on cleaned/paraphrased text rather than exact source tokens (a documented failure mode in the chaptering literature — Chapter-Llama and others hit this), run a **lexical realignment step**: for each returned boundary phrase, find its best-matching position in the original transcript via TF-IDF/string similarity, then snap the chapter's `start_time` to that matched segment's original timestamp. Skipping this step is the single most common bug in DIY chaptering implementations.
  4. Persist each resulting boundary as a `Chapter` row with an `order_index`.

### 2.4 Relevance-detection model (custom)

- **Input**: persisted transcript (segments + word-level timestamps), full video duration.
- **Output**: `RelevanceScore` rows — a continuous score in `[0,1]` per time-window, aligned to timestamps.
- **How, step by step (inference)**:
  1. Re-chunk the transcript into fixed **~10–20 second sliding windows with ~50% overlap**, built by walking word-level timestamps and accumulating words until each window's duration target is reached — independent of Whisper's arbitrary segment boundaries.
  2. Embed each window's text with a **frozen `BAAI/bge-base-en-v1.5`** sentence-transformer (self-hosted, no per-call cost).
  3. Feed the sequence of window embeddings through a trained **BiLSTM (hidden size 128–256)**, batching very long transcripts in overlapping ~45-minute super-chunks (15-minute overlap) purely for GPU memory management, stitching overlap-region scores by averaging.
  4. Output head: 2-layer MLP (FC-64 → ReLU → FC-1) with sigmoid, applied per timestep, producing the `[0,1]` score.
  5. Persist one `RelevanceScore` row per window, tagged with the `ModelVersion` that produced it.
- **How, step by step (training, offline, not per-video)**:
  1. Bootstrap pseudo-labels per window via heuristic rules (keyword/topic-anchor match, question-answer pair detection, speaker-change density, filler-word ratio) plus a zero-shot LLM scoring pass (prompt an LLM to score each window 0–100).
  2. Train the BiLSTM head via **distillation** against these noisy pseudo-labels (MSE or Huber loss — Huber is more robust to label noise), plus a temporal-smoothness regularizer penalizing large score deltas between adjacent windows (relevance is known to be temporally autocorrelated in podcast content — this regularizer is a cheap, well-justified inductive bias given scarce labels).
  3. As the OAuth-connected channel's retention data (§6) accumulates, migrate toward retention-heatmap-derived labels (Mr. HiSum-style — Phase 2 finding) as a better training signal than the heuristic bootstrap.
  4. Store resulting weights as a new `ModelVersion` row; mark `is_live=true` once validated against a held-out set.
- **Why this architecture, not something bigger**: a BiLSTM/TCN over frozen embeddings has far fewer parameters than a transformer encoder, trains stably on small/noisy-labeled datasets, and encodes the strong local-sequential structure relevance already has in podcast content — a self-attention stack would need much more data to avoid degenerate attention patterns. This is a v1-appropriate choice for a cold-start data situation, not a permanent ceiling.

### 2.5 Shorts generator

- **Input**: `RelevanceScore` curve, transcript timestamps (for sentence-boundary snapping), source video file, user-selected target aspect ratio (default `9:16`), caption-burn-in toggle.
- **Output**: `Clip` rows + exported video files in blob storage.
- **How, step by step**:
  1. **Peak-finding** over the relevance-score sequence: local-maxima detection with a minimum prominence threshold, producing candidate peak center-points.
  2. **Boundary expansion**: from each peak, expand outward in both directions until the score drops below a relative threshold (e.g. 70% of peak height) or a max-duration cap is hit.
  3. **Sentence-snap**: snap the resulting rough start/end to the nearest sentence/utterance boundary using segment/word-level punctuation, so cuts never land mid-sentence.
  4. **Duration constraints & dedup**: discard candidates below a minimum duration, truncate at a maximum, merge/deduplicate overlapping candidate spans.
  5. **Cut**: ffmpeg-cut the source video to each finalized span (stream-copy where no crop is needed, re-encode when cropping).
  6. **Subject detection**: run a lightweight face/person detector (per Phase 2 findings, projects like `Autocrop-vertical`/`openshorts` already implement this — worth adapting rather than building from scratch) per sampled frame across the clip, producing a bounding-box track.
  7. **Smoothing**: smooth the bounding-box track over time (exponential moving average or Kalman filter on box center/size) to avoid jittery cropping.
  8. **Crop + reframe**: compute a per-frame crop window from the smoothed track sized/positioned to the target aspect ratio, applied via ffmpeg's crop+scale filter chain during re-encode.
  9. **Optional caption burn-in**: if toggled on, generate an SRT/ASS subtitle slice scoped to the clip's time range from the persisted transcript, burned in via ffmpeg's subtitle filter in the same re-encode pass.
  10. Persist a `Clip` row (`start_time, end_time, aspect_ratio, caption_burned_in, storage_ref, crop_bounding_box_track` as JSONB, source `RelevanceScore` window references).
- Fully automatic — no human review step for v1, per the confirmed decision.

### 2.6 Show notes & metadata generator

- **Input**: transcript text (+ chapters, if generated).
- **Output**: one `ShowNotesDraft` row: `title_suggestions[]`, `description`, `tags[]`, `summary`.
- **How, step by step**:
  1. Assemble a prompt combining the transcript (chunked/summarized if it exceeds context length) and any generated chapters.
  2. Call an external LLM API — **recommend the Claude API** (a mid-tier current model, e.g. Claude Sonnet at implementation time) for quality/context-length fit; kept swappable, not hard-wired to one vendor — requesting structured JSON output matching the four fields above.
  3. Validate the response against the expected schema; retry/repair on malformed JSON.
  4. Persist as a `ShowNotesDraft` row; surface in the frontend for user review/edit.
  - **Open question, not yet in scope**: whether the platform pushes accepted show notes back to YouTube via `videos.update`, or stays purely advisory (user copies it manually). Not decided — flagging so it isn't silently assumed either way.

### 2.7 Title/thumbnail CTR predictor (custom)

- **Input**: one or more candidate `(title text, thumbnail image)` pairs for the same underlying upload.
- **Output**: `CTRPrediction` rows — a relative "attractiveness score" per candidate.
- **How, step by step**:
  1. Embed each candidate thumbnail with a **frozen SigLIP 2 (`google/siglip2-so400m-patch14-384`)** image tower — current strongest open-weights image-text embedding model as of this research (ahead of CLIP ViT-L/14, which most prior CTR-prediction work defaults to).
  2. Embed each candidate title with the same **frozen `BAAI/bge-base-en-v1.5`** text encoder used by the relevance model (representation reuse).
  3. Project both embeddings to a shared 256-d space via small linear layers (balances differing native dimensionalities), concatenate, and append a cheap engineered-feature side-channel: title length/word count, presence of numbers/question marks, thumbnail face count/size (lightweight face detector), thumbnail text-overlap area (OCR).
  4. Feed the concatenated vector through a fusion MLP (FC-512 → ReLU → Dropout(0.3) → FC-128 → ReLU → FC-1) producing a scalar score.
  5. **Trained with a pairwise ranking loss**, not absolute regression (see §5 for why) — so scores are meaningful relative to other candidates for the *same* upload, not as calibrated absolute CTR percentages, until enough real CTR data justifies an isotonic-regression calibration layer (a v2 add-on).
  6. Persist a `CTRPrediction` row per candidate; when multiple candidates are submitted together, rank by score.

### 2.8 Hook-strength scorer (custom)

- **Input**: transcript text of a clip's opening ~5–15s window (from persisted `TranscriptSegment`/`TranscriptWord` rows), plus the upload's title for a title-hook-alignment feature. *Text-only for v1, for consistency with the relevance model's text-only constraint — flagged as an inferred call, not an explicit decision.*
- **Output**: one `HookScore` row per `Clip`.
- **How, step by step**:
  1. Slice the opening-window transcript text from the clip's `start_time` to `start_time + 15s`.
  2. Compute an engineered feature vector via spaCy/regex/a small pretrained classifier: word/sentence count and average sentence length; readability score; question-mark and exclamation-mark counts (and whether the *first* sentence is a question); superlative-adjective and numeric-mention counts (listicle framing, e.g. "3 ways"); second-person-pronoun and imperative-verb counts (direct address); curiosity-gap connective matches ("but," "until," "the reason why") and unresolved-demonstrative-pronoun counts in the first sentence; named-entity density (spaCy NER); sentiment/emotion score (e.g. `j-hartmann/emotion-english-distilroberta-base`, using max-emotion-probability as an arousal proxy); words-per-second delivery pace (from word-level timestamps); title-keyword overlap with the upload's title (string/embedding similarity).
  3. Feed the feature vector into a trained **LightGBM** model (pairwise-ranking objective, same rationale as the CTR predictor) to produce the hook-strength score.
  4. Persist a `HookScore` row with `score`, the raw `feature_vector` (JSONB, for SHAP-style explainability — "add a question," "add a number" style feedback becomes possible for free), and `model_version_id`.

## 3. System architecture (conceptual)

**Frontend (Angular)**: upload flow, per-job status/progress, tool
result viewers (transcript editor, chapter list, shorts gallery with
per-clip preview/export, show-notes editor, CTR/hook score display for
draft titles/thumbnails/clips), and a market-analysis dashboard
(curated-channel trends + the connected channel's own analytics).

**Backend (FastAPI)**: exposes the REST API the frontend calls, and
owns five internal responsibilities: (1) **upload handling** — ingest
and store the raw video file; (2) **job orchestration** — every AI
tool above is a long-running task (transcription, model inference,
ffmpeg rendering), so this needs an async queue+worker system; the
specific technology (Celery/RQ/arq/etc.) is intentionally not decided
at this phase; (3) an **AI/model layer** presenting both external
tools (Whisper, pyannote/WhisperX, LLM API, ffmpeg+crop) and the three
custom models behind one internal interface, so job orchestration code
doesn't need to know which is custom vs. external; (4) the
**market-analysis service** (§6), on a scheduled trigger — mechanism
TBD; (5) an **offline model training/retraining pipeline** (ML stack
not decided) that consumes accumulated market-analysis and usage data
to produce new `ModelVersion` artifacts.

**Database (PostgreSQL)**: single source of truth for both platform
data and market data — full schema in §4.

**Explicitly left open / not decided at this phase**: job queue
technology, blob/object storage technology for video files and
generated clips, the ML training stack, the scheduler mechanism for
the daily pull and periodic retraining, and where/how custom models
are served for inference (in-process vs. a separate model-serving
process).

## 4. Data model — detailed schema

Types written in Postgres-native terms. `FK` = foreign key. All
tables get an implicit `id BIGSERIAL PRIMARY KEY` unless noted, plus
`created_at TIMESTAMPTZ DEFAULT now()` where not listed explicitly.

### 4.1 Platform data

**`users`** *(minimal — single operator for v1, not multi-tenant)*
| Column | Type | Notes |
|---|---|---|
| email | TEXT UNIQUE | |
| password_hash | TEXT | or delegate to an auth provider later |

**`uploads`**
| Column | Type | Notes |
|---|---|---|
| filename | TEXT | original filename |
| storage_path | TEXT | blob storage reference |
| duration_seconds | FLOAT | probed via ffprobe on ingest |
| uploaded_at | TIMESTAMPTZ | |
| status | TEXT (enum: uploaded/processing/ready/failed) | |

**`jobs`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| tool_type | TEXT (enum: transcribe/diarize/chapters/relevance/shorts/shownotes/ctr/hook) | |
| status | TEXT (enum: queued/running/succeeded/failed) | |
| progress_pct | INT NULL | |
| started_at, completed_at | TIMESTAMPTZ NULL | |
| error_message | TEXT NULL | |

**`transcript_segments`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| segment_index | INT | order within transcript |
| start_seconds, end_seconds | FLOAT | from Whisper's `start`/`end` |
| text | TEXT | |
| avg_logprob | FLOAT NULL | Whisper confidence proxy |
| no_speech_prob | FLOAT NULL | |
| speaker_label | TEXT NULL | populated by diarization (§2.2) |

**`transcript_words`**
| Column | Type | Notes |
|---|---|---|
| segment_id | FK → transcript_segments | |
| word_index | INT | order within segment |
| word | TEXT | |
| start_seconds, end_seconds | FLOAT NULL | NULL for unalignable characters (WhisperX limitation) |
| probability | FLOAT NULL | Whisper's per-word confidence |
| speaker_label | TEXT NULL | |

**`chapters`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| order_index | INT | |
| start_seconds | FLOAT | |
| title | TEXT | |

**`relevance_scores`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| window_start_seconds, window_end_seconds | FLOAT | |
| score | FLOAT | `[0,1]` |
| model_version_id | FK → model_versions | |

**`clips`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| start_seconds, end_seconds | FLOAT | source-video span |
| aspect_ratio | TEXT (enum: 9:16/1:1/4:5/16:9) | user-selected, default 9:16 |
| caption_burned_in | BOOLEAN | |
| storage_path | TEXT | exported clip file |
| crop_bounding_box_track | JSONB | array of `{t, x, y, w, h}` — smoothed subject-detection track |

**`show_notes_drafts`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads | |
| title_suggestions | JSONB | array of strings |
| description | TEXT | |
| tags | TEXT[] | |
| summary | TEXT | |
| llm_model_used | TEXT | e.g. `claude-sonnet-5` — for reproducibility/audit |

**`ctr_predictions`**
| Column | Type | Notes |
|---|---|---|
| upload_id | FK → uploads NULL | nullable — candidates can be evaluated standalone |
| candidate_title | TEXT | |
| candidate_thumbnail_path | TEXT | blob storage reference |
| predicted_score | FLOAT | relative score, not calibrated CTR% (see §2.7) |
| model_version_id | FK → model_versions | |

**`hook_scores`**
| Column | Type | Notes |
|---|---|---|
| clip_id | FK → clips | |
| score | FLOAT | |
| feature_vector | JSONB | engineered features, for SHAP-style explainability |
| model_version_id | FK → model_versions | |

**`model_versions`**
| Column | Type | Notes |
|---|---|---|
| model_name | TEXT (enum: relevance/ctr/hook) | |
| version_label | TEXT | e.g. semantic version or training timestamp |
| architecture_notes | TEXT | free text, e.g. "BiLSTM-256, bge-base-en-v1.5" |
| training_data_reference | JSONB | which snapshots/date range/pseudo-label method used |
| metrics | JSONB | validation metrics at training time |
| artifact_path | TEXT | blob storage reference to trained weights |
| is_live | BOOLEAN | which version backend inference currently uses |
| trained_at | TIMESTAMPTZ | |

### 4.2 Market data

**`market_channels`** *(curated public list)*
| Column | Type | Notes |
|---|---|---|
| youtube_channel_id | TEXT UNIQUE | |
| channel_title | TEXT | |
| added_date | DATE | |
| active | BOOLEAN | |

**`market_video_snapshots`** *(daily, from Data API v3)*
| Column | Type | Notes |
|---|---|---|
| market_channel_id | FK → market_channels | |
| youtube_video_id | TEXT | |
| snapshot_date | DATE | |
| title | TEXT | `snippet.title` |
| published_at | TIMESTAMPTZ | `snippet.publishedAt` |
| duration_seconds | INT | parsed from `contentDetails.duration` (ISO 8601, e.g. `PT15M33S`) |
| view_count, like_count, comment_count | BIGINT | `statistics.*`, cast from API's string type |
| category_id | TEXT | `snippet.categoryId` |
| tags | TEXT[] | `snippet.tags` |
| thumbnail_url | TEXT | `snippet.thumbnails.high.url` |
| — | — | UNIQUE (`youtube_video_id`, `snapshot_date`) |

*(`favoriteCount` deliberately excluded — deprecated since 2015, always `"0"`. `dislikeCount` deliberately excluded — made private by YouTube in Dec 2021, no longer publicly returned.)*

**`analytics_snapshots`** *(own OAuth-connected channel, daily, from Analytics API v2 `reports.query` — engagement/retention, NOT CTR — see correction note)*
| Column | Type | Notes |
|---|---|---|
| youtube_video_id | TEXT | |
| snapshot_date | DATE | |
| views | BIGINT | |
| estimated_minutes_watched | FLOAT | |
| average_view_duration_seconds | FLOAT | |
| average_view_percentage | FLOAT | |
| likes, comments, shares | BIGINT | |
| subscribers_gained, subscribers_lost | INT | |

**`analytics_retention_curve`** *(one row per elapsed-time bucket per video per pull — from `reports.query`, dimension `elapsedVideoTimeRatio`)*
| Column | Type | Notes |
|---|---|---|
| youtube_video_id | TEXT | required single-video filter — API only allows one video per retention-curve request |
| snapshot_date | DATE | |
| elapsed_video_time_ratio | FLOAT | `[0,1]` bucket |
| audience_watch_ratio | FLOAT | |
| relative_retention_performance | FLOAT NULL | only populated above a view-count threshold |
| started_watching, stopped_watching | INT | |
| total_segment_impressions | INT | |

**`analytics_reach_snapshots`** *(NEW table — from the separate YouTube Reporting API bulk job system, not `reports.query`; this is the correction)*
| Column | Type | Notes |
|---|---|---|
| youtube_video_id | TEXT | |
| report_date | DATE | |
| video_thumbnail_impressions | BIGINT | from `channel_reach_basic_a1` report |
| video_thumbnail_impressions_ctr | FLOAT | the actual real CTR ground truth |

**`reporting_jobs`** *(tracks the async bulk-job workflow the above table depends on)*
| Column | Type | Notes |
|---|---|---|
| report_type | TEXT | e.g. `channel_reach_basic_a1` |
| youtube_job_id | TEXT | job ID returned by the Reporting API |
| status | TEXT (enum: created/pending/ready/failed) | |
| last_polled_at | TIMESTAMPTZ NULL | |

**`oauth_credentials`** *(single row for v1 — platform-level, not per-user)*
| Column | Type | Notes |
|---|---|---|
| provider | TEXT | `youtube` |
| access_token, refresh_token | TEXT (encrypted at rest) | |
| scopes | TEXT[] | e.g. `yt-analytics.readonly` |
| connected_channel_id | TEXT | |
| token_expiry | TIMESTAMPTZ | |

## 5. Model strategy — versions and architectures

### External tools — specific versions

| Component | Choice | Why |
|---|---|---|
| Transcription | **faster-whisper, `large-v3-turbo`** (CTranslate2 backend); fall back to full `large-v3` for noisy/accented audio | `large-v3-turbo` is ~6–8× faster than `large-v3` (pruned 32→4 decoder layers, 809M params) with only 1–2% accuracy loss on clean English audio — the right tradeoff for a self-hosted pipeline. MIT-licensed. INT8-quantized `large-v3` fits in ~2.9GB VRAM. |
| Diarization | **pyannote.audio 4.0.7** (`speaker-diarization-community-1` pipeline, CC-BY-4.0) via **WhisperX 3.8.6** for the word-speaker merge | Current stable versions as of this research; note the version gotcha in §2.2 (silent failure if the HF license for the wrong pipeline generation is accepted). |
| Show notes / metadata / LLM-based chaptering | **Claude API** (a current mid-tier model, e.g. Claude Sonnet) | Kept swappable behind the model-layer interface (§3) — not a hard vendor lock-in, just the recommended default. |
| Shorts crop/caption mechanics | ffmpeg + a lightweight subject-detection crop, referencing **`Autocrop-vertical`** (YOLOv8 person detection + Haar-cascade face fallback) or **`openshorts`**'s reframing module as implementation references | Phase 2 found this mechanically well-solved in open source — build on these rather than from scratch. |

### Custom models — proposed architectures

**Relevance-detection model** — see full step-by-step in §2.4. Summary: frozen `bge-base-en-v1.5` window embeddings → BiLSTM(128–256) → MLP+sigmoid → `[0,1]` score per window; trained via distillation from heuristic + zero-shot-LLM pseudo-labels with a temporal-smoothness regularizer.

**Title/thumbnail CTR predictor** — see full step-by-step in §2.7. Summary: frozen SigLIP-2 (image) + `bge-base-en-v1.5` (text) → projected + concatenated + engineered-feature side-channel → fusion MLP → scalar score. **Trained with a pairwise ranking loss, not absolute regression** — this is the key design decision, because (a) true CTR ground truth barely exists (see §6 correction) so absolute values would be dominated by noise; (b) pairwise comparisons for the *same* video cancel out confounders (channel size, topic, upload time, algorithmic promotion) that a regressor would otherwise need lots of data to learn to ignore; (c) each video with N candidate variants yields O(N²) training pairs from only N labeled points, multiplying effective sample size; (d) it matches the actual product use case — ranking candidates against each other, not predicting an absolute percentage. Fall back to plain regression only once real CTR data (from `analytics_reach_snapshots`, §6) grows into the hundreds-to-thousands range.

**Hook-strength scorer** — see full step-by-step in §2.8. Summary: ~15–20 engineered features from opening-window transcript text → **LightGBM** (pairwise-ranking objective, same rationale as above), not a neural net. GBDTs are the well-established best-performing model class on small (tens-to-low-hundreds of rows expected initially), heterogeneous tabular features — this problem has no bootstrap/pseudo-labeling source analogous to the relevance model's zero-shot-LLM baseline, so data will stay scarce longest here, making sample efficiency the dominant design constraint. GBDTs also give free SHAP-style feature importance, useful for surfacing "why is this hook weak" feedback to the user. Revisit a small neural net (or fusing in a `bge` embedding of the opening text) only once labeled examples pass roughly the low-thousands.

## 6. Market-analysis pipeline — data sources and endpoints

### 6.1 Public Data API v3 (curated channel list)

| Endpoint | Params used | Fields stored | Quota |
|---|---|---|---|
| `videos.list` | `part=snippet,contentDetails,statistics`, up to 50 comma-separated video IDs per call | `snippet.title/publishedAt/channelId/tags/categoryId/thumbnails.high.url`; `contentDetails.duration` (ISO 8601, parsed to seconds); `statistics.viewCount/likeCount/commentCount` (returned as strings — cast to BIGINT; **`favoriteCount` excluded**, deprecated 2015, always `"0"`; **`dislikeCount` unavailable**, made private Dec 2021) | 1 unit/call — batching to 50 IDs makes this the cheap backbone of the daily pull |
| `search.list` | `part=snippet`, `channelId`, `type=video`, `order=date`, `publishedAfter` | new video discovery only | As of a ~June 2026 change, billed 1 unit/call but capped at a **separate ~100-calls/day bucket** independent of the main 10,000-unit pool (older docs/blogs still describe it as flat 100 units from the shared pool — both converge on the same practical ceiling of ~100 discovery calls/day; worth a live-docs recheck before final implementation since this changed recently) |
| `channels.list` | `part=snippet,statistics` | `statistics.subscriberCount/videoCount` (gate `subscriberCount` on `hiddenSubscriberCount` — if true, don't trust/display it); `snippet.title/customUrl` | 1 unit/call |
| RSS feed | `https://www.youtube.com/feeds/videos.xml?channel_id=...` | latest ~15 uploads per channel (no backfill) | Free, no quota — used as the primary "new upload" trigger (checked frequently, e.g. hourly), with `videos.list` pulling the actual metadata/stats once a new video ID is known |

Default daily quota: **10,000 units/day** for the combined pool (excluding `search.list`'s separate bucket).

### 6.2 YouTube Analytics API v2 (`reports.query`) — own OAuth-connected channel

**Does NOT provide impressions or CTR** (see correction note at the top of this document) — only engagement/retention metrics.

| Report | Dimensions | Metrics | Scope |
|---|---|---|---|
| Per-video daily engagement | `day, video` | `views, estimatedMinutesWatched, averageViewDuration, averageViewPercentage, likes, comments, shares, subscribersGained, subscribersLost` | `yt-analytics.readonly` |
| Audience retention curve | `elapsedVideoTimeRatio`, filtered to exactly one `video` ID (API requirement — no multi-video retention queries) | `audienceWatchRatio, relativeRetentionPerformance, startedWatching, stoppedWatching, totalSegmentImpressions` | `yt-analytics.readonly` |

### 6.3 YouTube Reporting API v1 (bulk async job) — real CTR/impressions, own channel

**This is the corrected, newly-identified requirement**: real thumbnail-impression and CTR data (`video_thumbnail_impressions`, `video_thumbnail_impressions_ctr`) is only available via `channel_reach_basic_a1` (or `channel_reach_combined_a1`, which adds `traffic_source_type/detail`, `operating_system`, `device_type` breakdowns) report types — a fundamentally different, asynchronous integration pattern from §6.2:

1. Create a scheduled reporting job for the desired report type.
2. Poll job status periodically (`reporting_jobs` table, §4.2).
3. Once a report file is ready, download and parse the CSV (`date, channel_id, video_id, video_thumbnail_impressions, video_thumbnail_impressions_ctr`).
4. Store rows in `analytics_reach_snapshots`.

Same scope as §6.2 (`yt-analytics.readonly` — no extra OAuth consent needed), but requires its own poller separate from the daily `reports.query` pull, since reports are produced on YouTube's schedule with inherent lag, not returned synchronously.

### 6.4 Feedback loop

A periodic (not necessarily daily) retraining job consumes accumulated snapshots from all three sources above — correlating clip/segment features against public-channel growth curves (`market_video_snapshots`), and against the connected channel's real retention (`analytics_snapshots`/`analytics_retention_curve`) and, now that it's correctly identified as a distinct source, real CTR (`analytics_reach_snapshots`) — to produce new `model_versions` for the relevance, CTR, and hook models. This loop only becomes meaningfully useful once there's a corpus of clips with known source-segment provenance to correlate against — a bootstrapping dependency, not a day-one capability.

## 7. Explicitly out of scope for v1

- Thumbnail candidate picker (best-frame extraction)
- Multimodal (audio/video) input to the relevance-detection model — and, by extension, the hook-strength scorer — text-only confirmed for v1
- Full multi-object/face tracking for shorts cropping, beyond the lightweight single-subject crop
- Retention drop-off predictor (gated on accumulated Analytics data)
- Learned clip-boundary refiner (v2 evolution of the heuristic peak-finder)
- Multi-language/translation support
- Multi-tenant "connect your own channel" OAuth flow for other future users
- Publishing generated show notes/metadata back to YouTube automatically (open question noted in §2.6 — not decided either way)
- Any specific choice of job-queue tech, blob storage tech, or scheduler mechanism — explicitly left open, not to be assumed when implementation starts
