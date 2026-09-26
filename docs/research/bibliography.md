# Bibliography — Phase 2 Research

Index of all Phase 2 research, organized by topic. Each linked file has
full sources/citations; this page summarizes what each contributes and
surfaces findings that may affect scope (see bottom section). Researched
Aug 2026, informed by the scope agreed in `docs/decisions/v1-scope.md`.

## By topic

### Relevance-detection model (custom — the core differentiator)
→ [`relevance-detection-prior-art.md`](./relevance-detection-prior-art.md)
- Maps the problem space: video summarization vs. highlight detection vs.
  moment retrieval vs. text-salience scoring vs. GIF/clip curation — our
  tool sits closest to highlight detection + text-salience scoring.
- Surveys how training labels are typically sourced (human-annotated,
  editor-selected clips, retention/replay proxy signals, weak/duration
  heuristics, domain-curated).
- Catalogs 8 public datasets (TVSum, SumMe, QVHighlights, Video2GIF,
  YouTube Highlights, **Mr. HiSum**, QMSum, PHD²) with size/modality/license.
- Reviews 6 open-source "auto-clip" repos, notably **ClipsAI**.

### Chapter/timestamp generation
→ [`chapter-segmentation-prior-art.md`](./chapter-segmentation-prior-art.md)
- Classical methods (TextTiling, C99, TopicTiling, Bayesian segmentation).
- Embedding-based methods (BERT/SBERT segmentation, TreeSeg, Embed-KCPD).
- Recent LLM-based chaptering papers (**Chapter-Llama**, ARC-Chapter,
  LoRA table-of-contents fine-tuning).
- 6 open-source transcript-to-chapters tools.

### Shorts-generation mechanics (reframe, caption burn-in, competitors)
→ [`shorts-tooling-landscape.md`](./shorts-tooling-landscape.md)
- Commercial landscape: OpusClip, Klap, Quso, Munch, Submagic, Vizard.ai,
  Descript — mature, crowded market.
- Open-source reframing: AutoFlip (dead), auto-vertical-reframe,
  Autocrop-vertical, AI-Shorts-Creator.
- Open-source caption burn-in: auto-subtitle and the general
  Whisper+FFmpeg pattern.
- End-to-end open-source pipelines: **AI-Youtube-Shorts-Generator**,
  **openshorts** (closest full prior art to our whole shorts tool).
- Diarization maturity check: pyannote.audio 3.1 + WhisperX confirmed
  current standard (already a settled decision, no new info needed here).

### Title/thumbnail CTR predictor & hook-strength scorer (custom)
→ [`ctr-hook-prediction-prior-art.md`](./ctr-hook-prediction-prior-art.md)
- CTR/thumbnail-title prediction: established architecture pattern
  (CNN/text-embedding fusion → regression), YouTube-specific datasets
  (EnTube ~24K videos, arXiv 2508.02516 LMM engagement predictor), large
  public datasets (YouNiverse 137M videos, Trending YouTube Stats ~150K)
  — but **none contain true impression-level CTR**.
- Hook/opening-strength scoring: genuinely sparse in the literature,
  structurally so (retention-curve data is proprietary platform data).
- Competitive tools: TubeBuddy (live A/B testing), VidIQ + smaller
  startups (unverifiable claims), OpusClip/Klap "hook scores" (users
  report inconsistent results).

### YouTube Data API quota mechanics (market-analysis pipeline)
→ [`youtube-data-api-quota-notes.md`](./youtube-data-api-quota-notes.md)
- Default 10,000 units/day; `videos.list` = 1 unit/call (batchable to 50
  IDs); `search.list` = 100 units/call (expensive, use sparingly).
- Free, quota-free discovery alternative: per-channel RSS/Atom feed
  (last ~15 uploads only, no backfill).

## Flags for scope reconsideration

These are genuinely new/non-obvious findings surfaced during research
that go beyond what Phase 1 assumed — not decisions, just flagged for
your consideration before/during Phase 3:

1. **Mr. HiSum dataset** (NeurIPS 2023, CC BY 4.0, 31,892 videos) derives
   continuous importance labels from YouTube's "Most Replayed" retention
   heatmap rather than human annotation — a much better label-sourcing
   template than assumed in Phase 1. Suggests the relevance model's
   training-signal question (still "undecided" in the scope doc) might
   be answerable via retention-heatmap-style data rather than only
   heuristic bootstrap or hand-labeling. Worth a feasibility check on
   legality/access before committing.
2. **ClipsAI** is a surprisingly mature, directly relevant open-source
   repo doing transcript-only clip-*boundary*-finding (TextTiling on
   WhisperX output) for exactly the podcast/interview domain — but it
   does not *rank/score* segments by relevance. That ranking gap is
   exactly what the custom relevance model adds; ClipsAI is relevant to
   both the relevance model and the chapter generator simultaneously and
   worth reviewing as a reference/dependency rather than building
   transcript segmentation from scratch.
3. **Chapter-Llama / ARC-Chapter** (2025 papers) are purpose-built for
   "hour-long video → chapters from transcript," more on-target than
   generic topic-segmentation literature. Chapter-Llama's "speech-guided
   frame selection" (cheap transcript signal steering expensive
   multimodal processing) is a reusable architectural idea if video
   signal gets added post-v1.
4. **Implementation gotcha**: across both chaptering papers and hobbyist
   tools, LLM-found boundaries on cleaned/paraphrased text need a
   separate lexical-alignment step (TF-IDF matching) to map back to
   original timestamps — LLMs aren't reliable at preserving exact
   source timestamps over long context. Worth carrying into Phase 3 design.
5. **No existing OSS tool trains a dedicated relevance-*scoring* model**
   — all use heuristic segmentation or zero-shot LLM prompting for
   selection. Confirms the custom-model approach is a genuine
   differentiator, but also means there's no existing OSS baseline to
   benchmark against — the nearest comparable baseline is "prompt a
   strong LLM to score segments 0-100" as a zero-shot control worth
   building early as a quality floor.
6. **No dataset combines all four properties needed** (long-form
   talk/podcast domain + transcript-based + continuous score + large
   scale) — a real gap. Reinforces that proxy-labeled (retention-based,
   per finding #1) or LLM-distilled data collection is likely necessary
   rather than transfer learning from public sets — applies to both the
   relevance model and, by extension, the hook-strength scorer.
7. **True CTR ground truth doesn't exist publicly anywhere** — YouTube
   never publishes impressions. This means your own OAuth-connected
   channel's Analytics data is likely the *only* real CTR ground truth
   available; the curated multi-channel dataset should be framed as a
   proxy-engagement dataset calibrated against that one real source,
   not treated as a CTR dataset in its own right. Affects how the CTR
   predictor's training data section should be written in Phase 3.
8. **Neuromarketing-style "hook score" approaches exist as a different
   technical family** (Meta's TRIBE v2 fMRI-based brain-encoding model,
   commercial derivatives like Higgsfield trained on fMRI/EEG data) —
   predicting physiological attention response rather than engagement
   labels. Not recommending pursuit, just flagging awareness.
9. **The shorts tool's mechanical layer (crop + caption burn-in) is
   already well-solved in open source**, not just commercially —
   `auto-vertical-reframe`, `Autocrop-vertical`, and the reframing
   modules inside `AI-Youtube-Shorts-Generator`/`openshorts` already
   implement essentially the lightweight subject-detection crop planned
   in Phase 1. Worth reviewing as direct implementation references
   rather than building from scratch — keeps engineering effort focused
   on the actual differentiator (the relevance model). `openshorts` in
   particular mirrors the planned FastAPI-based backend and is worth a
   closer look as an architectural reference in Phase 3.

## Coverage notes
- Each research track stayed within roughly 8-25 sources; a few
  sub-areas (CTR prediction, competitive-tool verification) sat at the
  edge of the "propose narrowing" threshold — flagged in their source
  files if a deeper follow-up pass would be warranted later.
- Whisper/pyannote/WhisperX tooling was treated as already-settled per
  Phase 1 and only maturity-checked, not deeply re-researched.

## Development-phase spikes
Short measured experiments run during a milestone's planning or
development sub-phase to replace estimates with numbers on the real
hardware. Each one records its setup, its script and its results.
- **M3, faster-whisper on the GTX 1060:** `large-v3-turbo` int8 ran at
  ~14.7× real-time with ~1.55 GB VRAM. Recorded inline in
  [`../decisions/0007-transcription-pipeline.md`](../decisions/0007-transcription-pipeline.md) §1.
- **M4, pyannote `community-1` GPU vs. CPU:** GPU at ~17× real-time; it fits alongside Whisper with `embedding_batch_size=8` (~2.3 GB peak). CPU is under 1× real-time.
  → [`m4-diarization-spike.md`](./m4-diarization-spike.md)
