# Prior Art: Segment Relevance / Highlight Detection from Long-Form Video or Transcript

Research pass for YouTube Suite's custom "relevance score per time-window" model (v1: text-only, timestamped transcript in → continuous score out, used to pick short-form clip candidates).

Date of research: 2026-08-13. Scope note: this is a broad literature/tooling area (video summarization + highlight detection + moment retrieval + clip-generation products all overlap); this file stays at survey depth rather than exhaustively cataloguing every paper. If deeper investigation is needed on any one row, treat it as its own follow-up task.

---

## 1. Academic framing of the problem

The task splits into several related but distinct academic sub-problems. Knowing which one a paper belongs to matters because it determines what modality/labels it assumes.

| Sub-problem | What it optimizes | Typical modality | Relevance to us |
|---|---|---|---|
| **Video summarization** (SumMe/TVSum-style) | Pick a short keyframe/keyshot subset that "represents" the whole video | Visual (+ sometimes audio) | Indirect — assumes visual saliency, not transcript semantics |
| **Video highlight detection** | Score/rank temporal segments by how "highlight-worthy" they are (not necessarily representative) | Visual, audio, sometimes multimodal w/ text query | Closest conceptual match — this is literally "continuous relevance score per time-window" |
| **Moment retrieval / temporal grounding** | Given a natural-language query, localize the video moment(s) it refers to | Video + text query | Different task (query-conditioned), but shares architecture/backbone ideas and often shares benchmarks (e.g. QVHighlights does both) |
| **Extractive text/transcript summarization & salience scoring** | Score sentences/spans of a document by importance for inclusion in a summary | Text only | Closest **modality** match for our v1 (text-only transcript) — 20+ years of NLP work on "sentence salience" is directly reusable |
| **GIF/clip curation from video** (Video2GIF-style) | Given a long video, pick the sub-clip(s) people would turn into a GIF/share | Visual (+ audio) | Closest **product** match — literally "find the clip-worthy moment" |

**Key papers/surveys found:**
- A 2025-era survey organizes ~46 deep-learning highlight-detection approaches around five themes: multimodal fusion, segment/shot-level scoring, graph-based modeling, transformers, and supervised/unsupervised/self-supervised training. Useful as an entry point into the current method landscape. (No single canonical arXiv ID surfaced for this exact survey in search; treat as a signpost to search further if deep method review is needed.)
- **Video Highlight Detection via Deep Ranking Modeling** (Springer/ACCV) — early (2018) framing of highlight detection as pairwise ranking rather than absolute regression; ranking-based supervision (A is more highlight-worthy than B) is cheaper to collect than absolute scores and may be a useful label strategy for us. https://link.springer.com/chapter/10.1007/978-3-319-75786-5_3
- **Highlight Detection with Pairwise Deep Ranking for First-Person Video** (Microsoft Research) — originating pairwise-ranking highlight paper. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/2219-1.pdf
- **Less is More: Learning Highlight Detection from Video Duration** (arXiv 1903.00859) — uses video *duration* as a weak supervision signal (shorter videos are assumed more consistently "highlight-dense"), avoiding manual highlight labels entirely. Relevant as an example of a cheap proxy-label trick.
- **AHA: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead** (arXiv 2509.16421, 2025) — frames highlight detection as *causal/online* (score up to time t without seeing the future), which is architecturally closer to what a streaming or low-latency chapter/clip picker would need than most offline highlight papers.
- **Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence** (2025) — unsupervised approach using recurrence of audio/visual patterns as a highlight signal, avoiding labels altogether. https://www.researchgate.net/publication/390618563
- **Query-Dependent Video Representation for Moment Retrieval and Highlight Detection** (CVPR 2023, "QD-DETR") — strong joint architecture on QVHighlights; representative of current SOTA moment-retrieval+highlight joint modeling. https://openaccess.thecvf.com/content/CVPR2023/papers/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.pdf

**Text-only / transcript salience (most directly transferable to our v1 scope):**
- Sentence/segment salience scoring is a mature extractive-summarization subfield: classic surface features (TF-IDF, position, length) through neural "jointly learn to score and select sentences" models. See **Neural Document Summarization by Jointly Learning to Score and Select Sentences** (arXiv 1807.02305) for a clean supervised formulation of exactly "assign each sentence/segment a scalar importance score."
- **Extractive Summarization of Call Transcripts** (arXiv 2103.10599) — directly analogous domain (spoken, multi-turn transcript rather than written prose); worth reading closely since call-transcript salience scoring is structurally the same problem as podcast-transcript relevance scoring.
- **SupMMD: Sentence Importance via Maximum Mean Discrepancy** (arXiv 2010.02568) — an alternative unsupervised/weakly-supervised salience-scoring formulation.
- Long-document salience classifiers have been shown to recover a meaningful fraction of human-marked salient sentences in 200+ sentence documents — evidence that a scalar-salience-per-span formulation is tractable at podcast-transcript length.

## 2. How such models are typically trained / where labels come from

| Label source | Examples | Pros | Cons for us |
|---|---|---|---|
| **Human-annotated highlight/importance scores** | TVSum (crowd 1–5 importance per 2s shot), SumMe (15-18 human reference summaries/video) | High quality, directly matches target definition | Very expensive/slow to scale; small datasets (tens of videos) |
| **Editor-selected clips / "already cut" data** | Video2GIF (GIF ⊂ source video), trailer-moment datasets ("Learning Trailer Moments in Full-Length Movies") | Free, naturally large-scale, matches "what got clipped" objective directly | Selection bias toward what's giftable/trailer-able, not necessarily "relevant"; mostly visual-content domains, not talking-head podcasts |
| **Proxy engagement signals (retention curves, replays, shares)** | **Mr. HiSum** uses YouTube's "Most Replayed" heatmap as ground truth (see below) | Scales to tens of thousands of videos essentially for free; directly measures *actual audience behavior*, which is arguably a better proxy for "clip-worthy" than human judgment | Confounded by thumbnail/algorithm effects, replay ≠ topical relevance (could be "what did she just say" confusion spikes), not available for all creators' own catalogs unless you have YouTube Analytics API access |
| **Weak/self-supervision from video duration or recurrence** | "Less is More" (duration as proxy), unsupervised audio/visual recurrence paper | No labels needed at all | Weakest signal quality; more useful for pretraining/initialization than final signal |
| **Domain-specific curated sets** | YouTube Highlights dataset (Sun et al. 2014) — surfing/skating/skiing/gymnastics/parkour/dog, ~100 videos/domain, segment labels {highlight, borderline, non-highlight} | Clean 3-way labels | Action-sport domain, not talk/podcast content; visual not transcript-based |

**Most actionable label-sourcing idea for our use case:** replicate the Mr. HiSum approach — use YouTube's own "Most Replayed" retention-heatmap data (available via scraping the player's heatmap SVG or YouTube Analytics API for owned/licensed channels) as a free, large-scale, real-audience-behavior proxy label for a transcript-relevance model, instead of (or in addition to) hand-labeling. This sidesteps the small-dataset problem that plagues academic highlight datasets.

## 3. Public datasets

| Dataset | Size | Modality | Labels | License | Notes |
|---|---|---|---|---|---|
| **TVSum** | 50 YouTube videos, 10 categories | Video (+ title/category metadata) | Crowd-sourced 1–5 importance score per 2s shot, 20 annotators/video | Research use (see dataset site for exact terms) | Small; structured/event-centric categories (tire change, making a sandwich, etc.), not talk content |
| **SumMe** | 25 YouTube videos, 1–6 min | Video | 15–18 human reference summaries/video | Research use | Small; personal/travel/sports videos, not talk content |
| **QVHighlights** | 10,310 queries / 18,367 moments over ~4,435 YouTube videos (3,478 train / 431 val / 426 test) | Video + natural-language query + ASR captions | Human-annotated relevant moment spans per query, saliency scores | Annotations CC BY-NC-SA 4.0, code MIT | Only public dataset combining moment retrieval + highlight detection; query-conditioned (not quite our "query-free" continuous score), but ASR-caption inclusion makes it partially transcript-relevant |
| **Video2GIF** | 100K+ GIF–video pairs (120K+ GIFs, 80K+ videos, 7,379 hours) | Video (GIF as label of "clip-worthy span") | Implicit: GIF's source span = positive highlight | Research dataset via GitHub (gyglim/video2gif_dataset) | Large scale, but GIFs are visually/comedically driven, not talk/podcast domain; label is binary/implicit not continuous |
| **YouTube Highlights (Sun et al. 2014)** | ~100 videos × 6 domains (surfing, skating, skiing, gymnastics, parkour, dog), ~1,430 min total | Video | 3-way segment label {highlight, borderline, non-highlight} | Research use | Classic small benchmark; action-sport, not talk domain |
| **Mr. HiSum** (NeurIPS 2023) | 31,892 videos (from YouTube-8M), labels aggregated from 50,000+ users/video | Video (frame-level features) | Continuous frame-importance score from YouTube "Most Replayed" data | **CC BY 4.0** (+ must respect YouTube ToS) | **Most relevant find**: large scale, continuous importance score (not just binary), and label methodology (retention/replay proxy) is conceptually identical to what we'd want for transcript relevance. Worth close study even though it's visual-feature based, not transcript-based — the labeling recipe is directly portable to transcripts if the underlying videos have transcripts. |
| **QMSum** | 1,808 query–summary pairs over 232 meetings (product/academic/political) | Text transcript | Manual topic segmentation + query-relevant span annotations + reference summaries | Research use (see ACL page) | Meeting-transcript, not YouTube/podcast, but structurally the closest **text-only, span-relevance-annotated, spoken-transcript** dataset found; useful for pretraining/transfer even if out-of-domain |
| **PHD²/Video2GIF personalized set** | User-created GIF corpus | Video | Personalized highlight preference | Research use | Adds a "personalization" angle (whose highlight?) we likely don't need for v1 |

**Gap observed:** there is no public dataset that is (a) long-form talk/podcast video, (b) transcript-based, (c) continuous relevance score, (d) large scale. The closest analogues split those four properties across different datasets (Mr. HiSum has scale + continuous score but is visual/YouTube-8M-general-domain; QMSum has transcript + span relevance but is small and meeting-domain, not YouTube). This likely means a custom-labeled or proxy-labeled (retention-based) dataset is necessary rather than pure transfer learning from an existing public set.

## 4. Existing open-source projects doing "find the best clip in a long video"

These are product-shaped repos, useful for understanding what heuristics/pipelines are already common in this exact product space (auto-clipping long-form video/podcasts):

| Project | Approach | Notes |
|---|---|---|
| **ClipsAI** (github.com/ClipsAI/clipsai) | Transcribes with WhisperX (word-level timestamps) → runs **TextTiling** on the transcript to find topic-shift boundaries → treats topic segments as clip candidates → reframes 16:9→9:16 tracking the speaker | Most directly relevant open-source repo: purely transcript-driven, no visual/audio saliency model, and it uses the same TextTiling algorithm relevant to our chapter-generation sub-task (see other file). Notably it does NOT rank/score segments by "relevance" — it only finds boundaries, leaving selection to the caller. That gap (boundary-finding vs. relevance-scoring) is exactly the piece our project intends to add. |
| **AI-Youtube-Shorts-Generator** (Anil-matcha) | Whisper transcription + LLM (prompted) highlight detection + auto vertical crop | LLM-prompted selection rather than a trained scoring model — i.e. "ask GPT which moments are viral" rather than a custom relevance model. Useful as a zero-shot baseline to compare a custom model against. |
| **AutoClip** (artbyjazi/autoclip) | Whisper transcription + LLM (local Ollama or API) to find clip-worthy moments, speaker-tracked reframing, caption burn-in | Same LLM-prompted-selection pattern; fully local option is notable for cost/privacy comparison. |
| **OpenShorts** (mutonby/openshorts) | AI moment detection + face tracking + subtitles + dubbing, Docker self-host (MIT) or paid cloud/GPU + MCP server | Another LLM/heuristic-based product; MIT-licensed self-host option worth a closer look if we want to see a full pipeline end-to-end. |
| **opensource-clipping** (NaufalRizqullah) | Whisper + Gemini + MediaPipe + Pyannote (face tracking, speaker diarization, kinetic captions, B-roll) | More elaborate multimodal pipeline; again LLM-prompted selection, not a trained scorer. |
| **Opus Clip** (commercial, not open-source) | Multimodal (words + visual action + tone of voice); scores clips 0–99 on "virality" from 4 signals: hook strength, emotional flow, perceived value, trend alignment | Not open-source, but useful as a target UX/scoring-output bar: a single interpretable score + sub-signal breakdown is a good product pattern to emulate regardless of what powers it internally. |

**Pattern across almost all open-source repos in this space:** none of them train a custom relevance-scoring model. They either (a) use TextTiling/heuristics purely for segmentation, leaving ranking to a human or simple heuristic, or (b) prompt a general-purpose LLM zero-shot/few-shot to pick "the best moments," rather than training a dedicated model on labeled/proxy data. This suggests the custom-model approach YouTube Suite is pursuing is a genuine differentiator relative to current open-source prior art, not a reinvention of something already commoditized.

---

## Flags for follow-up (if this area needs deeper investigation)

- This file stayed at ~25 sources across 4 sub-areas (highlight detection academic, label sourcing, datasets, OSS tools). If you want a rigorous SOTA comparison table (numbers, benchmarks) for highlight-detection architectures specifically, that alone is worth a dedicated narrower research pass — propose scoping it to "transformer-based moment/highlight architectures 2023–2026, benchmarked on QVHighlights" (~8-10 sources on its own).
- Mr. HiSum's "Most Replayed" labeling methodology deserves a focused feasibility check: can we legally/practically scrape or API-access retention-heatmap data for arbitrary YouTube videos (not just our own channel) to bootstrap training labels at scale? That's a legal/ToS + engineering question, not a literature question, and out of scope for this pass.
