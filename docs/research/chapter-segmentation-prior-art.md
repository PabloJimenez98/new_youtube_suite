# Prior Art: Transcript/Topic Segmentation for Chapter Generation

Research pass for YouTube Suite's chapter/timestamp generator (topic-segments a long-form transcript into human-readable chapters).

Date of research: 2026-08-13. Scope note: topic segmentation is a well-established NLP subfield (25+ years of work since TextTiling) plus a newer wave of LLM-based approaches; this file surveys the landscape at a level sufficient for architecture decisions, not an exhaustive literature review.

---

## 1. Classical / statistical approaches

| Approach | Core idea | Era | Notes |
|---|---|---|---|
| **TextTiling** (Hearst, 1997) | Split text into token-window blocks, compute lexical cosine similarity between adjacent blocks, place boundaries at similarity "valleys" | Foundational | The original unsupervised topic-segmentation algorithm; still used today as a baseline and, notably, in production inside the ClipsAI tool (see other file). Free of training data requirements — purely lexical co-occurrence. |
| **C99** (Choi, 2000) | Builds a similarity matrix between all sentence pairs, uses rank-based local similarity and divisive clustering to find boundaries | Foundational | Common baseline alongside TextTiling in comparison papers. |
| **TopicTiling** (Riedl & Biemann, ACL 2012) | TextTiling variant that uses LDA topic-model assignments per word instead of raw lexical overlap for the similarity signal | Classical+topic-model | https://aclanthology.org/W12-3307.pdf |
| **Bayesian segmentation (BayesSeg-family)** | Probabilistic generative models of topic shift | Classical | Referenced across survey literature as another pre-neural baseline family. |

**Survey covering this whole space (recommended starting point):**
**Recent Trends in Linear Text Segmentation: A Survey** (Ghinassi, Wang, Newell, Purver — EMNLP 2024 Findings, arXiv 2411.16613) — explicitly frames the motivation as "surge of text/video/audio online requiring segmentation as a first step for summarization/categorization," reviews available resources, and flags under-explored directions. Good single entry point before going deeper into any one method family. https://arxiv.org/abs/2411.16613 / https://aclanthology.org/2024.findings-emnlp.174/

## 2. Embedding-based segmentation (neural, pre-LLM-chaptering)

| Approach | Core idea | Notes |
|---|---|---|
| **Embedding-Enhanced TextTiling** (arXiv 1610.03955) | Replace/augment TextTiling's lexical similarity with dense word/sentence embedding similarity for dialogue session segmentation | Directly bridges classical TextTiling and modern embedding approaches |
| **Text Segmentation based on Semantic Word Embeddings** (arXiv 1503.05543) | Early (2015) use of word embeddings for boundary detection | Predates sentence-transformer era |
| **Unsupervised Topic Segmentation of Meetings with BERT Embeddings** (arXiv 2106.12978) | Uses BERT sentence embeddings + coherence scoring to segment meeting transcripts unsupervised | Meeting-transcript domain — close analogue to podcast transcripts |
| **Document Segmentation for Topic Modelling with Sentence Embeddings** (IEEE ACDSA 2024) | SBERT sentence embeddings + cosine-distance between adjacent segments as the boundary signal, building explicitly on TextTiling's structure | Represents current "standard recipe": swap TextTiling's lexical-overlap score for SBERT cosine distance |
| **Embed-KCPD: Unsupervised Text Segmentation via Kernel Change-Point Detection on Sentence Embeddings** (arXiv 2601.18788, 2026) | Training-free method treating sentence embeddings as a sequence and finding boundaries via penalized kernel change-point detection | Most recent (2026) embedding-based method found; training-free is attractive for a v1 that doesn't want to collect segmentation-labeled training data |
| **TreeSeg** (AugmendTech, arXiv 2407.12028 + open-source repo github.com/AugmendTech/treeseg) | **Hierarchical** topic segmentation of long transcripts using embeddings + divisive clustering — produces nested chapter/sub-chapter structure, not just flat boundaries | Notable: open-source, MIT-ish research repo, and directly targets "large transcripts" (meetings) with a hierarchical output — closer to what a nice chapter UI (chapters + sub-chapters) would want than flat TextTiling-style segmentation. Worth a closer look as a candidate approach or baseline. |
| **With a Little Help from my (Linguistic) Friends** (arXiv 2402.02837) | Topic segmentation of **multi-party casual conversations** using linguistic/discourse cues | Relevant if podcast transcripts include multiple speakers/hosts — most segmentation literature assumes single-narrator or clean document text, this one specifically targets conversational multi-party structure |

## 3. LLM-based chaptering (current state of the art for this exact product feature)

| Approach | Core idea | Notes |
|---|---|---|
| **Chapter-Llama** (CVPR 2025, arXiv 2504.00072) | Feeds speech transcript + timestamps into a large-context LLM; proposes a lightweight **speech-guided frame selection** strategy so the (much cheaper) transcript signal guides which video frames the multimodal model needs to look at, instead of processing every frame | **Most directly relevant paper found for this whole research task.** Purpose-built for "hour-long video → chapters," matches our exact target length/use case, and the speech-guided-selection trick (use cheap transcript signal to avoid expensive full multimodal processing) is directly applicable to keeping our v1 (text-only) approach efficient even if a later version adds video signal. https://openaccess.thecvf.com/content/CVPR2025/papers/Ventura_Chapter-Llama_Efficient_Chaptering_in_Hour-Long_Videos_with_LLMs_CVPR_2025_paper.pdf |
| **ARC-Chapter** (arXiv 2511.14349, Nov 2025) | Feeds a multimodal transcript with (optionally user-provided) chapter timestamps/titles into an LLM, asks it to reorganize content into structured chapters with titles, abstracts, and precise temporal boundaries — produces **navigable hierarchical summaries**, not just flat boundary lists | Very recent (post Chapter-Llama); pushes toward richer output (title + abstract + intro per chapter), which is closer to what a polished chapter-generator product feature needs than a bare boundary-timestamp list. https://arxiv.org/pdf/2511.14349 |
| **Automate Video Chaptering with LLMs and TF-IDF** (Medium/TDS, Yann-Aël Le Borgne) + companion repo **daveshap/YouTubeChapterGenerator** style pipelines | Practical (non-paper) recipe: use a cheap/fast LLM (e.g. Llama 3 8B) for paragraph/topic-boundary identification, a stronger LLM (e.g. GPT-4o-mini class) for chapter titles, and TF-IDF matching to re-attach timestamps after the LLM has restructured/cleaned the paragraph text | Practical engineering pattern worth noting: LLMs are good at finding *semantic* topic breaks in cleaned text but bad at preserving exact original timestamps, so a lexical (TF-IDF) re-alignment step is used to map LLM-identified boundaries back to the original timestamped transcript. This "LLM segments cleaned text, then align back to timestamps" pattern is a reusable architecture idea regardless of which model does the segmentation. https://towardsdatascience.com/automate-video-chaptering-with-llms-and-tf-idf-f6569fd4d32b/ ; repo: https://github.com/Yannael/automatic-video-chaptering |
| **Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation** (arXiv 2601.02128) | Fine-tunes (via LoRA) an LLM specifically for multi-level (hierarchical) transcript segmentation / table-of-contents generation | Directly relevant if we consider fine-tuning a small open model specifically for the chaptering task rather than prompting a general LLM zero-shot; worth a closer read before any fine-tuning decision. |

## 4. Existing open-source tools/libraries that already do transcript-to-chapters

| Tool | What it does | License/status | Notes |
|---|---|---|---|
| **ClipsAI** (github.com/ClipsAI/clipsai) | WhisperX transcription → **TextTiling** on transcript → topic-shift boundaries used as clip/segment candidates | Open-source (see repo for license) | Same repo referenced in the relevance-detection file; here it's relevant as a working example of classical TextTiling applied to real (WhisperX word-timestamped) transcripts in production, i.e. proof the classical approach is viable at this problem's scale without any ML training. |
| **daveshap/YouTubeChapterGenerator** | Script-based pipeline: clean/concatenate a downloaded transcript, then use GPT-3-era LLM prompting to propose chapters | Open-source | Simple, LLM-prompt-only baseline; no fine-tuning or embeddings involved. |
| **TreeSeg** (AugmendTech) | Hierarchical embedding-based segmentation, open-source, purpose-built for "large transcripts" | Open-source (github.com/AugmendTech/treeseg) | See section 2 — notable because it's both open-source *and* produces hierarchical output. |
| **symscribe** (ExtensityAI) | Whisper transcription + chapter segmentation + timestamp handling, exports chapters.csv/transcript.csv | Open-source | Smaller/less documented tool; worth noting as another existing implementation but likely less mature than ClipsAI/TreeSeg. |
| **DetailedChaptersGenerator** (normand1) | Whisper transcription → timestamped chapters, specifically tuned to key off discussed links/topics in podcast episodes | Open-source | Niche but shows the "transcribe → chapter" pattern is a common enough need that multiple independent hobbyist implementations exist. |
| **Auphonic** (commercial feature, not OSS) | Automatically generates shownotes, summaries, and chapters from recordings | Commercial/SaaS | Useful as a UX reference for output shape (shownotes + chapters combined), not as a technical prior-art source. |
| **Podium / Castmagic / Descript** (commercial) | AI-generated podcast chapters as one feature among many (also show notes, clips, social posts) | Commercial/SaaS | None publish their underlying method (near-certainly LLM-prompt-based given the broader market pattern); useful only as product/UX benchmarks, not technical prior art. |

## 5. Relevant datasets for evaluating/training a segmenter (not full re-list of Section 1's file, only segmentation-specific)

| Dataset | Relevance to chaptering |
|---|---|
| **AMI Meeting Corpus** (100 hrs) and **ICSI Meeting Corpus** (70 hrs) | Both have manual **hierarchical topic-segmentation annotations** (AMI: 3 levels deep, ICSI: 4 levels deep) plus word-level transcription timing — the closest available public ground truth for "segment a spoken, multi-speaker transcript into nested topics," even though the domain is meetings, not podcasts. Strong candidate for pretraining/validating a segmentation approach before podcast-specific data is available. https://groups.inf.ed.ac.uk/ami/corpus/ |
| **QMSum** (232 meetings, 1,808 query-summary pairs) | Includes manual topic segmentation *and* relevant-span annotation, so it doubles as a segmentation dataset and a relevance/salience dataset (see other file, Section 3) — same source useful for both YouTube Suite sub-projects. https://arxiv.org/abs/2104.05938 |
| **Chapter Captor: Text Segmentation in Novels** (arXiv 2011.04163) | Segmentation dataset from long-form novels rather than transcripts — out of domain, but relevant as another example of the "long document → chapter boundaries" task framing outside video/audio. |

---

## Synthesis: what this means for our chapter generator

- The field has moved in three waves: **(1) classical lexical (TextTiling/C99)** → **(2) embedding/cosine-distance-based** → **(3) LLM-prompted/fine-tuned**, and all three are still in active use (e.g. ClipsAI ships TextTiling in production today; Chapter-Llama and ARC-Chapter are 2025 LLM papers).
- A very workable v1 architecture pattern, seen repeatedly across both papers and OSS tools, is: **LLM (or embedding-based method) finds semantic boundaries on cleaned transcript text → separate alignment step maps those boundaries back to the original timestamps** (TF-IDF or fuzzy text matching), because LLMs are unreliable at preserving exact original timestamps/tokens when reasoning over long text.
- **TreeSeg** and **ARC-Chapter** both push toward *hierarchical* chapter structure rather than a flat list — worth deciding early whether v1 targets flat chapters (simpler, matches YouTube's native chapter UI) or nested chapters (richer but more complex evaluation/UX).
- No dedicated large-scale public "YouTube podcast transcript → chapters" dataset exists; AMI/ICSI (meetings) and QMSum are the closest proxies for pretraining or validation methodology, but any production model will likely need either LLM-distilled labels (have a strong LLM label a sample of real podcast transcripts) or lightweight human annotation on in-domain data.

## Flags for follow-up (if this area needs deeper investigation)

- This file covers roughly 20 sources across classical, embedding, and LLM-based segmentation plus ~7 OSS tools. That's already close to the "propose narrowing" threshold; if you want head-to-head benchmark numbers (e.g. Chapter-Llama vs. ARC-Chapter vs. TreeSeg vs. plain TextTiling+SBERT on a shared eval set), that comparison doesn't fully exist in the literature yet (different papers use different eval sets/domains) and would need a dedicated small benchmarking exercise rather than more literature search.
- Chapter-Llama and ARC-Chapter are both multimodal (transcript + frames) in their full form; a follow-up read of each paper's ablations specifically for the "transcript-only" variant's performance vs. the full multimodal model would directly inform how much accuracy we're leaving on the table by staying text-only for v1.
