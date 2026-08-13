# Decision: YouTube Suite v1 Scope

**Date:** 2026-08-13
**Status:** Decided (Phase 1 of brainstorm)

## Context
YouTube Suite is a toolkit for managing YouTube video-podcast content:
upload a video, run AI-powered tools against it. Some tools use custom
models the user builds themselves; others lean on existing/pretrained
tools where those are already a good fit. A separate market-analysis
component pulls YouTube data daily to inform model training. This
document captures what's in/out of scope for v1, decided through a
structured tool-by-tool discussion.

## In scope for v1

| Component | Shape |
|---|---|
| Transcript/subtitle generator | Self-hosted Whisper; downloadable `.srt`/`.vtt` + raw transcript, English-only |
| Speaker diarization | pyannote/WhisperX |
| Relevance-detection model (custom) | Text-only transcript input; continuous relevance score per time-window aligned to transcript timestamps |
| Shorts generator | Fully automatic; peak-finds on the relevance score curve, snaps cuts to sentence/utterance boundaries, respects min/max clip duration; crop to user-selected aspect ratio (default 9:16) via a lightweight single-subject-detection crop, smoothed over time (not full multi-object tracking); optional caption burn-in toggle per export |
| Chapter/timestamp generator | Topic-segments transcript into YouTube chapter markers |
| Show notes & metadata generator | External LLM API — title/description/tags/summary suggestions from transcript |
| Market-analysis pipeline | Curated public podcast-channel list via public YouTube Data API (primary, `videos.list`-heavy to respect quota) + a platform-level YouTube Analytics API connection via OAuth (shared DB, single account for now — not a multi-tenant "connect your channel" feature) |
| Title/thumbnail CTR predictor (custom) | Predicts click-through performance from a candidate title + thumbnail, trained on market-analysis data |
| Hook-strength scorer (custom) | Scores a short's opening seconds for how likely it is to hold attention; bootstrapped heuristically at first |

## Out of scope for v1 (maybe later)
- Thumbnail candidate picker (best-frame extraction from source video)
- Multimodal (audio/video) input to the relevance-detection model — text-only confirmed for v1
- Full multi-object/face tracking for shorts cropping (beyond the lightweight single-subject crop)
- Retention drop-off predictor — gated on the OAuth Analytics connection having accumulated enough real data
- Learned clip-boundary refiner (v2 evolution of the heuristic peak-finder) — needs usage data from a running v1 first
- Multi-language/translation support
- Multi-tenant "connect your own channel" OAuth flow for other future users

## Still undecided — carried into Phase 2 research
- Relevance-detection model's training-signal/label source (heuristic bootstrap vs. weak supervision vs. curated dataset)
- Hook-strength scorer's training-data specifics (tied to the same bootstrap question)
- Title/thumbnail CTR predictor's feature set/model architecture (e.g. thumbnail embedding + title embedding → regression — needs a prior-art check)
- Which/how many channels seed the curated public-channel list, and how that list is initially assembled

## Custom vs. external, per component (for reference)
| Component | Build approach |
|---|---|
| Transcription/subtitles | External — Whisper |
| Diarization | External — pyannote/WhisperX |
| Shorts cut/reframe mechanics | External/utility — ffmpeg + lightweight subject-detection crop |
| Show notes & metadata | External — LLM API |
| Relevance-detection model | **Custom** |
| Title/thumbnail CTR predictor | **Custom** |
| Hook-strength scorer | **Custom** |

## Rejected / deferred ideas considered during discussion
- Thumbnail candidate picker — deferred, not rejected
- Retention drop-off predictor — deferred pending Analytics data accumulation
- Learned clip-boundary refiner — deferred pending v1 usage data
