# Prior Art — Shorts-Generation Mechanics (Reframe / Caption / Competitors)

Scope: the mechanical parts of the shorts generator (segment selection via
the custom relevance model is covered separately in
`relevance-detection-prior-art.md`). Researched Aug 2026.

## 1. Commercial products ("long video → auto shorts")

Crowded, mature market — several established, well-funded players plus a
long tail of newer entrants. For competitive awareness only.

| Product | Notes |
|---|---|
| [OpusClip](https://www.opus.pro/) | Category leader. AI picks "viral" moments, auto-reframes with subject tracking, burns in animated captions, auto-posts. Claims 10M+ users. |
| [Klap](https://klap.app/) | Similar one-click workflow; positioned on speed/volume with smart captions and autoframing. |
| Quso (rebrand of Vidyo.ai, late 2024) | Expanded from pure clipper into broader repurposing platform (clips, captions, scheduler, brand kit). |
| Munch | Repurposing platform for agencies/brands; differentiates on trend/social-analytics intelligence layered on clip extraction. |
| Submagic | **Not** a highlight-selection tool — captioning + styling only, applied to already-cut clips. Useful comparison point since our caption-burn-in step overlaps its scope. |
| Vizard.ai | Auto-clips via "key moment" detection + transcript-based clip UI. Budget-oriented (~$14.50/mo). |
| Descript | Different paradigm (transcript-driven timeline editing); category leader for long-form/podcast editing generally, includes shorts repurposing. |

Takeaway: commercial tools combine subject detection/tracking for reframing
+ ASR-based captioning + an LLM/ML relevance model — the same building
blocks planned here, wrapped in SaaS UX/distribution.

## 2. Open-source building blocks

### 2a. Reframing/cropping (face/subject-detection based)
- **[Google AutoFlip](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/autoflip/README.md)** (MediaPipe) — the original, most-cited ML saliency/tracking reframing framework. **Deprecated, support ended March 2023**, only builds against OpenCV 3 — good reference, not a usable dependency today.
- **[auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe)** (Apr 2026, MIT) — PySceneDetect + YOLOv11-seg/ByteTrack for subjects, MediaPipe face/pose + saliency, smoothed pan/zoom path, content-type presets. Conceptually very close to our planned approach; early-stage (Beta, small repo).
- **[Autocrop-vertical](https://github.com/kamilstanuch/Autocrop-vertical)** — YOLOv8 person detection per scene (PySceneDetect) + Haar-cascade face fallback, OpenCV crop + FFmpeg re-encode. Most mature standalone reframing tool found (315 stars, active Feb 2026).
- **[AI-Shorts-Creator](https://github.com/NisaarAgharia/AI-Shorts-Creator)** — smaller `autocropper.py` doing face-detection 9:16 cropping as part of a larger script.

### 2b. Caption burn-in
- **[m1guelpf/auto-subtitle](https://github.com/m1guelpf/auto-subtitle)** — widely-forked reference: Whisper → SRT → FFmpeg burn-in overlay. The de facto "hello world" of this step.
- **whisper-burner**, **subtitle-burner** — smaller/less-verified projects found; pattern confirmed but no single dominant maintained library.
- General pattern: **Whisper/faster-whisper + FFmpeg `drawtext`/subtitle filter** is the standard, widely-reimplemented recipe — no need to adopt a specific library, just the pattern.

### 2c. End-to-end open-source "auto shorts" pipelines (closest prior art to our whole tool)
- **[SamurAIGPT/AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)** — explicitly "open-source alternative to Opus Clip/Vidyo.ai/Klap/SubMagic." Pipeline: download → Whisper → LLM highlight/virality scoring → chunk/dedup → top-N selection → OpenCV face-tracking crop with motion smoothing. 4.6k★/827 forks, MIT — most established open-source match.
- **[openshorts](https://github.com/mutonby/openshorts)** — more recent, actively developed (3.1k★/837 forks, MIT core). Pipeline: faster-whisper → PySceneDetect → Gemini for moment detection → FFmpeg extraction → dual-mode reframing (MediaPipe+YOLOv8 face/subject tracking, blurred-background fallback) → subtitles/overlays → auto-publish. Closer to a full SaaS architecture (FastAPI + React, Docker Compose) than a script — worth reviewing as an architectural reference given our own planned FastAPI backend.

(Several other repos found were "faceless" script→TTS→stock-visuals generators — a different problem, not included here.)

## 3. Speaker diarization (maturity check only — already a settled decision)
- **pyannote.audio 3.1** (MIT, Nov 2023) remains the dominant open-source diarization model, ~11–19% DER depending on benchmark, handles overlapping speech via its Powerset formulation.
- **WhisperX** (faster-whisper + pyannote + forced alignment) confirmed as the standard self-hosted transcribe+diarize combo, still in active current use.

## Non-obvious finding worth flagging
The mechanical crop+caption steps are **already well-solved in open source**,
not just commercially. `auto-vertical-reframe`, `Autocrop-vertical`, and the
reframing modules inside `AI-Youtube-Shorts-Generator`/`openshorts` already
implement essentially the lightweight subject-detection-based crop this
project planned to build (YOLOv8/v11 + MediaPipe/Haar-cascade + PySceneDetect).
**Worth reviewing these as direct implementation references or dependencies
for the shorts tool's mechanical layer, rather than building the crop logic
from scratch** — the differentiator should stay the relevance-detection
model, not the reframing mechanics. `openshorts` in particular mirrors our
own planned stack (FastAPI-based backend) and is worth a closer look as an
architectural reference in Phase 3.
