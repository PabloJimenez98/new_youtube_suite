# M8 spike: YuNet face detection and shot cuts on episode 8

**Date:** 2026-10-16
**Milestone:** M8 — Shorts generator (planning sub-phase)
**Status:** Complete. The thresholds below are adopted in
`docs/decisions/0013-shorts-generator.md`.

A throwaway experiment, not product code. It checks the subject-crop
design before it is built, because the plan's M8 risk note names crop
quality as the most likely bad surprise.

## 1. Questions

1. Does **YuNet** (`cv2.FaceDetectorYN`, opencv-python-headless 4.14,
   `face_detection_yunet_2023mar.onnx`) find the right face on the
   owner's footage, on CPU, at a low decode width?
2. Can a simple **HSV-histogram difference** separate the show's
   camera cuts from motion inside a shot?
3. Does ffmpeg 6.1's `crop` filter take per-timestamp `sendcmd`
   commands, so the crop window can move inside one encode?

## 2. Setup

- **Source:** upload 8. 1280×720 AV1 at 25 fps with Opus audio, 64 min.
  It is an edited multi-camera show: mostly one person per shot,
  frequent cuts, and sometimes a blurred shoulder or hand in the
  foreground.
- **Spans:** five 45 s spans around M7's top moments (31:52, 34:07,
  40:37, 56:30, 57:52).
- **Decoding:** PyAV decodes every frame scaled to 320×180.
  - Every frame gets an H-S histogram (32×32 bins), compared with the
    previous frame by Bhattacharyya distance.
  - Every 5th frame (5 fps) goes through YuNet, with score threshold
    0.6 and NMS 0.3.

## 3. Results

| Span | Time | Per min of clip | Face coverage | >1 face | Cut-like diffs (>0.3) |
|---|---|---|---|---|---|
| 34:07 | 5.4 s | 7.2 s | 100 % | 5 % | 9 |
| 57:52 | 5.6 s | 7.4 s | 100 % | 2 % | 4 |
| 31:52 | 7.0 s | 9.4 s | 100 % | 1 % | 9 |
| 40:37 | 6.0 s | 8.0 s | 100 % | 1 % | 12 |
| 56:30 | 5.5 s | 7.3 s | 100 % | 0 % | 5 |

- **Faces:** the detected face was the speaking subject in every
  sampled contact-sheet frame. No foreground blur or hands were picked.
  - Faces measure 18–51 px wide at 320 px. That is enough for YuNet,
    so 320 is kept.
  - In 3 samples a second face touched the frame edge (a partly visible
    second host). The edge penalty handles these.
- **Cuts:** the histogram distance is ≤ 0.12 within a shot (median
  ~0.025, p99 ≤ 0.09 apart from one span). At a camera cut it is
  0.37–0.48.
  - **A threshold of 0.25** separates the two cleanly on every span.
  - Some shots are short (~0.8–1 s). Every shot still has ≥ 4 detector
    samples.
- **Crop:** `sendcmd=f=cmds.txt,crop=w=404:h=720:x=0:y=0` with
  `t crop x N;` lines moves the window as scheduled. Checked on
  `testsrc` at three timestamps.
- **Cost:** decode plus detection runs at ~7–9 s per minute of clip.
  Five 60 s clips is under a minute of analysis. Encoding will dominate.

## 4. Conclusion

- **Adopt YuNet at 320 px, 5 fps, and a cut threshold of 0.25.**
- **Not built:** speaker-aware face choice. Coverage was 100 % with
  very few two-face frames on this show, so area × confidence with an
  edge penalty is enough for v1. Noted in `brainstorm.md`.
- **Caveat:** one show, one camera style. A static two-shot (both hosts
  in one wide frame) would behave differently. M9 hardening should try
  one such episode.
