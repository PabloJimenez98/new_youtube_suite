"""Subject framing for a clip (docs/decisions/0013 §2, docs/research/m8-crop-spike.md).

Two halves:
- `analyse` (impure): PyAV decodes the span at ANALYSIS_WIDTH, a
  histogram distance per frame finds camera cuts, and YuNet looks for
  faces on every DETECT_EVERY-th frame.
- `plan` (pure): shots -> crop or fit, a face per sample, a smoothed
  crop centre reset at every cut, and the per-frame crop commands.
Coordinates in `Plan` are source pixels; times are relative to the clip.
"""

from dataclasses import dataclass, field
from pathlib import Path

ANALYSIS_WIDTH = 320
DETECT_EVERY = 5
FACE_SCORE = 0.6
CUT_THRESHOLD = 0.25
EDGE_PENALTY = 0.5
EDGE_BAND = 0.03  # of the frame width: a box this close to a side counts as "at the edge"
SCORE_POWER = 4  # confidence counts heavily: foreground hands are big but unsure (~0.65 vs ~0.93)
TIE_RATIO = 0.8  # a face within 20 % of the best, nearer the previous centre, wins
MIN_COVERAGE = 0.5
EMA_ALPHA = 0.3
DEAD_ZONE = 0.05  # of the crop width
MAX_SHORT_SIDE = 1080

ASPECTS = {"9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "16:9": (16, 9)}


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


@dataclass(frozen=True)
class Sample:
    t: float
    faces: tuple[Face, ...]


@dataclass
class Analysis:
    """What the decoder saw: frame times, cut distances, face samples (source pixels)."""

    width: int
    height: int
    fps: float
    frame_times: list[float]
    diffs: list[float]  # distance to the previous frame; 0 for the first
    samples: list[Sample]


@dataclass(frozen=True)
class Shot:
    start: float
    end: float
    mode: str  # "crop" | "fit"


@dataclass
class Plan:
    width: int  # source
    height: int
    crop_w: int  # 0 when nothing is ever cropped
    out_w: int
    out_h: int
    shots: list[Shot]
    track: list[dict] = field(default_factory=list)  # {t, x, y, w, h}, per sample

    def as_json(self) -> dict:
        return {
            "source": [self.width, self.height],
            "output": [self.out_w, self.out_h],
            "shots": [{"start": round(s.start, 3), "end": round(s.end, 3), "mode": s.mode} for s in self.shots],
        }


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def geometry(width: int, height: int, aspect: str) -> tuple[int, int, int]:
    """(crop width, output width, output height) for the target ratio.

    The crop is full height. The output's short side is the source's
    short side, at most MAX_SHORT_SIDE (720p -> 9:16 at 720x1280). A
    target as wide as the source or wider has no crop (crop width 0).
    """
    a, b = ASPECTS[aspect]
    ratio = a / b
    short = min(width, height, MAX_SHORT_SIDE)
    out_w, out_h = (even(short), even(short / ratio)) if ratio <= 1 else (even(short * ratio), even(short))
    crop_w = 0 if ratio >= width / height - 1e-6 else even(height * ratio)
    return crop_w, out_w, out_h


def shots_from_diffs(frame_times: list[float], diffs: list[float], duration: float,
                     threshold: float = CUT_THRESHOLD) -> list[tuple[float, float]]:
    """Frames whose distance to the previous one is above the threshold start a new shot."""
    cuts = [t for t, d in zip(frame_times, diffs, strict=True) if d > threshold and t > 0]
    edges = [0.0, *cuts, duration]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b > a]


def pick_face(faces: tuple[Face, ...], width: int, prev_cx: float | None) -> Face | None:
    """Best by area x confidence^4 (halved near the frame edge); ties go to the face nearest the previous centre.

    The power on confidence comes from QA on episode 8: a blurred
    foreground hand was detected at 0.6-0.7 with three times the area of
    the real face (0.93), a few pixels in from the edge.
    """
    if not faces:
        return None
    band = EDGE_BAND * width

    def weight(f: Face) -> float:
        edge = f.x <= band or f.x + f.w >= width - band
        return f.w * f.h * f.score**SCORE_POWER * (EDGE_PENALTY if edge else 1.0)

    ranked = sorted(faces, key=weight, reverse=True)
    if prev_cx is None:
        return ranked[0]
    close = [f for f in ranked if weight(f) >= TIE_RATIO * weight(ranked[0])]
    return min(close, key=lambda f: abs(f.cx - prev_cx))


def smooth_centres(raw: list[float | None], crop_w: int) -> list[float]:
    """Hold over missing samples, ignore moves inside the dead zone, then EMA towards the target."""
    known = [c for c in raw if c is not None]
    if not known:
        return []
    target = known[0]
    pos = target
    out = []
    for c in raw:
        if c is not None and abs(c - target) > DEAD_ZONE * crop_w:
            target = c
        pos += EMA_ALPHA * (target - pos)
        out.append(pos)
    return out


def plan(analysis: Analysis, duration: float, aspect: str) -> Plan:
    """Shots, crop/fit per shot and the smoothed crop track for a clip `duration` seconds long."""
    w, h = analysis.width, analysis.height
    crop_w, out_w, out_h = geometry(w, h, aspect)
    spans = shots_from_diffs(analysis.frame_times, analysis.diffs, duration)
    shots: list[Shot] = []
    track: list[dict] = []
    for s, e in spans:
        samples = [x for x in analysis.samples if s <= x.t < e]
        if crop_w == 0 or not samples:
            shots.append(Shot(s, e, "fit"))
            track.extend({"t": round(x.t, 3), "x": 0, "y": 0, "w": w, "h": h} for x in samples)
            continue
        raw: list[float | None] = []
        prev = None
        for x in samples:
            face = pick_face(x.faces, w, prev)
            raw.append(face.cx if face else None)
            if face:
                prev = face.cx
        coverage = sum(c is not None for c in raw) / len(raw)
        if coverage < MIN_COVERAGE:
            shots.append(Shot(s, e, "fit"))
            track.extend({"t": round(x.t, 3), "x": 0, "y": 0, "w": w, "h": h} for x in samples)
            continue
        shots.append(Shot(s, e, "crop"))
        for x, c in zip(samples, smooth_centres(raw, crop_w), strict=True):
            left = int(round(min(max(c - crop_w / 2, 0), w - crop_w)))
            track.append({"t": round(x.t, 3), "x": left, "y": 0, "w": crop_w, "h": h})
    return Plan(w, h, crop_w, out_w, out_h, shots, track)


def runs(shots: list[Shot]) -> list[Shot]:
    """Consecutive shots with the same mode merged: one filter branch each."""
    out: list[Shot] = []
    for s in shots:
        if out and out[-1].mode == s.mode:
            out[-1] = Shot(out[-1].start, s.end, s.mode)
        else:
            out.append(s)
    return out


def crop_commands(p: Plan, run: Shot, fps: float) -> str:
    """sendcmd lines (times relative to the run) setting the crop x for every frame of a crop run.

    x is interpolated linearly between the track samples of the same
    shot, and held at a shot's first/last sample; at a cut it jumps.
    """
    lines = []
    shots = [s for s in p.shots if s.start >= run.start - 1e-6 and s.end <= run.end + 1e-6]
    n = int(round((run.end - run.start) * fps))
    for k in range(n):
        t = run.start + k / fps
        shot = next((s for s in shots if s.start <= t < s.end), shots[-1])
        pts = [q for q in p.track if shot.start <= q["t"] < shot.end]
        if not pts:
            continue
        x = interpolate(pts, t)
        lines.append(f"{k / fps:.4f} crop x {x};")
    return "\n".join(lines) + "\n"


def interpolate(points: list[dict], t: float) -> int:
    if t <= points[0]["t"]:
        return points[0]["x"]
    for a, b in zip(points, points[1:], strict=False):
        if a["t"] <= t <= b["t"]:
            f = (t - a["t"]) / max(1e-9, b["t"] - a["t"])
            return int(round(a["x"] + f * (b["x"] - a["x"])))
    return points[-1]["x"]


# --- analysis (impure) -----------------------------------------------------------


class FaceModelMissingError(Exception):
    """SHORTS_FACE_MODEL_PATH is unset or the file is missing."""


def analyse(path: Path, start: float, end: float, model_path: Path) -> Analysis:
    """Decode [start, end) at ANALYSIS_WIDTH: cut distances on every frame, YuNet on every DETECT_EVERY-th."""
    import av
    import cv2

    if not model_path or not Path(model_path).is_file():
        raise FaceModelMissingError(
            "No face detection model -- set SHORTS_FACE_MODEL_PATH (see README, 'Shorts')"
        )
    frame_times: list[float] = []
    diffs: list[float] = []
    samples: list[Sample] = []
    with av.open(str(path)) as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        width, height = vs.codec_context.width, vs.codec_context.height
        fps = float(vs.average_rate or vs.guessed_rate or 25)
        aw = ANALYSIS_WIDTH
        ah = even(aw * height / width)
        scale = width / aw
        detector = cv2.FaceDetectorYN.create(str(model_path), "", (aw, ah), FACE_SCORE, 0.3, 50)
        container.seek(int(max(0.0, start - 1.0) / vs.time_base), stream=vs, backward=True)
        prev = None
        n = 0
        for frame in container.decode(vs):
            t = float(frame.pts * vs.time_base) if frame.pts is not None else start + n / fps
            if t < start - 1e-3:
                continue
            if t >= end:
                break
            img = frame.to_ndarray(format="bgr24", width=aw, height=ah)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            diffs.append(0.0 if prev is None else float(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)))
            prev = hist
            frame_times.append(t - start)
            if n % DETECT_EVERY == 0:
                _, found = detector.detect(img)
                faces = tuple(
                    Face(float(f[0]) * scale, float(f[1]) * scale, float(f[2]) * scale, float(f[3]) * scale,
                         float(f[-1]))
                    for f in (found if found is not None else [])
                )
                samples.append(Sample(t - start, faces))
            n += 1
    return Analysis(width, height, fps, frame_times, diffs, samples)
