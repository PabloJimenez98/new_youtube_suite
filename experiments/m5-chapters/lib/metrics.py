"""Chapter-boundary evaluation metrics.

Definitions mirror Chapter-Llama's evaluation code (MIT,
github.com/lucas-ventura/chapter-llama, `tools/results/metrics_distance.py`
and `metrics_overlap.py`) so our numbers are comparable with theirs --
see docs/research/m5-chapter-model.md §3:

- **F1@Ns** (primary): per video, the 0 s start is dropped from both
  sides; a prediction and a reference "match" when they are less than N
  seconds apart (many-to-many, not one-to-one). Precision/recall are
  averaged over videos, then combined into F1.
- **tIoU-F1**: chapters as [start, end) segments; per video and per tIoU
  threshold in 0.5..0.95 (step 0.05), a segment is covered when some
  segment on the other side overlaps it with IoU >= threshold. Per-video
  F1 is averaged over videos, then over thresholds.
- **Mean tIoU**: greedy one-to-one assignment by descending IoU, mean
  IoU per video, averaged over videos.

Plus two classic segmentation metrics over the unit sequence (Pk,
WindowDiff) and the chapter-count error.

Every video-level function takes chapter *start times in seconds*;
`duration` closes the last chapter.
"""

from collections.abc import Mapping, Sequence

import numpy as np

TIOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.95 + 1e-9, 0.05), 2))


def _drop_zero(starts: Sequence[float]) -> list[float]:
    return [s for s in starts if s != 0]


def video_pr_at_distance(
    refs: Sequence[float], preds: Sequence[float], dist: float
) -> tuple[float, float] | None:
    """Precision/recall of one video's boundaries at tolerance `dist` s.

    Returns None when the reference has no boundary besides 0 -- recall
    is undefined there (Chapter-Llama would divide by zero; our data
    filter requires >= 3 chapters, so this only matters in tests).
    """
    refs = _drop_zero(refs)
    preds = _drop_zero(preds)
    if not refs:
        return None
    ref_hit: set[int] = set()
    pred_hit: set[int] = set()
    for j, p in enumerate(preds):
        for i, r in enumerate(refs):
            if abs(p - r) < dist:
                ref_hit.add(i)
                pred_hit.add(j)
    return len(pred_hit) / max(len(preds), 1), len(ref_hit) / len(refs)


def f1_at_distance(
    refs: Mapping[str, Sequence[float]], preds: Mapping[str, Sequence[float]], dist: float
) -> dict[str, float]:
    """Corpus-level P/R/F1 at `dist` seconds, in percent."""
    ps, rs = [], []
    for vid, vid_refs in refs.items():
        if vid not in preds:
            continue
        pr = video_pr_at_distance(vid_refs, preds[vid], dist)
        if pr is None:
            continue
        ps.append(pr[0])
        rs.append(pr[1])
    if not ps:
        raise ValueError("no evaluable videos")
    p = 100 * float(np.mean(ps))
    r = 100 * float(np.mean(rs))
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def starts_to_segments(starts: Sequence[float], duration: float) -> list[tuple[float, float]]:
    """Sorted chapter starts -> [start, end) segments; a 0 start is added if missing."""
    s = sorted(set(starts))
    if not s or s[0] != 0:
        s = [0.0, *s]
    ends = [*s[1:], duration]
    return [(a, b) for a, b in zip(s, ends, strict=True)]


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = min(max(a[1], b[1]) - min(a[0], b[0]), (a[1] - a[0]) + (b[1] - b[0]))
    return inter / (union + 1e-8)


def _overlaps(refs, preds) -> dict[tuple[int, int], float]:
    out = {}
    for i, r in enumerate(refs):
        for j, p in enumerate(preds):
            v = iou(r, p)
            if v > 0:
                out[(i, j)] = v
    return out


def tiou_f1(
    refs: Mapping[str, Sequence[float]],
    preds: Mapping[str, Sequence[float]],
    durations: Mapping[str, float],
    thresholds: Sequence[float] = TIOU_THRESHOLDS,
) -> float:
    """Chapter-Llama's headline F1 (percent), averaged over tIoU thresholds."""
    per_threshold = []
    for t in thresholds:
        f1s = []
        for vid, vid_refs in refs.items():
            if vid not in preds:
                continue
            r_seg = starts_to_segments(vid_refs, durations[vid])
            p_seg = starts_to_segments(preds[vid], durations[vid])
            hits = [k for k, v in _overlaps(r_seg, p_seg).items() if v >= t]
            p = len({j for _, j in hits}) / max(len(p_seg), 1)
            r = len({i for i, _ in hits}) / len(r_seg)
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        per_threshold.append(float(np.mean(f1s)))
    return 100 * float(np.mean(per_threshold))


def mean_tiou(
    refs: Mapping[str, Sequence[float]],
    preds: Mapping[str, Sequence[float]],
    durations: Mapping[str, float],
) -> float:
    """Mean IoU of a greedy one-to-one segment assignment (percent)."""
    per_video = []
    for vid, vid_refs in refs.items():
        if vid not in preds:
            continue
        r_seg = starts_to_segments(vid_refs, durations[vid])
        p_seg = starts_to_segments(preds[vid], durations[vid])
        used_r: set[int] = set()
        used_p: set[int] = set()
        chosen = []
        for (i, j), v in sorted(_overlaps(r_seg, p_seg).items(), key=lambda kv: -kv[1]):
            if i not in used_r and j not in used_p:
                used_r.add(i)
                used_p.add(j)
                chosen.append(v)
        per_video.append(float(np.mean(chosen)) if chosen else 0.0)
    return 100 * float(np.mean(per_video))


def count_error(
    refs: Mapping[str, Sequence[float]], preds: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    """Chapter-count MAE plus % of videos under / exactly / over chaptered."""
    errs = np.array(
        [
            len(starts_to_segments(r, float("inf"))) - len(starts_to_segments(preds[v], float("inf")))
            for v, r in refs.items()
            if v in preds
        ]
    )
    n = len(errs)
    return {
        "mae": float(np.abs(errs).mean()),
        "under_pct": 100 * float((errs > 0).sum()) / n,
        "equal_pct": 100 * float((errs == 0).sum()) / n,
        "over_pct": 100 * float((errs < 0).sum()) / n,
    }


def _labels(boundaries: Sequence[int], n_units: int) -> np.ndarray:
    """Unit-level segment ids from boundary unit indices (a chapter starts at each)."""
    marks = np.zeros(n_units, dtype=int)
    for b in boundaries:
        if 0 < b < n_units:
            marks[b] = 1
    return np.cumsum(marks)


def default_k(ref_boundaries: Sequence[int], n_units: int) -> int:
    """Half the mean reference segment length, in units (the usual Pk/WindowDiff k)."""
    n_segments = len({b for b in ref_boundaries if 0 < b < n_units}) + 1
    return max(1, round(n_units / n_segments / 2))


def pk(ref_boundaries: Sequence[int], pred_boundaries: Sequence[int], n_units: int, k: int | None = None) -> float:
    """Beeferman's Pk over unit indices (lower is better, in [0, 1])."""
    k = k or default_k(ref_boundaries, n_units)
    if n_units <= k:
        return 0.0
    r, p = _labels(ref_boundaries, n_units), _labels(pred_boundaries, n_units)
    same_r = r[:-k] == r[k:]
    same_p = p[:-k] == p[k:]
    return float(np.mean(same_r != same_p))


def windowdiff(
    ref_boundaries: Sequence[int], pred_boundaries: Sequence[int], n_units: int, k: int | None = None
) -> float:
    """Pevzner & Hearst's WindowDiff over unit indices (lower is better, in [0, 1])."""
    k = k or default_k(ref_boundaries, n_units)
    if n_units <= k:
        return 0.0
    r, p = _labels(ref_boundaries, n_units), _labels(pred_boundaries, n_units)
    return float(np.mean((r[k:] - r[:-k]) != (p[k:] - p[:-k])))
