"""Relevance curve -> clip spans (docs/decisions/0013-shorts-generator.md §1).

Pure: no video, no models. Smooth the window scores, find peaks, grow
each peak while the curve stays high, snap the edges to sentence
boundaries, enforce 20-60 s, drop overlaps, keep the best five.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

SMOOTH_WINDOWS = 3
MIN_PROMINENCE = 0.05
PEAK_DISTANCE_S = 60.0
EXPAND_RATIO = 0.7
MIN_S = 20.0
MAX_S = 60.0
MAX_OVERLAP = 0.3
MAX_CLIPS = 5

TERMINAL = (".", "?", "!", "…")


@dataclass(frozen=True)
class ScoreWindow:
    index: int
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class TimedWord:
    start: float | None
    end: float | None
    text: str
    segment_start: bool = False  # first word of a transcript segment


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    score: float  # smoothed peak height, what spans are ranked by
    peak_window_index: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def smooth(scores: list[float], k: int = SMOOTH_WINDOWS) -> np.ndarray:
    """Centred moving average; the edges average over what exists."""
    x = np.asarray(scores, dtype=float)
    if len(x) == 0 or k <= 1:
        return x
    half = k // 2
    return np.array([x[max(0, i - half) : i + half + 1].mean() for i in range(len(x))])


def find_peak_indices(curve: np.ndarray, hop_s: float) -> list[int]:
    """Peaks with enough prominence, >= PEAK_DISTANCE_S apart, highest first.

    The curve is padded with its minimum so a maximum at either end of
    the episode can count as a peak.
    """
    if len(curve) == 0:
        return []
    floor = float(curve.min())
    padded = np.concatenate([[floor], curve, [floor]])
    distance = max(1, math.ceil(PEAK_DISTANCE_S / hop_s)) if hop_s > 0 else 1
    idx, _ = find_peaks(padded, prominence=MIN_PROMINENCE, distance=distance)
    return sorted((int(i) - 1 for i in idx), key=lambda i: -curve[i])


def expand(curve: np.ndarray, windows: list[ScoreWindow], peak: int) -> tuple[float, float]:
    """Grow from the peak while the curve stays high, within MAX_S.

    "High" is EXPAND_RATIO of the way from the episode's median to the
    peak: v0 curves sit on a broad base (M7: mean 0.46), so a plain
    fraction of the peak height would let almost every clip run to MAX_S.
    """
    base = float(np.median(curve))
    floor = base + EXPAND_RATIO * (curve[peak] - base)
    lo = hi = peak
    while True:
        can_lo = lo > 0 and curve[lo - 1] >= floor and windows[hi].end - windows[lo - 1].start <= MAX_S
        can_hi = hi < len(curve) - 1 and curve[hi + 1] >= floor and windows[hi + 1].end - windows[lo].start <= MAX_S
        if not (can_lo or can_hi):
            break
        if can_lo and (not can_hi or curve[lo - 1] >= curve[hi + 1]):
            lo -= 1
        else:
            hi += 1
    return windows[lo].start, windows[hi].end


def fill_times(words: list[TimedWord]) -> list[TimedWord]:
    """Untimed words take the previous word's end (or the next word's start for a leading run)."""
    out: list[TimedWord] = []
    last = None
    for w in words:
        start = w.start if w.start is not None else last
        end = w.end if w.end is not None else start
        out.append(TimedWord(start, end, w.text, w.segment_start))
        if end is not None:
            last = end
    first = next((w.start for w in out if w.start is not None), 0.0)
    return [w if w.start is not None else TimedWord(first, first, w.text, w.segment_start) for w in out]


def sentences(words: list[TimedWord]) -> list[tuple[float, float]]:
    """(start, end) of each sentence: closed at terminal punctuation or before a segment start."""
    words = fill_times(words)
    out: list[tuple[float, float]] = []
    cur: list[TimedWord] = []
    for w in words:
        if w.segment_start and cur:
            out.append((cur[0].start, cur[-1].end))
            cur = []
        cur.append(w)
        if w.text.strip().rstrip("\"'»”)").endswith(TERMINAL):
            out.append((cur[0].start, cur[-1].end))
            cur = []
    if cur:
        out.append((cur[0].start, cur[-1].end))
    return [(s, max(s, e)) for s, e in out]


def snap(rough: tuple[float, float], sents: list[tuple[float, float]], peak_t: float,
         words: list[TimedWord]) -> tuple[float, float] | None:
    """Rough span -> whole sentences, MIN_S..MAX_S. None when MIN_S can't be reached."""
    if not sents:
        return None
    starts = [s for s, _ in sents]
    # The first sentence ending at or after the rough start, and the last one starting before the rough end.
    i = next((k for k, (_, e) in enumerate(sents) if e > rough[0]), len(sents) - 1)
    j = max((k for k, (s, _) in enumerate(sents) if s < rough[1]), default=i)
    j = max(i, j)

    # Grow to MIN_S one sentence at a time, alternating sides (the shorter neighbour first).
    while sents[j][1] - sents[i][0] < MIN_S and (i > 0 or j < len(sents) - 1):
        before = sents[i - 1][1] - sents[i - 1][0] if i > 0 else math.inf
        after = sents[j + 1][1] - sents[j + 1][0] if j < len(sents) - 1 else math.inf
        if before <= after:
            i -= 1
        else:
            j += 1
    start, end = sents[i][0], sents[j][1]
    if end - start < MIN_S:
        return None
    if end - start <= MAX_S:
        return start, end

    # Too long: the earliest sentence start (not after the peak) with a
    # sentence end past the peak within MAX_S; keep the latest such end.
    for k in range(i, j + 1):
        if starts[k] > peak_t:
            break
        ends = [e for _, e in sents[k : j + 1] if e >= peak_t and MIN_S <= e - starts[k] <= MAX_S]
        if ends:
            return starts[k], max(ends)
    # No sentence end fits: from the last sentence start before the peak,
    # cut at the last gap between words inside MAX_S.
    start = max((s for s in starts[i : j + 1] if s <= peak_t), default=starts[i])
    gaps = [w.end for w in fill_times(words) if MIN_S <= w.end - start <= MAX_S]
    return (start, max(gaps)) if gaps else (start, start + MAX_S)


def overlap_ratio(a: Span, b: Span) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, inter) / max(1e-9, min(a.duration, b.duration))


def select_spans(windows: list[ScoreWindow], words: list[TimedWord], max_clips: int = MAX_CLIPS) -> list[Span]:
    """The best clip spans, ranked by peak height."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w.start)
    curve = smooth([w.score for w in windows])
    hop = windows[1].start - windows[0].start if len(windows) > 1 else windows[0].end - windows[0].start
    sents = sentences(words)

    spans: list[Span] = []
    for p in find_peak_indices(curve, hop):
        snapped = snap(expand(curve, windows, p), sents, (windows[p].start + windows[p].end) / 2, words)
        if snapped is None:
            continue
        cand = Span(snapped[0], snapped[1], float(curve[p]), windows[p].index)
        if any(overlap_ratio(cand, s) > MAX_OVERLAP for s in spans):
            continue
        spans.append(cand)
        if len(spans) == max_clips:
            break
    return spans
