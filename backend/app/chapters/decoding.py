# Copied from experiments/m5-chapters/lib/decoding.py -- keep in sync (the
# experiment's tests cover the same logic; docs/decisions/0009 §2-4).
"""Per-unit boundary scores -> chapter start times (docs/decisions/0009 §3).

Both the trained model and the baseline emit one score per unit ("a
chapter starts at this unit"). The owner's granularity ("few, broad")
and YouTube's chapter rules are enforced here, not left to the model:

- the first chapter starts at 0:00;
- at least `min_chapters` (YouTube: 3) chapters, each at least
  `min_len_s` long (ours: 60 s; YouTube's own floor is 10 s);
- about `per_hour` chapters per hour of audio;
- optionally stop early once scores fall below `threshold`.

Peaks are picked greedily by descending score, skipping any candidate
closer than `min_len_s` to an already-chosen start or to the end.
Deterministic: ties break toward the earlier unit.

This module is copied into the backend (`app/chapters/decoding.py`).
"""

from collections.abc import Sequence
from dataclasses import dataclass

YOUTUBE_MIN_CHAPTER_S = 10.0


@dataclass(frozen=True)
class DecodeConfig:
    per_hour: float = 6.0
    min_chapters: int = 3
    max_chapters: int | None = None
    min_len_s: float = 60.0
    threshold: float | None = None


def target_count(duration: float, cfg: DecodeConfig) -> int:
    n = max(cfg.min_chapters, round(duration / 3600 * cfg.per_hour))
    return min(n, cfg.max_chapters) if cfg.max_chapters else n


def decode(
    scores: Sequence[float],
    unit_starts: Sequence[float],
    duration: float,
    cfg: DecodeConfig = DecodeConfig(),
) -> list[float]:
    """Chapter start times (seconds, ascending, first is 0.0).

    Returns just `[0.0]` (one chapter) when the audio is too short to
    hold `min_chapters` chapters of YouTube's 10 s minimum.
    """
    if len(scores) != len(unit_starts):
        raise ValueError("scores and unit_starts differ in length")
    min_len = min(cfg.min_len_s, duration / cfg.min_chapters)
    if min_len < YOUTUBE_MIN_CHAPTER_S:
        return [0.0]

    n_target = target_count(duration, cfg)
    chosen = [0.0]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    for i in order:
        if len(chosen) >= n_target:
            break
        t = float(unit_starts[i])
        if cfg.threshold is not None and scores[i] < cfg.threshold and len(chosen) >= cfg.min_chapters:
            break
        if duration - t < min_len:
            continue
        if all(abs(t - c) >= min_len for c in chosen):
            chosen.append(t)
    return sorted(chosen)
