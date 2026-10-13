"""VidChapters records -> units, boundary labels and references.

A record here is either a line of our `interim/<split>.jsonl` (see
`01_data`) or a video from the Chapter-Llama subset files; both carry
`duration`, chapters and an ASR dict {text, start, end}.
"""

import bisect
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.units import Atom, Unit, UnitConfig, build_units


@dataclass
class Video:
    vid: str
    duration: float
    chapter_starts: list[float]
    chapter_titles: list[str]
    asr: dict
    lang: str = "en"
    split: str = ""


def load_jsonl(path: Path) -> Iterator[Video]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            yield Video(
                r["vid"], r["duration"], [s for s, _ in r["chapters"]], [t for _, t in r["chapters"]],
                r["asr"], r.get("lang", "en"), r.get("split", ""),
            )


def load_subset(chapters_path: Path, asrs_path: Path) -> list[Video]:
    """Chapter-Llama subset files (`chapters_<s>.json` + `asrs_<s>.json`)."""
    chapters = json.loads(Path(chapters_path).read_text())
    asrs = json.loads(Path(asrs_path).read_text())
    out = []
    for vid, c in chapters.items():
        if vid not in asrs:
            continue
        pairs = sorted((float(k), v) for k, v in c["chapters"].items())
        out.append(Video(vid, c["duration"], [s for s, _ in pairs], [t for _, t in pairs], asrs[vid]))
    return out


def atoms(video: Video) -> list[Atom]:
    a = video.asr
    return [Atom(float(s), float(e), t) for t, s, e in zip(a["text"], a["start"], a["end"], strict=True)]


def units_for(video: Video, cfg: UnitConfig) -> list[Unit]:
    return build_units(atoms(video), cfg)


def boundary_units(unit_starts: list[float], chapter_starts: list[float]) -> list[int]:
    """Index of the unit nearest to each chapter start (0 s excluded, deduplicated).

    These are the positive labels for training. Chapters starting before
    the first unit, or mapping onto unit 0, are dropped: unit 0 is always
    the implicit first chapter.
    """
    idx = set()
    for t in chapter_starts:
        if t <= 0:
            continue
        j = bisect.bisect_left(unit_starts, t)
        cands = [k for k in (j - 1, j) if 0 <= k < len(unit_starts)]
        best = min(cands, key=lambda k: (abs(unit_starts[k] - t), k))
        if best > 0:
            idx.add(best)
    return sorted(idx)


def labels(n_units: int, boundaries: list[int]) -> np.ndarray:
    y = np.zeros(n_units, dtype=np.float32)
    y[boundaries] = 1.0
    return y
