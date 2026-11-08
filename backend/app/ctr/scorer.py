"""Weighted heuristic scorer, `heuristic-v0` (docs/decisions/0019 §2). Pure.

Each scored feature maps to a sub-score in [0, 1] through a
piecewise-linear shape; the total is the weighted sum x 100.

- The shapes are **priors**: no measured CTR evidence exists for any of
  them (docs/research/ctr-heuristic-evidence.md).
- The score is relative within one upload, not a CTR %.
- A candidate without a thumbnail is `partial`: it is scored from the
  title alone, rescaled to 0-100, and ranked in its own group.
"""

from dataclasses import dataclass

from app.ctr.features import ThumbnailFeatures, TitleFeatures

VERSION_LABEL = "heuristic-v0"

WEIGHTS = {"title_length": 0.35, "faces": 0.30, "face_size": 0.20, "text_coverage": 0.15}

# (x, y) breakpoints; flat outside the first/last point.
SHAPES: dict[str, list[tuple[float, float]]] = {
    "title_length": [(5, 0.0), (40, 1.0), (60, 1.0), (100, 0.4)],
    "face_size": [(0.0, 0.0), (0.10, 1.0), (0.35, 1.0), (0.60, 0.6)],
    "text_coverage": [(0.0, 0.7), (0.02, 1.0), (0.20, 1.0), (0.50, 0.3)],
}
FACE_COUNT_SCORES = {0: 0.2, 1: 0.9, 2: 1.0}  # 3+ -> FACE_COUNT_MANY
FACE_COUNT_MANY = 0.9

ZERO_WEIGHT_FEATURES = (
    "title_words",
    "has_digit",
    "has_question",
    "has_exclamation",
    "has_brackets",
    "caps_words",
    "second_person",
    "text_boxes",
    "brightness",
    "contrast",
    "saturation",
    "aspect_ratio",
)


def piecewise(x: float, points: list[tuple[float, float]]) -> float:
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def face_count_score(count: int) -> float:
    return FACE_COUNT_SCORES.get(count, FACE_COUNT_MANY)


@dataclass(frozen=True)
class Score:
    score: float  # 0-100
    partial: bool
    sub_scores: dict[str, float]

    def features_json(self, title: TitleFeatures, thumbnail: ThumbnailFeatures | None) -> dict:
        return {
            "version": VERSION_LABEL,
            "title": title.as_dict(),
            "thumbnail": thumbnail.as_dict() if thumbnail else None,
            "sub_scores": self.sub_scores,
            "weights": WEIGHTS,
        }


def score(title: TitleFeatures, thumbnail: ThumbnailFeatures | None) -> Score:
    sub = {"title_length": round(piecewise(title.title_chars, SHAPES["title_length"]), 4)}
    if thumbnail is not None:
        sub["faces"] = face_count_score(thumbnail.face_count)
        sub["face_size"] = round(piecewise(thumbnail.largest_face_area, SHAPES["face_size"]), 4)
        sub["text_coverage"] = round(piecewise(thumbnail.text_area, SHAPES["text_coverage"]), 4)
    used = sum(WEIGHTS[k] for k in sub)
    total = sum(WEIGHTS[k] * v for k, v in sub.items()) / used
    return Score(score=round(total * 100, 1), partial=thumbnail is None, sub_scores=sub)


def architecture_notes() -> str:
    shapes = "; ".join(f"{k}: {v}" for k, v in SHAPES.items())
    return (
        f"Heuristic v0 (docs/decisions/0019 §2): weighted sum x 100 of piecewise-linear sub-scores. "
        f"Weights {WEIGHTS}. Shapes {shapes}; faces {FACE_COUNT_SCORES}, 3+ -> {FACE_COUNT_MANY}. "
        f"Zero-weight (logged): {', '.join(ZERO_WEIGHT_FEATURES)}. Title-only candidates are partial."
    )
