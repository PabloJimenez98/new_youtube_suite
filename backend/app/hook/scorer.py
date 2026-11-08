"""Weighted heuristic hook scorer, `heuristic-v0` (docs/decisions/0020 §3). Pure.

Four scored dimensions, each a sub-score in [0, 1]; the total is the
weighted sum x 100.

- The shapes are **priors**: no measured retention evidence exists for
  any of them (docs/research/ctr-hook-prediction-prior-art.md §2).
- The score is relative among one upload's clips, not a retention %.
- Every other feature is stored with weight 0, for M15's LightGBM.
"""

from dataclasses import dataclass

from app.ctr.scorer import piecewise
from app.hook.features import HookFeatures

VERSION_LABEL = "heuristic-v0"

WEIGHTS = {"start": 0.30, "pace": 0.25, "framing": 0.25, "address": 0.20}

# (x, y) breakpoints; flat outside the first/last point.
SHAPES: dict[str, list[tuple[float, float]]] = {
    "start": [(0.3, 1.0), (2.0, 0.0)],  # lead silence, s
    "pace": [(1.0, 0.0), (2.3, 1.0), (3.8, 1.0), (5.0, 0.6)],  # words/s
}
FRAMING = {"question": 1.0, "curiosity": 0.7, "none": 0.3}
ADDRESS_SCORES = {0: 0.3, 1: 0.8}  # 2+ -> ADDRESS_MANY
ADDRESS_MANY = 1.0

ZERO_WEIGHT_FEATURES = (
    "words",
    "sentences",
    "avg_sentence_words",
    "readability",
    "questions",
    "exclamations",
    "curiosity_connectives",
    "superlatives",
    "numerics",
    "entity_density",
    "emotion",
    "emotion_max",
    "title_overlap",
)


def framing_score(f: HookFeatures) -> float:
    if f.first_is_question:
        return FRAMING["question"]
    if f.first_curiosity or f.first_unresolved_demonstratives:
        return FRAMING["curiosity"]
    return FRAMING["none"]


def address_score(f: HookFeatures) -> float:
    return ADDRESS_SCORES.get(f.second_person + f.imperatives, ADDRESS_MANY)


@dataclass(frozen=True)
class Score:
    score: float  # 0-100
    sub_scores: dict[str, float]


def score(f: HookFeatures) -> Score:
    sub = {
        "start": round(piecewise(f.lead_silence_s, SHAPES["start"]), 4),
        "pace": round(piecewise(f.words_per_second, SHAPES["pace"]), 4),
        "framing": framing_score(f),
        "address": address_score(f),
    }
    total = sum(WEIGHTS[k] * v for k, v in sub.items())
    return Score(score=round(total * 100, 1), sub_scores=sub)


def architecture_notes() -> str:
    shapes = "; ".join(f"{k}: {v}" for k, v in SHAPES.items())
    return (
        f"Heuristic v0 (docs/decisions/0020 §3): weighted sum x 100 of sub-scores over the first 15 s. "
        f"Weights {WEIGHTS}. Shapes {shapes}; framing {FRAMING}; address {ADDRESS_SCORES}, 2+ -> {ADDRESS_MANY}. "
        f"Zero-weight (logged): {', '.join(ZERO_WEIGHT_FEATURES)}. "
        "spaCy es/en_core_*_sm 3.8.0; emotion MilaNLProc/xlm-emo-t."
    )
