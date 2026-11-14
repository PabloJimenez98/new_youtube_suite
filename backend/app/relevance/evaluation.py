"""Offline evaluation, the promotion gate and the blind A/B sheet (docs/decisions/0024 §7).

Pure functions over numpy arrays. Every metric is per episode, then averaged.
A distilled model can at best match its teacher on the teacher's labels, so the
gate asks something narrower: does the BiLSTM agree with the teacher more than
the cheap heuristic alone does? Whether it is *better* is the owner's blind A/B.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

TOP_FRACTION = 0.2
TOP_MOMENTS = 5
MIN_GAP_SECONDS = 60.0  # the 0012 §6 rule
MOMENT_TOLERANCE_SECONDS = 15.0  # two windows' hop either way
AB_MIN_WINS = 3  # preferred or tied on ≥ 3 of the 5 A/B episodes


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho, 0.0 when either side is constant or too short (no ranking information)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    rho = spearmanr(a, b).statistic
    return 0.0 if math.isnan(rho) else float(rho)


def top_indices(scores: np.ndarray, fraction: float = TOP_FRACTION) -> set[int]:
    k = max(1, round(len(scores) * fraction))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return set(order[:k])


def top_fraction_precision(pred: np.ndarray, ref: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    if len(pred) == 0:
        return 0.0
    p, r = top_indices(pred, fraction), top_indices(ref, fraction)
    return len(p & r) / len(p)


def top_moments(starts: Sequence[float], scores: Sequence[float], k: int = TOP_MOMENTS,
                gap: float = MIN_GAP_SECONDS) -> list[int]:
    """Greedy best-first windows at least `gap` s apart (as `GET /relevance`'s top moments)."""
    picked: list[int] = []
    for i in sorted(range(len(scores)), key=lambda i: (-scores[i], starts[i])):
        if all(abs(starts[i] - starts[j]) >= gap for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    return picked


def moment_overlap(starts: Sequence[float], pred: Sequence[float], ref: Sequence[float], k: int = TOP_MOMENTS,
                   tol: float = MOMENT_TOLERANCE_SECONDS) -> float:
    """Share of the predicted top-k moments within `tol` s of a reference top-k moment."""
    p, r = top_moments(starts, pred, k), top_moments(starts, ref, k)
    if not p:
        return 0.0
    return sum(any(abs(starts[i] - starts[j]) <= tol for j in r) for i in p) / len(p)


def smoothness(scores: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(scores)))) if len(scores) > 1 else 0.0


def curve_metrics(pred: np.ndarray, target: np.ndarray, starts: np.ndarray) -> dict:
    return {
        "spearman": spearman(pred, target),
        "top20_precision": top_fraction_precision(pred, target),
        "top5_overlap": moment_overlap(list(starts), list(pred), list(target)),
        "smoothness": smoothness(pred),
        "std": float(np.std(pred)) if len(pred) else 0.0,
    }


@dataclass(frozen=True)
class EpisodeEval:
    upload_id: int
    target: np.ndarray  # the teacher's combined score
    heuristic: np.ndarray  # the teacher's heuristic component (the cheap baseline)
    llm: np.ndarray  # the teacher's LLM component (NaN where missing)
    starts: np.ndarray
    h_hat: np.ndarray
    l_hat: np.ndarray
    score: np.ndarray  # the BiLSTM's combined score


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def evaluate(episodes: list[EpisodeEval], constant: float) -> dict:
    """Per-episode metrics for the BiLSTM and the baselines, their means, and the offline gate."""
    per_episode = []
    for e in episodes:
        per_episode.append({
            "upload_id": e.upload_id,
            "bilstm": {**curve_metrics(e.score, e.target, e.starts),
                       "spearman_heuristic_head": spearman(e.h_hat, e.heuristic),
                       "spearman_llm_head": spearman(e.l_hat, e.llm)},
            "heuristic": curve_metrics(e.heuristic, e.target, e.starts),
            "constant": curve_metrics(np.full_like(e.target, constant), e.target, e.starts),
        })
    summary = {
        method: {key: _mean([p[method] for p in per_episode], key) for key in per_episode[0][method]}
        for method in ("bilstm", "heuristic", "constant")
    } if per_episode else {}
    b = summary.get("bilstm", {}).get("spearman", 0.0)
    h = summary.get("heuristic", {}).get("spearman", 0.0)
    return {
        "episodes": len(per_episode),
        "summary": summary,
        "gate": {"passed": bool(per_episode) and b > h, "bilstm_spearman": b, "heuristic_spearman": h,
                 "rule": "BiLSTM mean Spearman vs the teacher > the heuristic alone's (0024 §7)"},
        "per_episode": per_episode,
    }


# ---- blind A/B -----------------------------------------------------------------


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


@dataclass(frozen=True)
class AbEpisode:
    upload_id: int
    name: str
    starts: Sequence[float]
    texts: Sequence[str]
    teacher: Sequence[float]
    candidate: Sequence[float]


def ab_sheet(episodes: list[AbEpisode], candidate_label: str, teacher_label: str, seed: int,
             max_chars: int = 220) -> tuple[str, dict]:
    """(markdown sheet, answer key). A and B are shuffled per episode; the sheet never names a version."""
    rng = random.Random(seed)
    lines = [
        "# Relevance blind A/B (M15)",
        "",
        "For each episode, two lists of the 5 best moments (at least 60 s apart). Which list would",
        "give you better shorts? Write **A**, **B** or **tie** in the table at the end. Don't open the key",
        "until you're done.",
        "",
    ]
    key: dict = {"candidate": candidate_label, "teacher": teacher_label, "seed": seed, "episodes": {}}
    for n, e in enumerate(episodes, 1):
        sides = ["candidate", "teacher"]
        rng.shuffle(sides)
        key["episodes"][str(e.upload_id)] = {"A": sides[0], "B": sides[1], "name": e.name}
        lines += [f"## Episode {n}: {e.name}", ""]
        for letter, side in zip("AB", sides, strict=True):
            scores = e.candidate if side == "candidate" else e.teacher
            lines.append(f"**List {letter}**")
            lines.append("")
            for rank, i in enumerate(sorted(top_moments(e.starts, scores), key=lambda i: e.starts[i]), 1):
                text = e.texts[i] if len(e.texts[i]) <= max_chars else e.texts[i][: max_chars - 1] + "…"
                lines.append(f"{rank}. `{_clock(e.starts[i])}` {text}")
            lines.append("")
    lines += ["## Answers", "", "| Episode | Better list (A / B / tie) |", "|---|---|"]
    lines += [f"| {n}: {e.name} | |" for n, e in enumerate(episodes, 1)]
    return "\n".join(lines) + "\n", key


def score_ab(key: dict, answers: dict[str, str]) -> dict:
    """answers: {upload_id: 'A' | 'B' | 'tie'} -> wins/ties/losses for the candidate and the A/B verdict."""
    wins = ties = losses = 0
    for upload_id, answer in answers.items():
        answer = answer.strip().lower()
        if answer == "tie":
            ties += 1
            continue
        side = key["episodes"][str(upload_id)][answer.upper()]
        wins, losses = (wins + 1, losses) if side == "candidate" else (wins, losses + 1)
    return {"wins": wins, "ties": ties, "losses": losses, "passed": wins + ties >= AB_MIN_WINS}
