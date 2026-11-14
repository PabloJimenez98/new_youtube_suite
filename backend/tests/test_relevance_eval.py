"""Offline metrics, the gate, the blind A/B sheet and the split (0024 §4, §7). Pure."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.relevance.dataset import load_or_make_split, make_split, partition, split_hash
from app.relevance.evaluation import (
    AbEpisode,
    EpisodeEval,
    ab_sheet,
    evaluate,
    moment_overlap,
    score_ab,
    smoothness,
    spearman,
    top_fraction_precision,
    top_moments,
)
from app.relevance.retrain import RetrainError, parse_answers


def test_spearman_handles_ties_constants_and_nans() -> None:
    assert spearman(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1.0)
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1.0)
    assert spearman(np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4])) == 0.0
    assert spearman(np.array([1, 2, np.nan, 4]), np.array([1, 2, 99, 4])) == pytest.approx(1.0)
    assert spearman(np.array([1.0]), np.array([2.0])) == 0.0


def test_top_fraction_precision() -> None:
    ref = np.arange(10, dtype=float)  # top 20 % = {8, 9}
    assert top_fraction_precision(ref, ref) == 1.0
    assert top_fraction_precision(-ref, ref) == 0.0
    pred = ref.copy()
    pred[8] = -1  # top 2 now {9, 7}
    assert top_fraction_precision(pred, ref) == 0.5


def test_top_moments_are_60_s_apart_best_first() -> None:
    starts = [0.0, 7.5, 15.0, 70.0, 140.0]
    scores = [0.9, 0.95, 0.1, 0.5, 0.4]
    assert top_moments(starts, scores, k=3) == [1, 3, 4]


def test_moment_overlap_uses_a_15_s_tolerance() -> None:
    starts = [7.5 * i for i in range(100)]
    ref = np.zeros(100)
    ref[[0, 20, 40, 60, 80]] = 1.0
    shifted = np.zeros(100)
    shifted[[2, 22, 44, 60, 80]] = 1.0  # 15 s, 15 s, 30 s, exact, exact
    assert moment_overlap(starts, list(shifted), list(ref)) == pytest.approx(4 / 5)


def test_smoothness() -> None:
    assert smoothness(np.array([0.2, 0.2, 0.2])) == 0.0
    assert smoothness(np.array([0.0, 1.0, 0.0])) == 1.0


def _ep(upload_id: int, score_noise: float) -> EpisodeEval:
    rng = np.random.default_rng(upload_id)
    n = 80
    h = rng.random(n)
    l = rng.random(n)  # noqa: E741
    target = 0.4 * h + 0.6 * l
    starts = np.arange(n) * 7.5
    score = target + rng.normal(scale=score_noise, size=n)
    return EpisodeEval(upload_id, target, h, l, starts, h, l, score)


def test_gate_passes_when_the_bilstm_tracks_the_teacher_better_than_the_heuristic() -> None:
    good = evaluate([_ep(1, 0.01), _ep(2, 0.01)], constant=0.5)
    assert good["gate"]["passed"] and good["episodes"] == 2
    assert good["summary"]["bilstm"]["spearman"] > good["summary"]["heuristic"]["spearman"]
    assert good["summary"]["constant"]["spearman"] == 0.0
    assert {"spearman_heuristic_head", "spearman_llm_head"} <= set(good["per_episode"][0]["bilstm"])
    bad = evaluate([_ep(1, 5.0), _ep(2, 5.0)], constant=0.5)
    assert not bad["gate"]["passed"]
    assert not evaluate([], constant=0.5)["gate"]["passed"]


def _ab_episodes() -> list[AbEpisode]:
    starts = [7.5 * i for i in range(60)]
    texts = [f"texto {i}" for i in range(60)]
    return [AbEpisode(10 + k, f"Episodio {k}", starts, texts, list(np.linspace(0, 1, 60)),
                      list(np.linspace(1, 0, 60))) for k in range(5)]


def test_ab_sheet_hides_the_versions_and_the_key_maps_back() -> None:
    sheet, key = ab_sheet(_ab_episodes(), "bilstm-x", "teacher-y", seed=3)
    assert "bilstm-x" not in sheet and "teacher-y" not in sheet
    assert sheet.count("**List A**") == 5 and sheet.count("**List B**") == 5
    assert {v["A"] for v in key["episodes"].values()} == {"candidate", "teacher"}  # shuffled, both orders
    for v in key["episodes"].values():
        assert {v["A"], v["B"]} == {"candidate", "teacher"}
    assert ab_sheet(_ab_episodes(), "bilstm-x", "teacher-y", seed=3) == (sheet, key)


def test_ab_scoring_and_the_answer_table() -> None:
    sheet, key = ab_sheet(_ab_episodes(), "c", "t", seed=3)
    key["order"] = [10 + k for k in range(5)]
    cand_letter = {uid: ("A" if v["A"] == "candidate" else "B") for uid, v in key["episodes"].items()}
    answers = [cand_letter["10"], cand_letter["11"], "tie", "A" if cand_letter["13"] == "B" else "B", "tie"]
    filled = sheet
    for n, a in enumerate(answers, 1):
        filled = filled.replace(f"| {n}: Episodio {n - 1} | |", f"| {n}: Episodio {n - 1} | {a} |")
    parsed = parse_answers(filled, key)
    result = score_ab(key, parsed)
    assert (result["wins"], result["ties"], result["losses"]) == (2, 2, 1)
    assert result["passed"]  # 4 of 5 preferred or tied
    assert not score_ab(key, {"10": "tie", "11": "tie"} | {
        u: ("B" if cand_letter[u] == "A" else "A") for u in ("12", "13", "14")})["passed"]
    with pytest.raises(RetrainError, match="Unanswered"):
        parse_answers(sheet, key)


# ---- split -------------------------------------------------------------------------


ITEMS = [(i, "es") for i in range(1, 62)] + [(100, "en"), (101, "en")]


def test_split_sizes_stratification_and_determinism() -> None:
    s = make_split(ITEMS)
    assert (len(s["train"]), len(s["val"]), len(s["test"]), len(s["ab"])) == (48, 7, 8, 5)
    assert set(s["ab"]) <= set(s["test"])
    assert not (set(s["train"]) & set(s["val"]) | set(s["train"]) & set(s["test"]) | set(s["val"]) & set(s["test"]))
    assert sum(i in s["test"] for i in (100, 101)) == 1  # one English episode in test
    again = make_split(list(reversed(ITEMS)))
    assert {k: again[k] for k in ("train", "val", "test", "ab")} == {k: s[k] for k in ("train", "val", "test", "ab")}


def test_split_needs_enough_episodes() -> None:
    with pytest.raises(ValueError, match="at least"):
        make_split(ITEMS[:10])


def test_a_saved_split_stays_frozen_and_new_episodes_join_train(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    first, created = load_or_make_split(path, ITEMS)
    assert created and json.loads(path.read_text())["test"] == first["test"]
    second, created = load_or_make_split(path, ITEMS + [(500, "es")])
    assert not created
    assert {k: second[k] for k in ("val", "test", "ab")} == {k: first[k] for k in ("val", "test", "ab")}
    assert 500 in second["train"] and split_hash(second) == split_hash(first)


def test_partition_skips_unknown_ids() -> None:
    class E:
        def __init__(self, upload_id):
            self.upload_id = upload_id

    parts = partition([E(1), E(2)], {"train": [1, 3], "val": [2], "test": [], "ab": []})
    assert [e.upload_id for e in parts["train"]] == [1] and [e.upload_id for e in parts["val"]] == [2]
