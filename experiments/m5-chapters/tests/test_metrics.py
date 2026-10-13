import pytest

from lib import metrics as m


def test_video_pr_at_distance_drops_zero_and_matches_many_to_many():
    # refs 100, 200; preds 0 (dropped), 102, 104 (both near 100), 300 (miss)
    p, r = m.video_pr_at_distance([0, 100, 200], [0, 102, 104, 300], dist=5)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(1 / 2)


def test_distance_is_strict_less_than():
    assert m.video_pr_at_distance([100], [105], dist=5) == (0.0, 0.0)
    assert m.video_pr_at_distance([100], [104.9], dist=5) == (1.0, 1.0)


def test_video_without_reference_boundaries_is_skipped():
    assert m.video_pr_at_distance([0], [0, 50], dist=5) is None
    res = m.f1_at_distance({"a": [0], "b": [0, 100]}, {"a": [0, 50], "b": [0, 100]}, 5)
    assert res == {"precision": 100.0, "recall": 100.0, "f1": 100.0}


def test_f1_at_distance_averages_pr_over_videos_then_combines():
    refs = {"a": [0, 100], "b": [0, 100, 200]}
    preds = {"a": [0, 100], "b": [0, 300]}  # a: P1 R1 ; b: P0 R0
    res = m.f1_at_distance(refs, preds, 5)
    assert res["precision"] == pytest.approx(50)
    assert res["recall"] == pytest.approx(50)
    assert res["f1"] == pytest.approx(50)


def test_empty_prediction_scores_zero_precision():
    assert m.video_pr_at_distance([0, 100], [0], 5) == (0.0, 0.0)


def test_starts_to_segments_adds_zero_and_closes_with_duration():
    assert m.starts_to_segments([100, 50], 200) == [(0.0, 50), (50, 100), (100, 200)]


def test_iou():
    assert m.iou((0, 10), (0, 10)) == pytest.approx(1)
    assert m.iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert m.iou((0, 10), (10, 20)) == 0


def test_tiou_f1_perfect_and_shifted():
    refs, durs = {"a": [0, 100, 200]}, {"a": 300}
    assert m.tiou_f1(refs, {"a": [0, 100, 200]}, durs) == pytest.approx(100)
    # Every segment shifted by 25 s: IoUs 75/100, 75/125, 75/100 -> only low thresholds pass.
    shifted = m.tiou_f1(refs, {"a": [0, 125, 225]}, durs)
    assert 0 < shifted < 100


def test_mean_tiou_perfect():
    assert m.mean_tiou({"a": [0, 60]}, {"a": [0, 60]}, {"a": 120}) == pytest.approx(100)


def test_count_error():
    res = m.count_error({"a": [0, 10, 20], "b": [0, 10]}, {"a": [0, 10], "b": [0, 10]})
    assert res == {"mae": 0.5, "under_pct": 50.0, "equal_pct": 50.0, "over_pct": 0.0}


def test_pk_and_windowdiff():
    ref = [5, 10]
    assert m.pk(ref, ref, 15) == 0
    assert m.windowdiff(ref, ref, 15) == 0
    assert m.pk(ref, [], 15) > 0
    assert m.windowdiff(ref, [], 15) > 0
    assert m.default_k(ref, 15) == 2  # 3 segments of 5 units -> k = 2.5 -> 2
