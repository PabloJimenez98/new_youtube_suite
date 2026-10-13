from lib.evaluate import Prediction, evaluate


def test_perfect_prediction_scores_top_marks():
    p = Prediction("v", 300.0, [0.0, 100.0, 200.0], [0.0, 100.0, 200.0], [0.0, 100.0, 200.0])
    r = evaluate([p])
    assert r["F1@5s"] == 100 and r["tIoU-F1"] == 100
    assert r["Pk"] == 0 and r["WindowDiff"] == 0 and r["count_MAE"] == 0


def test_missing_boundaries_hurt_recall_not_precision():
    p = Prediction("v", 300.0, [0.0, 100.0, 200.0], [0.0, 100.0, 200.0], [0.0, 100.0])
    r = evaluate([p])
    assert r["P@5s"] == 100 and r["R@5s"] == 50
