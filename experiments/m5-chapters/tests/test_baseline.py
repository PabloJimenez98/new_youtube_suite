import numpy as np

from lib.baseline import boundary_scores
from lib.decoding import DecodeConfig, decode


def three_topics(rng, per_topic=60, dim=32, noise=0.3):
    centers = rng.normal(size=(3, dim))
    return np.vstack([c + noise * rng.normal(size=(per_topic, dim)) for c in centers])


def test_finds_the_two_true_boundaries_on_synthetic_topics():
    rng = np.random.default_rng(0)
    emb = three_topics(rng)
    scores = boundary_scores(emb, window=10)
    starts = [5.0 * i for i in range(len(emb))]  # 5 s units -> 900 s
    out = decode(scores, starts, 900, DecodeConfig(per_hour=12, min_chapters=3, min_len_s=60))
    assert out == [0.0, 300.0, 600.0]


def test_scores_are_in_unit_range_and_unit_zero_is_zero():
    rng = np.random.default_rng(1)
    s = boundary_scores(three_topics(rng), window=10)
    assert s[0] == 0 and s.min() >= 0 and s.max() == 1


def test_degenerate_inputs():
    assert boundary_scores(np.zeros((1, 4))).tolist() == [0.0]
    assert boundary_scores(np.ones((5, 4))).tolist() == [0.0] * 5
