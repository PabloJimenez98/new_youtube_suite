import numpy as np

from lib.decoding import DecodeConfig, decode, target_count


def grid(duration, step=5.0):
    return list(np.arange(0, duration, step))


def test_target_count_follows_per_hour_with_floor():
    cfg = DecodeConfig(per_hour=6, min_chapters=3)
    assert target_count(3600, cfg) == 6
    assert target_count(600, cfg) == 3
    assert target_count(7200, DecodeConfig(per_hour=6, max_chapters=8)) == 8


def test_picks_strongest_peaks_and_starts_at_zero():
    starts = grid(3600)
    scores = np.zeros(len(starts))
    for t in (600, 1200, 1800, 2400, 3000):
        scores[int(t / 5)] = 1.0
    out = decode(scores, starts, 3600, DecodeConfig(per_hour=6))
    assert out == [0.0, 600, 1200, 1800, 2400, 3000]


def test_min_length_suppresses_neighbouring_peaks():
    starts = grid(1800)
    scores = np.zeros(len(starts))
    scores[120] = 1.0  # 600 s
    scores[121] = 0.99  # 605 s, too close
    scores[240] = 0.5  # 1200 s
    out = decode(scores, starts, 1800, DecodeConfig(per_hour=6, min_len_s=60))
    assert out == [0.0, 600, 1200]


def test_no_chapter_too_close_to_the_end_or_start():
    starts = grid(600)
    scores = np.zeros(len(starts))
    scores[1] = 1.0  # 5 s: too close to 0
    scores[-1] = 1.0  # 595 s: too close to the end
    out = decode(scores, starts, 600, DecodeConfig(min_len_s=60))
    assert all(t >= 60 and 600 - t >= 60 for t in out[1:])
    assert all(b - a >= 60 for a, b in zip(out, out[1:]))


def test_threshold_stops_early_but_respects_min_chapters():
    starts = grid(3600)
    scores = np.full(len(starts), 0.01)
    scores[120] = 0.9
    out = decode(scores, starts, 3600, DecodeConfig(per_hour=6, threshold=0.5, min_chapters=3))
    assert len(out) == 3


def test_too_short_audio_is_a_single_chapter():
    assert decode([0.1, 0.9], [0.0, 10.0], 20.0) == [0.0]


def test_short_audio_relaxes_min_length_to_fit_three_chapters():
    starts = grid(90, step=1)
    scores = np.zeros(len(starts))
    scores[30] = scores[60] = 1.0
    out = decode(scores, starts, 90, DecodeConfig(min_len_s=60))
    assert out == [0.0, 30, 60]


def test_deterministic_tie_break():
    starts = grid(3600)
    scores = np.ones(len(starts))
    assert decode(scores, starts, 3600) == decode(scores, starts, 3600)
