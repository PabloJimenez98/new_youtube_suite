"""Clip-span selection on synthetic curves (docs/decisions/0013 §1). No video, no models."""

import numpy as np
import pytest

from app.shorts import selection
from app.shorts.selection import (
    ScoreWindow,
    Span,
    TimedWord,
    find_peak_indices,
    select_spans,
    sentences,
    smooth,
    snap,
)

HOP = 7.5


def curve(scores: list[float]) -> list[ScoreWindow]:
    return [ScoreWindow(i, i * HOP, i * HOP + 15.0, s) for i, s in enumerate(scores)]


def bump(n: int, centre: int, height: float = 0.9, base: float = 0.3, width: int = 2) -> list[float]:
    return [height if abs(i - centre) <= width // 2 else base for i in range(n)]


def speech(duration: float, sentence_s: float = 5.0, word_s: float = 0.5) -> list[TimedWord]:
    """A word every `word_s`; a sentence ends every `sentence_s`."""
    words = []
    per = int(round(sentence_s / word_s))
    for k in range(int(duration / word_s)):
        t = k * word_s
        end = k % per == per - 1
        words.append(TimedWord(t, t + word_s * 0.8, f"w{k}." if end else f"w{k}", segment_start=k % per == 0))
    return words


def test_smoothing_averages_neighbours_and_keeps_edges() -> None:
    assert smooth([0, 3, 0, 0]).tolist() == pytest.approx([1.5, 1.0, 1.0, 0.0])
    assert smooth([]).tolist() == []


def test_peaks_are_ordered_by_height_and_at_least_60s_apart() -> None:
    scores = [0.2] * 60
    scores[10], scores[13], scores[40] = 0.8, 0.9, 0.7  # 10 and 13 are 22.5 s apart
    peaks = find_peak_indices(np.array(scores), HOP)
    assert peaks == [13, 40]


def test_low_prominence_gives_no_peaks() -> None:
    scores = [0.5] * 30
    scores[15] = 0.53
    assert find_peak_indices(np.array(scores), HOP) == []


def test_flat_or_empty_curve_gives_no_spans() -> None:
    assert select_spans(curve([0.4] * 40), speech(320)) == []
    assert select_spans([], speech(10)) == []


def test_peaks_at_the_episode_edges_count() -> None:
    scores = [0.9, 0.9] + [0.3] * 30 + [0.8, 0.8]
    assert set(find_peak_indices(smooth(scores), HOP)) >= {0, 33}


def test_sentences_break_at_punctuation_and_segment_starts() -> None:
    words = [TimedWord(0, 1, "Hola."), TimedWord(1, 2, "¿Qué"), TimedWord(2, 3, "tal?"),
             TimedWord(3, 4, "bien"), TimedWord(4, 5, "nuevo", segment_start=True), TimedWord(None, None, "€")]
    assert sentences(words) == [(0, 1), (1, 3), (3, 4), (4, 5)]


def test_snap_moves_out_to_whole_sentences() -> None:
    words = speech(120)
    s, e = snap((31.0, 57.0), sentences(words), 44.0, words)
    assert (s, e) == (30.0, pytest.approx(59.9))


def test_short_spans_grow_to_min_duration() -> None:
    words = speech(120)
    s, e = snap((50.0, 52.0), sentences(words), 51.0, words)
    assert e - s >= selection.MIN_S and s <= 50.0 and e >= 52.0


def test_too_short_episode_gives_none() -> None:
    words = speech(12)
    assert snap((0.0, 5.0), sentences(words), 2.0, words) is None


def test_long_spans_are_cut_at_a_sentence_end_and_keep_the_peak() -> None:
    words = speech(300)
    s, e = snap((10.0, 200.0), sentences(words), 100.0, words)
    assert e - s <= selection.MAX_S and s <= 100.0 <= e
    assert e in {end for _, end in sentences(words)}


def test_a_sentence_longer_than_max_is_cut_between_words() -> None:
    words = speech(200, sentence_s=150.0)  # sentences of 150 s
    s, e = snap((10.0, 40.0), sentences(words), 20.0, words)
    assert selection.MIN_S <= e - s <= selection.MAX_S
    assert any(w.end == e for w in words)  # a word end, not mid-word


def test_expansion_stops_where_the_curve_drops() -> None:
    scores = [0.3] * 40
    for i in range(18, 23):
        scores[i] = 0.9
    spans = select_spans(curve(scores), speech(320))
    assert len(spans) == 1
    sp = spans[0]
    # The high windows cover 135-180 s; smoothing and the 70 % floor keep 142.5-172.5, snapped out.
    assert 135 <= sp.start <= 142.5 and 172.5 <= sp.end <= 180
    assert selection.MIN_S <= sp.duration <= selection.MAX_S


def test_overlapping_candidates_are_deduplicated() -> None:
    a = Span(0, 40, 0.9, 1)
    assert selection.overlap_ratio(a, Span(30, 70, 0.8, 9)) == pytest.approx(0.25)
    assert selection.overlap_ratio(a, Span(10, 50, 0.8, 9)) == pytest.approx(0.75)


def test_at_most_five_spans_ranked_by_score() -> None:
    n = 200
    scores = [0.2] * n
    for k, c in enumerate(range(10, n - 5, 20)):  # 10 peaks, 150 s apart
        scores[c] = scores[c + 1] = 0.5 + k * 0.04
    spans = select_spans(curve(scores), speech(n * HOP + 15))
    assert len(spans) == 5
    assert [s.score for s in spans] == sorted((s.score for s in spans), reverse=True)
    assert all(selection.MIN_S <= s.duration <= selection.MAX_S for s in spans)
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            assert selection.overlap_ratio(a, b) <= selection.MAX_OVERLAP
