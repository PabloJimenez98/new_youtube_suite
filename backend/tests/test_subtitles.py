"""Unit tests for `.srt` / `.vtt` rendering -- no DB, no model."""

from dataclasses import dataclass, field

import pytest

from app.transcription.subtitles import format_timestamp, to_srt, to_vtt


@dataclass
class W:
    word: str
    start_seconds: float | None
    end_seconds: float | None
    speaker_label: str | None


@dataclass
class Seg:
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str | None = None
    words: list[W] = field(default_factory=list)


@pytest.mark.parametrize(
    ("seconds", "sep", "expected"),
    [
        (0, ",", "00:00:00,000"),
        (1.5, ",", "00:00:01,500"),
        (61.25, ".", "00:01:01.250"),
        # Rounds before splitting: must carry into seconds, never ",1000".
        (59.9996, ",", "00:01:00,000"),
        (3599.9999, ".", "01:00:00.000"),
        (3 * 3600 + 25 * 60 + 7.089, ",", "03:25:07,089"),
        (-0.4, ",", "00:00:00,000"),
    ],
)
def test_format_timestamp(seconds: float, sep: str, expected: str) -> None:
    assert format_timestamp(seconds, sep) == expected


SEGMENTS = [
    Seg(0.0, 2.5, "Hello and welcome."),
    Seg(2.5, 5.123, "Today we talk about podcasts."),
]


def test_to_srt_basic() -> None:
    assert to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello and welcome.\n"
        "\n"
        "2\n00:00:02,500 --> 00:00:05,123\nToday we talk about podcasts.\n"
    )


def test_to_vtt_basic() -> None:
    assert to_vtt(SEGMENTS) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\nHello and welcome.\n"
        "\n"
        "00:00:02.500 --> 00:00:05.123\nToday we talk about podcasts.\n"
    )


def test_empty_input() -> None:
    assert to_srt([]) == ""
    assert to_vtt([]) == "WEBVTT\n\n"


def test_empty_text_segments_are_skipped_and_srt_indices_stay_contiguous() -> None:
    segments = [Seg(0, 1, "  "), Seg(1, 2, "One."), Seg(2, 3, ""), Seg(3, 4, "Two.")]
    srt = to_srt(segments)
    assert srt.startswith("1\n00:00:01,000 --> 00:00:02,000\nOne.\n")
    assert "\n2\n00:00:03,000 --> 00:00:04,000\nTwo.\n" in srt
    assert "\n3\n" not in srt
    assert to_vtt(segments).count("-->") == 2


def test_blank_lines_inside_text_are_collapsed() -> None:
    # A blank line would terminate the cue early in both formats.
    srt = to_srt([Seg(0, 1, "line one\n\n  line two  ")])
    assert srt == "1\n00:00:00,000 --> 00:00:01,000\nline one\nline two\n"


def test_vtt_escapes_markup_and_arrow() -> None:
    vtt = to_vtt([Seg(0, 1, "a < b & c --> d")])
    assert vtt.endswith("a &lt; b &amp; c --&gt; d\n")
    assert vtt.count("-->") == 1  # only the timing line's arrow


def test_srt_leaves_text_unescaped() -> None:
    assert to_srt([Seg(0, 1, "Q&A <3")]).endswith("Q&A <3\n")


def test_end_before_start_is_clamped() -> None:
    assert "00:00:02,000 --> 00:00:02,000" in to_srt([Seg(2.0, 1.5, "x")])


# ---- Speaker labels (M4) ----

LABELLED = [
    Seg(0.0, 2.5, "Hello and welcome.", "SPEAKER_00"),
    Seg(2.5, 4.0, "Thanks for having me.", "SPEAKER_01"),
    Seg(4.0, 5.0, "Unlabelled bit."),
]


def test_srt_prefixes_speaker_labels() -> None:
    assert to_srt(LABELLED) == (
        "1\n00:00:00,000 --> 00:00:02,500\n[SPEAKER_00] Hello and welcome.\n"
        "\n"
        "2\n00:00:02,500 --> 00:00:04,000\n[SPEAKER_01] Thanks for having me.\n"
        "\n"
        "3\n00:00:04,000 --> 00:00:05,000\nUnlabelled bit.\n"
    )


def test_vtt_uses_voice_spans_for_speaker_labels() -> None:
    assert to_vtt(LABELLED) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\n<v SPEAKER_00>Hello and welcome.\n"
        "\n"
        "00:00:02.500 --> 00:00:04.000\n<v SPEAKER_01>Thanks for having me.\n"
        "\n"
        "00:00:04.000 --> 00:00:05.000\nUnlabelled bit.\n"
    )


def test_vtt_voice_span_text_is_still_escaped() -> None:
    assert to_vtt([Seg(0, 1, "a < b", "SPEAKER_00")]).endswith("<v SPEAKER_00>a &lt; b\n")


def test_srt_speaker_prefix_only_on_first_line() -> None:
    srt = to_srt([Seg(0, 1, "line one\nline two", "SPEAKER_01")])
    assert srt.endswith("[SPEAKER_01] line one\nline two\n")


# ---- Per-speaker cue splitting (M4 follow-up, 0008 §6) ----


def _seg(*words: tuple[str, float | None, str | None], start: float = 0.0, end: float = 5.0) -> Seg:
    ws = [W(text, t, None if t is None else t + 0.3, label) for text, t, label in words]
    majority = max(
        {w.speaker_label for w in ws if w.speaker_label},
        key=lambda s: sum(w.speaker_label == s for w in ws),
        default=None,
    )
    return Seg(start, end, " ".join(w.word for w in ws), majority, ws)


def test_mixed_segment_splits_into_one_cue_per_speaker() -> None:
    seg = _seg(
        ("So", 0.0, "SPEAKER_00"), ("what", 0.4, "SPEAKER_00"), ("happened?", 0.8, "SPEAKER_00"),
        ("Nothing", 2.0, "SPEAKER_01"), ("really.", 2.5, "SPEAKER_01"),
    )
    assert to_srt([seg]) == (
        "1\n00:00:00,000 --> 00:00:02,000\n[SPEAKER_00] So what happened?\n"
        "\n"
        "2\n00:00:02,000 --> 00:00:05,000\n[SPEAKER_01] Nothing really.\n"
    )
    assert to_vtt([seg]) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n<v SPEAKER_00>So what happened?\n"
        "\n"
        "00:00:02.000 --> 00:00:05.000\n<v SPEAKER_01>Nothing really.\n"
    )


def test_srt_numbering_continues_across_split_cues() -> None:
    first = _seg(("a", 0.0, "SPEAKER_00"), ("b", 0.5, "SPEAKER_00"),
                 ("c", 1.0, "SPEAKER_01"), ("d", 1.5, "SPEAKER_01"), end=2.0)
    second = Seg(2.0, 3.0, "next", "SPEAKER_00")
    srt = to_srt([first, second])
    assert "\n3\n00:00:02,000 --> 00:00:03,000\n[SPEAKER_00] next\n" in srt
    assert srt.count(" --> ") == 3


def test_single_word_interjection_is_folded_not_split() -> None:
    seg = _seg(
        ("I", 0.0, "SPEAKER_00"), ("think", 0.4, "SPEAKER_00"),
        ("yeah", 0.8, "SPEAKER_01"),
        ("it", 1.2, "SPEAKER_00"), ("works", 1.6, "SPEAKER_00"),
    )
    # One-word runs may be diarization noise: behave exactly like an
    # unsplit segment (segment text, majority label).
    assert to_srt([seg]) == "1\n00:00:00,000 --> 00:00:05,000\n[SPEAKER_00] I think yeah it works\n"


def test_unlabelled_words_stay_with_their_run() -> None:
    # Regression: an unlabelled word must not hand the *next* speaker's
    # words to the previous run.
    seg = _seg(
        ("Right", 0.0, "SPEAKER_00"), ("€5", None, None),
        ("No", 1.0, "SPEAKER_01"), ("way", 1.4, "SPEAKER_01"),
    )
    assert to_vtt([seg]) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n<v SPEAKER_00>Right €5\n"
        "\n"
        "00:00:01.000 --> 00:00:05.000\n<v SPEAKER_01>No way\n"
    )


def test_leading_unlabelled_words_join_the_first_speaker() -> None:
    seg = _seg(
        ("Um", None, None), ("well", 0.3, "SPEAKER_01"), ("yes", 0.6, "SPEAKER_01"),
        ("Okay", 2.0, "SPEAKER_00"), ("then", 2.4, "SPEAKER_00"),
    )
    cues = to_srt([seg])
    assert "[SPEAKER_01] Um well yes\n" in cues
    assert "[SPEAKER_00] Okay then\n" in cues


def test_uniformly_labelled_segment_keeps_the_segment_text() -> None:
    # No split → identical to the one-cue-per-segment output, using the
    # segment's own text (not re-joined words).
    seg = _seg(("Hello,", 0.0, "SPEAKER_00"), ("world.", 0.5, "SPEAKER_00"))
    seg.text = "Hello, world."
    assert to_srt([seg]) == "1\n00:00:00,000 --> 00:00:05,000\n[SPEAKER_00] Hello, world.\n"


def test_words_without_any_labels_behave_like_m3() -> None:
    seg = _seg(("plain", 0.0, None), ("text", 0.5, None))
    assert to_srt([seg]) == "1\n00:00:00,000 --> 00:00:05,000\nplain text\n"



# --- long cues (M9, docs/decisions/0014 §1): batched Whisper gives ~30 s segments ---


def _long_segment(n: int = 60, step: float = 0.5, speaker: str | None = None) -> Seg:
    words = [W(f"pal{i}" + ("." if i % 12 == 11 else ""), i * step, i * step + 0.4, speaker) for i in range(n)]
    return Seg(0.0, n * step, " ".join(w.word for w in words), speaker, words)


def test_a_long_segment_is_split_into_short_cues() -> None:
    srt = to_srt([_long_segment()])
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert len(blocks) > 4
    for b in blocks:
        _, times, text = b.split("\n", 2)
        start, end = (sum(float(x) * f for x, f in zip(t.replace(",", ".").split(":"), (3600, 60, 1))) for t in times.split(" --> "))
        assert len(text.strip()) <= 84 and end - start <= 7.0 + 1e-6
    # Every word survives, in order.
    assert " ".join(b.split("\n", 2)[2].strip() for b in blocks) == _long_segment().text


def test_long_cues_prefer_to_break_after_punctuation() -> None:
    texts = [b.split("\n", 2)[2].strip() for b in to_srt([_long_segment()]).split("\n\n") if b.strip()]
    assert sum(t.endswith(".") for t in texts) >= len(texts) // 2


def test_long_cues_keep_the_speaker_label() -> None:
    vtt = to_vtt([_long_segment(speaker="SPEAKER_01")])
    cues = [c for c in vtt.split("\n\n")[1:] if c.strip()]
    assert len(cues) > 4 and all("<v SPEAKER_01>" in c for c in cues)


def test_a_short_segment_is_untouched() -> None:
    seg = Seg(1.0, 3.0, "Hola, ¿qué tal?", None, [W("Hola,", 1.0, 1.4, None), W("¿qué", 1.5, 1.8, None), W("tal?", 1.9, 2.5, None)])
    assert to_srt([seg]) == "1\n00:00:01,000 --> 00:00:03,000\nHola, ¿qué tal?\n"


def test_a_long_segment_without_words_stays_one_cue() -> None:
    assert to_srt([Seg(0.0, 30.0, "x " * 100)]).count("-->") == 1
