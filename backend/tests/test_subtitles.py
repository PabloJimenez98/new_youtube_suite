"""Unit tests for `.srt` / `.vtt` rendering -- no DB, no model."""

from dataclasses import dataclass

import pytest

from app.transcription.subtitles import format_timestamp, to_srt, to_vtt


@dataclass
class Seg:
    start_seconds: float
    end_seconds: float
    text: str


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
