"""Karaoke ASS captions for a clip (docs/decisions/0013 §3)."""

import pytest

from app.shorts.captions import CueWord, build_ass, clip_words, dialogues, escape, group_cues, ts
from app.shorts.selection import TimedWord


def cw(*specs: tuple[float, float, str]) -> list[CueWord]:
    return [CueWord(s, e, t) for s, e, t in specs]


def test_words_are_scoped_to_the_clip_and_made_relative() -> None:
    words = [TimedWord(9.0, 9.5, "antes"), TimedWord(10.2, 10.6, "hola"), TimedWord(None, None, "€"),
             TimedWord(19.8, 20.4, "fin"), TimedWord(20.5, 21.0, "fuera")]
    got = clip_words(words, 10.0, 20.0)
    assert [(round(w.start, 2), round(w.end, 2), w.text) for w in got] == [
        (0.2, 0.6, "hola"), (0.6, 0.6, "€"), (9.8, 10.0, "fin")
    ]


def test_cues_hold_at_most_six_words() -> None:
    words = cw(*[(i * 0.3, i * 0.3 + 0.2, "ab") for i in range(13)])
    assert [len(c) for c in group_cues(words)] == [6, 6, 1]


def test_cues_break_on_length_punctuation_and_pauses() -> None:
    long = cw((0, .2, "extraordinariamente"), (.3, .5, "complicado"), (.6, .8, "todo"))
    assert [len(c) for c in group_cues(long)] == [1, 2]  # the first two would be 30 chars > 28
    punct = cw((0, .2, "uno"), (.3, .5, "dos"), (.6, .8, "tres,"), (.9, 1, "cuatro"))
    assert [len(c) for c in group_cues(punct)] == [3, 1]
    early = cw((0, .2, "sí,"), (.3, .5, "claro"))  # punctuation before MIN_WORDS doesn't break
    assert [len(c) for c in group_cues(early)] == [2]
    pause = cw((0, .2, "uno"), (1.5, 1.7, "dos"))
    assert [len(c) for c in group_cues(pause)] == [1, 1]


def test_one_dialogue_per_word_highlights_that_word_until_the_next() -> None:
    cue = cw((0.0, 0.4, "qué"), (0.5, 0.9, "pasó"), (1.0, 1.6, "aquí"))
    lines = dialogues(cue)
    assert [(s, e) for s, e, _ in lines] == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.6)]
    assert lines[1][2] == "qué {\\c&H00FFFF&}pasó{\\c&HFFFFFF&} aquí"


def test_braces_and_backslashes_are_neutralised() -> None:
    assert escape("a{b}\\c\nd") == "a(b)/c d"


def test_timestamps() -> None:
    assert ts(0) == "0:00:00.00"
    assert ts(61.237) == "0:01:01.24"
    assert ts(3725.5) == "1:02:05.50"


def test_build_ass_has_header_style_and_events() -> None:
    words = [TimedWord(100 + i * 0.5, 100 + i * 0.5 + 0.4, w) for i, w in enumerate("¿Qué tal estás hoy, señor?".split())]
    ass = build_ass(words, 100.0, 110.0, 720, 1280)
    assert "PlayResX: 720\nPlayResY: 1280" in ass
    assert "Style: Default,DejaVu Sans,54," in ass
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == 5 and "señor?" in events[-1]
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.50,Default,")


@pytest.mark.parametrize("w,h,size", [(720, 1280, 54), (1280, 720, 54), (1080, 1080, 81)])
def test_font_scales_with_the_short_side(w: int, h: int, size: int) -> None:
    assert f"DejaVu Sans,{size}," in build_ass([], 0, 1, w, h)
