import pytest

from lib import titles as t
from lib.vidchapters import Video


@pytest.mark.parametrize("raw,clean", [
    ("  1. Intro  ", "Intro"),
    ("02) The   big question", "The big question"),
    ("- Wrap up", "Wrap up"),
    ('Dan: Walker\'s speech was "really bad', "Dan: Walker's speech was really bad"),
    ("Why (it matters", "Why it matters"),
    ("Q&A (part 2)", "Q&A (part 2)"),
    ("¿Qué es la IA?", "¿Qué es la IA?"),
    ("12:30", None),
    ("#", None),
])
def test_clean_title(raw, clean):
    assert t.clean_title(raw) == clean


def video():
    return Video("v", 300.0, [0.0, 100.0], ["Intro", "Main"],
                 {"text": ["hello there", "welcome", "now the topic", "more"], "start": [0.0, 50.0, 100.0, 200.0],
                  "end": [40.0, 90.0, 150.0, 250.0]})


def test_chapter_text_uses_sentences_inside_the_chapter():
    v = video()
    assert t.chapter_text(v, 0) == "hello there welcome"
    assert t.chapter_text(v, 1) == "now the topic more"


def test_make_pairs_filters_short_text_and_bad_titles():
    v = video()
    v.chapter_titles = ["Intro", "12:30"]
    pairs = t.make_pairs([v], min_text_chars=5)
    assert [(p["title"], p["i"]) for p in pairs] == [("Intro", 0)]


def test_score_perfect_and_bad():
    perfect = t.score(["the cat sat", "a dog ran"], ["the cat sat", "a dog ran"])
    bad = t.score(["the cat sat", "a dog ran"], ["zebra", "piano"])
    assert perfect["B1"] > 99 and perfect["CIDEr"] > bad["CIDEr"]
