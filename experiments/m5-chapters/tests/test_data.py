import json

from lib import data as d


def rec(duration=1800.0, chapters=None, channel="c1", title="t"):
    return {
        "duration": duration,
        "title": title,
        "channel_id": channel,
        "chapters": chapters if chapters is not None else {"0": "Intro", "600": "A", "1200": "B"},
    }


def test_parse_chapters_sorts_and_validates():
    assert d.parse_chapters(rec(chapters={"600": "A", "0": "Intro"})) == [(0.0, "Intro"), (600.0, "A")]
    assert d.parse_chapters(rec(chapters={})) is None
    assert d.parse_chapters(rec(chapters={"x": "bad"})) is None
    assert d.parse_chapters(rec(chapters={"0": "a", "1800": "past the end"})) is None
    assert d.parse_chapters(rec(chapters={"0": "a", "0.0": "dup"})) is None


def test_is_candidate_duration_count_and_min_gap():
    c = d.Criteria()
    r = rec()
    assert d.is_candidate(r, d.parse_chapters(r), c)
    short = rec(duration=600)
    assert not d.is_candidate(short, d.parse_chapters(short), c)
    few = rec(chapters={"0": "a", "900": "b"})
    assert not d.is_candidate(few, d.parse_chapters(few), c)
    moments = rec(chapters={"0": "a", "5": "b", "900": "c"})  # 5 s chapter
    assert not d.is_candidate(moments, d.parse_chapters(moments), c)
    assert not d.is_candidate(r, None, c)


def test_duration_bucket():
    assert [d.duration_bucket(m * 60) for m in (15, 45, 90)] == ["15-30", "30-60", "60-120"]


def test_assign_split_english_official_spanish_by_channel():
    official = {"v1": "test"}
    assert d.assign_split("v1", "en", "c", official) == "test"
    assert d.assign_split("unknown", "en", "c", official) is None
    # Same channel -> same Spanish split, whatever the video.
    s1 = d.assign_split("a", "es", "chan-x", official)
    s2 = d.assign_split("b", "es", "chan-x", official)
    assert s1 == s2 and s1 in {"train", "es_val", "es_test"}


def test_spanish_split_proportions_roughly_hold():
    splits = [d.assign_split(f"v{i}", "es", f"ch{i}", {}) for i in range(4000)]
    assert 0.35 < splits.count("es_test") / 4000 < 0.45
    assert 0.15 < splits.count("es_val") / 4000 < 0.25


def test_sample_english_is_stratified_and_deterministic():
    rows = [
        {"vid": f"v{i}", "split": "train", "bucket": b}
        for i in range(30)
        for b in ("15-30", "30-60")
    ]
    a = d.sample_english(rows, {"train": 5})
    assert len(a) == 10
    assert sum(r["bucket"] == "15-30" for r in a) == 5
    assert a == d.sample_english(list(reversed(rows)), {"train": 5})
    assert d.sample_english(rows, {"val": 5}) == []


def test_iter_asr_streams_only_wanted(tmp_path):
    p = tmp_path / "asrs.json"
    p.write_text(json.dumps({v: {"text": [v], "start": [0.0], "end": [1.0]} for v in "abc"}))
    assert [v for v, _ in d.iter_asr(p, {"a", "c"})] == ["a", "c"]


def test_iter_chapters_streams_records(tmp_path):
    p = tmp_path / "chapters.json"
    p.write_text(json.dumps({"x": rec(), "y": rec(duration=100)}))
    assert [(v, r["duration"]) for v, r in d.iter_chapters(p)] == [("x", 1800.0), ("y", 100.0)]


def test_asr_is_usable():
    ok = {"text": ["a", "b"], "start": [0.0, 900.0], "end": [5.0, 1000.0]}
    assert d.asr_is_usable(ok, 1800)
    assert not d.asr_is_usable({"text": ["a"], "start": [0.0], "end": [60.0]}, 1800)  # 3% coverage
    assert not d.asr_is_usable({"text": ["a", "b"], "start": [0.0], "end": [1.0]}, 10)
    assert not d.asr_is_usable({}, 10)


def test_asr_text_samples_the_middle():
    asr = {"text": [str(i) for i in range(100)]}
    assert d.asr_text(asr, max_sentences=4).split() == ["48", "49", "50", "51"]
