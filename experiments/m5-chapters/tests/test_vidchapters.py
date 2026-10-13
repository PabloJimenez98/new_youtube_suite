import json

from lib import vidchapters as vc
from lib.units import UnitConfig


def video():
    return vc.Video(
        "v", 100.0, [0.0, 42.0, 70.0], ["a", "b", "c"],
        {"text": ["One.", "Two.", "Three.", "Four."], "start": [0.0, 20.0, 40.0, 60.0], "end": [19.0, 39.0, 59.0, 79.0]},
    )


def test_units_for_uses_asr_sentences_as_atoms():
    units = vc.units_for(video(), UnitConfig(mode="sentence", min_s=1))
    assert [u.start for u in units] == [0.0, 20.0, 40.0, 60.0]


def test_boundary_units_nearest_and_excludes_zero():
    starts = [0.0, 20.0, 40.0, 60.0]
    assert vc.boundary_units(starts, [0.0, 42.0, 70.0]) == [2, 3]  # 70 is nearer 60 than nothing after
    assert vc.boundary_units(starts, [5.0]) == []  # maps onto unit 0
    assert vc.boundary_units(starts, [30.0]) == [1]  # tie -> earlier unit
    assert vc.boundary_units(starts, [41.0, 43.0]) == [2]  # dedup


def test_labels():
    assert vc.labels(4, [1, 3]).tolist() == [0, 1, 0, 1]


def test_load_subset_and_jsonl(tmp_path):
    ch = tmp_path / "c.json"
    asr = tmp_path / "a.json"
    ch.write_text(json.dumps({"v": {"duration": 100.0, "chapters": {"42": "b", "0": "a"}}, "w": {"duration": 1, "chapters": {}}}))
    asr.write_text(json.dumps({"v": video().asr}))
    (v,) = vc.load_subset(ch, asr)
    assert v.chapter_starts == [0.0, 42.0] and v.chapter_titles == ["a", "b"]

    jl = tmp_path / "x.jsonl"
    jl.write_text(json.dumps({"vid": "v", "duration": 100.0, "chapters": [[0.0, "a"], [42.0, "b"]], "asr": video().asr, "lang": "es", "split": "es_test"}) + "\n")
    (v2,) = vc.load_jsonl(jl)
    assert (v2.lang, v2.split, v2.chapter_starts) == ("es", "es_test", [0.0, 42.0])
