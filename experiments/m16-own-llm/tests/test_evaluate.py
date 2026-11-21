"""A3 evaluation helpers: §10 gate arithmetic on synthetic results, and the DPO-pair verifiers."""

import json
import sys
from pathlib import Path

import pytest

from llm import dpo_pairs, evaluate


def _ep(i, *, fallback=0, chapters=4, lang_bad=0, sn_ok=True, passes=None, failed=0):
    chs = [{"fallback": k < fallback, "lang_ok": k >= lang_bad, "title": f"t{k}"} for k in range(chapters)]
    return {"id": f"e{i}", "channel": "c", "lang": "es",
            "titles": {"chapters": chs},
            "shownotes": {"ok": sn_ok, "attempts": 1 if sn_ok else 3, "lang_ok": sn_ok, "error": "x (a; b)"},
            "relevance": {"batches": 5, "failed": failed, "passes": passes or [[0.1 * k for k in range(10)]] * 2}}


def _write(d: Path, eps):
    d.mkdir(parents=True)
    (d / "run.json").write_text("{}")
    for e in eps:
        (d / f"{e['id']}.json").write_text(json.dumps(e))


def test_spearman_and_mean_passes():
    assert evaluate.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert evaluate.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert evaluate.spearman([1, 2, 3], [1, 2, 3]) is None  # too few points
    assert evaluate.spearman([1] * 6, [1, 2, 3, 4, 5, 6]) is None  # constant -> NaN -> None
    assert evaluate.mean_passes([[1.0, None, 0.0], [3.0, 2.0, None]]) == [2.0, 2.0, 0.0]


def test_gates_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL", tmp_path)
    up = [[0.1 * k for k in range(10)], [0.1 * k + 0.01 for k in range(10)]]
    _write(tmp_path / "qwen", [_ep(i, passes=up) for i in range(3)])
    _write(tmp_path / "own", [_ep(i, passes=up) for i in range(3)])
    g = evaluate.gates("own")
    assert g["titles"]["pass"] and g["shownotes"]["pass"] and g["relevance"]["pass"]
    assert g["relevance"]["spearman_vs_qwen"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "own" / "gates.json").read_text())["episodes"] == 3


def test_gates_fail_each_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL", tmp_path)
    up = [[0.1 * k for k in range(10)]] * 2
    down = [[1 - 0.1 * k for k in range(10)]] * 2
    _write(tmp_path / "qwen", [_ep(i, passes=up) for i in range(2)])
    # 2 of 8 chapters fall back (25 % > 10 %); 1 of 2 show notes invalid; a failed batch; anti-correlated.
    _write(tmp_path / "own", [_ep(0, fallback=1, sn_ok=False, passes=down, failed=1), _ep(1, fallback=1, passes=down)])
    g = evaluate.gates("own")
    assert g["titles"]["fallback_rate"] == 0.25 and not g["titles"]["pass"]
    assert g["shownotes"]["valid_rate"] == 0.5 and not g["shownotes"]["pass"]
    assert g["relevance"]["failed"] == 1 and not g["relevance"]["pass"]
    assert g["relevance"]["spearman_vs_qwen"] == pytest.approx(-1.0)


def test_distinct3():
    assert dpo_pairs.distinct3("a b c d e") == 1.0
    assert dpo_pairs.distinct3("a b c a b c a b c a b c") < 0.5
    assert dpo_pairs.distinct3("a b") == 0.0


BACKEND = Path(__file__).resolve().parents[3] / "backend"


@pytest.fixture
def backend_on_path(monkeypatch):
    if not (BACKEND / "app" / "relevance" / "llm_scoring.py").exists():
        pytest.skip("backend not found")
    monkeypatch.syspath_prepend(str(BACKEND))
    yield
    for m in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        sys.modules.pop(m)


class FakeLid:
    def __init__(self, lang):
        self.lang = lang

    def __call__(self, text):
        return self.lang, 0.99


def test_verify_title_and_relevance(backend_on_path):
    lid = FakeLid("es")
    assert dpo_pairs.verify("title", "La vida de los jabones", "x", "es", lid, None) is None
    assert dpo_pairs.verify("title", "uno dos tres cuatro cinco seis siete ocho nueve", "x", "es", lid, None) == "too long"
    assert dpo_pairs.verify("title", "Soaps and detergents", "x", "es", FakeLid("en"), None) == "wrong language"
    schema = {"properties": {"scores": {"minItems": 6}}}
    teacher = json.dumps({"scores": [10, 20, 30, 40, 50, 60]})
    assert dpo_pairs.verify("relevance", json.dumps({"scores": [15, 25, 35, 45, 55, 65]}), teacher, "es", lid, schema) is None
    assert dpo_pairs.verify("relevance", json.dumps({"scores": [50] * 6}), teacher, "es", lid, schema) == "constant"
    assert dpo_pairs.verify("relevance", json.dumps({"scores": [60, 50, 40, 30, 20, 10]}), teacher, "es", lid,
                            schema).startswith("disagrees")
    assert dpo_pairs.verify("relevance", "{\"scores\": [1, 2]}", teacher, "es", lid, schema).startswith("invalid")
