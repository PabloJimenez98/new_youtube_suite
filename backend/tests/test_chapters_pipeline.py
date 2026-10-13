"""`app.chapters` -- pipeline over a tiny fake bundle, plus the guarantee
that the backend's copies of the experiment modules haven't drifted.

The fake bundle uses Hugging Face's tiny random test models (downloaded
once into the HF cache) and a random-init BiLSTM: outputs are garbage,
but every code path of a real bundle runs. The real exported bundle is
covered by `test_chapters_integration.py`.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from app.chapters import llm_titles, pipeline
from app.chapters.model import BoundaryTagger
from app.chapters.units import Atom

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_LIB = REPO / "experiments" / "m5-chapters" / "lib"
TINY_ENCODER = "hf-internal-testing/tiny-random-BertModel"
TINY_T5 = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"


@pytest.mark.parametrize("name", ["units", "decoding", "baseline", "model"])
def test_backend_copy_matches_experiment(name: str) -> None:
    backend = (REPO / "backend" / "app" / "chapters" / f"{name}.py").read_text().splitlines()
    experiment = (EXPERIMENT_LIB / f"{name}.py").read_text().splitlines()
    body = [line for line in backend if not line.startswith("# Copied from") and not line.startswith("# experiment's tests")]
    assert body == experiment, f"app/chapters/{name}.py drifted from experiments/m5-chapters/lib/{name}.py"


def test_llm_titles_copy_matches_experiment() -> None:
    """Same as above; the only allowed difference is where `clean_title` comes from (0010 §4)."""
    backend = (REPO / "backend" / "app" / "chapters" / "llm_titles.py").read_text().splitlines()
    experiment = (EXPERIMENT_LIB / "llm_titles.py").read_text().splitlines()
    body = [line.replace("from app.chapters.text import", "from lib.titles import") for line in backend
            if not line.startswith("# Copied from") and not line.startswith("# experiment's tests")]
    assert body == experiment, "app/chapters/llm_titles.py drifted from experiments/m5-chapters/lib/llm_titles.py"


def test_clean_title_copy_matches_experiment() -> None:
    def source(path: Path, names: set[str]) -> list[str]:
        text = path.read_text()
        tree = ast.parse(text)
        return [ast.get_source_segment(text, n) for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in names
                or isinstance(n, ast.Assign) and {t.id for t in n.targets if isinstance(t, ast.Name)} & names]

    names = {"clean_title", "_ENUM", "_PAIRS"}
    assert source(REPO / "backend" / "app" / "chapters" / "text.py", names) == source(EXPERIMENT_LIB / "titles.py", names)


def make_bundle(root: Path, kind: str) -> Path:
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(root / "encoder")
    enc = AutoModel.from_pretrained(TINY_ENCODER)
    enc.save_pretrained(root / "encoder")
    AutoTokenizer.from_pretrained(TINY_T5).save_pretrained(root / "titles")
    AutoModelForSeq2SeqLM.from_pretrained(TINY_T5).save_pretrained(root / "titles")
    if kind == "bilstm":
        torch.manual_seed(0)
        tagger = BoundaryTagger(emb_dim=enc.config.hidden_size, proj=8, hidden=4, layers=1, dropout=0.0)
        torch.save({"state_dict": tagger.state_dict(), "model_config": tagger.config}, root / "boundary.pt")
    manifest = {
        "version": f"test-{kind}",
        "boundary": {
            "kind": kind,
            "unit": {"mode": "sentence", "min_s": 2.0, "max_s": 15.0, "block_s": 10.0},
            "decode": {"per_hour": 6.0, "min_chapters": 3, "max_chapters": None, "min_len_s": 60.0, "threshold": None},
            "baseline_window": 4,
            "encoder_prefix": "",
        },
        "titles": {"max_input_tokens": 64, "max_target_tokens": 8, "num_beams": 2},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def atoms(minutes: int = 30) -> list[Atom]:
    """One ~5 s sentence every 5 s, with a few untimed words."""
    out = []
    for i in range(minutes * 12):
        t = 5.0 * i
        out.append(Atom(t, t + 2.0, f"topic{i // 60} sentence"))
        out.append(Atom(None, None, "€") if i % 50 == 0 else Atom(t + 2.0, t + 4.5, f"number {i}."))
    return out


@pytest.mark.parametrize("kind", ["baseline", "bilstm"])
def test_chapterer_runs_end_to_end_on_a_tiny_bundle(tmp_path: Path, kind: str) -> None:
    c = pipeline.Chapterer(make_bundle(tmp_path, kind))
    results = c.run(atoms(30), duration=1800.0)

    starts = [r.start for r in results]
    assert starts[0] == 0.0
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert len(results) == 3  # 30 min at 6/h -> the 3-chapter floor
    assert all(b - a >= 60 for a, b in zip(starts, [*starts[1:], 1800.0]))
    assert all(isinstance(r.title, str) and r.title for r in results)
    assert c.version == f"test-{kind}"


def test_chapterer_short_audio_is_one_chapter(tmp_path: Path) -> None:
    c = pipeline.Chapterer(make_bundle(tmp_path, "baseline"))
    results = c.run([Atom(0.0, 3.0, "Hi."), Atom(3.0, 8.0, "Bye.")], duration=8.0)
    assert [r.start for r in results] == [0.0]


def test_chapterer_no_speech_is_no_chapters(tmp_path: Path) -> None:
    c = pipeline.Chapterer(make_bundle(tmp_path, "baseline"))
    assert c.run([Atom(None, None, "x")], duration=60.0) == []


def test_unknown_boundary_kind_is_rejected(tmp_path: Path) -> None:
    root = make_bundle(tmp_path, "baseline")
    m = json.loads((root / "manifest.json").read_text())
    m["boundary"]["kind"] = "crf"
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(pipeline.ChaptersUnavailableError):
        pipeline.Chapterer(root)


def test_get_chapterer_without_bundle_fails_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_chapterer", None)
    monkeypatch.setattr(pipeline.get_settings(), "chapters_model_dir", tmp_path / "missing")
    with pytest.raises(pipeline.ChaptersUnavailableError, match="CHAPTERS_MODEL_DIR"):
        pipeline.get_chapterer()


# --- bundle v2: LLM titles with mT5 fallback (0010) ----------------------


class FakeTitler:
    """Stands in for `LlamaTitler`: records its calls, returns scripted titles."""

    def __init__(self, path: Path, **kw):
        self.path, self.kw, self.calls = path, kw, []
        self.script: list[str | None] = []

    def title(self, text, lang, previous=None, lid=None):
        self.calls.append(dict(text=text, lang=lang, previous=list(previous or []), lid=lid))
        t = self.script[len(self.calls) - 1] if len(self.calls) <= len(self.script) else f"LLM {len(self.calls)}"
        return SimpleNamespace(title=t)


class FakeLid:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, text: str) -> tuple[str, float]:
        return "es", 0.99


def make_llm_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    make_bundle(root, "baseline")
    m = json.loads((root / "manifest.json").read_text())
    m["titles"]["llm"] = {"file": "llm/model.gguf", "n_ctx": 2048, "max_words": 900, "max_tokens": 32, "prompt_version": "v1"}
    m["titles"]["lid"] = {"file": "lid.176.ftz"}
    (root / "manifest.json").write_text(json.dumps(m))
    monkeypatch.setattr(llm_titles, "LlamaTitler", FakeTitler)
    monkeypatch.setattr(pipeline, "LanguageId", FakeLid)
    return root


def test_llm_bundle_loads_the_titler_and_lid_from_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = pipeline.Chapterer(make_llm_bundle(tmp_path, monkeypatch), llm_threads=3)
    assert c.llm.path == tmp_path / "llm" / "model.gguf"
    assert c.llm.kw == dict(n_ctx=2048, n_threads=3, max_tokens=32, max_words=900)
    assert c.lid.path == tmp_path / "lid.176.ftz"


def test_llm_titles_in_order_with_previous_titles_and_episode_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = pipeline.Chapterer(make_llm_bundle(tmp_path, monkeypatch))
    results = c.run(atoms(30), duration=1800.0)

    assert [r.title for r in results] == ["LLM 1", "LLM 2", "LLM 3"]
    calls = c.llm.calls
    assert [k["previous"] for k in calls] == [[], ["LLM 1"], ["LLM 1", "LLM 2"]]
    assert all(k["lang"] == "es" and k["lid"] is c.lid for k in calls)
    # The LLM gets the whole chapter, not mT5's 2000-char cut (sample_text caps it).
    assert max(len(k["text"]) for k in calls) > 2000


def test_llm_falls_back_to_mt5_per_chapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = pipeline.Chapterer(make_llm_bundle(tmp_path, monkeypatch))
    c.llm.script = ["Intro", None, "Cierre"]
    mt5_inputs = []
    monkeypatch.setattr(c, "titles", lambda texts: mt5_inputs.append(texts) or ["mT5 title"] * len(texts))

    results = c.run(atoms(30), duration=1800.0)

    assert [r.title for r in results] == ["Intro", "mT5 title", "Cierre"]
    assert len(mt5_inputs) == 1 and len(mt5_inputs[0]) == 1  # only the failed chapter
    assert c.llm.calls[2]["previous"] == ["Intro"]  # a failed title isn't passed on


def test_bundle_without_llm_uses_mt5_only(tmp_path: Path) -> None:
    c = pipeline.Chapterer(make_bundle(tmp_path, "baseline"))
    assert c.llm is None and c.lid is None
