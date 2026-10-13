import json

import torch

from lib.decoding import DecodeConfig
from lib.export import add_llm_titles, export_bundle
from lib.model import BoundaryTagger
from lib.units import UnitConfig


def test_export_bundle_layout_and_current_link(tmp_path, monkeypatch):
    # Stand-in for the HF encoder download: any saveable tokenizer/model.
    from lib import export as ex

    class Fake:
        def save_pretrained(self, p):
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text("{}")

    monkeypatch.setattr(ex.AutoTokenizer, "from_pretrained", lambda n: Fake())
    monkeypatch.setattr(ex.AutoModel, "from_pretrained", lambda n: Fake())
    tagger = BoundaryTagger(emb_dim=8, proj=8, hidden=4, layers=1)
    ck = tmp_path / "best.pt"
    torch.save({"state_dict": tagger.state_dict(), "model_config": tagger.config, "train_config": {}}, ck)
    titles = tmp_path / "titles"
    titles.mkdir()
    (titles / "config.json").write_text("{}")

    out = export_bundle(tmp_path / "bundles", "v1", "bilstm", "e5-small", UnitConfig(), DecodeConfig(), 10, ck, titles,
                        {"max_input_tokens": 256, "max_target_tokens": 32, "num_beams": 4}, {"F1@5s": 1.0}, "runs/x")

    m = json.loads((out / "manifest.json").read_text())
    assert m["version"] == "v1" and m["boundary"]["kind"] == "bilstm"
    assert m["boundary"]["encoder_prefix"] == "passage: "
    assert set(m["boundary"]["unit"]) == {"mode", "min_s", "max_s", "block_s"}
    assert (out / "boundary.pt").exists() and (out / "titles" / "config.json").exists()
    assert set(torch.load(out / "boundary.pt")) == {"state_dict", "model_config"}
    assert (tmp_path / "bundles" / "current").resolve() == out.resolve()


def test_add_llm_titles_derives_a_v2_bundle(tmp_path):
    root = tmp_path / "bundles"
    v1 = root / "v1"
    (v1 / "titles").mkdir(parents=True)
    (v1 / "titles" / "config.json").write_text("{}")
    (v1 / "manifest.json").write_text(json.dumps({"version": "v1", "boundary": {"kind": "bilstm"},
                                                  "titles": {"num_beams": 4}, "run": "runs/05"}))
    (root / "current").symlink_to("v1")
    gguf, lid = tmp_path / "M-Q4_K_M.gguf", tmp_path / "lid.176.ftz"
    gguf.write_bytes(b"gguf"); lid.write_bytes(b"lid")

    out = add_llm_titles(root / "current", root, "v2", gguf, lid, {"n_ctx": 4096, "prompt_version": "v1"},
                         {"CIDEr": 1.0}, "runs/06")

    m = json.loads((out / "manifest.json").read_text())
    assert m["version"] == "v2" and m["boundary"] == {"kind": "bilstm"}
    assert m["titles"] == {"num_beams": 4, "llm": {"file": "llm/M-Q4_K_M.gguf", "n_ctx": 4096, "prompt_version": "v1"},
                           "lid": {"file": "lid.176.ftz"}}
    assert m["run"] == "runs/06" and m["derived_from"] == "runs/05"
    assert (out / "llm" / "M-Q4_K_M.gguf").read_bytes() == b"gguf" and (out / "lid.176.ftz").exists()
    assert (out / "titles" / "config.json").exists()
    assert (root / "current").resolve() == out.resolve()
    assert json.loads((root / "v1" / "manifest.json").read_text())["version"] == "v1"  # source untouched
