"""Write backend/tests/fixtures/own_llm/: a tiny random own-model GGUF (+ manifest) exported by our
real export path, so the backend can test loading and running an own model without a trained one.

    python tools/make_backend_fixture.py      # from experiments/m16-own-llm (WSL, experiments/.venv)

Deterministic (fixed seeds); ~0.2 MB. Its outputs are random text -- the backend tests only check
the plumbing (chat template from the GGUF, no BOS, EOS = <|im_end|>, grammar-constrained JSON).
"""

import json
from pathlib import Path

import torch

from llm.config import ModelConfig
from llm.export_gguf import export
from llm.model import Decoder
from llm.tokenizer import train_tokenizer

HERE = Path(__file__).resolve().parent.parent
OUT = HERE.parent.parent / "backend" / "tests" / "fixtures" / "own_llm"

TEXTS = [
    "Escribes títulos de capítulos de YouTube para un podcast. Responde solo con el título.",
    "You write YouTube chapter titles for a podcast. Reply with the title only.",
    'Reply with JSON only: {"scores": [1, 2, 3]} {"title_suggestions": ["a"], "summary": "b"}',
    "Hablamos de la montaña, la canción y el episodio de hoy. ¿Qué opinas?",
] * 200

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tok_path = OUT / "tokenizer.json"
    tok = train_tokenizer(TEXTS, vocab_size=320, out_path=tok_path)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
                      max_seq_len=512, ffn_hidden=64)
    torch.manual_seed(0)
    export(Decoder(cfg).state_dict(), cfg, tok_path, OUT / "yts-own-tiny.gguf", f32=True, name="yts-own-tiny")
    tok_path.unlink()
    (OUT / "manifest.json").write_text(json.dumps(
        {"file": "yts-own-tiny.gguf", "n_ctx": 512, "label": "yts-own-tiny",
         "note": "test fixture: random weights, written by experiments/m16-own-llm/tools/make_backend_fixture.py"},
        indent=1))
    print(sorted(p.name for p in OUT.iterdir()), sum(p.stat().st_size for p in OUT.iterdir()) // 1024, "KB")
