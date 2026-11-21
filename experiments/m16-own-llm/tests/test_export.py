"""GGUF export parity (0025 §10, test plan): llama.cpp must tokenize exactly like our tokenizer and
compute the same next-token logits as our PyTorch model, on CPU, for a tiny random model."""

import numpy as np
import pytest
import torch

from llm.config import ModelConfig
from llm.export_gguf import export, preload_llama_cpp
from llm.model import Decoder
from llm.tokenizer import CHAT_TEMPLATE, IM_END, IM_START, render_chat, train_tokenizer
from tests.test_model_data_tokenizer import CORPUS

preload_llama_cpp()
llama_cpp = pytest.importorskip("llama_cpp")

STRINGS = [
    "Hola, ¿qué tal? Esto es una prueba del tokenizador.",
    "El niño comió ñoquis en la montaña; después, la canción terminó a las 21:45.",
    "The podcast host asked 3 questions about AI 😂",
    "  espacios   múltiples\n\nnuevas líneas\ty tabuladores  ",
    "Números 1234567 y 3.14159, símbolos €$%&/()=?¿¡!",
    "Chapter titles should be short: at most eight words.",
    "naïve café — “comillas” «españolas» …",
    "MAYÚSCULAS y minúsculas, CamelCase y snake_case",
]


@pytest.fixture(scope="module")
def tok_and_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    tok = train_tokenizer(CORPUS + STRINGS * 20, vocab_size=500, out_path=d / "tokenizer.json")
    return tok, d / "tokenizer.json"


def build(tok, qk_norm=False, f32=True, tmp=None):
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=128, qk_norm=qk_norm)
    torch.manual_seed(0)
    m = Decoder(cfg).eval()
    with torch.no_grad():  # norms away from 1 so a wrong norm mapping can't hide
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.mul_(torch.empty_like(p).uniform_(0.5, 1.5))
    path = tmp / f"m_{qk_norm}_{f32}.gguf"
    export(m.state_dict(), cfg, tmp / "tokenizer.json", path, f32=f32)
    return m, path


def load(path, n_ctx=128):
    return llama_cpp.Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=0, logits_all=True, verbose=False,
                           n_threads=2)


def test_tokenization_identical(tok_and_path, tmp_path):
    tok, tpath = tok_and_path
    (tmp_path / "tokenizer.json").write_bytes(tpath.read_bytes())
    _, path = build(tok, tmp=tmp_path)
    llm = load(path)
    for s in STRINGS + CORPUS[:4]:
        ours = tok.encode(s).ids
        theirs = llm.tokenize(s.encode(), add_bos=False, special=False)
        assert theirs == ours, s
    # Special tokens: parsed as one id each when special=True.
    chat = render_chat([{"role": "user", "content": "Hola"}], add_generation_prompt=True)
    assert llm.tokenize(chat.encode(), add_bos=False, special=True) == tok.encode(chat).ids
    assert llm.metadata["tokenizer.chat_template"] == CHAT_TEMPLATE
    assert llm.token_eos() == tok.token_to_id(IM_END)


@pytest.mark.parametrize("qk_norm,f32,tol", [(False, True, 1e-3), (False, False, 5e-2), (True, True, 1e-3)])
def test_logits_parity(tok_and_path, tmp_path, qk_norm, f32, tol):
    tok, tpath = tok_and_path
    (tmp_path / "tokenizer.json").write_bytes(tpath.read_bytes())
    m, path = build(tok, qk_norm=qk_norm, f32=f32, tmp=tmp_path)
    llm = load(path)
    for s in STRINGS[:4]:
        ids = tok.encode(s).ids
        with torch.no_grad():
            ref = m(torch.tensor([ids]))[0].numpy()
        llm.reset()
        llm.eval(ids)
        got = np.array(llm.scores[: len(ids)])
        assert np.abs(got - ref).max() < tol, (s, float(np.abs(got - ref).max()))


def test_generation_stops_at_im_end(tok_and_path, tmp_path):
    """Greedy generation through the chat template ends on EOS (<|im_end|>) or max_tokens, without error."""
    tok, tpath = tok_and_path
    (tmp_path / "tokenizer.json").write_bytes(tpath.read_bytes())
    _, path = build(tok, tmp=tmp_path)
    llm = load(path)
    prompt = f"{IM_START}user\nHola{IM_END}\n{IM_START}assistant\n"
    out = llm.create_completion(llm.tokenize(prompt.encode(), add_bos=False, special=True), max_tokens=8,
                                temperature=0.0)
    assert out["choices"][0]["finish_reason"] in ("stop", "length")
