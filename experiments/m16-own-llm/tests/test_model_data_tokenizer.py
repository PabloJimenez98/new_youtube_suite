"""Model, data and tokenizer unit tests (0025 test plan, fast CPU suite)."""

from collections import Counter

import numpy as np
import torch

from llm.config import ModelConfig
from llm.data import MixedLoader, mix_pattern, read_index
from llm.model import Decoder, apply_rope, rope_tables
from llm.tokenizer import IM_END, IM_START, SPECIALS, render_chat, train_tokenizer

# ---------------------------------------------------------------- model


def tiny_model(**kw):
    c = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, **kw)
    torch.manual_seed(0)
    return Decoder(c)


def test_rope_matches_complex_rotation():
    cos, sin = rope_tables(16, 8, 10000.0)
    x = torch.randn(1, 16, 2, 8)
    ref = torch.view_as_real(torch.view_as_complex(x.reshape(1, 16, 2, 4, 2))
                             * torch.polar(torch.ones_like(cos), torch.atan2(sin, cos))[None, :, None, :])
    assert torch.allclose(apply_rope(x, cos, sin), ref.flatten(-2), atol=1e-5)


def test_forward_shapes_and_tied_head():
    m = tiny_model()
    idx = torch.randint(0, 128, (2, 10))
    assert m(idx).shape == (2, 10, 128)
    assert "output.weight" not in dict(m.named_parameters())
    assert m.num_params() - m.num_params(non_embedding=True) == 128 * 32


def test_causal():
    m = tiny_model()
    idx = torch.randint(0, 128, (1, 12))
    a = m(idx)[0, :6]
    idx2 = idx.clone()
    idx2[0, 8:] = (idx2[0, 8:] + 1) % 128
    assert torch.allclose(a, m(idx2)[0, :6], atol=1e-5)


def test_relu2_same_params_as_swiglu():
    # At the stage A width: 3 x 768 x 2048 (SwiGLU) == 2 x 768 x 3072 (ReLU²).
    sw, r2 = ModelConfig(d_model=768), ModelConfig(d_model=768, mlp="relu2")
    assert 3 * 768 * sw.hidden == 2 * 768 * r2.hidden
    assert not hasattr(tiny_model(mlp="relu2").blocks[0].ffn, "ffn_gate")


def test_qk_norm_variant_and_learning():
    for qk, mlp in ((False, "swiglu"), (True, "swiglu"), (False, "relu2")):
        m = tiny_model(qk_norm=qk, mlp=mlp)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x = torch.arange(32).repeat(4, 1) % 128
        first = None
        for _ in range(30):
            loss = m(x[:, :-1], x[:, 1:])
            first = first or loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert loss.item() < first * 0.5


def test_muon_learns_and_orthogonalises():
    from types import SimpleNamespace

    from llm.optim import build_optimizer, zeropower_newtonschulz5

    G = torch.randn(64, 32)
    X = zeropower_newtonschulz5(G).float()
    s = torch.linalg.svdvals(X)
    assert 0.5 < s.min() and s.max() < 1.3  # NS5 maps singular values into ~[0.7, 1.2]

    m = tiny_model()
    cfg = SimpleNamespace(optimizer="muon", lr=3e-3, muon_lr=0.02, muon_momentum=0.95, muon_weight_decay=0.0,
                          weight_decay=0.0, beta1=0.9, beta2=0.95)
    opt = build_optimizer(m, cfg, "cpu")
    n_muon = sum(p.numel() for p in opt.opts["muon"].param_groups[0]["params"])
    assert n_muon == m.num_params(non_embedding=True) - sum(p.numel() for p in m.parameters() if p.dim() < 2)
    x = torch.arange(32).repeat(4, 1) % 128
    first = None
    for _ in range(30):
        loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first * 0.5


def test_loss_mask():
    m = tiny_model()
    x = torch.randint(0, 128, (2, 9))
    mask = torch.zeros(2, 8)
    mask[:, -2:] = 1
    full = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 128), x[:, 1:].reshape(-1), reduction="none")
    assert torch.allclose(m(x[:, :-1], x[:, 1:], mask), full.view(2, 8)[:, -2:].mean(), atol=1e-6)


# ---------------------------------------------------------------- data


def test_mix_pattern_exact_shares():
    p = mix_pattern({"es": 0.7, "en": 0.3})
    assert len(p) == 10
    assert Counter(p) == {"es": 7, "en": 3}
    assert "en en" not in " ".join(p)  # interleaved, not blocked


def test_shards_roundtrip_and_index(shards):
    idx = read_index(shards / "es")
    assert idx["vocab_size"] == 256
    assert sum(e["tokens"] for e in idx["train"]) >= 60_000
    assert len(idx["train"]) > 1  # rolled over shard_tokens
    arr = np.fromfile(shards / "es" / idx["train"][0]["file"], dtype=np.uint16)
    assert arr.max() < 256
    assert (arr == 0).any()  # document separators


def test_loader_deterministic_and_resumable(shards):
    a = MixedLoader(shards, {"es": 0.7, "en": 0.3})
    batches = [a.next_batch(4, 16) for _ in range(5)]
    state = a.state_dict()
    after = [a.next_batch(4, 16) for _ in range(3)]
    b = MixedLoader(shards, {"es": 0.7, "en": 0.3})
    for x, y in batches:
        bx, by = b.next_batch(4, 16)
        assert torch.equal(x, bx) and torch.equal(y, by)
    c = MixedLoader(shards, {"es": 0.7, "en": 0.3})
    c.load_state_dict(state)
    for x, _ in after:
        assert torch.equal(x, c.next_batch(4, 16)[0])


def test_loader_targets_shifted_and_wraps(shards):
    a = MixedLoader(shards, {"en": 1.0})
    x, y = a.next_batch(2, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    for _ in range(2000):  # 30k tokens of 'en' -> must wrap at least once
        a.next_batch(1, 16)
    assert a.epochs()["en"] >= 1


# ---------------------------------------------------------------- tokenizer

CORPUS = [
    "El niño comió ñoquis en la montaña; después, la canción terminó a las 21:45.",
    "¿Qué opinas del episodio 1x08? ¡Fue increíble! Árbol, acción, pingüino.",
    "The podcast host asked 3 questions about AI, then laughed 😂 for 10 seconds.",
    "Chapter titles should be short: at most eight words, in the episode's language.",
] * 200


def test_tokenizer_roundtrip_and_specials(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=600, out_path=tmp_path / "tok.json")
    for s in CORPUS[:4] + ["Mañana: 1234567 €, naïve café — “comillas”."]:
        assert tok.decode(tok.encode(s).ids) == s
    for sp in SPECIALS:
        ids = tok.encode(f"hola{sp}adiós").ids
        assert tok.token_to_id(sp) in ids
    # 4 repeated sentences run out of pairs to merge before 600; the vocab never exceeds the target.
    assert 256 + len(SPECIALS) < tok.get_vocab_size() <= 600
    # Frequent accented words become single tokens rather than byte fragments.
    assert len(tok.encode(" canción").ids) <= 2


def test_chat_template():
    msgs = [{"role": "system", "content": "Eres útil."}, {"role": "user", "content": "Hola"}]
    s = render_chat(msgs, add_generation_prompt=True)
    assert s == (f"{IM_START}system\nEres útil.{IM_END}\n{IM_START}user\nHola{IM_END}\n{IM_START}assistant\n")
    assert render_chat(msgs + [{"role": "assistant", "content": "¡Hola!"}]).endswith(
        f"{IM_START}assistant\n¡Hola!{IM_END}\n")


def test_chat_template_matches_backend_render_prompt():
    """The backend renders the GGUF template with its own helper; both must agree."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from llm.tokenizer import CHAT_TEMPLATE

    msgs = [{"role": "user", "content": "Título, por favor"}]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    backend = env.from_string(CHAT_TEMPLATE).render(messages=msgs, add_generation_prompt=True,
                                                    enable_thinking=False, bos_token="", eos_token="")
    assert backend == render_chat(msgs, add_generation_prompt=True)
