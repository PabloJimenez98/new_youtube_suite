"""SFT data, loader, task-call filtering and an SFT run that starts from pretrained weights (0025 §9)."""

import json

import numpy as np
import pytest
import torch

from llm import config as cfgmod
from llm.data import SFTLoader
from llm.sft_data import build, encode_example
from llm.task_data import subtask, usable
from llm.tokenizer import IM_END, render_chat, train_tokenizer
from llm.train import Trainer, read_status
from tests.conftest import tiny_config, write_run
from tests.test_model_data_tokenizer import CORPUS

CHATS = [
    [{"role": "system", "content": "Escribes títulos de capítulos."},
     {"role": "user", "content": "Transcripción: hablamos de la montaña y la canción."},
     {"role": "assistant", "content": "La montaña y la canción"}],
    [{"role": "user", "content": "Reply with JSON only."},
     {"role": "assistant", "content": '{"scores": [1, 5, 9]}'}],
    [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "¡Hola! ¿Qué tal?"},
     {"role": "user", "content": "Bien"}, {"role": "assistant", "content": "Me alegro."}],
]


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    texts = CORPUS + [m["content"] for c in CHATS for m in c] * 50
    train_tokenizer(texts, vocab_size=400, out_path=d / "tokenizer.json")
    return d / "tokenizer.json"


def test_encode_matches_app_tokenization_and_masks_assistant(tok_path):
    from llm.tokenizer import load_tokenizer

    tok = load_tokenizer(tok_path)
    for chat in CHATS:
        ids, mask = encode_example(tok, chat)
        assert ids == tok.encode(render_chat(chat)).ids  # same ids the app's prompt produces
        target = tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=False)
        expected = "".join(m["content"] + IM_END for m in chat if m["role"] == "assistant")
        assert target == expected


def test_build_and_loader(tok_path, tmp_path):
    src = tmp_path / "a.jsonl"
    src.write_text("\n".join(json.dumps({"messages": c}) for c in CHATS * 20), encoding="utf-8")
    val = tmp_path / "v.jsonl"
    val.write_text(json.dumps({"messages": CHATS[0], "split": "val"}), encoding="utf-8")
    small = build([(src, 1.0)], tok_path, tmp_path / "small", seq_len=8, val_fraction=0.0)
    assert small["dropped_too_long"] > 0  # longer than a row: dropped, never truncated
    idx = build([(src, 2.0), (val, 1.0)], tok_path, tmp_path / "sft", seq_len=128, val_fraction=0.0)
    assert idx["examples"] == 120 and idx["val_examples"] == 1 and idx["dropped_too_long"] == 0
    ld = SFTLoader(tmp_path / "sft")
    x, y, m = ld.next_batch(3, 128)
    assert x.shape == y.shape == m.shape == (3, 128)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert m.sum() > 0
    st = ld.state_dict()
    nxt = ld.next_batch(2, 128)[0]
    ld2 = SFTLoader(tmp_path / "sft")
    ld2.load_state_dict(st)
    assert torch.equal(ld2.next_batch(2, 128)[0], nxt)
    with pytest.raises(ValueError):
        ld.next_batch(1, 32)


def test_usable_filters_teacher_calls():
    schema = {"properties": {"scores": {"minItems": 3}}}
    assert usable({"task": "title", "output": "Un título"})
    assert not usable({"task": "title", "output": None})
    assert usable({"task": "relevance", "output": '{"scores": [1, 2, 3]}', "json_schema": schema})
    assert not usable({"task": "relevance", "output": '{"scores": [1, 2]}', "json_schema": schema})
    assert not usable({"task": "relevance", "output": '{"scores": [1, 2', "json_schema": schema})
    assert usable({"task": "shownotes", "output": '{"summary": "x"}', "json_schema": {"type": "object"}})
    assert subtask({"task": "shownotes", "json_schema": {}}) == "shownotes_reduce"
    assert subtask({"task": "shownotes"}) == "shownotes_map"


def test_sft_run_from_pretrained_learns_and_resumes(tok_path, tmp_path, shards, runs):
    from llm.tokenizer import load_tokenizer

    V = load_tokenizer(tok_path).get_vocab_size()
    src = tmp_path / "a.jsonl"
    src.write_text("\n".join(json.dumps({"messages": c}) for c in CHATS * 40), encoding="utf-8")
    idx = build([(src, 1.0)], tok_path, tmp_path / "sft", seq_len=128, val_fraction=0.1)
    assert idx["val_rows"] > 0 and idx["dropped_too_long"] == 0

    # A "pretrained" model: a tiny pretraining run on the synthetic shards (token ids < 256 < V).
    model = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    pre = write_run(runs, tiny_config("pre", shards, model=model))
    assert Trainer(pre).run() == "done"

    def sft_cfg(name):
        return tiny_config(name, shards, model=model, init_from=str(pre / "final_model.pt"), lr=3e-3,
                           data={"kind": "sft", "sft_dir": str(tmp_path / "sft")}, seq_len=128,
                           batch_tokens=128 * 4 * 2, total_tokens=128 * 4 * 2 * 20, warmup_tokens=128 * 4 * 2,
                           eval_every_tokens=128 * 4 * 2 * 5)

    a = write_run(runs, sft_cfg("sft-a"))
    assert Trainer(a).run() == "done"
    evals = [json.loads(l) for l in (a / "metrics.jsonl").read_text().splitlines() if '"eval"' in l]
    assert evals[-1]["val_loss_sft"] < evals[0]["val_loss_sft"]
    assert any('"init_from"' in l for l in (a / "metrics.jsonl").read_text().splitlines())

    b = write_run(runs, sft_cfg("sft-b"))
    Trainer(b, on_step=lambda tr: (tr.run_dir / "PAUSE").touch() if tr.step == 7 else None).run()
    assert read_status(b)["state"] == "paused"
    (b / "PAUSE").unlink()
    assert Trainer(b).run() == "done"
    fa = torch.load(a / "final_model.pt", weights_only=False)["model"]
    fb = torch.load(b / "final_model.pt", weights_only=False)["model"]
    for k in fa:
        assert torch.equal(fa[k], fb[k]), k


def test_seq_logprob_and_dpo_loss_at_reference():
    import math

    from llm.dpo import dpo_loss, seq_logprob

    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    y = torch.randint(0, 7, (2, 5))
    m = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=torch.float32)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), y.reshape(-1), reduction="none").view(2, 5)
    assert torch.allclose(seq_logprob(logits, y, m), -(ce * m).sum(-1))
    p = torch.tensor([-3.0, -5.0])
    loss, _ = dpo_loss(p, p, p, p, beta=0.1)  # policy == reference: margin 0
    assert loss.item() == pytest.approx(math.log(2))


def test_dpo_run_learns_preference(tok_path, tmp_path, shards, runs):
    from llm import dpo
    from llm.tokenizer import load_tokenizer

    V = load_tokenizer(tok_path).get_vocab_size()
    prompt = CHATS[0][:2]
    pairs = tmp_path / "pairs.jsonl"
    lines = []
    for i in range(60):
        lines.append(json.dumps({"messages": prompt, "chosen": "La montaña y la canción",
                                 "rejected": "Títulos: 1. la la la la la", "split": "val" if i < 6 else "train"},
                                ensure_ascii=False))
    pairs.write_text("\n".join(lines), encoding="utf-8")
    idx = dpo.build([pairs], tok_path, tmp_path / "dpo", seq_len=128)
    assert idx["pairs"] == 54 and idx["val_pairs"] == 6

    model = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    ref = write_run(runs, tiny_config("ref", shards, model=model))
    Trainer(ref).run()
    dpo.compute_ref(tmp_path / "dpo", ref / "final_model.pt", device="cpu")

    cfg = tiny_config("dpo", shards, model=model, init_from=str(ref / "final_model.pt"), objective="dpo",
                      dpo_beta=0.5, lr=3e-3, data={"kind": "dpo", "sft_dir": str(tmp_path / "dpo")}, seq_len=128,
                      micro_batch=2, batch_tokens=128 * 2 * 2, total_tokens=128 * 2 * 2 * 15,
                      warmup_tokens=128 * 2 * 2, eval_every_tokens=128 * 2 * 2 * 5, eval_batches=3)
    d = write_run(runs, cfg)
    assert Trainer(d).run() == "done"
    evals = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if '"eval"' in l]
    assert evals[0]["val_loss_dpo"] < math_log2() + 0.05  # starts near log 2 (policy ~ reference)
    assert evals[-1]["val_loss_dpo"] < evals[0]["val_loss_dpo"]
    assert evals[-1]["val_pref_acc"] == 1.0


def math_log2():
    import math

    return math.log(2)
