"""Tiny CPU fixtures: synthetic shards and a d32/L2 run config (0025 test plan)."""

import os

import numpy as np
import pytest

from llm import config as cfgmod
from llm.data import ShardWriter, write_index

V = 256


def make_source(root, name, n_tokens, seed, shard_tokens=5_000, val_tokens=3_000):
    d = root / name
    rng = np.random.default_rng(seed)
    # Structured, learnable data: repeated short "documents" with a separator (id 0).
    vocab = np.arange(1, V)
    docs = [rng.choice(vocab, size=rng.integers(5, 40)) for _ in range(200)]
    val = ShardWriter(d, "val")
    while val.tokens < val_tokens:
        doc = docs[rng.integers(len(docs))]
        val.write(np.append(doc, 0), nbytes=len(doc) * 3)
    w = ShardWriter(d, "train", shard_tokens=shard_tokens)
    while w.tokens < n_tokens:
        doc = docs[rng.integers(len(docs))]
        w.write(np.append(doc, 0), nbytes=len(doc) * 3)
    write_index(d, V, w.close(), val.close()[0])
    return d


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    make_source(root, "es", 60_000, seed=1)
    make_source(root, "en", 30_000, seed=2)
    return root


@pytest.fixture
def runs(tmp_path, monkeypatch):
    r = tmp_path / "runs"
    r.mkdir()
    monkeypatch.setenv("M16_RUNS", str(r))
    return r


def tiny_config(name, shards_root, **kw) -> cfgmod.TrainConfig:
    base = dict(
        name=name,
        model=dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64),
        data=dict(root=str(shards_root), sources={"es": 0.7, "en": 0.3}),
        tokenizer=None, total_tokens=32 * 4 * 2 * 12, seq_len=32, micro_batch=4, batch_tokens=32 * 4 * 2,
        lr=3e-3, warmup_tokens=32 * 4 * 2 * 2, decay_fraction=0.3, device="cpu", compile=False,
        autosave_minutes=10_000, eval_every_tokens=32 * 4 * 2 * 5, eval_batches=2, log_every_steps=1,
        permanent_every_fraction=0.5,
    )
    base.update(kw)
    return cfgmod.from_dict(base)


def write_run(runs_root, cfg) -> "os.PathLike":
    d = runs_root / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    return d
