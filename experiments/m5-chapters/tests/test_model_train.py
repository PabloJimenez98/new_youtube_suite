import numpy as np
import pytest
import torch

from lib.model import BoundaryTagger
from lib.store import Store, StoreWriter
from lib.train import TrainConfig, predict_scores, soft_targets, train


def test_soft_targets():
    y = soft_targets(6, [2, 5], 0.3)
    assert y.tolist() == pytest.approx([0, 0.3, 1, 0.3, 0.3, 1])  # unit 0 never positive; 5 is last


def test_model_shapes_and_padding_independence():
    m = BoundaryTagger(emb_dim=8, proj=16, hidden=8, layers=1, dropout=0).eval()
    e = torch.randn(2, 10, 8)
    p = torch.linspace(0, 1, 10).repeat(2, 1)
    out = m(e, p, torch.tensor([10, 6]))
    assert out.shape == (2, 10)
    # The 6-unit video's scores don't depend on what sits in its padding.
    e2 = e.clone()
    e2[1, 6:] = 99
    assert torch.allclose(out[1, :6], m(e2, p, torch.tensor([10, 6]))[1, :6], atol=1e-5)


def synthetic_store(path, n_videos, rng, dim=16):
    """Topic changes every 20 units; boundaries = where the topic changes."""
    w = StoreWriter(path, dim)
    for i in range(n_videos):
        n = 100
        centers = rng.normal(size=(5, dim))
        emb = np.vstack([centers[k // 20] + 0.2 * rng.normal(size=dim) for k in range(n)])
        starts = [5.0 * k for k in range(n)]
        w.add(f"v{i}", emb, {"unit_starts": starts, "duration": 500.0, "labels": [20, 40, 60, 80],
                             "chapter_starts": [0.0, 100.0, 200.0, 300.0, 400.0]})
    w.close()
    return Store(path)


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    s = synthetic_store(tmp_path / "s", 3, rng)
    e, meta = s.get(2)
    assert e.shape == (100, 16) and meta["vid"] == "v2" and len(s) == 3


def test_training_learns_synthetic_topic_changes(tmp_path):
    rng = np.random.default_rng(0)
    tr = synthetic_store(tmp_path / "tr", 64, rng)
    va = synthetic_store(tmp_path / "va", 8, rng)
    cfg = TrainConfig(epochs=20, batch_size=8, lr=1e-2, proj=32, hidden=16, layers=1, dropout=0.0)
    from lib.decoding import DecodeConfig
    res = train(tr, va, cfg, tmp_path / "out", "cpu", DecodeConfig(per_hour=36, min_len_s=60))
    assert res["best"]["F1@5s"] > 90
    ck = torch.load(tmp_path / "out" / "best.pt")
    m = BoundaryTagger(**ck["model_config"])
    m.load_state_dict(ck["state_dict"])
    e, meta = va.get(0)
    s = predict_scores(m, e, meta["unit_starts"], meta["duration"], "cpu")
    assert set(np.argsort(-s)[:4]) == {20, 40, 60, 80}
