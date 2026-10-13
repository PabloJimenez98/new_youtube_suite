"""Training loop for the boundary tagger (0009 §3).

- **Targets** are softened: each true boundary unit gets 1.0 and its
  neighbours `neighbour_weight` (a unit off is a near miss, not a total
  miss, for F1@5s). Loss is BCE with `pos_weight` to counter the
  imbalance (roughly 1 boundary per ~30 units).
- **Batches** group videos of similar length (fewer padded steps).
- **Model selection** is by validation F1@5s *after decoding*, the same
  metric the shipping rule uses -- not by validation loss.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lib.decoding import DecodeConfig, decode
from lib.evaluate import Prediction, evaluate
from lib.model import BoundaryTagger
from lib.store import Store


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 10.0
    neighbour_weight: float = 0.3
    max_units: int = 2000
    proj: int = 256
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    seed: int = 0


def soft_targets(n: int, boundaries: list[int], neighbour_weight: float) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    for b in boundaries:
        for k, w in ((b - 1, neighbour_weight), (b + 1, neighbour_weight), (b, 1.0)):
            if 0 < k < n:
                y[k] = max(y[k], w)
    return y


def make_batch(store: Store, idxs: list[int], cfg: TrainConfig, device: str):
    items = [store.get(i) for i in idxs]
    T = min(cfg.max_units, max(len(e) for e, _ in items))
    B, D = len(items), store.dim
    emb = torch.zeros(B, T, D)
    pos = torch.zeros(B, T)
    y = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (e, meta) in enumerate(items):
        n = min(len(e), T)
        emb[i, :n] = torch.from_numpy(e[:n])
        pos[i, :n] = torch.tensor(meta["unit_starts"][:n]) / meta["duration"]
        y[i, :n] = torch.from_numpy(soft_targets(len(e), meta["labels"], cfg.neighbour_weight)[:n])
        lengths[i] = n
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return emb.to(device), pos.to(device), lengths, y.to(device), mask.to(device)


def length_batches(store: Store, batch_size: int, rng: random.Random) -> list[list[int]]:
    order = sorted(range(len(store)), key=lambda i: store.videos[i]["n"] + rng.random())
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.inference_mode()
def predict_scores(model: BoundaryTagger, emb: np.ndarray, unit_starts: list[float], duration: float, device: str) -> np.ndarray:
    model.eval()
    e = torch.from_numpy(np.asarray(emb, dtype=np.float32))[None].to(device)
    p = (torch.tensor(unit_starts, dtype=torch.float32) / duration)[None].to(device)
    return torch.sigmoid(model(e, p, torch.tensor([len(unit_starts)]))).squeeze(0).cpu().numpy()


def oracle_config(chapter_starts: list[float], duration: float, base: DecodeConfig) -> DecodeConfig:
    """Same decoding, but asking for exactly the true number of chapters."""
    n = len([t for t in chapter_starts if t > 0]) + 1
    return DecodeConfig(per_hour=n / (duration / 3600), min_chapters=min(3, n), min_len_s=base.min_len_s)


def evaluate_scores(score_fn, store: Store, decode_cfg: DecodeConfig, oracle: bool = False) -> dict:
    """score_fn(emb, meta) -> per-unit scores; evaluated after decoding."""
    preds = []
    for i in range(len(store)):
        e, m = store.get(i)
        cfg = oracle_config(m["chapter_starts"], m["duration"], decode_cfg) if oracle else decode_cfg
        s = score_fn(e, m)
        preds.append(Prediction(m["vid"], m["duration"], m["unit_starts"], m["chapter_starts"],
                                decode(s, m["unit_starts"], m["duration"], cfg)))
    return evaluate(preds)


def evaluate_store(model, store: Store, device: str, decode_cfg: DecodeConfig, oracle: bool = False) -> dict:
    return evaluate_scores(lambda e, m: predict_scores(model, e, m["unit_starts"], m["duration"], device),
                           store, decode_cfg, oracle)


def train(train_store: Store, val_store: Store, cfg: TrainConfig, out_dir: Path, device: str,
          decode_cfg: DecodeConfig = DecodeConfig()) -> dict:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = BoundaryTagger(train_store.dim, cfg.proj, cfg.hidden, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(train_store) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device), reduction="none")
    history, best = [], None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for idxs in length_batches(train_store, cfg.batch_size, rng):
            emb, pos, lengths, y, mask = make_batch(train_store, idxs, cfg, device)
            loss = (bce(model(emb, pos, lengths), y) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = evaluate_store(model, val_store, device, decode_cfg)
        val_oracle = evaluate_store(model, val_store, device, decode_cfg, oracle=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1), **val,
               "oracle_F1@5s": val_oracle["F1@5s"], "oracle_F1@3s": val_oracle["F1@3s"],
               "oracle_tIoU-F1": val_oracle["tIoU-F1"]}
        history.append(row)
        print(json.dumps(row))
        if best is None or val["F1@5s"] > best["F1@5s"]:
            best = row
            torch.save({"state_dict": model.state_dict(), "model_config": model.config,
                        "train_config": asdict(cfg), "val": val}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"best": best, "history": history}
