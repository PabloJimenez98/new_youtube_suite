"""BiLSTM distillation from the teacher's two components (docs/decisions/0024 §3).

Episodes in, a trained `RelevanceBiLSTM` out. One episode is one sequence;
batches of 8 episodes are padded and masked. Early stopping on the validation
Spearman of the combined score against the teacher's. A small grid over the
smoothness weight lambda, 3 seeds each: the best lambda by mean validation
Spearman, then that lambda's median seed.
"""

import copy
import logging
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from app.relevance.bilstm import RelevanceBiLSTM, combine_scores, distillation_loss, predict
from app.relevance.dataset import Episode
from app.relevance.evaluation import spearman

log = logging.getLogger("app.relevance")


@dataclass(frozen=True)
class TrainConfig:
    proj: int = 128
    hidden: int = 128
    layers: int = 1
    dropout: float = 0.3
    head: int = 64
    lr: float = 1e-3
    batch_episodes: int = 8
    max_epochs: int = 200
    patience: int = 40  # i1's 10 stopped most seeds on a flat curve (0024 Results, i1)
    huber_delta: float = 0.1
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.5)
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    combine_mode: str = "heads"  # how validation forms the score; see bilstm.combine_scores


@dataclass
class Run:
    lam: float
    seed: int
    val_spearman: float
    epochs: int
    model: RelevanceBiLSTM | None = field(default=None, repr=False)


@dataclass
class TrainOutcome:
    model: RelevanceBiLSTM
    lam: float
    seed: int
    val_spearman: float
    runs: list[Run]
    config: TrainConfig

    def summary(self) -> dict:
        return {
            "lambda": self.lam,
            "seed": self.seed,
            "val_spearman": self.val_spearman,
            "runs": [{"lambda": r.lam, "seed": r.seed, "val_spearman": r.val_spearman, "epochs": r.epochs}
                     for r in self.runs],
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.config).items()},
        }


def _batch(episodes: list[Episode], device: str):
    lengths = torch.tensor([len(e.windows) for e in episodes])
    t, d = int(lengths.max()), episodes[0].emb.shape[1]
    emb = torch.zeros(len(episodes), t, d)
    h = torch.zeros(len(episodes), t)
    l = torch.full((len(episodes), t), float("nan"))  # noqa: E741
    mask = torch.zeros(len(episodes), t)
    for i, e in enumerate(episodes):
        n = len(e.windows)
        emb[i, :n] = torch.from_numpy(e.emb)
        h[i, :n] = torch.from_numpy(e.heuristic)
        l[i, :n] = torch.from_numpy(e.llm)
        mask[i, :n] = 1
    return emb.to(device), lengths, h.to(device), l.to(device), mask.to(device)


def val_spearman(model: RelevanceBiLSTM, episodes: list[Episode], device: str, combine_mode: str = "heads") -> float:
    rhos = []
    for e in episodes:
        hh, ll = predict(model, e.emb, device)
        _, s = combine_scores(combine_mode, hh, ll, e.heuristic)
        rhos.append(spearman(s, e.target))
    return float(np.mean(rhos)) if rhos else 0.0


def train_one(train: list[Episode], val: list[Episode], lam: float, seed: int, cfg: TrainConfig) -> Run:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    emb_dim = train[0].emb.shape[1]
    model = RelevanceBiLSTM(emb_dim=emb_dim, proj=cfg.proj, hidden=cfg.hidden, layers=cfg.layers,
                            dropout=cfg.dropout, head=cfg.head).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_state, best_epoch, since = -2.0, None, 0, 0
    order = list(range(len(train)))
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_episodes):
            emb, lengths, h, l, mask = _batch([train[j] for j in order[i : i + cfg.batch_episodes]], cfg.device)  # noqa: E741
            h_hat, l_hat = model(emb, lengths)
            loss = distillation_loss(h_hat, l_hat, h, l, mask, lam, delta=cfg.huber_delta)
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = val_spearman(model, val or train, cfg.device, cfg.combine_mode)
        if score > best:
            best, best_state, best_epoch, since = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1
            if since >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("relevance train: lambda=%s seed=%s best val spearman %.4f at epoch %s (stopped %s)",
             lam, seed, best, best_epoch, epoch)
    return Run(lam, seed, best, best_epoch, model.eval())


def train_grid(train: list[Episode], val: list[Episode], cfg: TrainConfig) -> TrainOutcome:
    if not train:
        raise ValueError("No training episodes")
    if any(e.emb is None for e in train + val):
        raise ValueError("Every episode needs its embeddings (Episode.emb) before training")
    runs = [train_one(train, val, lam, seed, cfg) for lam in cfg.lambdas for seed in cfg.seeds]
    best_lam = max(cfg.lambdas, key=lambda lam: (np.mean([r.val_spearman for r in runs if r.lam == lam]), -lam))
    of_lam = sorted((r for r in runs if r.lam == best_lam), key=lambda r: (r.val_spearman, r.seed))
    chosen = of_lam[(len(of_lam) - 1) // 2]  # the median seed (the lower one for an even count)
    for r in runs:
        if r is not chosen:
            r.model = None
    return TrainOutcome(chosen.model, best_lam, chosen.seed, chosen.val_spearman, runs, cfg)
