"""Optimizers (0025 §3): AdamW, and Muon for 2-D hidden matrices + AdamW for the rest.

Muon (Keller Jordan, modded-nanogpt) orthogonalises each hidden weight's
momentum with a quintic Newton-Schulz iteration before the update. The
embedding (tied with the output head) and the norm weights stay on AdamW.

`build_optimizer` returns one object with `param_groups`, `step`, `zero_grad`,
`state_dict` and `load_state_dict`, so the trainer and its checkpoints don't
care which one is in use. Every group carries `base_lr`; the trainer sets
`lr = base_lr * schedule_factor` (WSD).
"""

import torch


def zeropower_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = st["momentum_buffer"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                v = zeropower_newtonschulz5(u, g["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(v.to(p.dtype), alpha=-g["lr"] * scale)


class Combined:
    """Several optimizers behind one interface (for checkpoints and the LR schedule)."""

    def __init__(self, **opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts.values() for g in o.param_groups]

    def step(self):
        for o in self.opts.values():
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts.values():
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {k: o.state_dict() for k, o in self.opts.items()}

    def load_state_dict(self, d):
        for k, o in self.opts.items():
            o.load_state_dict(d[k])


def build_optimizer(model, cfg, device_type: str) -> Combined:
    fused = device_type == "cuda"
    adam_kw = dict(betas=(cfg.beta1, cfg.beta2), fused=fused)
    emb = model.token_embd.weight
    if cfg.optimizer == "adamw":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        adam = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay, "base_lr": cfg.lr},
                                  {"params": no_decay, "weight_decay": 0.0, "base_lr": cfg.lr}],
                                 lr=cfg.lr, **adam_kw)
        return Combined(adamw=adam)
    if cfg.optimizer == "muon":
        hidden = [p for n, p in model.named_parameters() if p.dim() == 2 and p is not emb]
        norms = [p for p in model.parameters() if p.dim() < 2]
        adam = torch.optim.AdamW([{"params": [emb], "weight_decay": cfg.weight_decay, "base_lr": cfg.lr},
                                  {"params": norms, "weight_decay": 0.0, "base_lr": cfg.lr}],
                                 lr=cfg.lr, **adam_kw)
        muon = Muon([{"params": hidden, "base_lr": cfg.muon_lr}], lr=cfg.muon_lr, momentum=cfg.muon_momentum,
                    weight_decay=cfg.muon_weight_decay)
        return Combined(adamw=adam, muon=muon)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
