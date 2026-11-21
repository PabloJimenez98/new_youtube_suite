"""Run configuration: YAML in, dataclasses out, plus the hash that guards resume (0025 §7).

A run's `config.yaml` is copied into its run directory at `start`. `resume`
reads that copy. Fields in `RESUME_MUTABLE` may be edited between a pause
and a resume (e.g. extending `total_tokens` while the WSD schedule is still
in its stable phase, 0025 §8). Changing anything else makes `resume` refuse,
because the checkpoint would no longer match the run.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

RESUME_MUTABLE = {
    "total_tokens", "eval_every_tokens", "eval_batches", "autosave_minutes", "log_every_steps",
    "sample_prompts", "sample_tokens", "min_free_vram_gib", "keep_last", "permanent_every_fraction",
}


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int | None = None  # None -> 8/3 * d_model rounded down to a multiple of 64
    rope_theta: float = 100000.0
    qk_norm: bool = False
    norm_eps: float = 1e-5
    init_std: float = 0.02
    max_seq_len: int = 4096  # RoPE tables are precomputed up to this length
    # "swiglu" (3 matrices, hidden 8/3·d) or "relu2" (2 matrices, hidden 4·d: same parameter count).
    # relu2 is ablation 4 (0025 §3); llama.cpp's llama/qwen3 graphs can't run it.
    mlp: str = "swiglu"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def hidden(self) -> int:
        if self.ffn_hidden:
            return self.ffn_hidden
        return 4 * self.d_model if self.mlp == "relu2" else int(8 * self.d_model / 3) // 64 * 64


@dataclass
class DataConfig:
    # "pretrain" (token shards, `root` + `sources`), "sft" (`sft_dir`, llm.sft_data) or "dpo" (`sft_dir`, llm.dpo)
    kind: str = "pretrain"
    root: str = "~/yts-experiments/m16-llm/data/shards"
    # source name (a sub-directory of `root`) -> share of training rows
    sources: dict = field(default_factory=lambda: {"es": 0.7, "en": 0.3})
    sft_dir: str | None = None


@dataclass
class TrainConfig:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: str | None = "~/yts-experiments/m16-llm/tokenizer/tokenizer.json"
    # Weights to start from when the run has no checkpoint yet (post-training: a pretrained
    # final_model.pt or checkpoint). None = random initialisation (pretraining).
    init_from: str | None = None
    # "lm": next-token loss (pretraining, SFT with a loss mask). "dpo": preference loss on pairs
    # (data.kind "dpo", llm.dpo); `init_from` should be the SFT model the reference log-probs came from.
    objective: str = "lm"
    dpo_beta: float = 0.1
    total_tokens: int = 5_000_000_000
    seq_len: int = 2048
    micro_batch: int = 8
    batch_tokens: int = 524_288  # tokens per optimizer step (gradient accumulation fills it)
    # Context extension (0025 §8): from this fraction of the budget on, rows are `long_seq_len`.
    long_from_fraction: float | None = None
    long_seq_len: int = 4096
    long_micro_batch: int = 4
    optimizer: str = "adamw"  # "adamw" | "muon" (Muon on hidden matrices + AdamW on embedding/norms)
    lr: float = 1e-3  # AdamW groups
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    warmup_tokens: int = 200_000_000
    schedule: str = "wsd"  # "wsd" (warmup-stable-decay) or "cosine" (ablation 6)
    decay_fraction: float = 0.2  # WSD: linear decay to min_lr over the last fraction
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    mem_fraction: float = 0.92  # never let the driver spill into system RAM (notebook 00, run 1)
    min_free_vram_gib: float = 9.5
    autosave_minutes: float = 15.0
    keep_last: int = 2
    permanent_every_fraction: float = 0.1
    eval_every_tokens: int = 250_000_000
    eval_batches: int = 20
    log_every_steps: int = 10
    sample_prompts: list = field(default_factory=list)
    sample_tokens: int = 48

    def hash(self) -> str:
        d = asdict(self)
        for k in RESUME_MUTABLE:
            d.pop(k, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True)


def _build(cls, d: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def from_dict(d: dict) -> TrainConfig:
    d = dict(d)
    d["model"] = _build(ModelConfig, d.get("model") or {})
    d["data"] = _build(DataConfig, d.get("data") or {})
    return _build(TrainConfig, d)


def load(path: str | Path) -> TrainConfig:
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def diff_keys(a: TrainConfig, b: TrainConfig) -> list[str]:
    """Top-level (and model./data.) keys that differ and are not resume-mutable."""
    da, db = asdict(a), asdict(b)
    out = []
    for k in da:
        if k in RESUME_MUTABLE:
            continue
        if isinstance(da[k], dict) and isinstance(db.get(k), dict):
            out += [f"{k}.{kk}" for kk in da[k] if da[k][kk] != db[k].get(kk)]
        elif da[k] != db.get(k):
            out.append(k)
    return out
