"""Write the A1 ablation configs (0025 §3) from stage_a.yaml: same data, seed and 1B-token budget.

    python configs/make_ablations.py   # -> configs/ablations/*.yaml

Each run differs from the baseline in exactly one thing. They keep only their last
checkpoint and no permanent copies (the owner prefers deleting intermediates).
Ablation 7 (Gated DeltaNet hybrid) is added only if its kernels run on sm_120
(notebook 02).
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
base = yaml.safe_load((HERE / "stage_a.yaml").read_text(encoding="utf-8"))
base.update(
    total_tokens=1_000_000_000,
    warmup_tokens=50_000_000,
    eval_every_tokens=100_000_000,
    eval_batches=20,
    keep_last=1,
    permanent_every_fraction=2.0,  # never
    sample_prompts=["El podcast de esta semana habla de", "In this episode, we talk about"],
)

VARIANTS = {
    "a1-baseline": {},
    "a1b-lr2e-3": {"lr": 0.002},  # LR sensitivity of the baseline, so a win isn't just a better LR
    "a2-muon": {"optimizer": "muon"},
    "a3-qknorm": {"model": {"qk_norm": True}},
    "a4-relu2": {"model": {"mlp": "relu2"}},
    # Equal non-embedding parameters (~76M) as the baseline's d768/L12:
    "a5-deep": {"model": {"d_model": 576, "n_layers": 21, "n_heads": 9, "n_kv_heads": 3}},
    "a5b-wide": {"model": {"d_model": 1024, "n_layers": 7, "n_heads": 16, "n_kv_heads": 4}},
    "a6-cosine": {"schedule": "cosine"},
}


def merged(over: dict) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    for k, v in over.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


if __name__ == "__main__":
    out = HERE / "ablations"
    out.mkdir(exist_ok=True)
    for name, over in VARIANTS.items():
        cfg = merged(over)
        cfg["name"] = f"ablation-{name}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(out / f"{name}.yaml")
