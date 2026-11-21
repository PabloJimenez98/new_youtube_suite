"""Write the A1 scaling configs (0025 §3) on the winning recipe: stage_a.yaml at 0.5B and 2B tokens.

    python configs/make_scaling.py   # -> configs/scaling/*.yaml

Same pattern as make_ablations.py: same data prefix, seed, warmup and eval cadence, only the
token budget changes. The 1B point is ablation-a1b-lr2e-3, which already has this exact recipe
(notebook 02), so it isn't run again. Only the last checkpoint is kept and no permanent copies
(the owner prefers deleting intermediates).
"""

from pathlib import Path

import yaml

from make_ablations import base, merged

HERE = Path(__file__).resolve().parent
POINTS = {"scaling-0.5b": 500_000_000, "scaling-2b": 2_000_000_000}

if __name__ == "__main__":
    out = HERE / "scaling"
    out.mkdir(exist_ok=True)
    for name, tokens in POINTS.items():
        cfg = merged({"total_tokens": tokens})
        cfg["name"] = name
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(out / f"{name}.yaml")
