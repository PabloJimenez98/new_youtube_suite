"""Export the measured M16 stage A numbers for the How it works page.

    python tools/make_frontend_data.py [--runs ~/yts-experiments/m16-llm/runs]
        [--eval ~/yts-experiments/m16-llm/eval] [--out ../../frontend/src/app/how-it-works/own-llm-data.ts]

Reads only measured artefacts: each run's metrics.jsonl, eval/<label>/gates.json and
configs/stage_a.yaml (the parameter count is computed from the config, the same way as
llm/model.py builds the model: 100,682,496 for stage A, checked against the PyTorch model).
Writes a generated TypeScript module; never edit that file by hand, re-run this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent

ABLATIONS = [
    ("ablation-a1-baseline", "Baseline (lr 1e-3)"),
    ("ablation-a1b-lr2e-3", "Learning rate 2e-3"),
    ("ablation-a2-muon", "Muon optimizer"),
    ("ablation-a3-qknorm", "QK-norm"),
    ("ablation-a4-relu2", "ReLU² MLP"),
    ("ablation-a5-deep", "Deeper, thinner"),
    ("ablation-a5b-wide", "Wider, shallower"),
    ("ablation-a6-cosine", "Cosine schedule"),
]
# The 1B scaling point is the lr 2e-3 ablation: same recipe as A2, 1B tokens.
SCALING = [("scaling-0.5b", 0.5), ("ablation-a1b-lr2e-3", 1.0), ("scaling-2b", 2.0), ("stage-a-110m", 5.0)]


def events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def params(cfg: dict) -> dict:
    m = cfg["model"]
    d, L, V = m["d_model"], m["n_layers"], m["vocab_size"]
    hd = d // m["n_heads"]
    kv = m.get("n_kv_heads", m["n_heads"]) * hd
    hidden = m.get("ffn_hidden") or int(8 * d / 3) // 64 * 64
    attn = L * (2 * d * d + 2 * d * kv)
    ffn = L * 3 * d * hidden
    norms = L * 2 * d + d
    emb = V * d
    return {"total": emb + attn + ffn + norms, "embedding": emb, "attention": attn, "ffn": ffn, "norms": norms,
            "hidden": hidden, "headDim": hd, "kvHeads": m.get("n_kv_heads", m["n_heads"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/yts-experiments/m16-llm/runs")
    ap.add_argument("--eval", default="~/yts-experiments/m16-llm/eval")
    ap.add_argument("--out", default=str(HERE.parent.parent / "frontend/src/app/how-it-works/own-llm-data.ts"))
    a = ap.parse_args()
    runs, evals = Path(a.runs).expanduser(), Path(a.eval).expanduser()
    cfg = yaml.safe_load((HERE / "configs/stage_a.yaml").read_text())

    pre = events(runs / "stage-a-110m")
    train = [e for e in pre if e["event"] == "train"]
    step = max(1, len(train) // 120)
    pretrain = {
        "train": [[r(e["tokens"] / 1e9, 3), r(e["loss"], 3)] for e in train[::step]] + [[r(train[-1]["tokens"] / 1e9, 3), r(train[-1]["loss"], 3)]],
        "lr": [[r(e["tokens"] / 1e9, 3), float(f'{e["lr"]:.3g}')] for e in train[::step]],
        "val": [[r(e["tokens"] / 1e9, 3), r(e["val_bpb"]), r(e["val_bpb_es"]), r(e["val_bpb_en"])] for e in pre if e["event"] == "eval"],
        "tokS": round(sum(e["tok_s"] for e in train[5:]) / len(train[5:])),
    }

    def final_bpb(name: str) -> tuple[float, int]:
        ev = events(runs / name)
        last = [e for e in ev if e["event"] == "eval"][-1]
        tr = [e for e in ev if e["event"] == "train"][5:]
        return r(last["val_bpb"]), round(sum(e["tok_s"] for e in tr) / len(tr))

    base, _ = final_bpb(ABLATIONS[0][0])
    ablations = []
    for name, label in ABLATIONS:
        bpb, tok_s = final_bpb(name)
        ablations.append({"id": name.removeprefix("ablation-"), "label": label, "bpb": bpb, "delta": r(bpb - base), "tokS": tok_s})
    scaling = [{"tokensB": t, "bpb": final_bpb(n)[0]} for n, t in SCALING]

    def curve(name: str, key: str) -> list[list[float]]:
        return [[r(e["tokens"] / 1e6, 1), r(e[key])] for e in events(runs / name) if e["event"] == "eval"]

    post = {
        "sftGeneral": curve("a3-sft-general", "val_loss_sft"),
        "sftTasks": curve("a3-sft-tasks-e1", "val_loss_sft"),
        "dpoLoss": curve("a3-dpo", "val_loss_dpo"),
        "dpoAcc": curve("a3-dpo", "val_pref_acc"),
        "tokens": {n: [e for e in events(runs / n) if e["event"] == "done"][-1]["tokens"] for n in ("a3-sft-general", "a3-sft-tasks-e1", "a3-dpo")},
    }

    gates = {}
    for label in ("own-sft", "own-dpo"):
        g = json.loads((evals / label / "gates.json").read_text())
        gates[label] = {
            "episodes": g["episodes"],
            "measured": g["measured"],
            "titlesFallback": g["titles"]["fallback_rate"],
            "titlesLang": g["titles"]["lang_ok_rate"],
            "shownotesValid": g["shownotes"]["valid_rate"],
            "relevanceValid": g["relevance"]["valid_rate"],
            "relevanceSpearman": g["relevance"]["spearman_vs_qwen"],
            "qwenPassToPass": g["relevance"]["qwen_pass_to_pass"],
        }

    data = {
        "arch": {**{k: cfg["model"][k] for k in ("vocab_size", "d_model", "n_layers", "n_heads", "rope_theta", "max_seq_len")},
                 "params": params(cfg), "pretrainTokens": cfg["total_tokens"], "lr": cfg["lr"],
                 "batchTokens": cfg["batch_tokens"], "seqLen": cfg["seq_len"], "mix": cfg["data"]["sources"]},
        "pretrain": pretrain,
        "scaling": scaling,
        "ablations": ablations,
        "post": post,
        "gates": gates,
    }
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out = Path(a.out)
    out.write_text(
        "// GENERATED by experiments/m16-own-llm/tools/make_frontend_data.py from the measured\n"
        "// M16 stage A runs (metrics.jsonl, eval/*/gates.json, configs/stage_a.yaml).\n"
        "// Do not edit by hand: re-run the script.\n"
        "/* eslint-disable */\n"
        "// prettier-ignore\n"
        f"export const OWN_LLM_DATA = {body} as const;\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(body):,} chars)")


if __name__ == "__main__":
    main()
