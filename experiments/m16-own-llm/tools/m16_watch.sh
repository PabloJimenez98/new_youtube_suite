#!/bin/bash
# M16 at a glance, A3 phase: trainer, GPU, the A3 chain (tools/a3_chain.sh), every A3 step's state.
# Meant for watch, from WSL:
#   watch -n 10 -t bash ~/projects/new_youtube_suite/experiments/m16-own-llm/tools/m16_watch.sh
# or from Windows: scripts\llm-watch.cmd
# Read-only: it never starts, stops or changes anything.
# (A0/A1/A2 are done: corpus, ablations, scaling, stage A pretraining. Their sections were dropped
#  on 2026-11-18; their results are in NIGHT_LOG.md and notebook 02.)
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
M16=$(cd "$(dirname "$0")/.." && pwd)
L=~/yts-experiments/m16-llm/logs

echo "=== M16 · A3 post-training · $(date '+%F %T') ==="
echo
echo "--- Trainer (llm-train) -----------------------------------------------"
(cd "$M16" && "$P" -m llm.train status 2>&1)
echo
echo "--- GPU ---------------------------------------------------------------"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader 2>/dev/null | awk -F', ' '{printf "VRAM %s / %s · use %s · %s°C · %s\n",$1,$2,$3,$4,$5}'
echo
echo "--- A3 chain (tools/a3_chain.sh: evals + DPO pairs) -------------------"
echo "state     : $(pgrep -f 'a3_chain.sh' >/dev/null && echo running || echo 'not running')"
grep -v '^\s*$' "$L/a3_chain.log" 2>/dev/null | grep -E '^20[0-9]{2}-|^\[[0-9]+/|^(train|val) ' | tail -2 | cut -c1-150 \
  | sed 's/^/last      : /'
echo
echo "--- Disk --------------------------------------------------------------"
df -h ~ | awk 'NR==2 {printf "WSL ext4: %s used, %s free (%s)\n",$3,$4,$5}'
echo
python3 - <<'PY'
import json, time
from pathlib import Path

H = Path("~/yts-experiments/m16-llm").expanduser()
runs, sft, ev = H / "runs", H / "data" / "sft", H / "eval"
APP = Path("~/projects/new_youtube_suite/backend/models/llm").expanduser()
N_TEST = 49

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def row(state, label, info=""):
    print(f"  [{state:4s}] {label:34s} {info}")

def run(name, label):
    s = load(runs / name / "status.json")
    if not s:
        return row("", label, "pending")
    if s.get("state") == "done":
        v = {k: v for k, v in s.items() if k.startswith("val_") and isinstance(v, float)}
        return row("done", label, " ".join(f"{k[4:]}={x:.4f}" for k, x in v.items()))
    eta = s.get("eta_hours")
    row("run", label, f"{s.get('tokens', 0) / s['total_tokens']:.0%} · {s.get('state')}"
        + (f" · ETA {eta * 60:.0f} min" if eta else ""))

def evaluation(label, text, total=N_TEST):
    d = ev / label
    n = len([p for p in d.glob("*.json") if p.name not in ("run.json", "gates.json")]) if d.exists() else 0
    g = load(d / "gates.json")
    if g:
        parts = []
        for t in ("titles", "shownotes", "relevance"):
            if t in g:
                parts.append(f"{t} {'PASS' if g[t]['pass'] else 'fail'}")
        return row("done", text, " · ".join(parts))
    row("run" if 0 < n < total else ("done" if n >= total else ""), text, f"{n}/{total} episodes")

print("--- A3 steps ----------------------------------------------------------")
run("a3-sft-general", "1 SFT general")
run("a3-sft-tasks-e1", "2 SFT tasks (1 epoch)")
evaluation("own-sft", "3 eval SFT model (test split)")
evaluation("qwen", "4 Qwen relevance reference")
st = load(sft / "dpo_tasks" / "pairs.stats.json")
pairs = sft / "dpo_tasks" / "pairs.jsonl"
if st:
    n = sum(v for k, v in st["stats"].items() if k.endswith("_pairs"))
    row("done", "5 DPO pairs (SFT model samples)", f"{n:,} pairs")
elif pairs.exists():
    row("run", "5 DPO pairs (SFT model samples)", f"{sum(1 for _ in open(pairs)):,} pairs so far")
else:
    row("", "5 DPO pairs (SFT model samples)", "pending")
idx = load(sft / "dpo" / "index.json")
row("done" if idx and idx.get("has_ref") else ("run" if idx else ""), "6 DPO data + reference log-probs",
    f"{idx['pairs']:,} train · {idx['val_pairs']:,} val pairs" if idx else "pending")
run("a3-dpo", "7 DPO")
evaluation("own-dpo", "8 eval DPO model (test split)")
cur = load(APP / "current" / "manifest.json")
row("done" if cur else "", "9 installed in the app", cur.get("label", "") if cur else "pending (models/llm/current)")
print("  then: notebooks 06-08 · regression pass · C1: you try it in the app")
PY
