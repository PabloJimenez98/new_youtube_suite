"""Write the stage B notebooks (09-11): WRITTEN, NOT EXECUTED (0025 amendment, 2026-11-15).

They are deliberately left without outputs: stage B is out of M16's scope, so nothing in them is a
result. Each starts with a banner saying so. They reuse the A1-A3 code paths (llm.train,
llm.sft_data, llm.dpo_pairs, llm.evaluate) so that running stage B later is a matter of
executing them in order.

    python tools/make_stage_b_notebooks.py
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

BANNER = md("""> **⚠ NOT EXECUTED.** Stage B is out of M16's scope (decision 0025, amendment of 2026-11-15).
> This notebook was written so the work can be resumed later. It has **never been run**, and there
> are **no results** in it. Do not read anything here as a measurement.""")

COMMON = """import json
from pathlib import Path
from llm import HOME
RUNS, SFT, EVAL = HOME / 'runs', HOME / 'data' / 'sft', HOME / 'eval'
def load(p): return json.loads(Path(p).read_text())
def metrics(run): return [json.loads(l) for l in (RUNS / run / 'metrics.jsonl').read_text().splitlines()]"""

nb09 = nbf.v4.new_notebook()
nb09.cells = [
    md("# M16 · 09 — B0: stage B size and budget from the A1 scaling runs"),
    BANNER,
    md("""B0 (0025 §1): fit loss-vs-tokens on the three stage A scaling points (0.5B, 1B = `ablation-a1b-lr2e-3`,
2B) plus A2's 5B, extrapolate to 20–30B tokens at ~330M, and propose a budget for the owner (C2).
The measured points themselves are in notebook 02 and NIGHT_LOG; this notebook only fits them."""),
    code(COMMON + """
import numpy as np
pts = {}
for run in ('scaling-0.5b', 'ablation-a1b-lr2e-3', 'scaling-2b', 'stage-a-110m'):
    s = load(RUNS / run / 'status.json')
    pts[s['tokens']] = s['val_bpb']
D = np.array(sorted(pts)); L = np.array([pts[d] for d in D])
print(dict(zip(D, L)))"""),
    code("""# L(D) = E + B / D^beta, fitted on the 110M points (data-scaling at fixed size).
from scipy.optimize import curve_fit
f = lambda d, E, B, beta: E + B / (d / 1e9) ** beta
(E, B, beta), _ = curve_fit(f, D, L, p0=(0.8, 0.3, 0.5), maxfev=20000)
print(f'E={E:.4f} B={B:.4f} beta={beta:.3f}')
for d in (10e9, 20e9, 25e9, 30e9):
    print(f'110M at {d/1e9:.0f}B tokens: bpb {f(d, E, B, beta):.4f}')"""),
    md("""Size step 110M → 330M: with only one model size measured, the size term cannot be fitted; B0
would use the A1 depth/width runs as a sanity check and a Chinchilla-style prior for it, and
report the uncertainty to the owner. Wall clock at 18.6k tok/s (notebook 00):"""),
    code("""for d in (20e9, 25e9, 30e9):
    print(f'{d/1e9:.0f}B tokens: {d / 18_600 / 86400:.1f} days of GPU')"""),
]

nb10 = nbf.v4.new_notebook()
nb10.cells = [
    md("# M16 · 10 — B1: stage B pretraining monitor"),
    BANNER,
    md("""B1: `llm-train start configs\\stage_b.yaml` (pause/resume with `llm-pause` / `llm-resume`, as in
stage A). Before it: re-fetch and tokenize the Spanish FineWeb2-HQ text deleted in the amendment
(`prepare_data fetch --source es`, `prepare_data tokenize --source es`). This notebook plots the
run's curves and samples while it trains."""),
    code(COMMON + """
import matplotlib.pyplot as plt
m = metrics('stage-b-330m')
tr = [x for x in m if x['event'] == 'train']; ev = [x for x in m if x['event'] == 'eval']
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot([x['tokens'] / 1e9 for x in tr], [x['loss'] for x in tr]); ax[0].set_title('train loss')
for k in ('val_bpb', 'val_bpb_es', 'val_bpb_en'):
    ax[1].plot([x['tokens'] / 1e9 for x in ev], [x[k] for x in ev], marker='o', label=k)
ax[1].legend(); ax[1].set_title('val bpb'); plt.show()"""),
    code("""print((RUNS / 'stage-b-330m' / 'samples.md').read_text(encoding='utf-8').split('\\n## ')[-1][:2000])"""),
]

nb11 = nbf.v4.new_notebook()
nb11.cells = [
    md("# M16 · 11 — B2: stage B post-training and evaluation"),
    BANNER,
    md("""B2 repeats A3 (notebooks 06–08) on the stage B model, with what A3 taught:
- general SFT 1 epoch; task SFT **1 epoch** (3 overfit at 110M), replay ~15 % of the loss tokens;
- DPO pairs from `llm.dpo_pairs` on the stage B SFT model, reduce prompts sampled ≥ 4 times;
- relevance needs more than DPO on "disagrees with teacher" pairs: at 110M both SFT and DPO models
  produced content-independent score shapes (notebook 07). A per-block prompt or a scoring head
  would be the first thing to try;
- gates with `llm.evaluate` on the same 49 test episodes, then **C3**: blind review per task."""),
    code("""# Commands, in order (each one is what A3 ran, with stage B paths):
STEPS = '''
llm-train start configs\\\\b2\\\\sft_general.yaml        # init_from runs/stage-b-330m/final_model.pt
llm-train start configs\\\\b2\\\\sft_tasks.yaml          # 1 epoch, init_from the general SFT
python -m llm.export_gguf runs/b2-sft-tasks/final_model.pt tokenizer/tokenizer.json <app>/models/llm/yts-own-330m-sft/yts-own-330m-sft.gguf
python -m llm.dpo_pairs build --own-dir <app>/models/llm/yts-own-330m-sft --per-task 1000
python -m llm.dpo_pairs build --own-dir <app>/models/llm/yts-own-330m-sft --tasks shownotes_reduce --repeats 3 --out .../pairs_reduce_x3.jsonl
python -m llm.dpo build --out data/sft/dpo-b .../pairs.jsonl .../pairs_reduce_x3.jsonl
python -m llm.dpo ref data/sft/dpo-b runs/b2-sft-tasks/final_model.pt
llm-train start configs\\\\b2\\\\dpo.yaml
python -m llm.evaluate run --model own --label own-b-dpo --own-dir <app>/models/llm/yts-own-330m-dpo
python -m llm.evaluate gates --label own-b-dpo
'''
print(STEPS)"""),
    code(COMMON + """
from llm.evaluate import gates
for label in ('own-dpo', 'own-b-sft', 'own-b-dpo'):   # stage A DPO model as the baseline
    if (EVAL / label).exists():
        print(json.dumps(gates(label), indent=1))"""),
]

for name, nb in (("09_stage_b_sizing.ipynb", nb09), ("10_stage_b_pretraining.ipynb", nb10),
                 ("11_stage_b_post_training_eval.ipynb", nb11)):
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "m16_not_executed": True}
    nbf.write(nb, name)
    print("wrote", name, "(not executed)")
