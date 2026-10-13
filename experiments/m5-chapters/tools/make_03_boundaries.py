"""Generates 03_boundaries.ipynb (run from experiments/m5-chapters/)."""
from nb import build

CELLS = [
    ("md", """
# 03: Boundary model: BiLSTM over frozen embeddings vs. the baseline

**Hypothesis:** a BiLSTM trained on thousands of real chapterings beats the unsupervised TextTiling baseline (02) at locating boundaries.

**Shipping rule** (0009 §7): the BiLSTM ships only if it beats the baseline on **validation F1@5s** in the **product setting** (~6 chapters/h). The oracle-count results are reported too, because the creators' references are much denser than "few and broad" (see 02).

Settings fixed by 02 (preliminary, 300 English videos):
- **Encoder: `e5-small`.** It ties MiniLM (F1@5s 10.8 vs 10.9); e5-small is chosen for its MIT licence.
- **Unit: `sentence`** (2–15 s). It has the highest ceiling: F1@5s 83.3 / F1@3s 72.8, against 70.8 / 47.2 for `block10`. The baseline differences between units were marginal.

Data: `01_data` → `interim/{train,val,es_val}.jsonl`. Section 2 below is also the **official baseline run on our validation split**. It uses the same units and encoder, so `02_baseline.ipynb` is not re-run on `val`; the preliminary 02 already settled units and encoder. Test sets (`test`, `es_test`) are **not touched** until `05_eval`.
"""),
    ("code", """
import json, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from dotenv import load_dotenv
load_dotenv('../.env')
from lib import data as d, vidchapters as vc
from lib.units import UnitConfig
from lib.encode import Encoder
from lib.store import Store
from lib.build_store import build_store
from lib.baseline import boundary_scores
from lib.decoding import DecodeConfig
from lib.train import TrainConfig, train, evaluate_scores, evaluate_store, predict_scores
from lib.model import BoundaryTagger

ROOT = d.data_root()
ENCODER, UNIT = 'e5-small', UnitConfig(mode='sentence', min_s=2, max_s=15)
STORES = ROOT / 'embeddings' / f'{ENCODER}-sentence'
RUN = Path('runs/2026-09-28-boundaries'); RUN.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / 'checkpoints' / 'boundaries'
PRODUCT = DecodeConfig(per_hour=6, min_len_s=60)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE, STORES)
"""),
    ("md", "## 1. Encode train / val / es_val once (cached on the second drive)"),
    ("code", """
enc = None
for split in ('train', 'val', 'es_val'):
    if (STORES / split / 'index.json').exists():
        print(split, 'cached'); continue
    enc = enc or Encoder(ENCODER, device=DEVICE)
    videos = list(vc.load_jsonl(ROOT / f'interim/{split}.jsonl'))
    t0 = time.time(); n = build_store(videos, enc, UNIT, STORES / split)
    print(f'{split}: {len(videos)} videos, {n} units, {time.time()-t0:.0f}s')
del enc; torch.cuda.empty_cache()
stores = {s: Store(STORES / s) for s in ('train', 'val', 'es_val')}
pd.DataFrame([dict(split=s, videos=len(st), units=int(sum(v['n'] for v in st.videos)),
                   positives_per_video=np.mean([len(v['labels']) for v in st.videos]),
                   units_per_positive=sum(v['n'] for v in st.videos) / max(1, sum(len(v['labels']) for v in st.videos)))
              for s, st in stores.items()]).set_index('split').round(1)
"""),
    ("md", "## 2. The bar to beat: the baseline over these same units"),
    ("code", """
WINDOW = 10  # units ≈ 60 s with ~6 s sentences (best/near-best in 02)
base = lambda e, m: boundary_scores(e, window=WINDOW)
baseline = {}
for s in ('val', 'es_val'):
    for oracle in (False, True):
        baseline[s, oracle] = evaluate_scores(base, stores[s], PRODUCT, oracle)
pd.DataFrame(baseline).T.rename_axis(['split', 'oracle'])[['F1@5s', 'F1@3s', 'F1@30s', 'tIoU-F1', 'Pk', 'WindowDiff', 'count_MAE']]
"""),
    ("md", """
## 3. BiLSTM training

A few configurations, 10 epochs each. The best epoch is kept by **validation F1@5s (product setting)**.
"""),
    ("code", """
CONFIGS = {
    'default': TrainConfig(),
    'pos5': TrainConfig(pos_weight=5.0),
    'pos20': TrainConfig(pos_weight=20.0),
    'big': TrainConfig(proj=384, hidden=256),
}
runs = {}
for name, cfg in CONFIGS.items():
    print('=== ', name, cfg)
    t0 = time.time()
    runs[name] = train(stores['train'], stores['val'], cfg, CKPT / name, DEVICE, PRODUCT)
    print(f'{name}: {time.time()-t0:.0f}s, best epoch {runs[name]["best"]["epoch"]}, F1@5s {runs[name]["best"]["F1@5s"]}')
"""),
    ("code", """
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for name, r in runs.items():
    h = pd.DataFrame(r['history'])
    ax[0].plot(h.epoch, h.train_loss, label=name); ax[1].plot(h.epoch, h['F1@5s'], label=name); ax[2].plot(h.epoch, h['oracle_F1@5s'], label=name)
for a, t, key in zip(ax, ('train loss', 'val F1@5s (product)', 'val F1@5s (oracle count)'), (None, (False,), (True,))):
    a.set_title(t); a.set_xlabel('epoch')
    if key: a.axhline(baseline['val', key[0]]['F1@5s'], color='grey', ls='--', label='baseline')
    a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(RUN / 'learning_curves.png', dpi=110); plt.show()
"""),
    ("md", "## 4. Best model vs. baseline: English validation and **Spanish** (cross-lingual transfer)"),
    ("code", """
best_name = max(runs, key=lambda n: runs[n]['best']['F1@5s'])
ck = torch.load(CKPT / best_name / 'best.pt')
model = BoundaryTagger(**ck['model_config']); model.load_state_dict(ck['state_dict']); model.to(DEVICE)
rows = []
for s in ('val', 'es_val'):
    for oracle in (False, True):
        rows.append(dict(split=s, oracle=oracle, method='baseline', **baseline[s, oracle]))
        rows.append(dict(split=s, oracle=oracle, method=f'bilstm:{best_name}', **evaluate_store(model, stores[s], DEVICE, PRODUCT, oracle)))
cmp = pd.DataFrame(rows)
cmp[['split', 'oracle', 'method', 'F1@5s', 'F1@3s', 'F1@30s', 'tIoU-F1', 'mean_tIoU', 'Pk', 'WindowDiff', 'count_MAE']]
"""),
    ("code", """
val_p = cmp[(cmp.split == 'val') & ~cmp.oracle].set_index('method')['F1@5s']
ships = val_p[f'bilstm:{best_name}'] > val_p['baseline']
print(f'Shipping rule, val F1@5s (product): BiLSTM {val_p[f"bilstm:{best_name}"]:.2f} vs baseline {val_p["baseline"]:.2f} -> '
      + ('BiLSTM ships' if ships else 'BASELINE ships'))
"""),
    ("md", "## 5. Examples: scores from both methods on one English and one Spanish video"),
    ("code", """
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5))
for ax, s in zip(axes, ('val', 'es_val')):
    st = stores[s]; i = int(np.argmin([abs(v['duration'] - 45 * 60) for v in st.videos]))
    e, m = st.get(i); t = np.array(m['unit_starts']) / 60
    ax.plot(t, boundary_scores(e, window=WINDOW), lw=.7, label='baseline')
    ax.plot(t, predict_scores(model, e, m['unit_starts'], m['duration'], DEVICE), lw=.9, label='BiLSTM')
    for c in m['chapter_starts']: ax.axvline(c / 60, color='green', alpha=.5)
    ax.set_title(f'{s}: {m["vid"]} (green = creator chapters)'); ax.legend(fontsize=8)
axes[-1].set_xlabel('minutes'); plt.tight_layout(); plt.savefig(RUN / 'examples.png', dpi=110); plt.show()
"""),
    ("code", """
json.dump(dict(encoder=ENCODER, unit=UNIT.__dict__, decode_product=PRODUCT.__dict__, baseline_window_units=WINDOW,
               configs={k: v.__dict__ for k, v in CONFIGS.items()}), open(RUN / 'config.json', 'w'), indent=1)
json.dump(dict(best=best_name, best_epoch=runs[best_name]['best']['epoch'], bilstm_ships=bool(ships),
               comparison=json.loads(cmp.to_json(orient='records')),
               histories={k: r['history'] for k, r in runs.items()}), open(RUN / 'metrics.json', 'w'), indent=1)
print('saved to', RUN)
"""),
]

if __name__ == "__main__":
    build(CELLS, "03_boundaries.ipynb")
