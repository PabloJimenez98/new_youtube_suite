"""Generates 05_eval.ipynb (run from experiments/m5-chapters/)."""
from nb import build

CELLS = [
    ("md", """
# 05: Final evaluation and export

This is the only notebook that touches the **test sets**:
- `test`: English, VidChapters' official split
- `es_test`: Spanish, held out by channel

Each one is evaluated **once**, with what was selected on validation in 03 and 04.

1. **Boundaries:** BiLSTM vs baseline, in the product setting and with the oracle count, broken down by duration bucket.
2. **Titles:** mT5 vs `lead-8` with ground-truth boundaries.
3. **Owner's QA:** the real 64-min episode, with both boundary sets side by side and the generated titles.
4. **Export:** the bundle goes to `backend/models/chapters/<version>` and `current`. Its boundary `kind` follows the **shipping rule decided on validation in 03**, not the test result.
"""),
    ("code", """
import json, subprocess, time, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from dotenv import load_dotenv
load_dotenv('../.env')
from lib import data as d, vidchapters as vc, titles as T
from lib.units import UnitConfig, Atom, build_units
from lib.encode import Encoder
from lib.store import Store
from lib.build_store import build_store
from lib.baseline import boundary_scores
from lib.decoding import DecodeConfig, decode
from lib.train import evaluate_scores, predict_scores
from lib.model import BoundaryTagger
from lib.export import export_bundle

ROOT = d.data_root(); RUN = Path('runs/2026-09-28-eval'); RUN.mkdir(parents=True, exist_ok=True)
b03 = json.load(open('runs/2026-09-28-boundaries/metrics.json')); c03 = json.load(open('runs/2026-09-28-boundaries/config.json'))
ENCODER, UNIT = c03['encoder'], UnitConfig(**c03['unit'])
PRODUCT, WINDOW = DecodeConfig(**c03['decode_product']), c03['baseline_window_units']
KIND = 'bilstm' if b03['bilstm_ships'] else 'baseline'
BEST = b03['best']; CKPT = ROOT / 'checkpoints' / 'boundaries' / BEST / 'best.pt'
TITLES = ROOT / 'checkpoints' / 'titles' / 'best'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STORES = ROOT / 'embeddings' / f'{ENCODER}-sentence'
print(f'shipping (decided on val in 03): {KIND}; BiLSTM config {BEST}')
"""),
    ("md", "## 1. Boundaries on test / es_test"),
    ("code", """
enc = None
for split in ('test', 'es_test'):
    if not (STORES / split / 'index.json').exists():
        enc = enc or Encoder(ENCODER, device=DEVICE)
        build_store(list(vc.load_jsonl(ROOT / f'interim/{split}.jsonl')), enc, UNIT, STORES / split)
del enc; torch.cuda.empty_cache()
stores = {s: Store(STORES / s) for s in ('test', 'es_test')}
ck = torch.load(CKPT); model = BoundaryTagger(**ck['model_config']); model.load_state_dict(ck['state_dict']); model.to(DEVICE)
methods = {'baseline': lambda e, m: boundary_scores(e, window=WINDOW),
           'bilstm': lambda e, m: predict_scores(model, e, m['unit_starts'], m['duration'], DEVICE)}
rows = []
for s, st in stores.items():
    for oracle in (False, True):
        for name, fn in methods.items():
            rows.append(dict(split=s, oracle=oracle, method=name, **evaluate_scores(fn, st, PRODUCT, oracle)))
test_res = pd.DataFrame(rows)
test_res[['split', 'oracle', 'method', 'F1@5s', 'F1@3s', 'F1@30s', 'tIoU-F1', 'mean_tIoU', 'Pk', 'WindowDiff', 'count_MAE']]
"""),
    ("code", """
# By duration bucket (test, product setting)
def bucket(m): return '15-30' if m['duration'] < 1800 else '30-60' if m['duration'] < 3600 else '60-120'
class Sub:  # a Store view over a subset of indices
    def __init__(self, st, idx): self.st, self.idx, self.dim = st, idx, st.dim
    def __len__(self): return len(self.idx)
    def get(self, i): return self.st.get(self.idx[i])
    @property
    def videos(self): return [self.st.videos[i] for i in self.idx]
rows = []
for s, st in stores.items():
    for b in ('15-30', '30-60', '60-120'):
        idx = [i for i, v in enumerate(st.videos) if bucket(v) == b]
        if len(idx) < 10: continue
        for name, fn in methods.items():
            r = evaluate_scores(fn, Sub(st, idx), PRODUCT)
            rows.append(dict(split=s, bucket=b, n=len(idx), method=name, F1_5s=r['F1@5s'], F1_30s=r['F1@30s'], tIoU_F1=r['tIoU-F1']))
pd.DataFrame(rows).pivot_table(index=['split', 'bucket', 'n'], columns='method', values=['F1_5s', 'F1_30s']).round(2)
"""),
    ("md", "## 2. Titles on test (ground-truth boundaries)"),
    ("code", """
tok, tmodel = T.load_model(DEVICE, TITLES); tcfg = T.TitleConfig()
rng = random.Random(1); title_rows, examples = [], {}
for s in ('test', 'es_test'):
    ps = T.make_pairs(list(vc.load_jsonl(ROOT / f'interim/{s}.jsonl'))); ps = rng.sample(ps, min(800, len(ps)))
    hyps = T.generate(tok, tmodel, [p['text'] for p in ps], DEVICE, tcfg)
    title_rows.append(dict(set=s, method='mt5-small', n=len(ps), **T.score([p['title'] for p in ps], hyps)))
    title_rows.append(dict(set=s, method='lead-8', n=len(ps), **T.score([p['title'] for p in ps], [' '.join(p['text'].split()[:8]) for p in ps])))
    examples[s] = pd.DataFrame(dict(reference=[p['title'] for p in ps], generated=hyps))
del tmodel; torch.cuda.empty_cache()
pd.DataFrame(title_rows).round(2)
"""),
    ("md", """
## 3. Owner's QA: the real 64-min episode (upload 8 in the dev DB)

The transcript is read straight from Postgres (read-only). Words go through `build_units`, exactly as in the backend handler.
"""),
    ("code", """
UPLOAD_ID = 8
q = f'''select coalesce(json_agg(json_build_object('s', w.start_seconds, 'e', w.end_seconds, 't', w.word)
        order by s.segment_index, w.word_index), '[]') from transcript_words w join transcript_segments s on s.id = w.segment_id
        where s.upload_id = {UPLOAD_ID}'''
out = subprocess.run(['docker', 'exec', 'new_youtube_suite-db-1', 'psql', '-U', 'youtube_suite', '-d', 'youtube_suite', '-At', '-c', q],
                     capture_output=True, text=True, check=True).stdout
words = json.loads(out); atoms = [Atom(w['s'], w['e'], w['t']) for w in words]
duration = max(w['e'] or 0 for w in words)
units = build_units(atoms, UNIT); starts = [u.start for u in units]
emb = Encoder(ENCODER, device='cpu')([u.text for u in units])
ep = {}
for name, fn in methods.items():
    s = fn(emb, {'unit_starts': starts, 'duration': duration}) if name == 'baseline' else predict_scores(model.cpu(), emb, starts, duration, 'cpu')
    ep[name] = decode(s, starts, duration, PRODUCT)
tok, tmodel = T.load_model('cpu', TITLES)
def titles_for(cs):
    ends = [*cs[1:], duration]
    return T.generate(tok, tmodel, [' '.join(u.text for u in units if a <= u.start < b)[:2000] for a, b in zip(cs, ends)], 'cpu', tcfg)
fmt = lambda t: f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}'
for name, cs in ep.items():
    print(f'--- {name} ({len(cs)} chapters)')
    for t, title in zip(cs, titles_for(cs)): print(fmt(t), title)
"""),
    ("md", "## 4. Export the bundle"),
    ("code", """
version = f'{KIND}-e5s-sentence-{time.strftime("%Y%m%d")}+mt5s'
metrics = dict(val=[r for r in b03['comparison'] if r['split'] in ('val', 'es_val')],
               test=json.loads(test_res.to_json(orient='records')), titles_test=title_rows)
out = export_bundle(Path('../../backend/models/chapters'), version, KIND, ENCODER, UNIT, PRODUCT, WINDOW,
                    CKPT if KIND == 'bilstm' else None, TITLES,
                    dict(max_input_tokens=tcfg.max_input_tokens, max_target_tokens=tcfg.max_target_tokens, num_beams=4),
                    metrics, str(RUN))
print('exported', out, '->', sorted(p.name for p in out.iterdir()))
"""),
    ("code", """
json.dump(dict(kind=KIND, bilstm_config=BEST, version=version, boundaries_test=json.loads(test_res.to_json(orient='records')),
               titles_test=title_rows, episode={k: v for k, v in ep.items()}), open(RUN / 'metrics.json', 'w'), indent=1)
for k, df in examples.items(): df.to_csv(RUN / f'title_examples_{k}.csv', index=False)
print('saved to', RUN)
"""),
]

if __name__ == "__main__":
    build(CELLS, "05_eval.ipynb")
