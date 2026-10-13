"""Generates 04_titles.ipynb (run from experiments/m5-chapters/)."""
from nb import build

CELLS = [
    ("md", """
# 04: Title model: mT5-small (embeddings frozen)

**Task:** chapter text (the ASR sentences inside the chapter) → chapter title, in the same language as the text.

**Why frozen embeddings:** measured on the 1060 (research §5), full fine-tuning OOMs even at batch 1. Freezing the shared embedding and the LM head leaves ~44 M trainable params and peaks at ~1.8 GB (batch 4 × 256 tokens).

**Pairs:**
- from the `train` split: English capped at `EN_PAIRS`, and **all** Spanish **oversampled ×`ES_OVERSAMPLE`**, because only ~900 Spanish pairs exist (see the 01 log);
- validation on `val` (en) and `es_val` (es), with ground-truth boundaries. This is VidChapters' "title generation with GT boundaries" task.

**Reference point:** an extractive baseline (`lead-8`, the first 8 words of the chapter). The paper's number (Vid2Seq speech-only: CIDEr 120.5, B1 23.5) was measured on a different test set with PTB tokenisation, so it is **not** directly comparable.
"""),
    ("code", """
import json, random, time
from pathlib import Path
import pandas as pd, torch
from dotenv import load_dotenv
load_dotenv('../.env')
from lib import data as d, vidchapters as vc, titles as T

ROOT = d.data_root(); RUN = Path('runs/2026-09-28-titles'); RUN.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / 'checkpoints' / 'titles'
EN_PAIRS, VAL_EN, VAL_ES = 60000, 500, 300
ES_OVERSAMPLE = 8  # only ~900 Spanish pairs exist (01_data); keep mT5 answering in Spanish
CFG = T.TitleConfig(max_input_tokens=256, batch_size=4, grad_accum=4, lr=1e-3, epochs=1, eval_every=2000)
DEVICE = 'cuda'
rng = random.Random(0)
def pairs_for(split): return T.make_pairs(list(vc.load_jsonl(ROOT / f'interim/{split}.jsonl')))
train_all = pairs_for('train')
en = [p for p in train_all if p['lang'] == 'en']; es = [p for p in train_all if p['lang'] == 'es']
rng.shuffle(en)
train_pairs = en[:EN_PAIRS] + es * ES_OVERSAMPLE
val_en = rng.sample(pairs_for('val'), VAL_EN); val_es = pairs_for('es_val'); val_es = rng.sample(val_es, min(VAL_ES, len(val_es)))
print(f'train pairs: {len(train_pairs)} (en {min(len(en), EN_PAIRS)} of {len(en)}, es {len(es)} x{ES_OVERSAMPLE}); val en {len(val_en)}, val es {len(val_es)}')
pd.DataFrame(train_pairs[:5])[['lang', 'title', 'text']].assign(text=lambda x: x.text.str[:120])
"""),
    ("md", "## 1. Extractive reference: the chapter's first 8 words"),
    ("code", """
lead = lambda ps: [' '.join(p['text'].split()[:8]) for p in ps]
ref_scores = {name: T.score([p['title'] for p in ps], lead(ps)) for name, ps in (('val_en', val_en), ('val_es', val_es))}
pd.DataFrame(ref_scores).T.round(2)
"""),
    ("md", "## 2. Fine-tune (checkpoint = best CIDEr on validation, en + es mixed)"),
    ("code", """
tok, model = T.load_model(DEVICE)
t0 = time.time()
history = T.train_titles(tok, model, train_pairs, val_en + val_es, CFG, CKPT, DEVICE)
print(f'{(time.time()-t0)/60:.0f} min')
pd.DataFrame([{k: v for k, v in h.items() if k != 'samples'} for h in history])
"""),
    ("md", "## 3. Best checkpoint per language, next to the extractive reference"),
    ("code", """
tok, model = T.load_model(DEVICE, CKPT / 'best')
rows, examples = [], {}
for name, ps in (('val_en', val_en), ('val_es', val_es)):
    hyps = T.generate(tok, model, [p['text'] for p in ps], DEVICE, CFG)
    rows.append(dict(set=name, method='mt5-small', **T.score([p['title'] for p in ps], hyps)))
    rows.append(dict(set=name, method='lead-8', **ref_scores[name]))
    examples[name] = pd.DataFrame(dict(reference=[p['title'] for p in ps], generated=hyps, text=[p['text'][:100] for p in ps]))
pd.DataFrame(rows).round(2)
"""),
    ("code", "examples['val_en'].head(15)"),
    ("code", "examples['val_es'].head(15)"),
    ("code", """
json.dump(dict(cfg=CFG.__dict__, en_pairs=EN_PAIRS, es_oversample=ES_OVERSAMPLE, n_train=len(train_pairs), n_val_en=len(val_en), n_val_es=len(val_es)),
          open(RUN / 'config.json', 'w'), indent=1)
json.dump(dict(results=rows, history=[{k: v for k, v in h.items() if k != 'samples'} for h in history]),
          open(RUN / 'metrics.json', 'w'), indent=1)
for k, df in examples.items(): df.to_csv(RUN / f'examples_{k}.csv', index=False)
print('saved to', RUN)
"""),
]

if __name__ == "__main__":
    build(CELLS, "04_titles.ipynb")
