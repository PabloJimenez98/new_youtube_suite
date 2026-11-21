"""Write notebooks 06 (post-training), 07 (evaluation) and 08 (export + app install) for A3.

They only read what the runs wrote (runs/*/metrics.jsonl, samples.md, data/sft/*, eval/*); they
are executed afterwards with `jupyter nbconvert --execute --inplace`, so every number in them
is a measured one.

    python tools/make_a3_notebooks.py
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

COMMON = """import json, statistics
from pathlib import Path
from llm import HOME
RUNS, SFT, EVAL = HOME / 'runs', HOME / 'data' / 'sft', HOME / 'eval'
def load(p): return json.loads(Path(p).read_text())
def metrics(run): return [json.loads(l) for l in (RUNS / run / 'metrics.jsonl').read_text().splitlines()]
"""

nb06 = nbf.v4.new_notebook()
nb06.cells = [
    md("""# M16 · 06 — Post-training (A3): general SFT, task SFT, DPO

Decision: `docs/decisions/0025-own-llm-m16.md` §9. Every step starts from the previous one's
`final_model.pt`; the first from the A2 model (`stage-a-110m`, 5B tokens, val bpb 0.9629).
Configs: `configs/a3/`. Log of what was run and why: `NIGHT_LOG.md` (2026-11-18).

1. Data: what went into each step.
2. General SFT (`a3-sft-general`): validation loss.
3. Task SFT: the 3-epoch run that overfit (`a3-sft-tasks`) and the 1-epoch rerun (`a3-sft-tasks-e1`).
4. DPO (`a3-dpo`): the pairs, validation loss and preference accuracy.
5. Fixed-prompt samples before and after each step."""),
    code(COMMON),
    md("## 1. Data"),
    code("""g = load(SFT / 'packed-general' / 'index.json'); t = load(SFT / 'packed-tasks' / 'index.json')
ts = load(SFT / 'tasks' / 'stats.json')
print(f"general SFT: {g['rows']:,} rows x {g['seq_len']} · {g['examples']:,} examples · {g['target_tokens']/1e6:.1f}M loss tokens · fill {g['fill']:.0%}")
print(f"task SFT   : {t['rows']:,} rows · {t['examples']:,} examples · {t['target_tokens']/1e6:.2f}M loss tokens · fill {t['fill']:.0%}")
print('task examples by split/task:', {k: v for k, v in ts.items() if not k.startswith('dropped')})
print('sources of the task SFT build:', json.dumps(t['sources'], indent=1))"""),
    md("""Replay in the task SFT is ~15 % of the **loss** tokens, not of the rows: the task prompts are
long transcripts with short answers, so 15 % of the rows from general SFT would have been ~69 %
of what the loss sees (measured on a first build, NIGHT_LOG 2026-11-18 18:40)."""),
    md("## 2–4. Validation curves"),
    code("""import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, runs, key, title in ((axes[0], ['a3-sft-general'], 'val_loss_sft', 'general SFT'),
                             (axes[1], ['a3-sft-tasks', 'a3-sft-tasks-e1'], 'val_loss_sft', 'task SFT'),
                             (axes[2], ['a3-dpo'], 'val_loss_dpo', 'DPO')):
    for r in runs:
        if not (RUNS / r / 'metrics.jsonl').exists():
            continue
        ev = [m for m in metrics(r) if m['event'] == 'eval' and key in m and not m.get('final')]
        ax.plot([m['tokens'] / 1e6 for m in ev], [m[key] for m in ev], marker='o', label=r)
    ax.set_title(title); ax.set_xlabel('tokens (M)'); ax.set_ylabel(key); ax.legend()
plt.tight_layout(); plt.show()
for r in ('a3-sft-general', 'a3-sft-tasks', 'a3-sft-tasks-e1', 'a3-dpo'):
    if (RUNS / r / 'status.json').exists():
        s = load(RUNS / r / 'status.json')
        print(r, s['state'], {k: round(v, 4) for k, v in s.items() if k.startswith('val_') and isinstance(v, float)})"""),
    md("""The 3-epoch task SFT reached its best validation loss after about one epoch and then rose: it
overfit the 1.3M task loss tokens. It was discarded and rerun at one epoch (`a3-sft-tasks-e1`),
which is the model DPO starts from."""),
    md("## DPO pairs"),
    code("""ps = load(SFT / 'dpo_tasks' / 'pairs.stats.json')
print(json.dumps(ps['stats'], indent=1))
print('why our samples were rejected:'); print(json.dumps(ps['reasons'], indent=1, ensure_ascii=False))
pairs = [json.loads(l) for l in (SFT / 'dpo_tasks' / 'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
for task in ('title', 'shownotes_reduce', 'relevance'):
    p = next((x for x in pairs if x['task'] == task), None)
    if p:
        print(f"\\n=== {task} ({p['reason']}) ===\\nCHOSEN  : {p['chosen'][:400]}\\nREJECTED: {p['rejected'][:400]}")"""),
    md("## 5. Samples (greedy, fixed prompts, as the trainer writes them)"),
    code("""for r in ('a3-sft-general', 'a3-sft-tasks-e1', 'a3-dpo'):
    p = RUNS / r / 'samples.md'
    if p.exists():
        last = p.read_text(encoding='utf-8').split('\\n## ')[-1]
        print(f'##### {r} — {last[:1500]}\\n')"""),
    md("""## Conclusions

- General SFT turned the base model into one that follows the ChatML turn structure and answers in
  the asked language, with heavy repetition (val loss 1.534 → 1.259 over one epoch).
- Task SFT: one epoch is the right length for 1.3M task loss tokens (val 1.437); three overfit.
- DPO on 1,509 verifier-built pairs: val DPO loss 0.692 → 0.472, preference accuracy 0.85 on 20 val
  pairs. Its effect on the app's tools is measured in notebook 07: show notes improve, relevance does not."""),
]

nb07 = nbf.v4.new_notebook()
nb07.cells = [
    md("""# M16 · 07 — Evaluation: the §10 functional gates on the frozen test split

`python -m llm.evaluate` runs the **app's own pipelines** (chapter titles, show notes, zero-shot
relevance) on the 49 test episodes of the task corpus (two held-out channels, `poco-se-habla` and
`lenny-s-podcast`, plus the 0024 test episodes). The transcripts are the corpus ones (Whisper),
and none of these episodes were used in training.

- `own-sft`: our model after task SFT. `own-dpo`: after DPO.
- `qwen`: Qwen3.5-9B relevance, the reference. Its two passes give Qwen's pass-to-pass Spearman.

Gates (selectable): titles ≤ 10 % mT5 fallback and ≥ 95 % right language; show notes ≥ 95 % valid
within 3 attempts and ≥ 95 % right language; relevance 100 % valid batches and Spearman with Qwen
≥ 0.8 × Qwen's pass-to-pass Spearman. Failing a gate does not hide the model in the app (0025
amendment): it is reported here and at C1."""),
    code(COMMON + "from llm.evaluate import gates, spearman, mean_passes"),
    code("""rows = {}
for label in ('own-sft', 'own-dpo'):
    if (EVAL / label).exists():
        rows[label] = gates(label)
        print(json.dumps(rows[label], indent=1))"""),
    md("### Per channel"),
    code("""def per_channel(label):
    out = {}
    for p in sorted((EVAL / label).glob('*.json')):
        if p.name in ('run.json', 'gates.json'): continue
        e = load(p); c = out.setdefault(e['channel'], {'eps': 0, 'chapters': 0, 'fb': 0, 'sn_ok': 0, 'rho': []})
        c['eps'] += 1
        c['chapters'] += len(e['titles']['chapters']); c['fb'] += sum(x['fallback'] for x in e['titles']['chapters'])
        c['sn_ok'] += e['shownotes']['ok']
        q = EVAL / 'qwen' / p.name
        if q.exists():
            r = spearman(mean_passes(e['relevance']['passes']), mean_passes(load(q)['relevance']['passes']))
            if r is not None: c['rho'].append(r)
    for ch, c in out.items():
        print(f"{label:8s} {ch:22s} eps {c['eps']:2d} · title fallback {c['fb']}/{c['chapters']} · "
              f"show notes valid {c['sn_ok']}/{c['eps']} · relevance rho vs Qwen "
              f"{statistics.mean(c['rho']) if c['rho'] else float('nan'):.3f}")
for label in rows: per_channel(label)"""),
    md("### Why show notes failed"),
    code("""from collections import Counter
import re
for label in rows:
    errs = Counter()
    for p in (EVAL / label).glob('*.json'):
        if p.name in ('run.json', 'gates.json'): continue
        s = load(p)['shownotes']
        if not s['ok']:
            inner = s['error'].split(' attempts (', 1)[-1].rstrip(')')
            for part in inner.split('; '):
                errs[re.sub(r'\\d+', '#', part)[:80]] += 1
    print(label, dict(errs.most_common()))"""),
    md("### Side by side: one episode per test channel"),
    code("""for label in rows:
    seen = set()
    for p in sorted((EVAL / label).glob('*.json')):
        if p.name in ('run.json', 'gates.json'): continue
        e = load(p)
        if e['channel'] in seen: continue
        seen.add(e['channel'])
        print(f"== {label} · {e['channel']} · {e['id']} ({e['lang']})")
        for c in e['titles']['chapters'][:6]: print('   title:', c['title'])
        s = e['shownotes']
        print('   show notes:', 'OK' if s['ok'] else 'FAILED', json.dumps(s.get('draft', {}).get('title_suggestions'), ensure_ascii=False))
        if s['ok']: print('   summary:', s['draft']['summary'][:300])"""),
    md("### Qwen's chapter titles for the same episodes (from the corpus run), for comparison"),
    code("""from llm.corpus import CORPUS
for p in sorted((EVAL / 'own-sft').glob('*.json'))[:4]:
    if p.name in ('run.json', 'gates.json'): continue
    e = load(p); q = load(CORPUS / 'processed' / e['id'] / 'chapters.json')
    print(f"== {e['channel']} {e['id']}")
    for a, b in zip(e['titles']['chapters'], q): print(f"   own : {a['title']}\\n   qwen: {b['title']}")"""),
    md("""## Conclusions (measured 2026-11-19, 49 test episodes)

| gate | own-sft | own-dpo | threshold |
|---|---|---|---|
| titles: mT5 fallback | 0 % | 0 % | ≤ 10 % |
| titles: right language | 99.6 % | 98.4 % | ≥ 95 % |
| show notes: valid in ≤ 3 attempts | 57 % | **86 %** | ≥ 95 % |
| relevance: valid batches | 100 % | 100 % | 100 % |
| relevance: Spearman vs Qwen | 0.211 | −0.073 | ≥ 0.8 × 0.490 = 0.392 |

- **Titles pass** for both. They are in the right language and on topic, but repetitive
  ("Holded Wallet y Holded Wallet…") and less specific than Qwen's.
- **Show notes fail the gate.** DPO took them from 57 % to 86 % valid; what still fails is too few
  distinct tags or titles, and a few truncated JSON strings.
- **Relevance fails, and DPO made it worse.** Both models emit smooth, content-independent shapes
  (ramps, arches) learnt from the form of Qwen's score lists, not from the blocks' content. The SFT
  model correlates with Qwen only on the English channel (0.41 on lenny-s-podcast).
- Qwen's own pass-to-pass Spearman is only 0.49, so the reference is noisy too.

**Installed in the app: `yts-own-110m-dpo`** (`models/llm/current`). It is the better of the two on
the tool that differs (show notes) and ties on titles; relevance fails with either. Per the 0025
amendment the model stays selectable despite the failed gates, and Qwen stays the default."""),
]

nb08 = nbf.v4.new_notebook()
nb08.cells = [
    md("""# M16 · 08 — GGUF export and installation in the app

`python -m llm.export_gguf` writes an f16 GGUF (architecture `llama`, our BPE tokenizer with the
`llama-bpe` pre-tokenizer, ChatML template, EOS `<|im_end|>`). It goes into
`backend/models/llm/<label>/` with a `manifest.json` (`file`, `n_ctx`, `label`); the app reads
`LLM_OWN_DIR` (default `models/llm/current`, a symlink to the chosen export). Qwen stays the
global default; our model is selectable in Settings and per tool (0025 §11 and amendment)."""),
    code(COMMON + """APP = Path('~/projects/new_youtube_suite/backend/models/llm').expanduser()
for d in sorted(APP.iterdir()):
    if (d / 'manifest.json').exists():
        m = load(d / 'manifest.json'); f = d / m['file']
        print(f"{d.name:22s} {'-> ' + str(d.resolve().name) if d.is_symlink() else '':24s} {f.stat().st_size/1e6:7.1f} MB  {m}")"""),
    md("### Parity: PyTorch vs llama.cpp on the same prompt"),
    code("""import torch, os, sys
from llm import config as cfgmod
from llm.model import Decoder
from llm.tokenizer import load_tokenizer
from llm.export_gguf import preload_llama_cpp
cur = APP / 'current'; man = load(cur / 'manifest.json')
run = {'yts-own-110m-sft': 'a3-sft-tasks-e1', 'yts-own-110m-dpo': 'a3-dpo'}[man['label']]
ck = torch.load(RUNS / run / 'final_model.pt', map_location='cpu', weights_only=False)
model = Decoder(cfgmod.from_dict(ck['config']).model); model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}); model.eval()
tok = load_tokenizer(HOME / 'tokenizer' / 'tokenizer.json')
prompt = '<|im_start|>user\\nResume en una frase: el episodio habla de cómo empezar un podcast.<|im_end|>\\n<|im_start|>assistant\\n'
ids = tok.encode(prompt).ids
with torch.no_grad(): ref = model(torch.tensor([ids]))[0, -1].float()
preload_llama_cpp()
from llama_cpp import Llama
llm = Llama(model_path=str(cur / man['file']), n_ctx=512, logits_all=True, verbose=False, n_gpu_layers=0)
llm.eval(ids); got = torch.tensor(llm.scores[len(ids) - 1])
print('top-5 torch    :', ref.topk(5).indices.tolist()); print('top-5 llama.cpp:', got.topk(5).indices.tolist())
print('max |logit diff|:', float((ref - got).abs().max()))"""),
]

for name, nb in (("06_post_training.ipynb", nb06), ("07_evaluation.ipynb", nb07), ("08_export_and_app.ipynb", nb08)):
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, name)
    print("wrote", name)
