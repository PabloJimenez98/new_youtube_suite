"""Generates 06_titles_llm.ipynb (run from experiments/m5-chapters/)."""
from nb import build

CELLS = [
    ("md", """
# 06: Titles with local pretrained LLMs

**Why:** mT5's Spanish titles were not usable on the owner's episode (05). The owner approved comparing local pretrained instruction LLMs as a replacement. Plan, candidates and selection rule are in the README ("plan for 06").

**Setup:**
- llama.cpp (`llama-cpp-python`, CPU build), GGUF Q4_K_M
- one call per chapter; the same prompt for every model (`lib/llm_titles.py`)
- greedy decoding, one sampled retry, then fallback to mT5

**Where things are evaluated** (so model selection never looks at test):
1. **Validation sample**, for the gates: 150 `val` (en) + all `es_val` (es) pairs, with ground-truth boundaries. Reported: B1/B4/CIDEr, language accuracy, fallback rate, repetition, length, CPU speed.
2. **The owner's episode:** titles from every method for the 6 BiLSTM chapters, written to a **blind sheet** (shuffled, unlabeled). The key is in a separate file.
3. **Test:** only the winner, next to mT5, on the same 800/620-pair sample as 05. Run once, after the owner picks, in the last section.
"""),
    ("code", """
import json, random, time, subprocess, os
from pathlib import Path
import pandas as pd, psutil
from dotenv import load_dotenv
load_dotenv('../.env')
from lib import data as d, vidchapters as vc, titles as T, llm_titles as L
from lib.units import Atom

ROOT = d.data_root(); RUN = Path('runs/2026-09-28-titles-llm'); RUN.mkdir(parents=True, exist_ok=True)
LLM_DIR = ROOT / 'llm'
MODELS = {
    'qwen3.5-2b': 'Qwen3.5-2B-Q4_K_M.gguf',
    'qwen3.5-4b': 'Qwen3.5-4B-Q4_K_M.gguf',
    'gemma4-e2b': 'gemma-4-E2B-it-Q4_K_M.gguf',
    'gemma4-e4b': 'gemma-4-E4B-it-Q4_K_M.gguf',
    'salamandra-7b': 'salamandra-7b-instruct-2606.Q4_K_M.gguf',
    'qwen3.5-9b': 'Qwen3.5-9B-Q4_K_M.gguf',
}
THREADS = 6            # physical cores; production-realistic, CPU only
MAX_WORDS = 900        # chapter text budget per call (~1.2-1.5k tokens)
lid = d.LanguageId(ROOT.parent / 'models' / 'lid.176.ftz')
MT5 = ROOT / 'checkpoints' / 'titles' / 'best'
def wait_for(path, poll=60):
    # the GGUFs download in MODELS order; hf_hub_download only renames into place when complete
    while not path.exists(): time.sleep(poll)
    return path

def full_pairs(split):
    # like T.make_pairs, but without the 2000-char cap: the LLMs get the whole chapter (sample_text caps it)
    out = []
    for v in vc.load_jsonl(ROOT / f'interim/{split}.jsonl'):
        for i, raw in enumerate(v.chapter_titles):
            title, text = T.clean_title(raw), T.chapter_text(v, i, max_chars=10**7)
            if title and len(text) >= 40:
                out.append({'vid': v.vid, 'lang': v.lang, 'i': i, 'text': text, 'title': title})
    return out
rng = random.Random(2)
val_en = rng.sample(full_pairs('val'), 150)
val_es = full_pairs('es_val')
SAMPLE = {'val_en': val_en, 'val_es': val_es}
print({k: len(v) for k, v in SAMPLE.items()}, 'median words/chapter:',
      {k: int(pd.Series([len(p['text'].split()) for p in v]).median()) for k, v in SAMPLE.items()})
"""),
    ("md", "## 1. References on the same sample: mT5 (as shipped in 05) and `lead-8`"),
    ("code", """
import torch
tok, mt5 = T.load_model('cuda' if torch.cuda.is_available() else 'cpu', MT5)
hyps = {}
for name, ps in SAMPLE.items():
    hyps[('mt5-small', name)] = T.generate(tok, mt5, [p['text'][:2000] for p in ps], mt5.device.type)
    hyps[('lead-8', name)] = [' '.join(p['text'].split()[:8]) for p in ps]
del mt5; torch.cuda.empty_cache()
"""),
    ("md", """
## 2. Every LLM on the validation sample (CPU only)

Results are cached per model in `RUN/gen_<model>.jsonl`, so a crash or a restart resumes where it stopped.
"""),
    ("code", """
proc = psutil.Process()
speed = {}
for key, fname in MODELS.items():
    cache = RUN / f'gen_{key}.jsonl'
    done = [json.loads(l) for l in cache.open()] if cache.exists() else []
    todo = [(n, j) for n, ps in SAMPLE.items() for j in range(len(ps))][len(done):]
    if todo:
        wait_for(LLM_DIR / fname); t0 = time.time(); m = L.LlamaTitler(LLM_DIR / fname, n_threads=THREADS, max_words=MAX_WORDS); load_s = time.time() - t0
        with cache.open('a') as f:
            for n, j in todo:
                p = SAMPLE[n][j]; r = m.title(p['text'], p['lang'], lid=lid)
                row = dict(set=n, j=j, title=r.title, raw=r.raw, seconds=r.seconds, prompt_tokens=r.prompt_tokens,
                           attempts=r.attempts, rss_gb=proc.memory_info().rss / 1e9, load_s=load_s)
                f.write(json.dumps(row, ensure_ascii=False) + '\\n'); f.flush(); done.append(row)
        del m
    df = pd.DataFrame(done)
    speed[key] = dict(s_per_chapter=df.seconds.median(), prompt_tokens=df.prompt_tokens.median(),
                      rss_gb=df.rss_gb.max(), load_s=df.load_s.iloc[0])
    for n in SAMPLE:
        sub = df[df.set == n].sort_values('j')
        fb = [t if t else h for t, h in zip(sub.title, hyps[('mt5-small', n)])]   # fallback to mT5, as in production
        hyps[(key, n)] = fb
        speed[key][f'fallback_{n}'] = float(sub.title.isna().mean())
    print(key, {k: round(v, 2) for k, v in speed[key].items()}, flush=True)
"""),
    ("md", "## 3. Validation metrics and gates"),
    ("code", """
def raw_repeats(t):
    return L.collapse_repeats(t) != ' '.join(t.split())
rows = []
for (method, n), hs in hyps.items():
    ps = SAMPLE[n]
    lang = ps[0]['lang']
    rows.append(dict(method=method, set=n, **T.score([p['title'] for p in ps], hs),
                     lang_ok=100 * (1 - sum(L.wrong_language(h, lang, lid) for h in hs) / len(hs)),
                     repeats=100 * sum(raw_repeats(h) for h in hs) / len(hs),
                     words=sum(len(h.split()) for h in hs) / len(hs)))
val = pd.DataFrame(rows)
table = val.pivot(index='method', columns='set', values=['CIDEr', 'B1', 'lang_ok', 'repeats', 'words']).round(1)
sp = pd.DataFrame(speed).T
EP_CHAPTERS = 6
sp['episode_min_est'] = (sp.s_per_chapter * EP_CHAPTERS * 1.0 / 60).round(1)
gates = pd.DataFrame({
    'lang_ok_es>=95': table[('lang_ok', 'val_es')] >= 95,
    'CIDEr_en>=0.8*mt5': table[('CIDEr', 'val_en')] >= 0.8 * table.loc['mt5-small', ('CIDEr', 'val_en')],
}).join(sp.episode_min_est.le(15).rename('cpu<=15min'), how='left')
display(table); display(sp.round(2)); display(gates)
"""),
    ("md", """
## 4. The owner's episode: blind sheet

Chapter starts are the BiLSTM's from 05. Every method titles every chapter, on CPU, with the timings measured for real. The sheet shuffles the candidates per chapter and doesn't name them. The key is written to `blind_key.json`, which the owner shouldn't open before judging.
"""),
    ("code", """
e05 = json.load(open('runs/2026-09-28-eval/metrics.json'))
starts = e05['episode']['bilstm']
UPLOAD_ID = 8
q = f'''select coalesce(json_agg(json_build_object('s', w.start_seconds, 'e', w.end_seconds, 't', w.word)
        order by s.segment_index, w.word_index), '[]') from transcript_words w join transcript_segments s on s.id = w.segment_id
        where s.upload_id = {UPLOAD_ID}'''
out = subprocess.run(['docker', 'exec', 'new_youtube_suite-db-1', 'psql', '-U', 'youtube_suite', '-d', 'youtube_suite', '-At', '-c', q],
                     capture_output=True, text=True, check=True).stdout
words = json.loads(out); duration = max(w['e'] or 0 for w in words)
ends = [*starts[1:], duration]
texts = [' '.join(w['t'].strip() for w in words if a <= w['s'] < b) for a, b in zip(starts, ends)]
ep_lang, conf = lid(' '.join(texts)[:5000]); print('episode language:', ep_lang, round(conf, 2), '| words/chapter:', [len(t.split()) for t in texts])
ep = {}
tok, mt5 = T.load_model('cpu', MT5); ep['mt5-small'] = dict(titles=T.generate(tok, mt5, [t[:2000] for t in texts], 'cpu'), seconds=None); del mt5
for key, fname in MODELS.items():
    m = L.LlamaTitler(LLM_DIR / fname, n_threads=THREADS, max_words=MAX_WORDS)
    t0, prev, titles, raws = time.time(), [], [], []
    for t in texts:
        r = m.title(t, ep_lang, prev, lid=lid); titles.append(r.title); raws.append(r.raw)
        if r.title: prev.append(r.title)
    ep[key] = dict(titles=titles, raw=raws, seconds=time.time() - t0)
    del m
    print(key, f"{ep[key]['seconds'] / 60:.1f} min", titles, flush=True)
fmt = lambda t: f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}'
rb = random.Random(7); key_map, lines = {}, ['# Blind review: chapter titles for upload 8', '',
    'For each chapter, rank the candidates or mark the ones you would publish. Candidates are shuffled per chapter.', '']
for c, (s, t) in enumerate(zip(starts, texts)):
    cands = [(meth, v['titles'][c] or '(no title)') for meth, v in ep.items()]
    rb.shuffle(cands); letters = 'ABCDEFGHIJ'
    key_map[c] = {letters[k]: meth for k, (meth, _) in enumerate(cands)}
    lines += [f'## Chapter {c + 1}, {fmt(s)}', '', f'> {t[:400]}…', '']
    lines += [f'- **{letters[k]}.** {title}' for k, (_, title) in enumerate(cands)] + ['']
(RUN / 'blind_sheet.md').write_text('\\n'.join(lines))
(RUN / 'blind_key.json').write_text(json.dumps(key_map, indent=1))
print('blind sheet ->', RUN / 'blind_sheet.md')
"""),
    ("code", """
json.dump(dict(models=MODELS, threads=THREADS, max_words=MAX_WORDS, n_val_en=len(val_en), n_val_es=len(val_es),
               prompt_version='v1'), open(RUN / 'config.json', 'w'), indent=1)
json.dump(dict(val=json.loads(val.to_json(orient='records')), speed=speed,
               episode={k: dict(seconds=v['seconds']) for k, v in ep.items()}),
          open(RUN / 'metrics.json', 'w'), indent=1)
print('saved to', RUN)
"""),
]

if __name__ == "__main__":
    build(CELLS, "06_titles_llm.ipynb")
