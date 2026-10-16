"""M9: llama.cpp CPU tuning on a real relevance batch (docs/research/m9-performance.md).

For each (n_threads, n_batch) config: load the GGUF, then time one
prefill-only call (max_tokens=1) and one full grammar-constrained call
on the same prompt (a batch of `BLOCKS` 15 s blocks from a real episode).

    cd backend && .venv/bin/python ../experiments/m9-perf/bench_llm.py WORDS.tsv OUT.json BLOCKS T:B[:G] [T:B[:G] ...]   # G = layers on the GPU (CUDA build)

WORDS.tsv: start<TAB>end<TAB>word<TAB>segment_start (dumped from the DB).
"""

import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar

from app.chapters.llm_titles import render_prompt
from app.llm import runtime
from app.relevance.llm_scoring import llm_blocks, messages, schema
from app.relevance.windows import Word, build_blocks

GGUF = Path("models/chapters/current/llm/Qwen3.5-4B-Q4_K_M.gguf")


def main() -> None:
    words_tsv, out_path, n_blocks, *configs = sys.argv[1:]
    n_blocks = int(n_blocks)
    words = []
    for line in open(words_tsv):
        a, _b, w, _s = line.rstrip("\n").split("\t")
        words.append(Word(float(a) if a else None, w))
    blocks = build_blocks(words, max(w.start or 0 for w in words) + 1)
    texts = [t for t, _ in llm_blocks(blocks) if t][100 : 100 + n_blocks]
    msgs = messages(texts, "es", [None] * len(texts))
    results = json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    for cfg in configs:
        threads, n_batch, gpu = ([int(x) for x in cfg.split(":")] + [0])[:3]
        llm = Llama(model_path=str(GGUF), n_ctx=8192, n_threads=threads, n_threads_batch=threads,
                    n_batch=n_batch, n_ubatch=n_batch, n_gpu_layers=gpu, verbose=False)
        template, bos, eos = runtime.chat_tokens(llm)
        ids = llm.tokenize(render_prompt(template, msgs, bos, eos).encode(), add_bos=False, special=True)
        grammar = LlamaGrammar.from_json_schema(json.dumps(schema(len(texts))), verbose=False)
        t0 = time.perf_counter()
        llm.create_completion(ids, max_tokens=1, temperature=0.0)
        prefill = time.perf_counter() - t0
        llm.reset()
        t0 = time.perf_counter()
        out = llm.create_completion(ids, max_tokens=8 * len(texts) + 24, temperature=0.0, grammar=grammar)
        total = time.perf_counter() - t0
        gen = out["usage"]["completion_tokens"]
        res = {"threads": threads, "n_batch": n_batch, "gpu_layers": gpu, "blocks": len(texts), "prompt_tokens": len(ids),
               "prefill_s": round(prefill, 1), "prefill_tok_s": round(len(ids) / prefill, 1),
               "total_s": round(total, 1), "gen_tokens": gen,
               "gen_tok_s": round(gen / max(1e-6, total - prefill), 2)}
        results[f"{cfg}/{len(texts)}"] = res
        print(res, flush=True)
        Path(out_path).write_text(json.dumps(results, indent=2))
        del llm


if __name__ == "__main__":
    main()
